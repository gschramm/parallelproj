# %%
from __future__ import annotations
from parallelproj import Array

import array_api_compat.cupy as xp

import parallelproj
from array_api_compat import to_device
import array_api_compat.numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cupyx.scipy.ndimage import gaussian_filter as gaussian_filter_cp
from scipy.ndimage import gaussian_filter

# choose a device (CPU or CUDA GPU)
if "numpy" in xp.__name__:
    # using numpy, device must be cpu
    dev = "cpu"
elif "cupy" in xp.__name__:
    # using cupy, only cuda devices are possible
    dev = xp.cuda.Device(0)
elif "torch" in xp.__name__:
    # using torch valid choices are 'cpu' or 'cuda'
    if parallelproj.cuda_present:
        dev = "cuda"
    else:
        dev = "cpu"

# %%
gain = 1.0  # gain factor controlling sensitivity -> higher for more counts
fwhm_tof_mm = 30.0
res_fwhm_mm = 4.5

# number of MLEM iterations
num_mlem_updates = 5
num_mlacf_newton_updates = 10
num_outer_iterations = 40
mlacf_update_type: str = (
    "poisson-newton"  # "poisson-newton" or "unweighted-gauss-analytic", "weighted-gauss-analytic"  or "None"
)

# %%
if mlacf_update_type not in [
    "poisson-newton",
    "unweighted-gauss-analytic",
    "weighted-gauss-analytic",
    "None",
]:
    raise ValueError(f"Unknown MLACF update type: {mlacf_update_type}")

# %%
# Setup of the forward model :math:`\bar{y}(x) = A x + s`
# --------------------------------------------------------
#
# We setup a linear forward operator :math:`A` consisting of an
# image-based resolution model, a non-TOF PET projector and an attenuation model
#
# .. note::
#     The MLEM implementation below works with all linear operators that
#     subclass :class:`.LinearOperator` (e.g. the high-level projectors).

num_rings = 1
scanner = parallelproj.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=350.0,
    num_sides=32,
    num_lor_endpoints_per_side=16,
    lor_spacing=4.0,
    ring_positions=xp.asarray([0.0], dtype=xp.float32),
    symmetry_axis=2,
)

# %%
# setup the LOR descriptor that defines the sinogram

img_shape = (250, 250, 1)
voxel_size = (2.0, 2.0, 2.0)

lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=100,
    max_ring_difference=2,
    sinogram_order=parallelproj.SinogramSpatialAxisOrder.RVP,
)

proj = parallelproj.RegularPolygonPETProjector(
    lor_desc, img_shape=img_shape, voxel_size=voxel_size
)

# setup a simple test image containing a few "hot rods"
x_true = xp.ones(proj.in_shape, device=dev, dtype=xp.float32)
c0 = proj.in_shape[0] // 2
c1 = proj.in_shape[1] // 2
x_true[(c0 - 2) : (c0 + 2), (c1 - 2) : (c1 + 2), :] = 5.0
x_true[4, c1, 2:] = 5.0
x_true[c0, 4, :-2] = 5.0

x_true[:60, :, :] = 0
x_true[-60:, :, :] = 0
x_true[:, :20, :] = 0
x_true[:, -20:, :] = 0

# %%
# Attenuation image and sinogram setup
# ------------------------------------

# setup an attenuation image
x_att = 0.01 * xp.astype(x_true > 0, xp.float32)
x_att[110:140, 110:140] = 0.016
x_att[60:90, 60:90] = 0.003

# calculate the attenuation sinogram
att_sino = gain * xp.exp(-proj(x_att))


# %%
# Complete PET forward model setup
# --------------------------------
#
# We combine an image-based resolution model,
# a non-TOF or TOF PET projector and an attenuation model
# into a single linear operator.

sig_tof_mm = fwhm_tof_mm / 2.35
tof_bin_width_mm = sig_tof_mm

num_tofbins = (
    2 * int(1.41 * 0.5 * voxel_size[0] * proj.in_shape[0] / tof_bin_width_mm) + 1
)

proj.tof_parameters = parallelproj.TOFParameters(
    num_tofbins=num_tofbins, tofbin_width=tof_bin_width_mm, sigma_tof=sig_tof_mm
)

att_op = parallelproj.TOFNonTOFElementwiseMultiplicationOperator(
    proj.out_shape, att_sino
)

res_model = parallelproj.GaussianFilterOperator(
    proj.in_shape, sigma=res_fwhm_mm / (2.35 * proj.voxel_size)
)

# compose all 3 operators into a single linear operator
proj = parallelproj.CompositeLinearOperator((proj, res_model))


# %%
# Simulation of projection data
# -----------------------------
#
# We setup an arbitrary ground truth :math:`x_{true}` and simulate
# noise-free and noisy data :math:`y` by adding Poisson noise.

# simulated noise-free data
noise_free_data = att_op(proj(x_true))

# generate a contant contamination sinogram
contamination = 0.75 * gaussian_filter_cp(noise_free_data, sigma=5.0) + 0.1 * float(
    xp.mean(noise_free_data)
)

noise_free_data += contamination

# add Poisson noise
np.random.seed(1)
y = xp.asarray(
    np.random.poisson(parallelproj.to_numpy_array(noise_free_data)),
    device=dev,
    dtype=xp.float64,
)

# %%
# EM update to minimize :math:`f(x)`
# ----------------------------------
#
# The EM update that can be used in MLEM or OSEM is given by cite:p:`Dempster1977` :cite:p:`Shepp1982` :cite:p:`Lange1984` :cite:p:`Hudson1994`
#
# .. math::
#     x^+ = \frac{x}{(A^k)^H 1} (A^k)^H \frac{y^k}{A^k x + s^k}
#
# to calculate the minimizer of :math:`f(x)` iteratively.
#
# To monitor the convergence we calculate the relative cost
#
# .. math::
#    \frac{f(x) - f(x^*)}{|f(x^*)|}
#
# and the distance to the optimal point
#
# .. math::
#    \frac{\|x - x^*\|}{\|x^*\|}.
#
#
# We setup a function that calculates a single MLEM/OSEM
# update given the current solution, a linear forward operator,
# data, contamination and the adjoint of ones.


def em_update(
    x_cur: Array,
    data: Array,
    p: parallelproj.LinearOperator,
    m: parallelproj.LinearOperator | None,
    s: Array,
    adjoint_ones: Array,
) -> Array:
    """EM update

    Parameters
    ----------
    x_cur : Array
        current solution
    data : Array
        data
    p : parallelproj.LinearOperator
        projection operator excluding mult. corrections
    m : parallelproj.LinearOperator
        (non-TOF) multiplicative corrections operator
    s : Array
        contamination
    adjoint_ones : Array
        adjoint of ones

    Returns
    -------
    Array, float
        update and logL at input x (to save computation)
    """

    if m is None:
        ybar = p(x_cur) + s
        res = x_cur * p.adjoint(data / ybar) / adjoint_ones
    else:
        ybar = m(p(x_cur)) + s
        res = x_cur * p.adjoint(m.adjoint(data / ybar)) / adjoint_ones

    logL = float((data * xp.log(ybar) - ybar).sum())
    return res, logL


# %%
# Run the MLEM iterations
# -----------------------

# initialize x
x_mlem = xp.ones(proj.in_shape, dtype=xp.float32, device=dev)
# calculate A^H 1
adjoint_ones_mlem = proj.adjoint(
    att_op.adjoint(xp.ones(proj.out_shape, dtype=xp.float32, device=dev))
)

num_iter = num_outer_iterations * num_mlem_updates

logL_mlem = np.zeros(num_iter, dtype=np.float32)

for i in range(num_iter):
    print(f"MLEM iteration {(i + 1):03} / {num_iter:03}", end="\r")
    x_mlem, logL = em_update(x_mlem, y, proj, att_op, contamination, adjoint_ones_mlem)
    logL_mlem[i] = logL

print()


# %%
# Run the MLEM iterations without accounting for attenuation sinogram
# -------------------------------------------------------------------

# initialize variables
x_mlacf = xp.ones(proj.in_shape, dtype=xp.float32, device=dev)

att_sino_cur = 0 * att_sino + gain
att_sino_init = att_sino_cur.copy()
att_op_cur = parallelproj.TOFNonTOFElementwiseMultiplicationOperator(
    proj.out_shape, att_sino_cur
)
#############################

logL_mlacf = np.zeros(num_outer_iterations * num_mlem_updates, dtype=np.float32)

for i_outer in range(num_outer_iterations):
    print(
        f"Outer MALCF {mlacf_update_type} iteration {(i_outer + 1):03} / {num_outer_iterations:03}",
        end="\r",
    )
    # calculate A^H 1 with current attenuation sinogram
    adjoint_ones_cur = proj.adjoint(
        att_op_cur.adjoint(xp.ones(proj.out_shape, dtype=xp.float32, device=dev))
    )

    # run activity MLEM updates using current attenuation sinogram and adjoint ones
    for i in range(num_mlem_updates):
        x_mlacf, logL = em_update(
            x_mlacf, y, proj, att_op_cur, contamination, adjoint_ones_cur
        )
        logL_mlacf[i_outer * num_mlem_updates + i] = logL

    #############################
    # MLACF attenuation sinogram update

    # TOF projection excluding attenuation
    p_it = proj(x_mlacf)
    # TOF projection excluding attenuation summed over TOF bins
    p_i = p_it.sum(-1)
    # mask for MLACF attenuation sino update
    mask = p_i > (0.01 * p_i.max())

    if mlacf_update_type == "poisson-newton":
        # initialize with the weighted gaussian analytic update

        for i_mlacf in range(num_mlacf_newton_updates):
            ybar = att_op_cur(p_it) + contamination

            # calculate the the function f we want to minimize
            f = p_i - ((y * p_it) / ybar).sum(-1)
            fprime = ((y * (p_it**2)) / (ybar**2)).sum(-1)

            update = xp.zeros_like(f)
            inds = xp.where(fprime != 0)
            update[inds] = f[inds] / fprime[inds]

            # update the attn sino and the attenuation operator
            att_sino_cur -= update
            # clip negative values
            att_sino_cur = xp.clip(att_sino_cur, 0, None)
            att_op_cur = parallelproj.TOFNonTOFElementwiseMultiplicationOperator(
                proj.out_shape, att_sino_cur
            )
    elif mlacf_update_type == "unweighted-gauss-analytic":
        inds = xp.where(mask)
        att_sino_cur[inds] = ((y[inds] - contamination[inds]) * p_it[inds]).sum(-1) / (
            p_it**2
        ).sum(-1)[inds]

        # clip negative values
        att_sino_cur = xp.clip(att_sino_cur, 0, None)

        att_op_cur = parallelproj.TOFNonTOFElementwiseMultiplicationOperator(
            proj.out_shape, att_sino_cur
        )
    elif mlacf_update_type == "weighted-gauss-analytic":
        inds = xp.where(mask)
        att_sino_cur[inds] = (y[inds] - contamination[inds]).sum(-1) / p_i[inds]

        # clip negative values
        att_sino_cur = xp.clip(att_sino_cur, 0, None)

        att_op_cur = parallelproj.TOFNonTOFElementwiseMultiplicationOperator(
            proj.out_shape, att_sino_cur
        )
    elif mlacf_update_type == "None":
        pass
    else:
        raise ValueError(f"Unknown MLACF update type: {mlacf_update_type}")

print()
# %%

x_true_np = parallelproj.to_numpy_array(x_true)
x_mlem_np = parallelproj.to_numpy_array(x_mlem)
x_mlacf_np = parallelproj.to_numpy_array(x_mlacf)
x_att_np = parallelproj.to_numpy_array(x_att)

# rescale the MLACF solution based on total activity
scaling_fac = x_mlem_np.sum() / x_mlacf_np.sum()
x_mlacf_np *= scaling_fac

# load the estimated singram
att_sino_mlacf_np = parallelproj.to_numpy_array(att_sino_cur) / scaling_fac
att_sino_np = parallelproj.to_numpy_array(att_sino)

ps_fwhm_mm = 6.0
x_true_np_smooth = gaussian_filter(
    x_true_np, sigma=ps_fwhm_mm / (2.35 * np.array(voxel_size))
)
x_mlem_np_smooth = gaussian_filter(
    x_mlem_np, sigma=ps_fwhm_mm / (2.35 * np.array(voxel_size))
)
x_mlacf_np_smooth = gaussian_filter(
    x_mlacf_np, sigma=ps_fwhm_mm / (2.35 * np.array(voxel_size))
)

# %%

kws = dict(cmap="Greys", vmin=0, vmax=1.2 * float(x_true_np.max()))

diff_max = 0.1 * x_true_np.max()

sl_z = proj.in_shape[2] // 2

sino_plane = att_sino.shape[2] // 2

fig, ax = plt.subplots(3, 5, figsize=(15, 9), tight_layout=True)
ax[0, 0].imshow(x_true_np[:, :, sl_z], **kws)
ax[0, 0].set_title("true activity img")
ax[1, 0].imshow(x_att_np[:, :, sl_z], cmap="Greys")
ax[1, 0].set_title("true attenuation img")

ax[0, 1].imshow(x_mlem_np[:, :, sl_z], **kws)
ax[0, 1].set_title("MLEM w true attn sino")
ax[1, 1].imshow(x_mlacf_np[:, :, sl_z], **kws)
ax[1, 1].set_title(f"{mlacf_update_type} MLACF")
ax[2, 1].imshow(
    x_mlem_np[:, :, sl_z] - x_mlacf_np[:, :, sl_z],
    cmap="bwr",
    vmin=-diff_max,
    vmax=diff_max,
)
ax[2, 1].set_title("MLEM - MLACF")

ax[0, 2].imshow(x_mlem_np_smooth[:, :, sl_z], **kws)
ax[0, 2].set_title(f"smoothed MLEM w true attn sino")
ax[1, 2].imshow(x_mlacf_np_smooth[:, :, sl_z], **kws)
ax[1, 2].set_title(f"smoothed MLACF")
ax[2, 2].imshow(
    x_mlem_np_smooth[:, :, sl_z] - x_mlacf_np_smooth[:, :, sl_z],
    cmap="bwr",
    vmin=-diff_max,
    vmax=diff_max,
)
ax[2, 2].set_title("sm. MLEM - sm. MLACF")


ax[0, 3].imshow(att_sino_np[:, :, sino_plane].T ** 0.5, cmap="Greys", vmin=0, vmax=1.0)
ax[0, 3].set_title("sqrt(true attn sino)")
ax[1, 3].imshow(
    att_sino_mlacf_np[:, :, sino_plane].T ** 0.5, cmap="Greys", vmin=0, vmax=1.0
)
ax[1, 3].set_title("sqrt(MLACF est. attn sino)")
ax[2, 3].imshow(
    (att_sino_np - att_sino_mlacf_np)[:, :, sino_plane].T,
    cmap="bwr",
    vmin=-0.05,
    vmax=0.05,
)
ax[2, 3].set_title("true - MLACF attn sino")

ax[0, 4].plot(logL_mlem, label=f"logL MLEM {logL_mlem[-1]:.4E}")
ax[0, 4].plot(logL_mlacf, label=f"logL MLACF {logL_mlacf[-1]:.4E}")
pmin = logL_mlem[20:].min()
pmax = max(logL_mlem.max(), logL_mlacf.max())
ax[0, 4].legend(fontsize="small")
ax[0, 4].set_ylim(pmin, pmax)

for i in [(2, 0), (1, 4), (2, 4)]:
    ax[i].set_axis_off()

fig.show()
