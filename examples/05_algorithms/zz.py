# %%
from __future__ import annotations
from parallelproj import Array

import array_api_compat.cupy as xp

import parallelproj
from array_api_compat import to_device
import array_api_compat.numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

# number of MLEM iterations
num_mlem_updates = 10
num_mlacf_newton_updates = 10
num_outer_iterations = 20
mlacf_update_type = "poisson-newton"  # "poisson-newton" or "unweighted-gauss-analytic", "weighted-gauss-analytic"  other options result in no update


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
    radius=300.0,
    num_sides=28,
    num_lor_endpoints_per_side=16,
    lor_spacing=4.0,
    ring_positions=xp.asarray([0.0], dtype=xp.float32),
    symmetry_axis=2,
)

# %%
# setup the LOR descriptor that defines the sinogram

img_shape = (200, 200, 1)
voxel_size = (2.0, 2.0, 2.0)

lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=10,
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

x_true[:50, :, :] = 0
x_true[-50:, :, :] = 0
x_true[:, :20, :] = 0
x_true[:, -20:, :] = 0

# %%
# Attenuation image and sinogram setup
# ------------------------------------

# setup an attenuation image
x_att = 0.01 * xp.astype(x_true > 0, xp.float32)
x_att[110:150, 110:150] = 0.02
x_att[50:90, 50:90] = 0.003

# calculate the attenuation sinogram
att_sino = gain * xp.exp(-proj(x_att))


# %%
# Complete PET forward model setup
# --------------------------------
#
# We combine an image-based resolution model,
# a non-TOF or TOF PET projector and an attenuation model
# into a single linear operator.

proj.tof_parameters = parallelproj.TOFParameters(
    num_tofbins=51, tofbin_width=12.0, sigma_tof=12.0
)

att_op = parallelproj.TOFNonTOFElementwiseMultiplicationOperator(
    proj.out_shape, att_sino
)

res_model = parallelproj.GaussianFilterOperator(
    proj.in_shape, sigma=4.5 / (2.35 * proj.voxel_size)
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
contamination = xp.full(
    noise_free_data.shape,
    0.5 * float(xp.mean(noise_free_data)),
    device=dev,
    dtype=xp.float32,
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
    Array
        _description_
    """

    if m is None:
        ybar = p(x_cur) + s
        res = x_cur * p.adjoint(data / ybar) / adjoint_ones
    else:
        ybar = m(p(x_cur)) + s
        res = x_cur * p.adjoint(m.adjoint(data / ybar)) / adjoint_ones

    return res


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

for i in range(num_iter):
    print(f"MLEM iteration {(i + 1):03} / {num_iter:03}", end="\r")
    x_mlem = em_update(x_mlem, y, proj, att_op, contamination, adjoint_ones_mlem)


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

for i_outer in range(num_outer_iterations):
    print(f"Outer iteration {(i_outer + 1):03} / {num_outer_iterations:03}")
    # calculate A^H 1 with current attenuation sinogram
    adjoint_ones_cur = proj.adjoint(
        att_op_cur.adjoint(xp.ones(proj.out_shape, dtype=xp.float32, device=dev))
    )

    # run activity MLEM updates using current attenuation sinogram and adjoint ones
    for i in range(num_mlem_updates):
        print(f"MLEM update {(i + 1):03} / {num_mlem_updates:03}", end="\r")
        x_mlacf = em_update(
            x_mlacf, y, proj, att_op_cur, contamination, adjoint_ones_cur
        )
    print()

    #############################
    # MLACF attenuation sinogram update

    # TOF projection excluding attenuation
    p_it = proj(x_mlacf)
    # TOF projection excluding attenuation summed over TOF bins
    p_i = p_it.sum(-1)
    # mask for MLACF attenuation sino update
    mask = p_i > (0.01 * p_i.max())

    if mlacf_update_type == "poisson-newton":
        print("poisson-newton attenuation sinogram update")
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
        print()
    elif mlacf_update_type == "unweighted-gauss-analytic":
        print("unweighted gauss-analytic attenuation sinogram update")
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
        print("weighted gauss-analytic attenuation sinogram update")
        inds = xp.where(mask)
        att_sino_cur[inds] = (y[inds] - contamination[inds]).sum(-1) / p_i[inds]

        # clip negative values
        att_sino_cur = xp.clip(att_sino_cur, 0, None)

        att_op_cur = parallelproj.TOFNonTOFElementwiseMultiplicationOperator(
            proj.out_shape, att_sino_cur
        )
    else:
        print("no attenuation sinogram update")

# %%
# calculate the Poisson log-likelihood of the final MLEM and MLACF solutions

exp_mlem = att_op(proj(x_mlem)) + contamination
logL_mlem = float((y * xp.log(exp_mlem) - exp_mlem).sum())

exp_mlacf = att_op_cur(proj(x_mlacf)) + contamination
logL_mlacf = float((y * xp.log(exp_mlacf) - exp_mlacf).sum())

print(f"MLEM log-likelihood: {logL_mlem}")
print(f"MLACF {mlacf_update_type} log-likelihood: {logL_mlacf}")

# %%

x_true_np = parallelproj.to_numpy_array(x_true)
x_mlem_np = parallelproj.to_numpy_array(x_mlem)
x_mlacf_np = parallelproj.to_numpy_array(x_mlacf)
x_att_np = parallelproj.to_numpy_array(x_att)

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

sl_z = proj.in_shape[2] // 2

fig, ax = plt.subplots(2, 4, figsize=(12, 6), tight_layout=True)
ax[0, 0].imshow(x_true_np[:, :, sl_z], cmap="Greys")
ax[0, 1].imshow(x_mlem_np[:, :, sl_z], cmap="Greys")
ax[0, 2].imshow(x_mlacf_np[:, :, sl_z], cmap="Greys")
ax[0, 3].imshow(x_att_np[:, :, sl_z], cmap="Greys")
ax[1, 1].imshow(x_mlem_np_smooth[:, :, sl_z], cmap="Greys")
ax[1, 2].imshow(x_mlacf_np_smooth[:, :, sl_z], cmap="Greys")
fig.show()
