# %%
from __future__ import annotations
from parallelproj import Array

import array_api_compat.cupy as xp

import parallelproj
from array_api_compat import to_device
import array_api_compat.numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
gain = 100.0  # gain factor controlling sensitivity -> higher for more counts

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
# calculate the attenuation sinogram
att_sino = gain * xp.exp(-proj(x_att))


# %%
# Complete PET forward model setup
# --------------------------------
#
# We combine an image-based resolution model,
# a non-TOF or TOF PET projector and an attenuation model
# into a single linear operator.

# enable TOF - comment if you want to run non-TOF
proj.tof_parameters = parallelproj.TOFParameters(
    num_tofbins=51, tofbin_width=12.0, sigma_tof=12.0
)

# setup the attenuation multiplication operator which is different
# for TOF and non-TOF since the attenuation sinogram is always non-TOF
if proj.tof:
    att_op = parallelproj.TOFNonTOFElementwiseMultiplicationOperator(
        proj.out_shape, att_sino
    )
else:
    att_op = parallelproj.ElementwiseMultiplicationOperator(att_sino)

res_model = parallelproj.GaussianFilterOperator(
    proj.in_shape, sigma=4.5 / (2.35 * proj.voxel_size)
)

# compose all 3 operators into a single linear operator
pet_lin_op = parallelproj.CompositeLinearOperator((att_op, proj, res_model))


# %%
# Simulation of projection data
# -----------------------------
#
# We setup an arbitrary ground truth :math:`x_{true}` and simulate
# noise-free and noisy data :math:`y` by adding Poisson noise.

# simulated noise-free data
noise_free_data = pet_lin_op(x_true)

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
    op: parallelproj.LinearOperator,
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
    op : parallelproj.LinearOperator
        linear forward operator
    s : Array
        contamination
    adjoint_ones : Array
        adjoint of ones

    Returns
    -------
    Array
        _description_
    """
    ybar = op(x_cur) + s
    return x_cur * op.adjoint(data / ybar) / adjoint_ones


# %%
# Run the MLEM iterations
# -----------------------

# number of MLEM iterations
num_iter = 100

# initialize x
x = xp.ones(pet_lin_op.in_shape, dtype=xp.float32, device=dev)
# calculate A^H 1
adjoint_ones = pet_lin_op.adjoint(
    xp.ones(pet_lin_op.out_shape, dtype=xp.float32, device=dev)
)

for i in range(num_iter):
    print(f"MLEM iteration {(i + 1):03} / {num_iter:03}", end="\r")
    x = em_update(x, y, pet_lin_op, contamination, adjoint_ones)


# %%
# Run the MLEM iterations without accounting for attenuation sinogram
# -------------------------------------------------------------------

if proj.tof:
    att_op_nac = parallelproj.TOFNonTOFElementwiseMultiplicationOperator(
        proj.out_shape, att_sino * 0 + gain
    )
else:
    att_op_nac = parallelproj.ElementwiseMultiplicationOperator(att_sino * 0 + gain)

# compose all 3 operators into a single linear operator
pet_lin_op_nac = parallelproj.CompositeLinearOperator((att_op_nac, proj, res_model))


# initialize x
x_nac = xp.ones(pet_lin_op_nac.in_shape, dtype=xp.float32, device=dev)
# calculate A^H 1
adjoint_ones_nac = pet_lin_op_nac.adjoint(
    xp.ones(pet_lin_op.out_shape, dtype=xp.float32, device=dev)
)

for i in range(num_iter):
    print(f"MLEM iteration {(i + 1):03} / {num_iter:03}", end="\r")
    x_nac = em_update(x, y, pet_lin_op_nac, contamination, adjoint_ones_nac)


# %%

fig, ax = plt.subplots(1, 4, figsize=(16, 4), tight_layout=True)
ax[0].imshow(x_true[:, :, 0].get(), cmap="Greys")
ax[1].imshow(x[:, :, 0].get(), cmap="Greys")
ax[2].imshow(x_nac[:, :, 0].get(), cmap="Greys")
fig.show()
