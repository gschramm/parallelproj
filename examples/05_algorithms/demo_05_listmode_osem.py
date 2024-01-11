"""
TOF listmode OSEM with projection data
======================================

This example demonstrates the use of the MLEM algorithm to minimize the negative Poisson log-likelihood function.

.. math::
    f(x) = \sum_{i=1}^m \\bar{y}_i - \\bar{y}_i (x) \log(y_i)

using the linear forward model

.. math::
    \\bar{y}(x) = A x + s

.. tip::
    parallelproj is python array API compatible meaning it supports different 
    array backends (e.g. numpy, cupy, torch, ...) and devices (CPU or GPU).
    Choose your preferred array API ``xp`` and device ``dev`` below.
"""
# %%
from __future__ import annotations
from numpy.array_api._array_object import Array

import numpy.array_api as xp

# import array_api_compat.numpy as xp

# import array_api_compat.cupy as xp
# import array_api_compat.torch as xp

import parallelproj
from array_api_compat import to_device
import array_api_compat.numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import copy

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
# Setup of the forward model :math:`\bar{y}(x) = A x + s`
# --------------------------------------------------------
#
# We setup a linear forward operator :math:`A` consisting of an
# image-based resolution model, a non-TOF PET projector and an attenuation model
#
# .. note::
#     The OSEM implementation below works with all linear operators that
#     subclass :class:`.LinearOperator` (e.g. the high-level projectors).

num_rings = 5
scanner = parallelproj.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=65.0,
    num_sides=12,
    num_lor_endpoints_per_side=15,
    lor_spacing=2.3,
    ring_positions=xp.linspace(-10, 10, num_rings),
    symmetry_axis=2,
)

# %%
# setup the LOR descriptor that defines the sinogram

img_shape = (40, 40, 8)
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

x_true[:2, :, :] = 0
x_true[-2:, :, :] = 0
x_true[:, :2, :] = 0
x_true[:, -2:, :] = 0

# %%
# setup an attenuation image and calculate the attenuation sinogram
# -----------------------------------------------------------------

# setup an attenuation image
x_att = 0.01 * xp.astype(x_true > 0, xp.float32)
# calculate the attenuation sinogram
att_sino = xp.exp(-proj(x_att))

# %%
# setup the complete PET forward model
# ------------------------------------
#
# We combine an image-based resolution model,
# a non-TOF or TOF PET projector and an attenuation model
# into a single linear operator.

# enable TOF - comment if you want to run non-TOF
proj.tof_parameters = parallelproj.TOFParameters(
    num_tofbins=13, tofbin_width=12.0, sigma_tof=12.0
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
    np.random.poisson(np.asarray(to_device(noise_free_data, "cpu"))),
    device=dev,
    dtype=xp.uint16,
)

# %%

sino = y

if sino.ndim == 4:
    tof = True
    num_tofbins = sino.shape[-1]
else:
    tof = False
    num_tofbins = 1

a = []

for view in range(lor_desc.num_views):
    xstart, xend = lor_desc.get_lor_coordinates(views=xp.asarray([view], device=dev))
    xstart = xp.reshape(xstart, (-1, 3))
    xend = xp.reshape(xend, (-1, 3))

    sino_view = xp.take(
        sino, xp.asarray([view], device=dev), axis=lor_desc.view_axis_num
    )
    sino_view = xp.squeeze(sino_view, axis=lor_desc.view_axis_num)

    for it in range(sino.shape[-1]):
        if tof:
            ss = sino_view[..., it]
        else:
            ss = sino_view

        ss = xp.reshape(ss, (ss.size,))

        event_sino_inds = xp.asarray(
            np.repeat(np.arange(ss.shape[0]), to_device(ss, "cpu")), device=dev
        )

        event_start_coords = xp.take(xstart, event_sino_inds, axis=0)
        event_end_coords = xp.take(xend, event_sino_inds, axis=0)

        if tof:
            event_tofbin = xp.full(
                ss.shape, it - num_tofbins // 2, device=dev, dtype=xp.int16
            )
        else:
            event_tofbin = None

        a.append(event_start_coords)
