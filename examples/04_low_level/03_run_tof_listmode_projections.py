"""
Low-level TOF listmode projection example
=========================================

A minimal example that shows how to use the joseph3d TOF forward and back projector in sinogram mode.

.. tip::
    parallelproj is python array API compatible meaning it supports different 
    array backends (e.g. numpy, cupy, torch, ...) and devices (CPU or GPU).
    Choose your preferred array API ``xp`` and device ``dev`` below.
"""
# %%
import array_api_compat.numpy as xp

# import array_api_compat.cupy as xp
# import array_api_compat.torch as xp

import parallelproj
from array_api_compat import to_device, device

# choose a device (CPU or CUDA GPU)
if "numpy" in xp.__name__:
    # using numpy, device must be cpu
    dev = "cpu"
elif "cupy" in xp.__name__:
    # using cupy, only cuda devices are possible
    dev = xp.cuda.Device(0)
elif "torch" in xp.__name__:
    # using torch valid choices are 'cpu' or 'cuda'
    dev = "cuda"

# %%
# Setup a simple test image
# -------------------------

# setup the image dimensions
n0, n1, n2 = (7, 7, 7)
img_dim = (n0, n1, n2)

# define the voxel sizes (in physical units)
voxel_size = to_device(xp.asarray([2.0, 2.0, 2.0], dtype=xp.float32), dev)
# define the origin of the image (location of voxel (0,0,0) in physical units)
img_origin = (
    -to_device(xp.asarray(img_dim, dtype=xp.float32), dev) / 2 + 0.5
) * voxel_size

# create a simple test image
img = to_device(xp.zeros((n0, n1, n2), dtype=xp.float32), dev)
img[n0 // 2, n1 // 2, n2 // 2] = 1

# %%
# Setup the LOR start and end points
# ----------------------------------

# Every line of response (LOR) along which we want to project is
# defined by its start point (3 element array) and end point (3 element array).
# Here we define 2 LORs and group all start and end points in two
# 2D arrays of shape (2,3).

# We first define the LORs start/end points in voxel coordinates (for convenience)
# and convert them later to physical units (as required for the projectors)

# define start/end points in voxel coordinates
vstart = to_device(
    xp.asarray(
        [
            [n0 // 2, -1, n2 // 2],  #
            [n0 // 2, n1 // 2, -1],  #
        ],
        dtype=xp.float32,
    ),
    dev,
)

vend = to_device(
    xp.asarray(
        [
            [n0 // 2, n1, n2 // 2],
            [n0 // 2, n1 // 2, n2],  #
        ],
        dtype=xp.float32,
    ),
    dev,
)

# convert the LOR coordinates to world coordinates (physical units)
xstart = vstart * voxel_size + img_origin
xend = vend * voxel_size + img_origin

# %%
# setup the TOF related parameters
# --------------------------------

# the width of the TOF bins in spatial physical units
# same unit as voxel size
tofbin_width = 1.5

# the number of TOF bins
num_tof_bins = 17

# number of sigmas after which TOF kernel is truncated
nsigmas = 3.0

# FWHM of the Gaussian TOF kernel in physical units
fwhm_tof = 6.0

# sigma of the Gaussian TOF kernel in physical units
# if this is an array of length 1, the same sigma is used
# for all LORs
sigma_tof = to_device(xp.asarray([fwhm_tof / 2.35], dtype=xp.float32), dev)

# TOF center offset for the central TOF bin in physical units
# if this is an array of length 1, the same offset is used
# for all LORs
tofcenter_offset = to_device(xp.asarray([0], dtype=xp.float32), dev)

# setup an array containing the TOF bin of each event
tof_bin = to_device(xp.zeros(xstart.shape[0], dtype=xp.int16), dev)
tof_bin[-1] = 1

# %%
# Call the forward projector
# --------------------------

img_fwd = parallelproj.joseph3d_fwd_tof_lm(
    xstart,
    xend,
    img,
    img_origin,
    voxel_size,
    tofbin_width,
    sigma_tof,
    tofcenter_offset,
    nsigmas,
    tof_bin,
)

print(img_fwd)
print(type(img_fwd))
print(device(img_fwd))
print("")

# %%
# Call the adjoint of the forward projector
# -----------------------------------------

# setup a list of ones to be back projected
lst = to_device(xp.ones(img_fwd.shape, dtype=xp.float32), dev)

back_img = parallelproj.joseph3d_back_tof_lm(
    xstart,
    xend,
    img_dim,
    img_origin,
    voxel_size,
    lst,
    tofbin_width,
    sigma_tof,
    tofcenter_offset,
    nsigmas,
    tof_bin,
)

print(back_img[:, :, 3])
print(type(back_img))
print(device(back_img))
