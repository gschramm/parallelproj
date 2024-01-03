"""
low-level non-TOF projection example
====================================

The example below shows how to do a simple non-TOF forward projection along a set of
known lines of response with known start and end points.
"""
# %%
# parallelproj supports the numpy, cupy and pytorch array API and different devices
# choose your preferred array API uncommenting the corresponding line

import array_api_compat.numpy as xp
#import array_api_compat.cupy as xp
#import array_api_compat.torch as xp

import parallelproj
from array_api_compat import to_device, device

# choose a device (CPU or CUDA GPU)
if 'numpy' in xp.__name__:
    # using numpy, device must be cpu
    dev = 'cpu'
elif 'cupy' in xp.__name__:
    # using cupy, only cuda devices are possible
    dev = xp.cuda.Device(0)
elif 'torch' in xp.__name__:
    # using torch valid choices are 'cpu' or 'cuda'
    dev = 'cuda'

# %%
# setup a simple test image
# -------------------------

# setup the image dimensions
n0, n1, n2 = (2, 3, 4)
img_dim = (n0, n1, n2)

# define the voxel sizes (in physical units)
voxel_size = to_device(xp.asarray([4., 3., 2.], dtype=xp.float32), dev)
# define the origin of the image (location of voxel (0,0,0) in physical units)
img_origin = (-to_device(xp.asarray(img_dim, dtype=xp.float32), dev) / 2 +
              0.5) * voxel_size

# create a simple test image
img = to_device(
    xp.reshape(xp.arange(n0 * n1 * n2, dtype=xp.float32), (n0, n1, n2)), dev)

# %%
# setup the LOR start and end points
# ----------------------------------
#
# Every line of response (LOR) along which we want to project is
# defined by its start point (3 element array) and end point (3 element array).
# Here we define 10 LORs and group all start and end points in two
# 2D arrays of shape (10,3).
#
# We first define the LORs start/end points in voxel coordinates (for convenience)
# and convert them later to physical units (as required for the projectors)

# define start/end points in voxel coordinates
vstart = to_device(
    xp.asarray(
        [
            [0, -1, 0],  # 
            [0, -1, 0],  #
            [0, -1, 1],  #
            [0, -1, 0.5],  #
            [0, 0, -1],  #
            [-1, 0, 0],  #
            [n0 - 1, -1, 0],  # 
            [n0 - 1, -1, n2 - 1],  #
            [n0 - 1, 0, -1],  #
            [n0 - 1, n1 - 1, -1]
        ],
        dtype=xp.float32),
    dev)

vend = to_device(
    xp.asarray(
        [
            [0, n1, 0],  #           
            [0, n1, 0],  #           
            [0, n1, 1],  #          
            [0, n1, 0.5],  #         
            [0, 0, n2],  #          
            [n0, 0, 0],  #          
            [n0 - 1, n1, 0],  #      
            [n0 - 1, n1, n2 - 1],  # 
            [n0 - 1, 0, n2],  #     
            [n0 - 1, n1 - 1, n2]
        ],
        dtype=xp.float32),
    dev)

# convert the LOR coordinates to world coordinates (physical units)
xstart = vstart * voxel_size + img_origin
xend = vend * voxel_size + img_origin

# %%
# call the forward projector
# --------------------------

# allocate memory for the forward projection array
# call the forward projector
img_fwd = parallelproj.joseph3d_fwd(xstart, xend, img, img_origin, voxel_size)

print(img_fwd)
print(type(img_fwd))
print(device(img_fwd))
print('')

# %%
# call the adjoint of the forward projector
# -----------------------------------------

# setup a "sinogram" full of ones
sino = to_device(xp.ones(img_fwd.shape, dtype=xp.float32), dev)

# call the back projector
back_img = parallelproj.joseph3d_back(xstart, xend, img_dim, img_origin,
                                      voxel_size, sino)

print(back_img)
print(type(back_img))
print(device(back_img))
