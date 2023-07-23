"""minimal example that shows how to use the joseph3d non-TOF forward and back projector"""
import parallelproj

# parallelproj tells us whether cupy is available and supported
# if it is, we use cupy, otherwise numpy as array module (xp)
# to be compatible with the python array api, we use the minimal
# numpy.array_api implementation
if parallelproj.cupy_enabled:
    import cupy as xp
else:
    import numpy.array_api as xp

## parallelproj also supports torch tensors
#if parallelproj.torch_enabled:
#    import torch as xp

#---------------------------------------------------------------
#--- setup a simple test image ---------------------------------
#---------------------------------------------------------------

# setup the image dimensions
n0, n1, n2 = (2, 3, 4)
img_dim = (n0, n1, n2)

# define the voxel sizes (in physical units)
voxel_size = xp.asarray([4., 3., 2.], dtype=xp.float32)
# define the origin of the image (location of voxel (0,0,0) in physical units)
img_origin = ((-xp.asarray(img_dim, dtype=xp.float32) / 2 + 0.5) * voxel_size)

# create a simple test image
img = xp.reshape(xp.arange(n0 * n1 * n2, dtype=xp.float32), (n0, n1, n2))

#---------------------------------------------------------------
#--- setup the LOR start and end points ------------------------
#---------------------------------------------------------------

# Every line of response (LOR) along which we want to project is
# defined by its start point (3 element array) and end point (3 element array).
# Here we define 10 LORs and group all start and end points in two
# 2D arrays of shape (10,3).

# We first define the LORs start/end points in voxel coordinates (for convenience)
# and convert them later to physical units (as required for the projectors)

# define start/end points in voxel coordinates
vstart = xp.asarray([
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
])

vend = xp.asarray([
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
])

# convert the LOR coordinates to world coordinates (physical units)
xstart = xp.asarray(vstart * voxel_size + img_origin, dtype=xp.float32)
xend = xp.asarray(vend * voxel_size + img_origin, dtype=xp.float32)

#---------------------------------------------------------------
#--- call the forward projector --------------------------------
#---------------------------------------------------------------

# allocate memory for the forward projection array
# call the forward projector
img_fwd = parallelproj.joseph3d_fwd(xstart, xend, img, img_origin, voxel_size)

print(img_fwd)

#---------------------------------------------------------------
#--- call the adjoint of the forward projector -----------------
#---------------------------------------------------------------

# setup a "sinogram" full of ones
sino = xp.ones_like(img_fwd)

# call the back projector
back_img = parallelproj.joseph3d_back(xstart, xend, img_dim, img_origin,
                                      voxel_size, sino)

print(back_img)