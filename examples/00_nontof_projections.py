"""minimal example that shows how to use the joseph3d non-TOF forward and back projector"""
import parallelproj

# parallelproj tells us whether cupy is available and supported
# if it is, we use cupy, otherwise numpy as array module (xp)
if parallelproj.cupy_enabled:
    import cupy as xp
else:
    import numpy as xp

#---------------------------------------------------------------
#--- setup a simple test image ---------------------------------
#---------------------------------------------------------------

# setup the image dimensions
n0, n1, n2 = (2, 3, 4)
img_dim = xp.array([n0, n1, n2])

# define the voxel sizes (in physical units)
voxel_size = xp.array([4., 3., 2.], dtype=xp.float32)
# define the origin of the image (location of voxel (0,0,0) in physical units)
img_origin = ((-img_dim / 2 + 0.5) * voxel_size).astype(xp.float32)

# create a simple test image
img = xp.arange(n0 * n1 * n2, dtype=xp.float32).reshape((n0, n1, n2))


#---------------------------------------------------------------
#--- setup the LOR start and end points ------------------------
#---------------------------------------------------------------

# Rvery line of response (LOR) along which we want to project is
# define by its start point (3 element array) and end point (3 element array).
# Here we define 10 LORs, such and group all start and end points in two
# 2D arrays of shape (10,3).

# We first define the LORs start/end points in voxel coordinates (for convenience)
# and convert them later to physical units (as required for the projectors)

# define start/end points in voxel coordinates
vstart = xp.array([
    [0, -1, 0], # 
    [0, -1, 0], #
    [0, -1, 1], #
    [0, -1, 0.5], #
    [0, 0, -1], #
    [-1, 0, 0], #
    [n0 - 1, -1, 0], # 
    [n0 - 1, -1, n2 - 1], #
    [n0 - 1, 0, -1], #
    [n0 - 1, n1 - 1, -1]
])

vend = xp.array([
    [0, n1, 0], #           
    [0, n1, 0], #           
    [0, n1, 1], #          
    [0, n1, 0.5], #         
    [0, 0, n2], #          
    [n0, 0, 0], #          
    [n0 - 1, n1, 0], #      
    [n0 - 1, n1, n2 - 1], # 
    [n0 - 1, 0, n2], #     
    [n0 - 1, n1 - 1, n2]
])

# convert the LOR coordinates to world coordinates (physical units)
xstart = (vstart * voxel_size + img_origin).astype(xp.float32)
xend = (vend * voxel_size + img_origin).astype(xp.float32)


#---------------------------------------------------------------
#--- call the forward projector --------------------------------
#---------------------------------------------------------------

# allocate memory for the forward projection array
img_fwd = xp.zeros(xstart.shape[0], dtype=xp.float32)

# call the forward projector
parallelproj.joseph3d_fwd(xstart, xend, img, img_origin, voxel_size,
                          img_fwd)


#---------------------------------------------------------------
#--- call the forward projector --------------------------------
#---------------------------------------------------------------

# setup a "sinogram" full of ones
sino = xp.ones_like(img_fwd)

# allocate memory for the back projection array
back_img = xp.zeros_like(img)

# call the back projector
parallelproj.joseph3d_back(xstart, xend, back_img, img_origin, voxel_size,
                           sino)