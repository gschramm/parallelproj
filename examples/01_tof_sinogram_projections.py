"""minimal example that shows how to use the joseph3d TOF forward and back projector in sinogram mode"""
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
n0, n1, n2 = (7, 7, 7)
img_dim = xp.array([n0, n1, n2])

# define the voxel sizes (in physical units)
voxel_size = xp.array([2., 2., 2.], dtype=xp.float32)
# define the origin of the image (location of voxel (0,0,0) in physical units)
img_origin = ((-img_dim / 2 + 0.5) * voxel_size).astype(xp.float32)

# create a simple test image
img = xp.zeros((n0,n1,n2), dtype=xp.float32)
img[n0//2,n1//2,n2//2] = 1


#---------------------------------------------------------------
#--- setup the LOR start and end points ------------------------
#---------------------------------------------------------------

# Every line of response (LOR) along which we want to project is
# defined by its start point (3 element array) and end point (3 element array).
# Here we define 2 LORs and group all start and end points in two
# 2D arrays of shape (2,3).

# We first define the LORs start/end points in voxel coordinates (for convenience)
# and convert them later to physical units (as required for the projectors)

# define start/end points in voxel coordinates
vstart = xp.array([
    [n0//2, -1, n2//2], # 
    [n0//2, n1//2, -1], # 
])

vend = xp.array([
    [n0//2, n1, n2//2], #           
    [n0//2, n1//2, n2], # 
])

# convert the LOR coordinates to world coordinates (physical units)
xstart = (vstart * voxel_size + img_origin).astype(xp.float32)
xend = (vend * voxel_size + img_origin).astype(xp.float32)


#---------------------------------------------------------------
#--- setup the TOF related parameters --------------------------
#---------------------------------------------------------------

# the width of the TOF bins in spatial physical units
# same unit as voxel size
tofbin_width = 1.5

# the number of TOF bins 
num_tof_bins = 17

# number of sigmas after which TOF kernel is truncated
nsigmas = 3.

# FWHM of the Gaussian TOF kernel in physical units
fwhm_tof = 6.

# sigma of the Gaussian TOF kernel in physical units
# if this is an array of length 1, the same sigma is used
# for all LORs
sigma_tof = xp.array([fwhm_tof / (2 * xp.sqrt(2 * xp.log(2)))],
                     dtype=xp.float32)

# TOF center offset for the central TOF bin in physical units
# if this is an array of length 1, the same offset is used
# for all LORs
tofcenter_offset = xp.array([0], dtype=xp.float32)

#---------------------------------------------------------------
#--- call the forward projector --------------------------------
#---------------------------------------------------------------

# allocate memory for the forward projection array 
# its shape is (number of LORs, number of TOF bins)
img_fwd = xp.zeros((xstart.shape[0], num_tof_bins), dtype=xp.float32)

parallelproj.joseph3d_fwd_tof_sino(xstart, xend, img, img_origin,
                                   voxel_size, img_fwd, tofbin_width,
                                   sigma_tof, tofcenter_offset, nsigmas,
                                   num_tof_bins)



#---------------------------------------------------------------
#--- call the adjoint of the forward projector -----------------
#---------------------------------------------------------------

# allocate memory for the back projection array
back_img = xp.zeros((n0,n1,n2), dtype=xp.float32)

# setup a "TOF sinogram"
sino = xp.zeros_like(img_fwd)
sino[:,num_tof_bins//2] = 1

parallelproj.joseph3d_back_tof_sino(xstart, xend, back_img, img_origin,
                                    voxel_size, sino, tofbin_width,
                                    sigma_tof, tofcenter_offset, nsigmas,
                                    num_tof_bins)