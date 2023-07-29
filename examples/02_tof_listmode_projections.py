"""minimal example that shows how to use the joseph3d TOF forward and back projector in listmode"""
import parallelproj

# parallelproj supports the numpy, cupy and pytorch array API
# choose your preferred array API uncommenting the corresponding line
import array_api_compat.numpy as xp
#import array_api_compat.cupy as xp
#import array_api_compat.torch as xp

#---------------------------------------------------------------
#--- setup a simple test image ---------------------------------
#---------------------------------------------------------------

# setup the image dimensions
n0, n1, n2 = (7, 7, 7)
img_dim = (n0, n1, n2)

# define the voxel sizes (in physical units)
voxel_size = xp.asarray([2., 2., 2.], dtype=xp.float32)
# define the origin of the image (location of voxel (0,0,0) in physical units)
img_origin = ((-xp.asarray(img_dim, dtype=xp.float32) / 2 + 0.5) * voxel_size)

# create a simple test image
img = xp.zeros((n0, n1, n2), dtype=xp.float32)
img[n0 // 2, n1 // 2, n2 // 2] = 1

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
vstart = xp.asarray([
    [n0 // 2, -1, n2 // 2],  # 
    [n0 // 2, n1 // 2, -1],  # 
])

vend = xp.asarray([
    [n0 // 2, n1, n2 // 2],  #           
    [n0 // 2, n1 // 2, n2],  # 
])

# convert the LOR coordinates to world coordinates (physical units)
xstart = (xp.asarray(vstart, dtype=xp.float32) * voxel_size + img_origin)
xend = (xp.asarray(vend, dtype=xp.float32) * voxel_size + img_origin)

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
sigma_tof = xp.asarray([fwhm_tof / 2.35], dtype=xp.float32)

# TOF center offset for the central TOF bin in physical units
# if this is an array of length 1, the same offset is used
# for all LORs
tofcenter_offset = xp.asarray([0], dtype=xp.float32)

# setup an array containing the TOF bin of each event
tof_bin = xp.zeros(xstart.shape[0], dtype=xp.int16)
tof_bin[-1] = 1

#---------------------------------------------------------------
#--- call the forward projector --------------------------------
#---------------------------------------------------------------

img_fwd = parallelproj.joseph3d_fwd_tof_lm(xstart, xend, img, img_origin,
                                           voxel_size, tofbin_width, sigma_tof,
                                           tofcenter_offset, nsigmas, tof_bin)

#---------------------------------------------------------------
#--- call the adjoint of the forward projector -----------------
#---------------------------------------------------------------

# setup a list of ones to be back projected
lst = xp.ones_like(img_fwd)

back_img = parallelproj.joseph3d_back_tof_lm(xstart, xend, img_dim, img_origin,
                                             voxel_size, lst, tofbin_width,
                                             sigma_tof, tofcenter_offset,
                                             nsigmas, tof_bin)
