import os
import pyparallelproj as ppp
import numpy as np

from scipy.ndimage import gaussian_filter

#---------------------------------------------------------------------------------
def pet_fwd_model(img, proj, attn_sino, sens_sino, isub, fwhm_mm = 0):

  if fwhm_mm > 0:
    img = gaussian_filter(img, fwhm_mm/(2.35*voxsize))

  sino = sens_sino[isub, ...]*attn_sino[isub, ...]*proj.fwd_project(img, subset = isub)

  return sino

#---------------------------------------------------------------------------------
def pet_back_model(subset_sino, proj, attn_sino, sens_sino, isub, fwhm_mm = 0):

  back_img = proj.back_project(sens_sino[isub, ...]*attn_sino[isub, ...]*subset_sino, subset = isub)

  if fwhm_mm > 0:
    back_img = gaussian_filter(back_img, fwhm_mm/(2.35*voxsize))

  return back_img

#---------------------------------------------------------------------------------
# input parameters

# number of subsets
nsubsets = 1
# subset to use for adjoint test
subset   = 0
# fwhm [mm] used for resolution modeling
fwhm_mm  = 4.5

# setup a scanner
scanner = ppp.RegularPolygonPETScanner(ncrystals_per_module = np.array([16,1]),
                                       nmodules             = np.array([28,1]))

voxsize = np.array([2.,2.,2.])
n0      = 120
n1      = 120
n2      = max(1,int((scanner.xc2.max() - scanner.xc2.min()) / voxsize[2]))


#----------------
#----------------


# setup a toy emission and attenuation image
img = np.zeros((n0,n1,n2), dtype = np.float32)
img[(n0//4):(3*n0//4),(n1//4):(3*n1//4),:] = 1
img *= np.random.rand(n0,n1,n2).astype(np.float32)

att_img = np.zeros((n0,n1,n2), dtype = np.float32)
att_img[(n0//4):(3*n0//4),(n1//4):(3*n1//4),:] = 0.01*voxsize[0]

img_origin = (-(np.array(img.shape) / 2) +  0.5) * voxsize


######## define nontof projections
sino_params = ppp.PETSinogramParameters(scanner)
proj        = ppp.SinogramProjector(scanner, sino_params, img.shape, nsubsets = nsubsets, 
                                    voxsize = voxsize, img_origin = img_origin)

######## define tof projections
tofsino_params = ppp.PETSinogramParameters(scanner, ntofbins = 27, tofbin_width = 28.)
tofproj        = ppp.SinogramProjector(scanner, tofsino_params, img.shape, nsubsets = nsubsets, 
                                       voxsize = voxsize, img_origin = img_origin,
                                       tof = True, sigma_tof = 60./2.35, n_sigmas = 3)



#----------------
#----------------

# forward project the attenuation image to generate the attenuation sinogram
sino_shape = sino_params.shape
attn_sino  = np.zeros((nsubsets,sino_shape[0],sino_shape[1] // nsubsets, sino_shape[2], sino_shape[3]),
                      dtype = np.float32)

for i in range(nsubsets):
  attn_sino[i,...] = np.exp(-proj.fwd_project(att_img, subset = i))

# create a sensitivity sinogram
sens_sino = np.ones(attn_sino.shape, dtype = np.float32)


#----------------
#----------------

# apply the forward model
img_fwd_tof = pet_fwd_model(img, tofproj, attn_sino, sens_sino, subset, fwhm_mm = fwhm_mm)

# setup a random sinogram and backproject it
tsino    = np.random.rand(*tofproj.subset_sino_shapes[subset])
back_tof = pet_back_model(tsino, tofproj, attn_sino, sens_sino, subset, fwhm_mm = fwhm_mm)

# test if models are adjoint
print((img*back_tof).sum())
print((img_fwd_tof*tsino).sum())
