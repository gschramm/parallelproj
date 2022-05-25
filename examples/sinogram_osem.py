import pyparallelproj as ppp
import numpy as np
import matplotlib.pyplot as plt
from warnings import warn

try:
  import cupy as cp
except:
  warn('cupy is not available')

import scipy.ndimage as ndi
import cupyx.scipy.ndimage as ndi_cupy

from time import time

import argparse

#---------------------------------------------------------------------------------
# parse the command line

parser = argparse.ArgumentParser()
parser.add_argument('--counts',   help = 'counts to simulate',    default = 1e6, type = float)
parser.add_argument('--niter',    help = 'number of iterations',  default = 4,   type = int)
parser.add_argument('--nsubsets',   help = 'number of subsets',     default = 28,  type = int)
parser.add_argument('--fwhm_mm',  help = 'psf modeling FWHM mm',  default = 4.5, type = float)
parser.add_argument('--fwhm_data_mm',  help = 'psf for data FWHM mm',  default = 4.5, type = float)
parser.add_argument('--phantom', help = 'phantom to use', default = 'brain2d')
parser.add_argument('--seed',    help = 'seed for random generator', default = 1, type = int)
args = parser.parse_args()

#---------------------------------------------------------------------------------

counts        = args.counts
niter         = args.niter
nsubsets      = args.nsubsets
fwhm_mm       = args.fwhm_mm
fwhm_data_mm  = args.fwhm_data_mm
phantom       = args.phantom
seed          = args.seed

on_gpu        = True

if on_gpu:
  xp = cp
  ndimage_module = ndi_cupy
else:
  xp = np
  ndimage_module = ndi


#---------------------------------------------------------------------------------

xp.random.seed(seed)

# setup a scanner with one ring
scanner = ppp.RegularPolygonPETScanner(ncrystals_per_module = np.array([16,1]),
                                       nmodules             = np.array([28,1]),
                                       on_gpu               = on_gpu)

# setup a test image
voxsize = np.array([2.,2.,2.])
n2      = max(1,int((scanner.xc2.max() - scanner.xc2.min()) / voxsize[2]))

# convert fwhm from mm to pixels
fwhm      = fwhm_mm / voxsize
fwhm_data = fwhm_data_mm / voxsize

# setup a test image
if phantom == 'ellipse2d':
  n   = 200
  img = xp.zeros((n,n,n2), dtype = xp.float32)
  tmp = ppp.ellipse2d_phantom(n = n, c = 3)
  for i2 in range(n2):
    if xp.__name__ == 'numpy':
      img[:,:,i2] = tmp
    else: 
      img[:,:,i2] = xp.asarray(tmp)
elif phantom == 'brain2d':
  n   = 128
  img = xp.zeros((n,n,n2), dtype = xp.float32)
  tmp = ppp.brain2d_phantom(n = n)
  for i2 in range(n2):
    if xp.__name__ == 'numpy':
      img[:,:,i2] = tmp
    else: 
      img[:,:,i2] = xp.asarray(tmp)

img_origin = (-(np.array(img.shape) / 2) +  0.5) * voxsize

#-----------------------------------------------------------------------------------------------------------
# data generation

# setup an attenuation image
att_img = (img > 0) * 0.01

# generate TOF sinogram parameters and the TOF projector
sino_params = ppp.PETSinogramParameters(scanner, ntofbins = 17, tofbin_width = 15.)
proj        = ppp.SinogramProjector(scanner, sino_params, img.shape, nsubsets = nsubsets, 
                                    voxsize = voxsize, img_origin = img_origin,
                                    tof = True, sigma_tof = 60./2.35, n_sigmas = 3.)

# create the attenuation sinogram
proj.set_tof(False)
attn_sino = xp.exp(-proj.fwd_project(att_img))
proj.set_tof(True)

# generate the sensitivity sinogram
sens_sino = xp.ones(proj.sino_params.nontof_shape, dtype = xp.float32)

# setup the acquisition models for data generation
res_model_data = ppp.ImageBasedResolutionModel(fwhm_data, ndimage_module = ndimage_module) 
acq_model_data = ppp.PETAcqModel(proj, attn_sino, sens_sino, image_based_res_model = res_model_data)

# forward project the image
img_fwd= acq_model_data.forward(img)

# scale sum of fwd image to counts
if counts > 0:
  scale_fac = (counts / img_fwd.sum())
  img_fwd  *= scale_fac 
  img      *= scale_fac 

  # contamination sinogram with scatter and randoms
  # useful to avoid division by 0 in the ratio of data and exprected data
  contam_sino = xp.full(img_fwd.shape, 0.2*img_fwd.mean(), dtype = xp.float32)
  
  em_sino = xp.random.poisson(img_fwd + contam_sino)
else:
  scale_fac = 1.

  # contamination sinogram with sctter and randoms
  # useful to avoid division by 0 in the ratio of data and exprected data
  contam_sino = xp.full(img_fwd.shape, 0.2*img_fwd.mean(), dtype = xp.float32)

  em_sino = img_fwd + contam_sino

del res_model_data
del acq_model_data


#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
# OSEM reconstruction

# setup the acquisition models for data generation
res_model = ppp.ImageBasedResolutionModel(fwhm, ndimage_module = ndimage_module) 
acq_model = ppp.PETAcqModel(proj, attn_sino, sens_sino, image_based_res_model = res_model)

osem = ppp.OSEM(em_sino, acq_model, contam_sino, xp)
osem.init() # initialize OSEM (e.g. calculate the sensivity image for every subset)
osem.run(niter, calculate_cost = True)

##-----------------------------------------------------------------------------------------------------------
## OSEM EMTV recon with prior
#
#prior      = ppp.GradientBasedPrior(ppp.GradientOperator(xp), ppp.GradientNorm(xp, name = 'l2_l1'), 3e-2)
#osem_emtv  = ppp.OSEM_EMTV(em_sino, acq_model, contam_sino, prior, xp)
#osem_emtv.init()
#
#osem_emtv.run(30, calculate_cost = True)

#-----------------------------------------------------------------------------------------------------------
# visualizations

ims1 = dict(vmin = 0, vmax = 1.3*img.max(), cmap = plt.cm.Greys)
ims2 = dict(vmin = -0.2*img.max(), vmax = 0.2*img.max(), cmap = plt.cm.seismic)

fig, ax = plt.subplots(1,5, figsize = (15,3))

if xp.__name__ == 'numpy':
  ax[0].imshow(img[...,n2//2], **ims1)
  ax[1].imshow(osem.x[...,n2//2], **ims1)
  ax[2].imshow(ndi.gaussian_filter(osem.x[...,n2//2], fwhm[0] / 2.35), **ims1)
  ax[3].imshow(osem.x[...,n2//2] - img[...,n2//2], **ims2) 
else:
  ax[0].imshow(xp.asnumpy(img[...,n2//2]), **ims1)
  ax[1].imshow(xp.asnumpy(osem.x[...,n2//2]), **ims1)
  ax[2].imshow(ndi.gaussian_filter(xp.asnumpy(osem.x[...,n2//2]), fwhm[0] / 2.35), **ims1)
  ax[3].imshow(xp.asnumpy(osem.x[...,n2//2] - img[...,n2//2]), **ims2) 
ax[4].plot(np.arange(1,osem.cost.shape[0]+1), osem.cost)

ax[0].set_title('ground truth')
ax[1].set_title('OSEM')
ax[2].set_title('p.sm. OSEM')
ax[3].set_title('bias')
ax[4].set_title('neg Poisson logL')

fig.tight_layout()
fig.show()
