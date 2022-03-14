# script to investigate the convergence of SPDHG for 2D TOF PET

import os
import matplotlib.pyplot as plt
import pyparallelproj as ppp
from pyparallelproj.algorithms import osem, spdhg
from pyparallelproj.phantoms   import ellipse2d_phantom, brain2d_phantom

import numpy as np
from scipy.ndimage import gaussian_filter

from pyparallelproj.utils import GradientOperator, GradientNorm

import argparse

from scipy.ndimage import binary_fill_holes, binary_dilation

#---------------------------------------------------------------------------------

def power_iterations(x, proj, attn_sino, sens_sino, fwhm, npower = 500, precond = True):

  if precond:
    fwd_ones = ppp.pet_fwd_model(np.ones(x.shape, dtype = np.float32), proj, attn_sino, sens_sino, fwhm = fwhm)
    fwd_ones[fwd_ones == 0] = fwd_ones[fwd_ones > 0].min()

    back_ones = ppp.pet_back_model(np.ones(proj.sino_params.shape, dtype = np.float32), 
                                   proj, attn_sino, sens_sino, fwhm = fwhm)

    sqr_T = np.sqrt(1./back_ones)
    sqr_S = np.sqrt(1./fwd_ones)
  else:
    sqr_T = 1
    sqr_S = 1

  for i in range(npower):
    fwd  = sqr_S*ppp.pet_fwd_model(x*sqr_T, proj, attn_sino, sens_sino, fwhm = fwhm)
    back = sqr_T*ppp.pet_back_model(sqr_S*fwd, proj, attn_sino, sens_sino, fwhm = fwhm)
  
    pnsq = np.linalg.norm(back)
    print(i,np.sqrt(pnsq))
  
    x = back / pnsq

  return x  


#---------------------------------------------------------------------------------
# parse the command line

parser = argparse.ArgumentParser()
parser.add_argument('--fwhm_mm',  help = 'psf modeling FWHM mm',  default = 4.5, type = float)
parser.add_argument('--fwhm_data_mm',  help = 'psf for data FWHM mm',  default = 4.5, type = float)
parser.add_argument('--phantom', help = 'phantom to use', default = 'brain2d')
parser.add_argument('--npower', help = 'number of power iterations', default = 500, type = int)
args = parser.parse_args()

#---------------------------------------------------------------------------------

fwhm_mm       = args.fwhm_mm
fwhm_data_mm  = args.fwhm_data_mm
phantom       = args.phantom
npower        = args.npower

#---------------------------------------------------------------------------------
np.random.seed(1)

# setup a scanner with one ring
scanner = ppp.RegularPolygonPETScanner(ncrystals_per_module = np.array([16,1]),
                                       nmodules             = np.array([28,1]))

# setup a test image
voxsize = np.array([2.,2.,2.])
n2      = max(1,int((scanner.xc2.max() - scanner.xc2.min()) / voxsize[2]))

# convert fwhm from mm to pixels
fwhm      = fwhm_mm / voxsize
fwhm_data = fwhm_data_mm / voxsize

# setup a test image
if phantom == 'ellipse2d':
  n   = 200
  img = np.zeros((n,n,n2), dtype = np.float32)
  tmp = ellipse_phantom(n = n, c = 3)
  for i2 in range(n2):
    img[:,:,i2] = tmp
elif phantom == 'brain2d':
  n   = 128
  img = np.zeros((n,n,n2), dtype = np.float32)
  tmp = brain2d_phantom(n = n)
  for i2 in range(n2):
    img[:,:,i2] = tmp

img_origin = (-(np.array(img.shape) / 2) +  0.5) * voxsize

# setup an attenuation image
att_img = np.zeros(img.shape, dtype = np.float32)
att_img[...,0] = binary_fill_holes(img[...,0] > 0) * 0.01

# generate TOF sinogram parameters and the TOF projector
sino_params = ppp.PETSinogramParameters(scanner, ntofbins = 17, tofbin_width = 15.)
proj        = ppp.SinogramProjector(scanner, sino_params, img.shape, nsubsets = 1, 
                                    voxsize = voxsize, img_origin = img_origin,
                                    tof = True, sigma_tof = 60./2.35, n_sigmas = 3.)


# create the attenuation sinogram
proj.set_tof(False)
attn_sino = np.exp(-proj.fwd_project(att_img))
proj.set_tof(True)
# generate the sensitivity sinogram
sens_sino = np.ones(proj.sino_params.nontof_shape, dtype = np.float32)

eigen = np.zeros((npower,) + img.shape, dtype = img.dtype)

# estimate the norm of the operator
test_img = np.random.randn(*img.shape)

# sensitivity image
back_ones = ppp.pet_back_model(np.ones(proj.sino_params.shape, dtype = np.float32), 
                               proj, attn_sino, sens_sino, fwhm = fwhm)

back_ones_no_att = ppp.pet_back_model(np.ones(proj.sino_params.shape, dtype = np.float32), 
                                      proj, sens_sino, sens_sino, fwhm = fwhm)



eigen  = power_iterations(test_img, proj, sens_sino, sens_sino, fwhm, npower, precond = False)
eigen2 = power_iterations(test_img, proj, sens_sino, sens_sino, fwhm, npower, precond = True)
eigen3 = power_iterations(test_img, proj, attn_sino, sens_sino, fwhm, npower, precond = False)
eigen4 = power_iterations(test_img, proj, attn_sino, sens_sino, fwhm, npower, precond = True)

#-----------------------------------------------------------------------------------------------------

fig, ax = plt.subplots(2,4, figsize = (16,8))
i00 = ax[0,0].imshow(att_img, cmap = plt.cm.magma)
i01 = ax[0,1].imshow(back_ones_no_att, cmap = plt.cm.magma)
i02 = ax[0,2].imshow(back_ones, cmap = plt.cm.magma)

ax[0,0].set_title('attenuation image')
ax[0,1].set_title('A^T 1, no attn.')
ax[0,2].set_title('A^T 1, with attn.')
ax[0,3].set_axis_off()

fig.colorbar(i00, ax = ax[0,0], location = 'bottom')
fig.colorbar(i01, ax = ax[0,1], location = 'bottom')
fig.colorbar(i02, ax = ax[0,2], location = 'bottom')

i10 = ax[1,0].imshow(eigen.squeeze(), cmap = plt.cm.magma)
i11 = ax[1,1].imshow(eigen2.squeeze(), cmap = plt.cm.magma)
i12 = ax[1,2].imshow(eigen3.squeeze(), cmap = plt.cm.magma)
i13 = ax[1,3].imshow(eigen4.squeeze(), cmap = plt.cm.magma)

ax[1,0].set_title('EV (A A^T) no attn., no precond')
ax[1,1].set_title('EV (A A^T) no attn., precond')
ax[1,2].set_title('EV (A A^T) attn., no precond')
ax[1,3].set_title('EV (A A^T) attn., precond')

fig.colorbar(i10, ax = ax[1,0], location = 'bottom')
fig.colorbar(i11, ax = ax[1,1], location = 'bottom')
fig.colorbar(i12, ax = ax[1,2], location = 'bottom')
fig.colorbar(i13, ax = ax[1,3], location = 'bottom')

fig.tight_layout()
fig.show()
