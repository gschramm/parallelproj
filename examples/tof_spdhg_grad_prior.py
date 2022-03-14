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

#---------------------------------------------------------------------------------
# parse the command line

parser = argparse.ArgumentParser()
parser.add_argument('--counts',   help = 'counts to simulate',    default = 1e6, type = float)
parser.add_argument('--niter',    help = 'number of iterations',  default = 20,  type = int)
parser.add_argument('--nsubsets',   help = 'number of subsets',     default = 56,  type = int)
parser.add_argument('--fwhm_mm',  help = 'psf modeling FWHM mm',  default = 4.5, type = float)
parser.add_argument('--fwhm_data_mm',  help = 'psf for data FWHM mm',  default = 4.5, type = float)
parser.add_argument('--phantom', help = 'phantom to use', default = 'brain2d')
parser.add_argument('--seed',    help = 'seed for random generator', default = 1, type = int)
parser.add_argument('--rel_gamma', help = 'relative step size ratio',  default = 30, type = float)
parser.add_argument('--beta',  help = 'prior strength',  default = 0.1, type = float)
parser.add_argument('--norm',  help = 'name of gradient norm',  default = 'l2_l1', 
                               choices = ['l2_sq', 'l2_l1'])
parser.add_argument('--use_structure',  help = 'use structural info for prior',  action = 'store_true')
args = parser.parse_args()

#---------------------------------------------------------------------------------

counts        = args.counts
niter         = args.niter
nsubsets      = args.nsubsets
fwhm_mm       = args.fwhm_mm
fwhm_data_mm  = args.fwhm_data_mm
phantom       = args.phantom
seed          = args.seed
beta          = args.beta
norm          = args.norm
use_structure = args.use_structure
rel_gamma     = args.rel_gamma

#---------------------------------------------------------------------------------

np.random.seed(seed)

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
att_img = (img > 0) * 0.01

# generate TOF sinogram parameters and the TOF projector
sino_params = ppp.PETSinogramParameters(scanner, ntofbins = 17, tofbin_width = 15.)
proj        = ppp.SinogramProjector(scanner, sino_params, img.shape, nsubsets = nsubsets, 
                                    voxsize = voxsize, img_origin = img_origin,
                                    tof = True, sigma_tof = 60./2.35, n_sigmas = 3.)


# create the attenuation sinogram
proj.set_tof(False)
attn_sino = np.exp(-proj.fwd_project(att_img))
proj.set_tof(True)
# generate the sensitivity sinogram
sens_sino = np.ones(proj.sino_params.nontof_shape, dtype = np.float32)

# estimate the norm of the operator
test_img = np.random.rand(*img.shape)
for i in range(5):
  fwd  = ppp.pet_fwd_model(test_img, proj, attn_sino, sens_sino, fwhm = fwhm)
  back = ppp.pet_back_model(fwd, proj, attn_sino, sens_sino, fwhm = fwhm)

  pnsq = np.linalg.norm(back)
  print(i,np.sqrt(pnsq))

  test_img = back / pnsq

# normalize sensitivity sinogram to get PET forward model for 1 view with norm 1
# this is important otherwise the step size T in SPDHG get dominated by the gradient
sens_sino /= (np.sqrt(pnsq)/np.sqrt(8))
# forward project the image
img_fwd= ppp.pet_fwd_model(img, proj, attn_sino, sens_sino, fwhm = fwhm_data)

# scale sum of fwd image to counts
if counts > 0:
  scale_fac = (counts / img_fwd.sum())
  img_fwd  *= scale_fac 
  img      *= scale_fac 

  # contamination sinogram with scatter and randoms
  # useful to avoid division by 0 in the ratio of data and exprected data
  contam_sino = np.full(img_fwd.shape, 0.75*img_fwd.mean(), dtype = np.float32)
  
  em_sino = np.random.poisson(img_fwd + contam_sino)
else:
  scale_fac = 1.

  # contamination sinogram with sctter and randoms
  # useful to avoid division by 0 in the ratio of data and exprected data
  contam_sino = np.full(img_fwd.shape, 0.75*img_fwd.mean(), dtype = np.float32)

  em_sino = img_fwd + contam_sino


#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------

# setup joint gradient field
if use_structure:
  G0            = GradientOperator()
  grad_operator = GradientOperator(joint_grad_field = G0.fwd(-(img**0.5)))
else:
  grad_operator = GradientOperator()

grad_norm = GradientNorm(name = norm)

#-----------------------------------------------------------------------------------------------------
# callback function to calculate cost after every iteration

def calc_cost(x):
  exp  = ppp.pet_fwd_model(x, proj, attn_sino, sens_sino, fwhm = fwhm) + contam_sino
  cost = (exp - em_sino*np.log(exp)).sum()

  if beta > 0:
    cost += beta*grad_norm.eval(grad_operator.fwd(x))

  return cost

def _cb(x, **kwargs):
  it = kwargs.get('iteration',0)
  if 'cost' in kwargs:
    kwargs['cost'][it-1] = calc_cost(x)
  if 'xm' in kwargs:
    kwargs['xm'].append(x)
#-----------------------------------------------------------------------------------------------------

cost = np.zeros(niter)
xm   = [] 
recon = spdhg(em_sino, attn_sino, sens_sino, contam_sino, proj, niter,
              gamma = rel_gamma/img.max(), fwhm = fwhm, verbose = True, rho = 1,
              xstart = None, grad_operator = grad_operator, grad_norm = grad_norm, beta = beta,
              callback = _cb, callback_kwargs = {'cost': cost, 'xm':xm})
xm = np.array(xm)

cost2 = np.zeros(niter)
xm2   = [] 
recon2 = spdhg(em_sino, attn_sino, sens_sino, contam_sino, proj, niter,
              gamma = rel_gamma/img.max(), fwhm = fwhm, verbose = True, rho = 4./3,
              xstart = None, grad_operator = grad_operator, grad_norm = grad_norm, beta = beta,
              callback = _cb, callback_kwargs = {'cost': cost2, 'xm':xm2})
xm2 = np.array(xm2)

cost3 = np.zeros(niter)
xm3   = [] 
recon3 = spdhg(em_sino, attn_sino, sens_sino, contam_sino, proj, niter,
              gamma = rel_gamma/img.max(), fwhm = fwhm, verbose = True, rho = 4.,
              xstart = None, grad_operator = grad_operator, grad_norm = grad_norm, beta = beta,
              callback = _cb, callback_kwargs = {'cost': cost3, 'xm':xm3})
xm3 = np.array(xm3)

cost4 = np.zeros(niter)
xm4   = [] 
recon4 = spdhg(em_sino, attn_sino, sens_sino, contam_sino, proj, niter,
              gamma = rel_gamma/img.max(), fwhm = fwhm, verbose = True, rho = 16.,
              xstart = None, grad_operator = grad_operator, grad_norm = grad_norm, beta = beta,
              callback = _cb, callback_kwargs = {'cost': cost4, 'xm':xm4})
xm4 = np.array(xm4)





#--- visualization
ims_kwargs = dict(cmap = plt.cm.Greys, vmin = 0, vmax = 1.2*img.max())

fig, ax = plt.subplots(3,4, figsize = (12,9))
ax[0,0].imshow(img.squeeze(), **ims_kwargs)
ax[0,0].set_title(f'ground truth')

ymin = min(cost.min(), cost2.min(), cost3.min(), cost4.min())
ymax = cost[3:].max()

ax[0,1].plot(np.arange(1,niter+1), cost,  label = 'rho = 1')
ax[0,1].plot(np.arange(1,niter+1), cost2, label = 'rho = 4/3')
ax[0,1].plot(np.arange(1,niter+1), cost3, label = 'rho = 4')
ax[0,1].plot(np.arange(1,niter+1), cost4, label = 'rho = 16')
ax[0,1].grid(ls = ':')
ax[0,1].set_xlabel('epoch')
ax[0,1].set_ylabel('cost')
ax[0,1].set_ylim(ymin, ymax)
ax[0,1].legend()

ax[0,2].plot(np.arange(1,21), cost[:20],  label = 'rho = 1')
ax[0,2].plot(np.arange(1,21), cost2[:20], label = 'rho = 4/3')
ax[0,2].plot(np.arange(1,21), cost3[:20], label = 'rho = 4')
ax[0,2].plot(np.arange(1,21), cost4[:20], label = 'rho = 16')
ax[0,2].grid(ls = ':')
ax[0,2].set_xlabel('epoch')
ax[0,2].set_ylabel('cost')
ax[0,2].set_ylim(ymin, ymax)
ax[0,2].legend()

ax[0,3].plot(np.arange(niter + 1 - 5,niter+1), cost[-5:],  label = 'rho = 1')
ax[0,3].plot(np.arange(niter + 1 - 5,niter+1), cost2[-5:], label = 'rho = 4/3')
ax[0,3].plot(np.arange(niter + 1 - 5,niter+1), cost3[-5:], label = 'rho = 4')
ax[0,3].plot(np.arange(niter + 1 - 5,niter+1), cost4[-5:], label = 'rho = 16')
ax[0,3].grid(ls = ':')
ax[0,3].set_xlabel('epoch')
ax[0,3].set_ylabel('cost')
ax[0,3].set_ylim(ymin, cost[-5:].max())
ax[0,3].legend()

ax[1,0].imshow(recon.squeeze(),  **ims_kwargs)
ax[1,1].imshow(recon2.squeeze(), **ims_kwargs)
ax[1,2].imshow(recon3.squeeze(), **ims_kwargs)
ax[1,3].imshow(recon4.squeeze(), **ims_kwargs)
ax[1,0].set_title(f'rho = 1, {niter} ep., {nsubsets} ss')
ax[1,1].set_title(f'rho = 4/3, {niter} ep., {nsubsets} ss')
ax[1,2].set_title(f'rho = 4, {niter} ep., {nsubsets} ss')
ax[1,3].set_title(f'rho = 16, {niter} ep., {nsubsets} ss')

ax[2,0].imshow(xm[10,...].squeeze(),  **ims_kwargs)
ax[2,1].imshow(xm2[10,...].squeeze(), **ims_kwargs)
ax[2,2].imshow(xm3[10,...].squeeze(), **ims_kwargs)
ax[2,3].imshow(xm4[10,...].squeeze(), **ims_kwargs)
ax[2,0].set_title(f'rho = 1, 10 ep., {nsubsets} ss')
ax[2,1].set_title(f'rho = 4/3, 10 ep., {nsubsets} ss')
ax[2,2].set_title(f'rho = 4, 10 ep., {nsubsets} ss')
ax[2,3].set_title(f'rho = 16, 10 ep., {nsubsets} ss')

fig.tight_layout()
fig.show()
