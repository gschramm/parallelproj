# small demo for stochastic primal hybrid dual gradient algorithm without regulatization 
# Ehrhaardt et al. PMB 2019 "Faster PET reconstruction with non-smooth priors by 
# randomization and preconditioning"

# open questions:
# (1) tuning of gamma vs counts -> 1/img.max() -> 1e4/counts for ||A|| = 1
# (2) random vs ordered subsets

import os
import matplotlib.pyplot as plt
import pyparallelproj as ppp
import numpy as np
import argparse

from pymirc.image_operations import grad, div

#---------------------------------------------------------------------------------
# parse the command line

parser = argparse.ArgumentParser()
parser.add_argument('--ngpus',     help = 'number of GPUs to use', default = 0,   type = int)
parser.add_argument('--counts',    help = 'counts to simulate',    default = 1e6, type = float)
parser.add_argument('--niter',     help = 'number of iterations',  default = 10,  type = int)
parser.add_argument('--nsubsets',  help = 'number of subsets',     default = 28,  type = int)
parser.add_argument('--likeli',    help = 'calc logLikelihodd',    action = 'store_true')
parser.add_argument('--scat_frac', help = 'scatter fraction',      default = 0.2, type = float)
parser.add_argument('--beta',      help = 'beta for TV',           default = 1e-2, type = float)
parser.add_argument('--fwhm_mm',   help = 'psf modeling FWHM mm',  default = 4.5, type = float)
parser.add_argument('--fwhm_data_mm',  help = 'psf for data FWHM mm',  default = 4.5, type = float)
args = parser.parse_args()

#---------------------------------------------------------------------------------

ngpus     = args.ngpus
counts    = args.counts
niter     = args.niter
nsubsets  = args.nsubsets
beta      = args.beta
track_likelihood = args.likeli
scat_frac = args.scat_frac
fwhm_mm      = args.fwhm_mm
fwhm_data_mm = args.fwhm_data_mm

#---------------------------------------------------------------------------------

np.random.seed(1)

# setup a scanner with one ring
scanner = ppp.RegularPolygonPETScanner(ncrystals_per_module = np.array([16,1]),
                                       nmodules             = np.array([28,1]))

# setup a test image
voxsize = np.array([2.,2.,2.])
n0      = 120
n1      = 120
n2      = max(1,int((scanner.xc2.max() - scanner.xc2.min()) / voxsize[2]))

# convert fwhm from mm to pixels
fwhm      = fwhm_mm / voxsize
fwhm_data = fwhm_data_mm / voxsize

# setup a random image
img = np.zeros((n0,n1,n2), dtype = np.float32)
img[(n0//4):(3*n0//4),(n1//4):(3*n1//4),:] = 1
img_origin = (-(np.array(img.shape) / 2) +  0.5) * voxsize

# setup an attenuation image
att_img = (img > 0) * 0.01 * voxsize[0]

# generate nonTOF sinogram parameters and the nonTOF projector for attenuation projection
sino_params_nt = ppp.PETSinogramParameters(scanner)
proj_nt        = ppp.SinogramProjector(scanner, sino_params_nt, img.shape, nsubsets = nsubsets, 
                                    voxsize = voxsize, img_origin = img_origin, ngpus = ngpus)
sino_shape_nt  = sino_params_nt.shape

attn_sino = np.zeros((nsubsets, sino_shape_nt[0], sino_shape_nt[1] // nsubsets, 
                      sino_shape_nt[2], sino_shape_nt[3]), dtype = np.float32)

for i in range(nsubsets):
    attn_sino[i, ...] = np.exp(-proj_nt.fwd_project(att_img, subset=i))

# generate TOF sinogram parameters and the TOF projector
sino_params = ppp.PETSinogramParameters(scanner, ntofbins = 17, tofbin_width = 15.)
proj        = ppp.SinogramProjector(scanner, sino_params, img.shape, nsubsets = nsubsets, 
                                    voxsize = voxsize, img_origin = img_origin, ngpus = ngpus,
                                    tof = True, sigma_tof = 60./2.35, n_sigmas = 3.)

# allocate array for the subset sinogram
sino_shape = sino_params.shape

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# power iterations to estimate the norm of one subset fwd + back projection
# of the raw projector

test_img = np.random.rand(*img.shape).astype(np.float32)
for pi in range(20):
  fwd  = attn_sino[0, ...]*proj.fwd_project(test_img, subset = 0)
  back = proj.back_project(attn_sino[0, ...]*fwd, subset = 0)
  norm = np.linalg.norm(back)
  
  test_img = back / norm

# calculate the norm of the full projector
pr_norm = np.sqrt(nsubsets*norm)

# norm of the gradient operator = sqrt(ndim*4)
grad_norm = np.sqrt(2*4)
#---------------------------------------------------------------------------

# generate the sensitivity sinogram
sens_sino = np.ones((nsubsets, sino_shape_nt[0], sino_shape_nt[1] // nsubsets, 
                      sino_shape_nt[2], sino_shape_nt[3]), dtype = np.float32) / pr_norm

img_fwd    = np.zeros((nsubsets, sino_shape[0], sino_shape[1] // nsubsets, sino_shape[2], 
                       sino_shape[3]), dtype = np.float32)

# forward project the image
for i in range(nsubsets):
  img_fwd[i,...] = ppp.pet_fwd_model(img, proj, attn_sino, sens_sino, i, fwhm = fwhm_data)

# scale sum of fwd image to counts
if counts > 0:
  scale_fac = (counts / img_fwd.sum())
  img_fwd  *= scale_fac 
  img      *= scale_fac 

  # contamination sinogram with scatter and randoms
  contam_sino = np.full(img_fwd.shape, 0.2*img_fwd.mean(), dtype = np.float32)
  
  em_sino = np.random.poisson(img_fwd + contam_sino)
else:
  scale_fac = 1.

  # contamination sinogram with scatter and randoms
  contam_sino = np.full(img_fwd.shape, 0.2*img_fwd.mean(), dtype = np.float32)

  em_sino = img_fwd + contam_sino


#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
# SPDHG algorithm

rho   = 0.999

gamma = 1./img.max()

# calculate the "step sizes" S_i, T_i  for the projector
S_i = np.zeros(img_fwd.shape, dtype = np.float32)

ones_img = np.ones(img.shape, dtype = np.float32)
for i in range(nsubsets):
  S_i[i,...] = (gamma*rho) / ppp.pet_fwd_model(ones_img, proj, attn_sino, sens_sino, i, fwhm = fwhm)
# clip inf values
S_i[S_i == np.inf] = S_i[S_i != np.inf].max()

# calculate S for the gradient operator
S_g = (gamma*rho/grad_norm)


ones_sino = np.ones((sino_shape[0], sino_shape[1] // nsubsets, sino_shape[2], 
                     sino_shape[3]), dtype = np.float32)
T_i = np.zeros((nsubsets + 1,) + img.shape, dtype = np.float32)
for i in range(nsubsets):
  T_i[i,...] = (rho/(nsubsets*gamma)) / ppp.pet_back_model(ones_sino, proj, attn_sino, sens_sino, i, fwhm = fwhm)

# calculate T for the gradient operator
T_i[-1,...] = np.full(img.shape, rho/(2*gamma*grad_norm), dtype = np.float32)

# take the element-wise min of the T_i's of all subsets
T = T_i.min(axis = 0)


#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

# start SPDHG iterations
x      = np.zeros(img.shape, dtype = np.float32)
z      = np.zeros(img.shape, dtype = np.float32)
zbar   = np.zeros(img.shape, dtype = np.float32)
y      = np.zeros(img_fwd.shape, dtype = np.float32)

# allocate arrays for gradient operations
x_grad      = np.zeros((img.ndim,) + img.shape, dtype = np.float32)
y_grad      = np.zeros((img.ndim,) + img.shape, dtype = np.float32)
y_grad_plus = np.zeros((img.ndim,) + img.shape, dtype = np.float32)

if track_likelihood:
  logL = np.zeros(niter)

plt.ion()
fig, ax = plt.subplots(1,3, figsize = (12,4))
ax[0].imshow(img[...,n2//2],   vmin = 0, vmax = 1.3*img.max(), cmap = plt.cm.Greys)
ax[0].set_title('ground truth')
ir = ax[1].imshow(x[...,n2//2], vmin = 0, vmax = 1.3*img.max(), cmap = plt.cm.Greys)
ax[1].set_title('intial recon')
ib = ax[2].imshow(x[...,n2//2] - img[...,n2//2], vmin = -0.2*img.max(), vmax = 0.2*img.max(), 
                  cmap = plt.cm.bwr)
ax[2].set_title('bias')
fig.tight_layout()
fig.canvas.draw()

for it in range(niter):
  subset_sequence = np.random.permutation(np.arange(2*nsubsets))
  for ss in range(2*nsubsets):
    # select a random subset
    i = subset_sequence[ss]

    if i < nsubsets:
      print(f'iteration {it + 1} update {ss+1} subset {i}')

      x = np.clip(x - T*zbar, 0, None)

      y_plus = y[i,...] + S_i[i,...]*(ppp.pet_fwd_model(x, proj, attn_sino, sens_sino, i, fwhm = fwhm) + contam_sino[i,...])

      # apply the prox for the dual of the poisson logL
      y_plus = 0.5*(y_plus + 1 - np.sqrt((y_plus - 1)**2 + 4*S_i[i,...]*em_sino[i,...]))

      dz = ppp.pet_back_model(y_plus - y[i,...], proj, attn_sino, sens_sino, i, fwhm = fwhm)

      # update variables
      z = z + dz
      y[i,...] = y_plus.copy()
      zbar = z + dz*2*nsubsets
    else:
      print(f'iteration {it + 1} update {ss+1} gradient update')

      grad(x, x_grad)
      y_grad_plus = (y_grad + S_g*x_grad).reshape(img.ndim,-1)

      # proximity operator for dual of TV
      gnorm = np.linalg.norm(y_grad_plus, axis = 0)
      y_grad_plus /= np.maximum(np.ones(gnorm.shape, np.float32), gnorm / beta)
      y_grad_plus = y_grad_plus.reshape(x_grad.shape)

      dz = -1*div(y_grad_plus - y_grad)

      # update variables
      z = z + dz
      y_grad = y_grad_plus.copy()
      zbar = z + dz*2


    ir.set_data(x[...,n2//2])
    ax[1].set_title(f'itertation {it+1} update {ss+1}')
    ib.set_data(x[...,n2//2] - img[...,n2//2])
    fig.canvas.draw()

    # calculate the likelihood
    if track_likelihood and ss == (nsubsets - 1):
      exp = np.zeros(img_fwd.shape, dtype = np.float32)
      for ii in range(nsubsets):
        exp[ii,...] = ppp.pet_fwd_model(x, proj, attn_sino, sens_sino, ii, fwhm = fwhm)+contam_sino[ii,...]
      grad(x, x_grad)

      logL[it] = (exp - em_sino*np.log(exp)).sum() + beta*np.linalg.norm(x_grad, axis = 0).sum()
      print(f'neg logL {logL[it]}')

#--------------------------------------------------------------------------------------------------

if track_likelihood:
  fig2, ax2 = plt.subplots(1,1, figsize = (4,4))
  ax2.plot(np.arange(niter) + 1, logL)
  fig2.tight_layout()
  fig2.show()
