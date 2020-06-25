# small demo for stochastic primal hybrid dual gradient algorithm without regulatization 
# Ehrhaardt et al. PMB 2019 "Faster PET reconstruction with non-smooth priors by 
# randomization and preconditioning"

import os
import matplotlib.pyplot as py
import pyparallelproj as ppp
import numpy as np
import argparse

from pymirc.image_operations import grad, div

#---------------------------------------------------------------------------------
# parse the command line

parser = argparse.ArgumentParser()
parser.add_argument('--ngpus',    help = 'number of GPUs to use', default = 0,   type = int)
parser.add_argument('--counts',   help = 'counts to simulate',    default = 1e5, type = float)
parser.add_argument('--niter',    help = 'number of iterations',  default = 5,   type = int)
parser.add_argument('--nsubsets', help = 'number of subsets',     default = 28,  type = int)
parser.add_argument('--likeli',   help = 'calc logLikelihodd',    action = 'store_true')
args = parser.parse_args()

#---------------------------------------------------------------------------------

ngpus     = args.ngpus
counts    = args.counts
niter     = args.niter
nsubsets  = args.nsubsets
track_likelihood = args.likeli

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


# setup a random image
img = np.zeros((n0,n1,n2), dtype = np.float32)
img[(n0//4):(3*n0//4),(n1//4):(3*n1//4),:] = 1
img_origin = (-(np.array(img.shape) / 2) +  0.5) * voxsize

# generate sinogram parameters and the projector
sino_params = ppp.PETSinogramParameters(scanner, ntofbins = 17, tofbin_width = 15.)
proj        = ppp.SinogramProjector(scanner, sino_params, img.shape, nsubsets = nsubsets, 
                                    voxsize = voxsize, img_origin = img_origin, ngpus = ngpus,
                                    tof = True, sigma_tof = 60./2.35, n_sigmas = 3.)

# allocate array for the subset sinogram
sino_shape = sino_params.shape
img_fwd    = np.zeros((nsubsets, sino_shape[0], sino_shape[1] // nsubsets, sino_shape[2], sino_shape[3]),
                      dtype = np.float32)


#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# power iterations to estimate the norm of one subset fwd + back projection

test_img = np.random.rand(*img.shape).astype(np.float32)
for pi in range(20):
  fwd  = proj.fwd_project(test_img, subset = 0)
  back = proj.back_project(fwd, subset = 0)
  norm = np.linalg.norm(back)
  
  test_img = back / norm

pr_norm = np.sqrt(norm)

# power iterations to estimate the norm of gradient
test_img = np.random.rand(*img.shape).astype(np.float32)
for pi in range(20):
  fwd = np.zeros((test_img.ndim,) + test_img.shape, dtype = np.float32) 
  grad(test_img, fwd)
  back = -div(fwd)
  norm = np.linalg.norm(back)
  
  test_img = back / norm

grad_norm = np.sqrt(norm)


#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# simulate data

# forward project the image
for i in range(nsubsets):
  img_fwd[i,...] = proj.fwd_project(img, subset = i)

# generate sensitity sinogram (product from attenuation and normalization sinogram)
# this sinogram is usually non TOF!
# we scale the sens sino in a way to get approx the same norm for a subset projection
# and the gradient operator
sens_sino  = (grad_norm / pr_norm) * np.ones(img_fwd.shape[:-1], dtype = np.float32)

# scale sum of fwd image to counts
if counts > 0:
  scale_fac = (counts / ((grad_norm / pr_norm) * img_fwd.sum()))
  img_fwd  *= scale_fac 
  img      *= scale_fac 

  # contamination sinogram with scatter and randoms
  # useful to avoid division by 0 in the ratio of data and exprected data
  contam_sino = np.full(img_fwd.shape, 0.2*(grad_norm / pr_norm)*img_fwd.mean(), dtype = np.float32)
  
  em_sino = np.random.poisson(sens_sino[...,np.newaxis]*img_fwd + contam_sino)
else:
  scale_fac = 1.

  # contamination sinogram with sctter and randoms
  # useful to avoid division by 0 in the ratio of data and exprected data
  contam_sino = np.full(img_fwd.shape, 0.2*img_fwd.mean(), dtype = np.float32)

  em_sino = sens_sino[...,np.newaxis]*img_fwd + contam_sino


#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
# SPDHG algorithm

rho   = 0.999
gamma = 1.

# calculate the "step sizes" S_i, T_i and T
S_i = np.zeros((nsubsets, sino_shape[0], sino_shape[1] // nsubsets, sino_shape[2], sino_shape[3]),
                dtype = np.float32)

ones_img = np.ones(img.shape, dtype = np.float32)
for i in range(nsubsets):
  S_i[i,...] = (gamma*rho) / (sens_sino[i,...][...,np.newaxis]*proj.fwd_project(ones_img, subset = i))

# clip inf values
S_i[S_i == np.inf] = S_i[S_i != np.inf].max()

T_i = np.zeros((nsubsets,) + img.shape, dtype = np.float32)
for i in range(nsubsets):
  T_i[i,...] = (rho/(nsubsets*gamma)) / proj.back_project(sens_sino[i,...].repeat(proj.ntofbins).reshape(sens_sino[i,...].shape + (proj.ntofbins,)), subset = i) 

# take the element-wise min of the T_i's of all subsets
T = T_i.min(axis = 0)


# start SPDHG iterations
x  = np.zeros(img.shape, dtype = np.float32)

z      = np.zeros(img.shape, dtype = np.float32)
zbar   = np.zeros(img.shape, dtype = np.float32)
y      = np.zeros(img_fwd.shape, dtype = np.float32)

py.ion()
fig, ax = py.subplots(1,3, figsize = (12,4))
ax[0].imshow(img[...,n2//2],   vmin = 0, vmax = 1.3*img.max(), cmap = py.cm.Greys)
ax[0].set_title('ground truth')
ir = ax[1].imshow(x[...,n2//2], vmin = 0, vmax = 1.3*img.max(), cmap = py.cm.Greys)
ax[1].set_title('intial x')
ib = ax[2].imshow(x[...,n2//2] - img[...,n2//2], vmin = -0.2*img.max(), vmax = 0.2*img.max(), 
                  cmap = py.cm.bwr)
ax[2].set_title('bias')
fig.tight_layout()
fig.canvas.draw()

if track_likelihood:
  logL = np.zeros(niter)

for it in range(niter):
  for ss in range(nsubsets):
    x = np.clip(x - T*zbar, 0, None)

    # select a random subset
    i = np.random.randint(0,nsubsets)
    print(f'iteration {it + 1} subset {i} step {ss}')

    y_plus = y[i,...] + S_i[i,...]*(sens_sino[i,...][...,np.newaxis]*proj.fwd_project(x, subset = i) + contam_sino[i,...])

    # apply the prox for the dual of the poisson logL
    y_plus = 0.5*(y_plus + 1 - np.sqrt((y_plus - 1)**2 + 4*S_i[i,...]*em_sino[i,...]))

    dz = proj.back_project(sens_sino[i,...][...,np.newaxis]*(y_plus - y[i,...]), subset = i)

    # update variables
    z = z + dz
    y[i,...] = y_plus.copy()
    zbar = z + dz*nsubsets

    ir.set_data(x[...,n2//2])
    ax[1].set_title(f'itertation {it+1} subset {i+1}')
    ib.set_data(x[...,n2//2] - img[...,n2//2])
    fig.canvas.draw()

  if track_likelihood:
    exp = np.zeros(img_fwd.shape, dtype = np.float32)
    for ii in range(nsubsets):
      exp[ii,...] = (sens_sino[ii,...][...,np.newaxis]*proj.fwd_project(x, subset = ii) + 
                    contam_sino[ii,...])
    logL[it] = (exp - em_sino*np.log(exp)).sum()
    print(f'neg logL {logL[it]}')


if track_likelihood:
  fig2, ax2 = py.subplots(1,1, figsize = (4,4))
  ax2.plot(np.arange(niter) + 1, logL)
  fig2.tight_layout()
  fig2.show()
