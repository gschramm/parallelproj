# small demo for sinogram TOF OS-MLEM

import os
import matplotlib.pyplot as plt
import pyparallelproj as ppp
from pyparallelproj.algorithms import osem, spdhg

import numpy as np
from scipy.ndimage import gaussian_filter

import argparse

#---------------------------------------------------------------------------------
# parse the command line

parser = argparse.ArgumentParser()
parser.add_argument('--ngpus',    help = 'number of GPUs to use', default = 0,   type = int)
parser.add_argument('--counts',   help = 'counts to simulate',    default = 1e6, type = float)
parser.add_argument('--niter',    help = 'number of iterations',  default = 2,   type = int)
parser.add_argument('--nsubsets', help = 'number of subsets',     default = 28,  type = int)
parser.add_argument('--likeli',   help = 'calc logLikelihodd',    action = 'store_true')
parser.add_argument('--fwhm_mm',  help = 'psf modeling FWHM mm',  default = 4.5, type = float)
parser.add_argument('--fwhm_data_mm',  help = 'psf for data FWHM mm',  default = 4.5, type = float)
parser.add_argument('--gamma',     help = 'gamma parameter',       default = 1., type = float)
args = parser.parse_args()

#---------------------------------------------------------------------------------

ngpus            = args.ngpus
counts           = args.counts
niter            = args.niter
nsubsets         = args.nsubsets
fwhm_mm          = args.fwhm_mm
fwhm_data_mm     = args.fwhm_data_mm
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
proj_nt        = ppp.SinogramProjector(scanner, sino_params_nt, img.shape, nsubsets = 1, 
                                    voxsize = voxsize, img_origin = img_origin, ngpus = ngpus)

attn_sino = np.exp(-proj_nt.fwd_project(att_img))

# generate the sensitivity sinogram
sens_sino = np.ones(sino_params_nt.shape, dtype = np.float32)

# generate TOF sinogram parameters and the TOF projector
sino_params = ppp.PETSinogramParameters(scanner, ntofbins = 17, tofbin_width = 15.)
proj        = ppp.SinogramProjector(scanner, sino_params, img.shape, nsubsets = 1, 
                                    voxsize = voxsize, img_origin = img_origin, ngpus = ngpus,
                                    tof = True, sigma_tof = 60./2.35, n_sigmas = 3.)

# allocate array for the subset sinogram
img_fwd    = np.zeros(sino_params.shape, dtype = np.float32)

# forward project the image
img_fwd= ppp.pet_fwd_model(img, proj, attn_sino, sens_sino, 0, fwhm = fwhm_data)

# scale sum of fwd image to counts
if counts > 0:
  scale_fac = (counts / img_fwd.sum())
  img_fwd  *= scale_fac 
  img      *= scale_fac 

  # contamination sinogram with scatter and randoms
  # useful to avoid division by 0 in the ratio of data and exprected data
  contam_sino = np.full(img_fwd.shape, 0.2*img_fwd.mean(), dtype = np.float32)
  
  em_sino = np.random.poisson(img_fwd + contam_sino)
else:
  scale_fac = 1.

  # contamination sinogram with sctter and randoms
  # useful to avoid division by 0 in the ratio of data and exprected data
  contam_sino = np.full(img_fwd.shape, 0.2*img_fwd.mean(), dtype = np.float32)

  em_sino = img_fwd + contam_sino

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
#--- OS-MLEM reconstruction

if track_likelihood:
  cost_osem  = np.zeros(niter)
  cost_spdhg = np.zeros(niter)
else:
  cost_osem  = None
  cost_spdhg = None

fig, ax = plt.subplots(1,3, figsize = (12,4))
ax[0].imshow(img[...,n2//2],   vmin = 0, vmax = 1.3*img.max(), cmap = plt.cm.Greys)
ax[0].set_title('ground truth')
ir = ax[1].imshow(0*img[...,n2//2], vmin = 0, vmax = 1.3*img.max(), cmap = plt.cm.Greys)
ax[1].set_title('recon')
ib = ax[2].imshow(img[...,n2//2] - img[...,n2//2], vmin = -0.2*img.max(), vmax = 0.2*img.max(), 
                  cmap = plt.cm.bwr)
ax[2].set_title('bias')
fig.tight_layout()
fig.canvas.draw()


#-----------------------------------------------------------------------------------------------
# callback functions to calculate likelihood and show recon updates

def update_img(x):
  ir.set_data(x[...,n2//2])
  ib.set_data(x[...,n2//2] - img[...,n2//2])
  plt.pause(1e-6)

def calc_likeli(x):
  logL = 0

  for i in range(proj.nsubsets):
    # get the slice for the current subset
    ss = proj.subset_slices[i]
    exp = ppp.pet_fwd_model(x, proj, attn_sino[ss], sens_sino[ss], i, fwhm = fwhm) + contam_sino[ss]
    logL += (exp - em_sino[ss]*np.log(exp)).sum()

  return logL

def _cb(x, cost = None):
  if (cost is not None) and track_likelihood:
    cost.append(calc_likeli(x))
  update_img(x)

#-----------------------------------------------------------------------------------------------
init_recon = None

cost_mlem = []
#recon_mlem = osem(em_sino, attn_sino, sens_sino, contam_sino, proj, niter*nsubsets, 
#                  fwhm = fwhm, verbose = True, xstart = init_recon,
#                  callback = _cb, callback_kwargs = {'cost': cost_mlem})

# initialize the subsets for the projector
proj.init_subsets(nsubsets)

cost_osem = []
recon_osem = osem(em_sino, attn_sino, sens_sino, contam_sino, proj, niter, 
                  fwhm = fwhm, verbose = True, xstart = init_recon,
                  callback = _cb, callback_kwargs = {'cost': cost_osem})

ystart = np.zeros(em_sino.shape, dtype = np.float32)
ystart[em_sino == 0] = 1

gammas = [1e-2,3e-2,1e-1,3e-1,1e0,3e0,1e1]
costs_spdhg = np.zeros((len(gammas),niter))
costs_spdhg_sparse = np.zeros((len(gammas),niter))

recons_spdhg        = np.zeros((len(gammas),) + img.shape, dtype = np.float32)
recons_spdhg_sparse = np.zeros((len(gammas),) + img.shape, dtype = np.float32)

for ig, gamma in enumerate(gammas):
  cost_spdhg = []
  recons_spdhg[ig,...] = spdhg(em_sino, attn_sino, sens_sino, contam_sino, proj, niter,
                               gamma = gamma, fwhm = fwhm, verbose = True, xstart = init_recon, 
                               callback = _cb, callback_kwargs = {'cost': cost_spdhg})
  costs_spdhg[ig,:] = cost_spdhg

  cost_spdhg_sparse = [] 
  recons_spdhg_sparse[ig,...] = spdhg(em_sino, attn_sino, sens_sino, contam_sino, proj, niter,
                                      gamma = gamma, fwhm = fwhm, verbose = True,
                                      xstart = init_recon, ystart = ystart, 
                                      callback = _cb, callback_kwargs = {'cost': cost_spdhg_sparse})
  costs_spdhg_sparse[ig,:] = cost_spdhg_sparse



# show the results

if track_likelihood:
  fig2, ax2 = plt.subplots(1,len(gammas), figsize = (16,3))
  it = np.arange(niter) + 1

  for ig, gamma in enumerate(gammas):
    ax2[ig].plot(it,cost_osem, label = 'OSEM')
    ax2[ig].plot(it,costs_spdhg[ig,:], label = f'SPD')
    ax2[ig].plot(it,costs_spdhg_sparse[ig,:], label = f'SPD-S')
    ax2[ig].set_title(f'{gamma}')

  for axx in ax2.flatten():
    axx.legend()
    axx.grid(ls = ':')
    axx.set_ylim(min(min(cost_osem), costs_spdhg.min(), costs_spdhg_sparse.min()), 
                 1.5*max(cost_osem) - 0.5*min(cost_osem))
  fig2.tight_layout()
  fig2.savefig(f'counts_{counts:.1E}_niter_{niter}_nsub{nsubsets}_cost.pdf')
  fig2.show()            

fig3, ax3 = plt.subplots(4,len(gammas) + 1, figsize = (16,8))
vmax = 1.5*img.max()
sig  = 1.5

ax3[0,0].imshow(recon_osem, vmax = vmax, cmap = plt.cm.Greys)
ax3[0,0].set_title('OSEM')
ax3[2,0].imshow(gaussian_filter(recon_osem,sig), vmax = vmax, cmap = plt.cm.Greys)
ax3[2,0].set_title('ps OSEM')

for ig, gamma in enumerate(gammas):
  ax3[0,ig+1].imshow(recons_spdhg[ig,...], vmax = vmax, cmap = plt.cm.Greys)
  ax3[1,ig+1].imshow(recons_spdhg_sparse[ig,...], vmax = vmax, cmap = plt.cm.Greys)
  ax3[0,ig+1].set_title(f'SPD {gamma}')
  ax3[1,ig+1].set_title(f'SPD-S {gamma}')

  ax3[2,ig+1].imshow(gaussian_filter(recons_spdhg[ig,...],sig), vmax = vmax, cmap = plt.cm.Greys)
  ax3[3,ig+1].imshow(gaussian_filter(recons_spdhg_sparse[ig,...],sig), vmax = vmax, cmap = plt.cm.Greys)
  ax3[2,ig+1].set_title(f'ps SPD {gamma}')
  ax3[3,ig+1].set_title(f'ps SPD-S {gamma}')

for axx in ax3.flatten():
  axx.set_axis_off()

fig3.tight_layout()
fig3.savefig(f'counts_{counts:.1E}_niter_{niter}_nsub{nsubsets}.png')
fig3.show()
