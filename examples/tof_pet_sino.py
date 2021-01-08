# small demo for sinogram TOF OS-MLEM

import os
import matplotlib.pyplot as plt
import pyparallelproj as ppp
from pyparallelproj.algorithms import osem, spdhg

import numpy as np
from scipy.ndimage import gaussian_filter

from phantoms import ellipse_phantom

import argparse

#---------------------------------------------------------------------------------
# parse the command line

parser = argparse.ArgumentParser()
parser.add_argument('--ngpus',    help = 'number of GPUs to use', default = 0,   type = int)
parser.add_argument('--counts',   help = 'counts to simulate',    default = 1e5, type = float)
parser.add_argument('--niter',    help = 'number of iterations',  default = 50,  type = int)
parser.add_argument('--niter_mlem', help = 'number of MLEM iterations', default = 5000,  type = int)
parser.add_argument('--nsubsets',   help = 'number of subsets',     default = 28,  type = int)
parser.add_argument('--warm'  ,   help = 'warm start with 1 OSEM it', action = 'store_true')
parser.add_argument('--interactive', help = 'show recons updates', action = 'store_true')
parser.add_argument('--fwhm_mm',  help = 'psf modeling FWHM mm',  default = 4.5, type = float)
parser.add_argument('--fwhm_data_mm',  help = 'psf for data FWHM mm',  default = 4.5, type = float)
parser.add_argument('--phantom', help = 'phantom to use', default = 'ellipse')
parser.add_argument('--seed',    help = 'seed for random generator', default = 1, type = int)
args = parser.parse_args()

#---------------------------------------------------------------------------------

ngpus            = args.ngpus
counts           = args.counts
niter            = args.niter
niter_mlem       = args.niter_mlem
nsubsets         = args.nsubsets
fwhm_mm          = args.fwhm_mm
fwhm_data_mm     = args.fwhm_data_mm
warm             = args.warm
interactive      = args.interactive
phantom          = args.phantom
seed             = args.seed

#---------------------------------------------------------------------------------

np.random.seed(seed)

# setup a scanner with one ring
scanner = ppp.RegularPolygonPETScanner(ncrystals_per_module = np.array([16,1]),
                                       nmodules             = np.array([28,1]))

# setup a test image
voxsize = np.array([2.,2.,2.])
n       = 200
n0      = n
n1      = n
n2      = max(1,int((scanner.xc2.max() - scanner.xc2.min()) / voxsize[2]))

# convert fwhm from mm to pixels
fwhm      = fwhm_mm / voxsize
fwhm_data = fwhm_data_mm / voxsize

# setup a random image
img = np.zeros((n0,n1,n2), dtype = np.float32)

if phantom == 'ellipse':
  for i2 in range(n2):
    img[:,:,i2] = ellipse_phantom(n = n, c = 3)

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

# estimate the norm of the operator
test_img = np.random.rand(*img.shape)
for i in range(10):
  fwd  = ppp.pet_fwd_model(test_img, proj, attn_sino, sens_sino, 0, fwhm = fwhm)
  back = ppp.pet_back_model(fwd, proj, attn_sino, sens_sino, 0, fwhm = fwhm)

  norm = np.linalg.norm(back)
  print(i,norm)

  test_img = back / norm

# normalize sensitivity sinogram to get PET forward model with norm 1
sens_sino /= np.sqrt(norm)

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

if interactive:
  fig, ax = plt.subplots(1,3, figsize = (12,4))
  ax[0].imshow(img[...,n2//2],   vmin = 0, vmax = 1.3*img.max(), cmap = plt.cm.Greys)
  ax[0].set_title('ground truth')
  ir = ax[1].imshow(0*img[...,n2//2], vmin = 0, vmax = 1.3*img.max(), cmap = plt.cm.Greys)
  ax[1].set_title('recon')
  ib = ax[2].imshow(img[...,n2//2] - img[...,n2//2], vmin = -0.2*img.max(), vmax = 0.2*img.max(), 
                    cmap = plt.cm.bwr)
  ax[2].set_title('bias')
  fig.tight_layout()
  plt.pause(1e-6)


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

def _cb(x, **kwargs):
  it = kwargs.get('iteration',0)
  if interactive: 
    update_img(x)
  if 'cost' in kwargs:
    kwargs['cost'][it-1] = calc_likeli(x)
  if 'psnr' in kwargs:
    MSE = ((x - kwargs['xref'])**2).mean()
    kwargs['psnr'][it-1] = 20*np.log10(kwargs['xref'].max()/np.sqrt(MSE))

#-----------------------------------------------------------------------------------------------

# do long MLEM as reference
mlem_fname = os.path.join('data',
                          f'mlem_{phantom}_niter_{niter_mlem}_counts_{counts:.1E}_seed_{seed}.npz')

if os.path.exists(mlem_fname):
  tmp = np.load(mlem_fname)
  recon_mlem = tmp['recon_mlem']
  cost_mlem  = tmp['cost_mlem']
else:
  cost_mlem = np.zeros(niter_mlem)
  recon_mlem = osem(em_sino, attn_sino, sens_sino, contam_sino, proj, niter_mlem, 
                    fwhm = fwhm, verbose = True,
                    callback = _cb, callback_kwargs = {'cost': cost_mlem})

  np.savez(mlem_fname, recon_mlem = recon_mlem, cost_mlem = cost_mlem)

ref_recon = recon_mlem

# initialize the subsets for the projector
proj.init_subsets(nsubsets)

if warm:
  init_recon = osem(em_sino, attn_sino, sens_sino, contam_sino, proj, 1, 
                    fwhm = fwhm, verbose = True)
else:
  init_recon = None

cost_osem = np.zeros(niter)
psnr_osem = np.zeros(niter)

cbk = {'cost':cost_osem, 'xref':ref_recon, 'psnr':psnr_osem}
recon_osem = osem(em_sino, attn_sino, sens_sino, contam_sino, proj, niter, 
                  fwhm = fwhm, verbose = True, xstart = init_recon,
                  callback = _cb, callback_kwargs = cbk)

ystart = np.zeros(em_sino.shape, dtype = np.float32)
ystart[em_sino == 0] = 1

gammas = np.array([0.1,0.3,1,3,10,30,100]) / (counts/1e3)

costs_spdhg = np.zeros((len(gammas),niter))
costs_spdhg_sparse = np.zeros((len(gammas),niter))

psnr_spdhg = np.zeros((len(gammas),niter))
psnr_spdhg_sparse = np.zeros((len(gammas),niter))

recons_spdhg        = np.zeros((len(gammas),) + img.shape, dtype = np.float32)
recons_spdhg_sparse = np.zeros((len(gammas),) + img.shape, dtype = np.float32)

for ig, gamma in enumerate(gammas):
  cbs = {'cost':costs_spdhg[ig,:], 'xref':ref_recon, 'psnr':psnr_spdhg[ig,:]}
  recons_spdhg[ig,...] = spdhg(em_sino, attn_sino, sens_sino, contam_sino, proj, niter,
                               gamma = gamma, fwhm = fwhm, verbose = True, 
                               xstart = init_recon, 
                               callback = _cb, callback_kwargs = cbs)

  cbss = {'cost':costs_spdhg_sparse[ig,:], 'xref':ref_recon, 'psnr':psnr_spdhg_sparse[ig,:]}
  recons_spdhg_sparse[ig,...] = spdhg(em_sino, attn_sino, sens_sino, contam_sino, proj, niter,
                                      gamma = gamma, fwhm = fwhm, verbose = True,
                                      xstart = init_recon, ystart = ystart, 
                                      callback = _cb, callback_kwargs = cbss)

# show the cost function
it = np.arange(niter) + 1
base_str = f'{phantom}_counts_{counts:.1E}_niter_{niter}_nsub_{nsubsets}'

fig2, ax2 = plt.subplots(1,len(gammas), figsize = (16,3), sharex = True, sharey = True)
for ig, gamma in enumerate(gammas):
  ax2[ig].plot(it,cost_osem, label = 'OSEM')
  ax2[ig].plot(it,costs_spdhg[ig,:], label = f'SPD')
  ax2[ig].plot(it,costs_spdhg_sparse[ig,:], label = f'SPD-S')
  ax2[ig].set_title(f'{gamma}')

ax2[0].set_ylabel('cost')
ax2[0].legend()
for axx in ax2.flatten():
  axx.set_xlabel('iteration')
  axx.grid(ls = ':')
  axx.set_ylim(min(min(cost_osem), costs_spdhg.min(), costs_spdhg_sparse.min()), 
               1.5*max(cost_osem) - 0.5*min(cost_osem))
fig2.tight_layout()
fig2.savefig(os.path.join('figs',f'{base_str}_cost.pdf'))
fig2.savefig(os.path.join('figs',f'{base_str}_cost.png'))
fig2.show()            

# show the PSNR
fig4, ax4 = plt.subplots(1,len(gammas), figsize = (16,3), sharex = True, sharey = True)
for ig, gamma in enumerate(gammas):
  ax4[ig].plot(it,psnr_osem, label = 'OSEM')
  ax4[ig].plot(it,psnr_spdhg[ig,:], label = f'SPD')
  ax4[ig].plot(it,psnr_spdhg_sparse[ig,:], label = f'SPD-S')
  ax4[ig].set_title(f'{gamma}')

ax4[0].set_ylabel('PSNR')
ax4[0].legend()
for axx in ax4.flatten():
  axx.set_xlabel('iteration')
  axx.grid(ls = ':')

fig4.tight_layout()
fig4.savefig(os.path.join('figs',f'{base_str}_psnr.pdf'))
fig4.savefig(os.path.join('figs',f'{base_str}_psnr.png'))
fig4.show()            

# show the reconstructions
fig3, ax3 = plt.subplots(4,len(gammas) + 1, figsize = (16,8))
vmax = 1.5*img.max()
sig  = 2

ax3[0,0].imshow(recon_mlem, vmax = vmax, cmap = plt.cm.Greys)
ax3[0,0].set_title('MLEM')
ax3[1,0].imshow(recon_osem, vmax = vmax, cmap = plt.cm.Greys)
ax3[1,0].set_title('OSEM')

ax3[2,0].imshow(gaussian_filter(recon_mlem,sig), vmax = vmax, cmap = plt.cm.Greys)
ax3[2,0].set_title('ps MLEM')
ax3[3,0].imshow(gaussian_filter(recon_osem,sig), vmax = vmax, cmap = plt.cm.Greys)
ax3[3,0].set_title('ps OSEM')

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
fig3.savefig(os.path.join('figs',f'{base_str}.png'))
fig3.show()
