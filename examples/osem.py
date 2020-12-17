# small demo for sinogram TOF OS-MLEM

import os
import matplotlib.pyplot as py
import pyparallelproj as ppp
import numpy as np
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
proj_nt        = ppp.SinogramProjector(scanner, sino_params_nt, img.shape, nsubsets = nsubsets, 
                                    voxsize = voxsize, img_origin = img_origin, ngpus = ngpus)
sino_shape_nt  = sino_params_nt.shape

attn_sino = np.zeros((nsubsets, sino_shape_nt[0], sino_shape_nt[1] // nsubsets, 
                      sino_shape_nt[2], sino_shape_nt[3]), dtype = np.float32)

for i in range(nsubsets):
    attn_sino[i, ...] = np.exp(-proj_nt.fwd_project(att_img, subset=i))

# generate the sensitivity sinogram
sens_sino = np.ones((nsubsets, sino_shape_nt[0], sino_shape_nt[1] // nsubsets, 
                      sino_shape_nt[2], sino_shape_nt[3]), dtype = np.float32)

# generate TOF sinogram parameters and the TOF projector
sino_params = ppp.PETSinogramParameters(scanner, ntofbins = 17, tofbin_width = 15.)
proj        = ppp.SinogramProjector(scanner, sino_params, img.shape, nsubsets = nsubsets, 
                                    voxsize = voxsize, img_origin = img_origin, ngpus = ngpus,
                                    tof = True, sigma_tof = 60./2.35, n_sigmas = 3.)

# allocate array for the subset sinogram
sino_shape = sino_params.shape
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

# initialize recon
recon = np.full((n0,n1,n2), em_sino.sum() / (n0*n1*n2), dtype = np.float32)

if track_likelihood:
  logL = np.zeros(niter)

py.ion()
fig, ax = py.subplots(1,3, figsize = (12,4))
ax[0].imshow(img[...,n2//2],   vmin = 0, vmax = 1.3*img.max(), cmap = py.cm.Greys)
ax[0].set_title('ground truth')
ir = ax[1].imshow(recon[...,n2//2], vmin = 0, vmax = 1.3*img.max(), cmap = py.cm.Greys)
ax[1].set_title('intial recon')
ib = ax[2].imshow(recon[...,n2//2] - img[...,n2//2], vmin = -0.2*img.max(), vmax = 0.2*img.max(), 
                  cmap = py.cm.bwr)
ax[2].set_title('bias')
fig.tight_layout()
fig.canvas.draw()

# calculate the sensitivity images for each subset
sens_img  = np.zeros((nsubsets,) + img.shape, dtype = np.float32)
ones_sino = np.ones((sino_shape[0], sino_shape[1] // nsubsets, sino_shape[2], 
                     sino_shape[3]), dtype = np.float32)

for i in range(nsubsets):
  sens_img[i,...] = ppp.pet_back_model(ones_sino, proj, attn_sino, sens_sino, i, fwhm = fwhm)

# run MLEM iterations
for it in range(niter):
  for i in range(nsubsets):
    print(f'iteration {it + 1} subset {i+1}')
    exp_sino = ppp.pet_fwd_model(recon, proj, attn_sino, sens_sino, i, fwhm = fwhm) + contam_sino[i,...]
    ratio  = em_sino[i,...] / exp_sino
    recon *= (ppp.pet_back_model(ratio, proj, attn_sino, sens_sino, i, fwhm = fwhm) / sens_img[i,...]) 
    
    ir.set_data(recon[...,n2//2])
    ax[1].set_title(f'itertation {it+1} subset {i+1}')
    ib.set_data(recon[...,n2//2] - img[...,n2//2])
    fig.canvas.draw()


  if track_likelihood:
    exp = np.zeros(img_fwd.shape, dtype = np.float32)
    for i in range(nsubsets):
      exp[i,...] = ppp.pet_fwd_model(recon, proj, attn_sino, sens_sino, i, fwhm = fwhm) + contam_sino[i,...]
    logL[it] = (exp - em_sino*np.log(exp)).sum()
    print(f'neg logL {logL[it]}')

if track_likelihood:
  fig2, ax2 = py.subplots(1,1, figsize = (4,4))
  ax2.plot(np.arange(niter) + 1, logL)
  fig2.tight_layout()
  fig2.show()
