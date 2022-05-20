# small demo for sinogram TOF OS-MLEM

import os
import matplotlib.pyplot as plt
import pyparallelproj as ppp
from pyparallelproj.phantoms import ellipse2d_phantom, brain2d_phantom

import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndi_cupy
import scipy.ndimage as ndi
import argparse

import pymirc.viewer as pv
from scipy.ndimage import gaussian_filter
from cupyx.scipy.ndimage import gaussian_filter as gaussian_filter_cupy

from time import time

#---------------------------------------------------------------------------------
def osem(em_sino, acq_model, niter, verbose = False, xstart = None):

  if isinstance(em_sino, np.ndarray):
    xp = np
  else:
    xp = cp

  img_shape  = tuple(acq_model.proj.img_dim)

  # calculate the sensitivity images for each subset
  sens_img  = xp.zeros((proj.nsubsets,) + img_shape, dtype = xp.float32)
 
  for isub in range(proj.nsubsets):
    sens_img[isub,...] = acq_model.back(xp.ones(proj.subset_sino_shapes[isub] , dtype = xp.float32), isub = isub)
  
  # initialize recon
  if xstart is None:
    recon = xp.full(img_shape, 1., dtype = xp.float32)
  else:
    recon = xstart.copy()

  # run OSEM iterations
  for it in range(niter):
    for isub in range(proj.nsubsets):
      if verbose: print(f'iteration {it+1} subset {isub+1}')
      # get the slice for the current subset
      ss = proj.subset_slices[isub]

      exp_sino = acq_model.fwd(recon, isub = isub) + contam_sino[ss]
      ratio    = em_sino[ss] / exp_sino
      recon   *= (acq_model.back(ratio, isub = isub) / sens_img[isub,...])
    
  return recon


#---------------------------------------------------------------------------------
class PETAcqModel:
  def __init__(self, proj, attn_sino, sens_sino, fwhm = 0):
    self.proj        = proj
    self.attn_sino   = attn_sino
    self.sens_sino   = sens_sino
    self.fwhm        = fwhm

    if isinstance(attn_sino, np.ndarray):
      self._ndi = ndi
    else:
      self._ndi = ndi_cupy

  def fwd(self, img, isub = None):
    if np.any(fwhm > 0):
      img = self._ndi.gaussian_filter(img, fwhm/2.35)

    if isub is None:
      sino = self.sens_sino*self.attn_sino*self.proj.fwd_project(img)
    else:
      ss   = self.proj.subset_slices[isub]
      sino = self.sens_sino[ss]*self.attn_sino[ss]*self.proj.fwd_project_subset(img, isub)

    return sino

  def back(self, sino, isub = None):
    if isub is None:
      back_img = self.proj.back_project(self.sens_sino*self.attn_sino*sino)
    else:
      ss   = self.proj.subset_slices[isub]
      back_img = self.proj.back_project_subset(self.sens_sino[ss]*self.attn_sino[ss]*sino, subset = isub)

    if np.any(fwhm > 0):
      back_img = self._ndi.gaussian_filter(back_img, fwhm/2.35)

    return back_img


#---------------------------------------------------------------------------------
# parse the command line

parser = argparse.ArgumentParser()
parser.add_argument('--counts',   help = 'counts to simulate',    default = 1e7, type = float)
parser.add_argument('--niter',    help = 'number of iterations',  default = 2,   type = int)
parser.add_argument('--nsubsets',   help = 'number of subsets',     default = 28,  type = int)
parser.add_argument('--fwhm_mm',  help = 'psf modeling FWHM mm',  default = 4.5, type = float)
parser.add_argument('--fwhm_data_mm',  help = 'psf for data FWHM mm',  default = 4.5, type = float)
parser.add_argument('--seed',    help = 'seed for random generator', default = 1, type = int)
args = parser.parse_args()

#---------------------------------------------------------------------------------

counts        = args.counts
niter         = args.niter
nsubsets      = args.nsubsets
fwhm_mm       = args.fwhm_mm
fwhm_data_mm  = args.fwhm_data_mm
seed          = args.seed

# choose backend (numpy or cupy)
on_gpu = True

if on_gpu:
  xp = cp
else:
  xp = np

#---------------------------------------------------------------------------------

np.random.seed(seed)

# setup a scanner with one ring
scanner = ppp.RegularPolygonPETScanner(ncrystals_per_module = np.array([16,9]),
                                       nmodules             = np.array([28,3]),
                                       on_gpu               = on_gpu)

# setup a test image
voxsize = np.array([2.,2.,2.])
n2      = max(1,int((scanner.xc2.max() - scanner.xc2.min()) / voxsize[2]))

# convert fwhm from mm to pixels
fwhm      = fwhm_mm / voxsize
fwhm_data = fwhm_data_mm / voxsize

# setup a test image
n   = 128
img = xp.zeros((n,n,n2), dtype = xp.float32)
tmp = brain2d_phantom(n = n)
for i2 in np.arange(2,n2-2):
  if on_gpu:
    img[:,:,i2] = cp.asarray(tmp.T)
  else:
    img[:,:,i2] = tmp.T

img_origin = (-(np.array(img.shape) / 2) +  0.5) * voxsize

# setup an attenuation image
att_img = (img > 0) * 0.01

# generate TOF sinogram parameters and the TOF projector
sino_params = ppp.PETSinogramParameters(scanner, ntofbins = 17, tofbin_width = 15., rtrim = 140)
proj        = ppp.SinogramProjector(scanner, sino_params, img.shape, nsubsets = nsubsets, 
                                    voxsize = voxsize, img_origin = img_origin,
                                    tof = True, sigma_tof = 60./2.35, n_sigmas = 3.)

# create the attenuation sinogram
proj.set_tof(False)
attn_sino = np.exp(-proj.fwd_project(att_img))
proj.set_tof(True)
# generate the sensitivity sinogram
sens_sino = xp.ones(proj.sino_params.nontof_shape, dtype = xp.float32)

# create the PET acquisition model
paq = PETAcqModel(proj, attn_sino, sens_sino, fwhm = fwhm)

# forward project the image
img_fwd = paq.fwd(img)

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
  contam_sino = xp.full(img_fwd.shape, 0.2*img_fwd.mean(), dtype = np.float32)

  em_sino = img_fwd + contam_sino

del img_fwd

#-----------------------------------------------------------------------------------------------

t0 = time()
recon_osem = osem(em_sino, paq, niter, verbose = True)
t1 = time()

print(t1-t0)

if on_gpu:
  recon_osem = cp.asnumpy(recon_osem)

vi = pv.ThreeAxisViewer(gaussian_filter(recon_osem,1))
