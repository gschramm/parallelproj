import pyparallelproj as ppp
import h5py
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

import cupyx.scipy.ndimage as ndi_cupy
import scipy.ndimage as ndi

from pathlib import Path

class LMPETAcqModel:
  def __init__(self, proj, events, attn_list, sens_list, fwhm = 0):
    self.proj        = proj
    self.events      = events
    self.attn_list   = attn_list
    self.sens_list   = sens_list
    self.fwhm        = fwhm

    if isinstance(attn_list, np.ndarray):
      self._ndi = ndi
    else:
      self._ndi = ndi_cupy

  def fwd(self, img, isub = 0, nsubsets = 1):
    if np.any(fwhm > 0):
      img = self._ndi.gaussian_filter(img, fwhm/2.35)

    ss = slice(isub, None, nsubsets)
    img_fwd = self.sens_list[ss]*self.attn_list[ss]*self.proj.fwd_project_lm(img, self.events[ss])

    return img_fwd

  def back(self, values, isub = 0, nsubsets = 1):
    ss = slice(isub, None, nsubsets)
    back_img = self.proj.back_project_lm(self.sens_list[ss]*self.attn_list[ss]*values, self.events[ss])

    if np.any(fwhm > 0):
      back_img = self._ndi.gaussian_filter(back_img, fwhm/2.35)

    return back_img

#--------------------------------------------------------------------------------------------

def osem_lm(lm_acq_model, contam_list, sens_img, niter, nsubsets, xstart = None, verbose = True):

  if isinstance(lm_acq_model.events, np.ndarray):
    xp = np
  else:
    xp = cp

  img_shape  = tuple(proj.img_dim)

  # initialize recon
  if xstart is None:
    recon = xp.full(img_shape, 1., dtype = xp.float32)
  else:
    recon = xstart.copy()

  # run OSEM iterations
  for it in range(niter):
    for i in range(nsubsets):
      if verbose: print(f'iteration {it+1} subset {i+1}')
    
      exp_list = lm_acq_model.fwd(recon, i, nsubsets) + contam_list[i::nsubsets] 

      recon   *= (lm_acq_model.back(nsubsets/exp_list, i, nsubsets) / sens_img)

  return recon


#--------------------------------------------------------------------------------------------

on_gpu = True

voxsize   = np.array([2., 2., 2.], dtype = np.float32)
img_shape = (166,166,94)
verbose   = True

# FHHM for resolution model in voxel
fwhm  = 4.5 / (2.35*voxsize)

# prior strength
beta = 6

niter    = 100
nsubsets = 224

np.random.seed(1)

#--------------------------------------------------------------------------------------------

if on_gpu:
  xp = cp
else:
  xp = np

#--------------------------------------------------------------------------------------------
# setup scanner and projector

# define the 4 ring DMI geometry
scanner = ppp.RegularPolygonPETScanner(
               R                    = 0.5*(744.1 + 2*8.51),
               ncrystals_per_module = np.array([16,9]),
               crystal_size         = np.array([4.03125,5.31556]),
               nmodules             = np.array([34,4]),
               module_gap_axial     = 2.8,
               on_gpu               = on_gpu)


# define sinogram parameter - also needed to setup the LM projector

# speed of light in mm/ns
speed_of_light = 300.

# time resolution FWHM in ns
time_res_FWHM = 0.385

# sigma TOF in mm
sigma_tof = (speed_of_light/2) * (time_res_FWHM/2.355)

# the TOF bin width in mm is 13*0.01302ns times the speed of light (300mm/ns) divided by two
sino_params = ppp.PETSinogramParameters(scanner, rtrim = 65, ntofbins = 29, 
                                        tofbin_width = 13*0.01302*speed_of_light/2)

# define the projector
proj = ppp.SinogramProjector(scanner, sino_params, img_shape,
                             voxsize = voxsize, tof = True, 
                             sigma_tof = sigma_tof, n_sigmas = 3.)


#--------------------------------------------------------------------------------------------
# calculate sensitivity image

sens_img = np.load('../../lm-spdhg/python/data/dmi/NEMA_TV_beta_6_ss_224/sens_img.npy')

#--------------------------------------------------------------------------------------------

# read the actual LM data and the correction lists

# read the LM data
if verbose:
  print('Reading LM data')

with h5py.File('../../lm-spdhg/python/data/dmi/lm_data.h5', 'r') as data:
  sens_list   =  data['correction_lists/sens'][:]
  atten_list  =  data['correction_lists/atten'][:]
  contam_list =  data['correction_lists/contam'][:]
  LM_file     =  Path(data['header/listfile'][0].decode("utf-8"))

with h5py.File(LM_file, 'r') as data:
  events = data['MiceList/TofCoinc'][:]

# swap axial and trans-axial crystals IDs
events = events[:,[1,0,3,2,4]]

# for the DMI the tof bins in the LM files are already meshed (only every 13th is populated)
# so we divide the small tof bin number by 13 to get the bigger tof bins
# the definition of the TOF bin sign is also reversed 

events[:,-1] = -(events[:,-1]//13)

nevents = events.shape[0]

## shuffle events since events come semi sorted
if verbose: 
  print('shuffling LM data')
ie = np.arange(nevents)
np.random.shuffle(ie)
events = events[ie,:]
sens_list   = sens_list[ie]
atten_list  = atten_list[ie]  
contam_list = contam_list[ie]

if on_gpu:
  events      = cp.asarray(events)
  sens_img    = cp.asarray(sens_img)
  sens_list   = cp.asarray(sens_list)
  atten_list  = cp.asarray(atten_list)
  contam_list = cp.asarray(contam_list)

#--------------------------------------------------------------------------------------------

paq = LMPETAcqModel(proj, events, atten_list, sens_list, fwhm = fwhm)

recon = osem_lm(paq, contam_list, sens_img, 2, 28)
