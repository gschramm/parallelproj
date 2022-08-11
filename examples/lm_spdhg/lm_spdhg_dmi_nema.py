import pyparallelproj as ppp
import h5py
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

import scipy.ndimage as ndi
import cupyx.scipy.ndimage as ndi_cupy

from pathlib import Path
from time import time

on_gpu = True
voxsize = np.array([2., 2., 2.], dtype=np.float32)
img_shape = (166, 166, 94)
verbose = True

fwhm = 4.5 / (2.35 * voxsize)  # FWHM for resolution model in voxels

niter = 10
beta = 12
nsubsets = 56
rho = 4.

nevents = 10000000
norm_name = 'l2_l1'

#--------------------------------------------------------------------------------------------
np.random.seed(1)

if on_gpu:
    xp = cp
    ndimage_module = ndi_cupy
else:
    xp = np
    ndimage_module = ndi

#--------------------------------------------------------------------------------------------
# setup scanner and projector

# define the 4 ring DMI geometry
scanner = ppp.RegularPolygonPETScanner(R=0.5 * (744.1 + 2 * 8.51),
                                       ncrystals_per_module=np.array([16, 9]),
                                       crystal_size=np.array(
                                           [4.03125, 5.31556]),
                                       nmodules=np.array([34, 4]),
                                       module_gap_axial=2.8,
                                       on_gpu=on_gpu)

# define sinogram parameter - also needed to setup the LM projector
# speed of light in mm/ns
speed_of_light = 300.

# time resolution FWHM of the DMI in ns
time_res_FWHM = 0.385

# sigma TOF in mm
sigma_tof = (speed_of_light / 2) * (time_res_FWHM / 2.355)

# the TOF bin width in mm is 13*0.01302ns times the speed of light (300mm/ns) divided by two
sino_params = ppp.PETSinogramParameters(scanner,
                                        rtrim=65,
                                        ntofbins=29,
                                        tofbin_width=13 * 0.01302 *
                                        speed_of_light / 2)

# define the projector
proj = ppp.SinogramProjector(scanner,
                             sino_params,
                             img_shape,
                             voxsize=voxsize,
                             tof=True,
                             sigma_tof=sigma_tof,
                             n_sigmas=3.)

#--------------------------------------------------------------------------------------------
# calculate sensitivity image

sens_img = np.load(
    '../../lm-spdhg/python/data/dmi/NEMA_TV_beta_6_ss_224/sens_img.npy')

#--------------------------------------------------------------------------------------------
# read the actual LM data and the correction lists

# read the LM data
if verbose:
    print('Reading LM data')

with h5py.File('../../lm-spdhg/python/data/dmi/lm_data.h5', 'r') as data:
    sens_list = data['correction_lists/sens'][:]
    atten_list = data['correction_lists/atten'][:]
    contam_list = data['correction_lists/contam'][:]
    LM_file = Path(data['header/listfile'][0].decode("utf-8"))

with h5py.File(LM_file, 'r') as data:
    events = data['MiceList/TofCoinc'][:]

# swap axial and trans-axial crystals IDs
events = events[:, [1, 0, 3, 2, 4]]

# for the DMI the tof bins in the LM files are already meshed (only every 13th is populated)
# so we divide the small tof bin number by 13 to get the bigger tof bins
# the definition of the TOF bin sign is also reversed

events[:, -1] = -(events[:, -1] // 13)

nevents = events.shape[0]

## shuffle events since events come semi sorted
if verbose:
    print('shuffling LM data')
ie = np.arange(nevents)
np.random.shuffle(ie)
events = events[ie, :]
sens_list = sens_list[ie]
atten_list = atten_list[ie]
contam_list = contam_list[ie]

## use only part of the events
if nevents is not None:
    sens_list = sens_list[:nevents]
    atten_list = atten_list[:nevents]
    contam_list = contam_list[:nevents] * (nevents / events.shape[0])
    events = events[:nevents, :]

if on_gpu:
    print('copying data to GPU')
    events = cp.asarray(events)
    sens_img = cp.asarray(sens_img)
    sens_list = cp.asarray(sens_list)
    atten_list = cp.asarray(atten_list)
    contam_list = cp.asarray(contam_list)

#--------------------------------------------------------------------------------------------
# OSEM as intializer

res_model = ppp.ImageBasedResolutionModel(fwhm, ndimage_module=ndimage_module)
lm_acq_model = ppp.LMPETAcqModel(proj,
                                 events,
                                 atten_list,
                                 sens_list,
                                 image_based_res_model=res_model)

print('LM-OSEM')

t0 = time()
lm_osem = ppp.LM_OSEM(lm_acq_model, contam_list, xp)
lm_osem.init(sens_img, 28)
lm_osem.run(2)
t1 = time()
print(f'recon time: {(t1-t0):.2f}s')

#----------------------------------------------------------------------------------------------
# LM OSEM-TV

print('LM-OSEM-EMTV')
prior = ppp.GradientBasedPrior(ppp.GradientOperator(xp),
                               ppp.GradientNorm(xp, name=norm_name), beta)

lm_osem_emtv = ppp.LM_OSEM_EMTV(lm_acq_model, contam_list, prior, xp)
lm_osem_emtv.init(sens_img, 28, x=lm_osem.x)
lm_osem_emtv.run(10)

#--------------------------------------------------------------------------------------------
# LM-SPDHG

print('LM-SPDHG')
event_counter = ppp.EventMultiplicityCounter(xp)

if xp.__name__ == 'numpy':
    img_norm = float(ndi.gaussian_filter(lm_osem.x, 2).max())
else:
    img_norm = float(ndi_cupy.gaussian_filter(lm_osem.x, 2).max())

lm_spdhg = ppp.LM_SPDHG(lm_acq_model, contam_list, event_counter, prior, xp)
lm_spdhg.init(sens_img, nsubsets, x=lm_osem.x, gamma=30. / img_norm, rho=rho)

r = xp.zeros((niter, ) + img_shape, dtype=xp.float32)

t2 = time()
for i in range(niter):
    lm_spdhg.run(1)
    r[i, ...] = lm_spdhg.x.copy()
t3 = time()
print(f'recon time: {(t3-t2):.2f}s')

#----------------------------------------------------------------------------------------------

import pymirc.viewer as pv
if xp.__name__ == 'numpy':
    vi = pv.ThreeAxisViewer([lm_osem.x, lm_spdhg.x],
                            imshow_kwargs={'vmax': img_norm})
else:
    vi = pv.ThreeAxisViewer(
        [cp.asnumpy(lm_osem.x), cp.asnumpy(lm_spdhg.x)],
        imshow_kwargs={'vmax': img_norm})
