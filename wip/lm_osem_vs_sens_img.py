"""a short demo on how to generate LM PET data
   we first generate a sinogram, add Poisson noise and then convert it to LM data"""

import pyparallelproj as ppp
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------
# input parameters
niter = 10
nsubsets = 8
counts = 1e7
fwhm_data_mm = 4.5
fwhm_mm = 2.

xp = np
ndimage_module = ndi

xp.random.seed(1)
plt.ion()
#---------------------------------------------------------------------------------
# setup a scanner
scanner = ppp.RegularPolygonPETScanner(ncrystals_per_module=np.array([16, 1]),
                                       nmodules=np.array([9, 1]),
                                       crystal_size=np.array([2, 2]),
                                       R=55.)

# setup a test image
voxsize = np.array([1., 1., 1.])
n0 = 70
n1 = 70
n2 = max(1, int((scanner.xc2.max() - scanner.xc2.min()) / voxsize[2]))

fwhm_data = fwhm_data_mm / voxsize
fwhm = fwhm_mm / voxsize

# setup a random image
img = xp.zeros((n0, n1, n2), dtype=xp.float32)
img[(n0 // 4):(3 * n0 // 4), (n1 // 4):(3 * n1 // 4), :] = 1
img_origin = (-(xp.array(img.shape) / 2) + 0.5) * voxsize

# create the attenuation image
att_img = 0.01 * (img > 0)

# generate sinogram parameters and the projector
sino_params = ppp.PETSinogramParameters(scanner)
proj = ppp.SinogramProjector(scanner,
                             sino_params,
                             img.shape,
                             nsubsets=nsubsets,
                             voxsize=voxsize,
                             img_origin=img_origin,
                             tof=False)

# create the attenuation sinogram
attn_sino = xp.exp(-proj.fwd_project(att_img))

# generate the sensitivity sinogram
sens_sino = xp.ones(proj.sino_params.nontof_shape, dtype=xp.float32)

# setup the acquisition models for data generation
res_model_data = ppp.ImageBasedResolutionModel(fwhm_data)
acq_model_data = ppp.PETAcqModel(proj,
                                 attn_sino,
                                 sens_sino,
                                 image_based_res_model=res_model_data)

# forward project the image
img_fwd = acq_model_data.forward(img)

scale_fac = (counts / img_fwd.sum())
img_fwd *= scale_fac
img *= scale_fac

# contamination sinogram with scatter and randoms
# useful to avoid division by 0 in the ratio of data and exprected data
contam_sino = xp.full(img_fwd.shape, 0.2 * img_fwd.mean(), dtype=xp.float32)

em_sino = xp.random.poisson(img_fwd + contam_sino)

#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
# OSEM reconstruction

# setup the acquisition models for data generation
res_model = ppp.ImageBasedResolutionModel(fwhm, ndimage_module=ndimage_module)
acq_model = ppp.PETAcqModel(proj,
                            attn_sino,
                            sens_sino,
                            image_based_res_model=res_model)

osem = ppp.OSEM(em_sino, acq_model, contam_sino, xp)
# initialize OSEM (e.g. calculate the sensivity image for every subset)
osem.init()
osem.run(niter, calculate_cost=True)

x_osem = osem.x.copy()

#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------

# events is a list of all events
# each event if characterize by 5 integers:
# [start_crystal_id_tr, start_crystal_id_ax, end_crystal_id_tr, end_crystal_id_ax, tofbin]
events, multi_index = sino_params.sinogram_to_listmode(em_sino,
                                                       return_multi_index=True)

attn_list = attn_sino[multi_index[:, 0], multi_index[:, 1], multi_index[:, 2],
                      0]
sens_list = sens_sino[multi_index[:, 0], multi_index[:, 1], multi_index[:, 2],
                      0]
contam_list = contam_sino[multi_index[:, 0], multi_index[:, 1],
                          multi_index[:, 2], multi_index[:, 3]]

# define the listmode acquisiton model

lm_acq_model = ppp.LMPETAcqModel(proj,
                                 events,
                                 attn_list,
                                 sens_list,
                                 image_based_res_model=res_model)

# back project LM events and compare to back projection of em sino
back_sino = acq_model.adjoint(em_sino)
back_lm = lm_acq_model.adjoint(xp.ones(events.shape[0], dtype=xp.float32))

# calculate the sensitivity image
sens_img = acq_model.adjoint(xp.ones(em_sino.shape, dtype=xp.float32))

# calculate the "wrong" sens image where we neglect the effect of attenuation in the backprojection
sens_img_wrong = acq_model.adjoint(
    xp.ones(em_sino.shape, dtype=xp.float32) / attn_sino)

# run LM OSEM with correct sens image
lm_osem = ppp.LM_OSEM(lm_acq_model, contam_list, xp, verbose=True)
lm_osem.init(sens_img, nsubsets)
lm_osem.run(niter)
x_lm_osem = lm_osem.x.copy()

# run LM OSEM with correct sens image
lm_osem_wrong = ppp.LM_OSEM(lm_acq_model, contam_list, xp, verbose=True)
lm_osem_wrong.init(sens_img_wrong, nsubsets)
lm_osem_wrong.run(niter)
x_lm_osem_wrong = lm_osem_wrong.x.copy()