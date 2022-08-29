"""a short demo on how to generate LM PET data
   we first generate a sinogram, add Poisson noise and then convert it to LM data"""

import enum
import argparse

import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

import pyparallelproj as ppp


class Phantom(enum.Enum):
    DISK_20MM = 'DISK_20MM'
    DISK_45MM = 'DISK_45MM'
    DISK_55MM = 'DISK_55MM'
    FOUR_SYRINGES = 'FOUR_SYRINGES'


def rod_inds(n0, n1, n2, voxsize, r_mm, offset=(0, 0, 0)):
    x0 = np.linspace(-0.5 * (n0 - 1) * voxsize[0], 0.5 * (n0 - 1) * voxsize[0],
                     n0)
    x1 = np.linspace(-0.5 * (n1 - 1) * voxsize[1], 0.5 * (n1 - 1) * voxsize[1],
                     n1)
    x2 = np.linspace(-0.5 * (n2 - 1) * voxsize[2], 0.5 * (n2 - 1) * voxsize[2],
                     n2)

    X0, X1, X2 = np.meshgrid(x0, x1, x2, indexing='ij')

    R = np.sqrt((X0 - offset[0])**2 + (X1 - offset[1])**2)

    return np.where(R <= r_mm)


#---------------------------------------------------------------------------------
# input parameters

parser = argparse.ArgumentParser()
parser.add_argument('--niter', type=int, default=10)
parser.add_argument('--nsubsets', type=int, default=8)
parser.add_argument('--counts', type=int, default=1_000_000)
parser.add_argument('--fwhm_data_mm', type=float, default=2.5)
parser.add_argument('--fwhm_mm', type=float, default=1.)
parser.add_argument('--ps_fwhm_mm', type=float, default=3.)
parser.add_argument(
    '--phantom',
    type=str,
    default='QC',
    choices=['DISK_20MM', 'DISK_45MM', 'DISK_55MM', 'FOUR_SYRINGES'])

args = parser.parse_args()

niter = args.niter
nsubsets = args.nsubsets
counts = args.counts
fwhm_data_mm = args.fwhm_data_mm
fwhm_mm = args.fwhm_mm
ps_fwhm_mm = args.ps_fwhm_mm
phantom = Phantom(args.phantom)

xp = np
ndimage_module = ndi

xp.random.seed(1)
#---------------------------------------------------------------------------------
# setup a scanner
scanner = ppp.RegularPolygonPETScanner(ncrystals_per_module=np.array([16, 1]),
                                       nmodules=np.array([9, 1]),
                                       crystal_size=np.array([2.2, 2.2]),
                                       R=55.)

# setup a test image
voxsize = np.array([1., 1., 1.])
n0 = 80
n1 = 80
n2 = max(1, int((scanner.xc2.max() - scanner.xc2.min()) / voxsize[2]))

fwhm_data = fwhm_data_mm / voxsize
fwhm = fwhm_mm / voxsize

if phantom.name.startswith('DISK_'):
    r_mm = float(phantom.name.split('_')[1].split('MM')[0]) / 2
    img = xp.zeros((n0, n1, n2), dtype=xp.float32)
    inds1 = rod_inds(n0, n1, n2, voxsize, r_mm=r_mm)
    img[inds1] = 1.
elif phantom == Phantom.FOUR_SYRINGES:
    img = xp.zeros((n0, n1, n2), dtype=xp.float32)
    inds1 = rod_inds(n0, n1, n2, voxsize, r_mm=10, offset=(20, 0))
    inds2 = rod_inds(n0, n1, n2, voxsize, r_mm=10, offset=(-20, 0))
    inds3 = rod_inds(n0, n1, n2, voxsize, r_mm=10, offset=(0, 20))
    inds4 = rod_inds(n0, n1, n2, voxsize, r_mm=10, offset=(0, -20))
    img[inds1] = 1.
    img[inds2] = 1.
    img[inds3] = 1.
    img[inds4] = 1.

img_origin = (-(xp.array(img.shape) / 2) + 0.5) * voxsize

# create the attenuation image
att_img = 0.01 * (img > 0)

# generate sinogram parameters and the projector
sino_params = ppp.PETSinogramParameters(scanner, rtrim=36)
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

# post smooth recons

if ps_fwhm_mm > 0:
    x_osem = ndi.gaussian_filter(x_osem, ps_fwhm_mm / (2.35 * voxsize))
    x_lm_osem = ndi.gaussian_filter(x_lm_osem, ps_fwhm_mm / (2.35 * voxsize))
    x_lm_osem_wrong = ndi.gaussian_filter(x_lm_osem_wrong,
                                          ps_fwhm_mm / (2.35 * voxsize))

x_lm_osem_wrong_scaled = x_lm_osem_wrong * x_lm_osem.sum(
) / x_lm_osem_wrong.sum()

#------------------------------------------------------------------------------------
# show results
ratio = x_lm_osem_wrong / x_lm_osem
ratio[img == 0] = 0

fig, ax = plt.subplots(3, 4, figsize=(12, 8))
imkw = dict(cmap=plt.cm.Greys, vmin=0, vmax=1.2 * img.max())

ax[0, 0].imshow(img.squeeze(), **imkw)
ax[1, 0].imshow(att_img.squeeze(),
                cmap=plt.cm.Greys,
                vmin=0,
                vmax=1.2 * att_img.max())
ax[2, 0].imshow(x_lm_osem.squeeze(), **imkw)

ax[0, 1].imshow(x_lm_osem.squeeze(), **imkw)
ax[1, 1].imshow(x_lm_osem_wrong.squeeze(), **imkw)
ax[2, 1].imshow(x_lm_osem_wrong_scaled.squeeze(), **imkw)

ax[0, 2].imshow(sens_img,
                cmap=plt.cm.Greys,
                vmin=0,
                vmax=1.05 * sens_img_wrong.max())
ax[1, 2].imshow(sens_img_wrong,
                cmap=plt.cm.Greys,
                vmin=0,
                vmax=1.05 * sens_img_wrong.max())
ax[2, 2].imshow(sens_img_wrong / sens_img, cmap=plt.cm.Greys)

ax[0, 3].plot(img[:, n1 // 2, 0] / img.max(), label='true')
ax[0, 3].plot(x_lm_osem[:, n1 // 2, 0] / img.max(), label='LM')
ax[0, 3].plot(x_lm_osem_wrong[:, n1 // 2, 0] / img.max(), label='LM wrong')
ax[0, 3].plot(x_lm_osem_wrong_scaled[:, n1 // 2, 0] / img.max(),
              label='LM wrong sc.')
ax[0, 3].legend(ncol=2, fontsize='x-small', loc='lower center')

ax[1, 3].plot(ratio[:, n1 // 2, 0])

ax[0, 0].set_title('true emission image', fontsize='small')
ax[1, 0].set_title('true attenuation image', fontsize='small')
ax[2, 0].set_title('OSEM', fontsize='small')

ax[0, 1].set_title('LM OSEM', fontsize='small')
ax[1, 1].set_title('LM OSEM wrong sens. img.', fontsize='small')
ax[2, 1].set_title('LM OSEM wrong sens. img. scaled', fontsize='small')

ax[0, 2].set_title('correct sens. img', fontsize='small')
ax[1, 2].set_title('sens. img w/o att', fontsize='small')
ax[2, 2].set_title('ration of sens. imgs.', fontsize='small')

ax[0, 3].set_title('profiles through emission', fontsize='small')
ax[1, 3].set_title('profile through ratio LM_wrong_sens / LM',
                   fontsize='small')

ax[0, 3].grid(ls=':')
ax[1, 3].grid(ls=':')

if phantom.name.startswith('DISK_'):
    x0 = np.linspace(-0.5 * (n0 - 1) * voxsize[0], 0.5 * (n0 - 1) * voxsize[0],
                     n0)
    x1 = np.linspace(-0.5 * (n1 - 1) * voxsize[1], 0.5 * (n1 - 1) * voxsize[1],
                     n1)
    x2 = np.linspace(-0.5 * (n2 - 1) * voxsize[2], 0.5 * (n2 - 1) * voxsize[2],
                     n2)
    X0, X1, X2 = np.meshgrid(x0, x1, x2, indexing='ij')
    R = np.sqrt(X0**2 + X1**2)

    ax[2, 3].plot(R.ravel(), img.ravel() / img.max(), '.', ms=1)
    ax[2, 3].plot(R.ravel(), x_lm_osem.ravel() / img.max(), '.', ms=1)
    ax[2, 3].plot(R.ravel(), x_lm_osem_wrong.ravel() / img.max(), '.', ms=1)
    ax[2, 3].plot(R.ravel(),
                  x_lm_osem_wrong_scaled.ravel() / img.max(),
                  '.',
                  ms=1)
    ax[2, 3].set_xlabel('distance from center [mm]')
    ax[2, 3].set_ylabel('a [arb. units]')
    ax[2, 3].set_title('radial prof. plot', fontsize='small')
    ax[2, 3].grid(ls=':')
else:
    ax[2, 3].set_axis_off()

for axx in ax[:, :-1].flatten():
    axx.set_axis_off()

fig.tight_layout()
fig.show()