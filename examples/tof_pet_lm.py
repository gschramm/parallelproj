# small demo for sinogram TOF OS-MLEM

import os
import matplotlib.pyplot as plt
import pyparallelproj as ppp
from pyparallelproj.algorithms import osem_lm
from pyparallelproj.phantoms import ellipse2d_phantom, brain2d_phantom

import numpy as np
import argparse

#---------------------------------------------------------------------------------
# parse the command line

parser = argparse.ArgumentParser()
parser.add_argument('--counts',
                    help='counts to simulate',
                    default=1e6,
                    type=float)
parser.add_argument('--niter',
                    help='number of iterations',
                    default=4,
                    type=int)
parser.add_argument('--nsubsets',
                    help='number of subsets',
                    default=28,
                    type=int)
parser.add_argument('--fwhm_mm',
                    help='psf modeling FWHM mm',
                    default=4.5,
                    type=float)
parser.add_argument('--fwhm_data_mm',
                    help='psf for data FWHM mm',
                    default=4.5,
                    type=float)
parser.add_argument('--phantom', help='phantom to use', default='brain2d')
parser.add_argument('--seed',
                    help='seed for random generator',
                    default=1,
                    type=int)
args = parser.parse_args()

#---------------------------------------------------------------------------------

counts = args.counts
niter = args.niter
nsubsets = args.nsubsets
fwhm_mm = args.fwhm_mm
fwhm_data_mm = args.fwhm_data_mm
phantom = args.phantom
seed = args.seed

#---------------------------------------------------------------------------------

np.random.seed(seed)

# setup a scanner with one ring
scanner = ppp.RegularPolygonPETScanner(ncrystals_per_module=np.array([16, 1]),
                                       nmodules=np.array([28, 1]))

# setup a test image
voxsize = np.array([2., 2., 2.])
n2 = max(1, int((scanner.xc2.max() - scanner.xc2.min()) / voxsize[2]))

# convert fwhm from mm to pixels
fwhm = fwhm_mm / voxsize
fwhm_data = fwhm_data_mm / voxsize

# setup a test image
if phantom == 'ellipse2d':
    n = 200
    img = np.zeros((n, n, n2), dtype=np.float32)
    tmp = ellipse_phantom(n=n, c=3)
    for i2 in range(n2):
        img[:, :, i2] = tmp
elif phantom == 'brain2d':
    n = 128
    img = np.zeros((n, n, n2), dtype=np.float32)
    tmp = brain2d_phantom(n=n)
    for i2 in range(n2):
        img[:, :, i2] = tmp

img_origin = (-(np.array(img.shape) / 2) + 0.5) * voxsize

# setup an attenuation image
att_img = (img > 0) * 0.01

# generate TOF sinogram parameters and the TOF projector
sino_params = ppp.PETSinogramParameters(scanner, ntofbins=17, tofbin_width=15.)
proj = ppp.SinogramProjector(scanner,
                             sino_params,
                             img.shape,
                             nsubsets=nsubsets,
                             voxsize=voxsize,
                             img_origin=img_origin,
                             tof=True,
                             sigma_tof=60. / 2.35,
                             n_sigmas=3.)

# create the attenuation sinogram
proj.set_tof(False)
attn_sino = np.exp(-proj.fwd_project(att_img))
proj.set_tof(True)
# generate the sensitivity sinogram
sens_sino = np.ones(proj.sino_params.nontof_shape, dtype=np.float32)

# forward project the image
img_fwd = ppp.pet_fwd_model(img, proj, attn_sino, sens_sino, fwhm=fwhm_data)

# scale sum of fwd image to counts
if counts > 0:
    scale_fac = (counts / img_fwd.sum())
    img_fwd *= scale_fac
    img *= scale_fac

    # contamination sinogram with scatter and randoms
    # useful to avoid division by 0 in the ratio of data and exprected data
    contam_sino = np.full(img_fwd.shape,
                          0.2 * img_fwd.mean(),
                          dtype=np.float32)

    em_sino = np.random.poisson(img_fwd + contam_sino)
else:
    scale_fac = 1.

    # contamination sinogram with sctter and randoms
    # useful to avoid division by 0 in the ratio of data and exprected data
    contam_sino = np.full(img_fwd.shape,
                          0.2 * img_fwd.mean(),
                          dtype=np.float32)

    em_sino = img_fwd + contam_sino

#-------------------------------------------------------------------------------------
# calculate the sensitivity images for each subset
sens_img = ppp.pet_back_model(np.ones(proj.sino_params.shape,
                                      dtype=np.float32),
                              proj,
                              attn_sino,
                              sens_sino,
                              fwhm=fwhm)

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

# generate list mode events and the corresponting values in the contamination and sensitivity
# sinogram

events, multi_index = sino_params.sinogram_to_listmode(em_sino,
                                                       return_multi_index=True)

contam_list = contam_sino[multi_index[:, 0], multi_index[:, 1],
                          multi_index[:, 2], multi_index[:, 3]]
sens_list = sens_sino[multi_index[:, 0], multi_index[:, 1], multi_index[:, 2],
                      0]
attn_list = attn_sino[multi_index[:, 0], multi_index[:, 1], multi_index[:, 2],
                      0]

plt.ion()
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(img[..., n2 // 2],
             vmin=0,
             vmax=1.3 * img.max(),
             cmap=plt.cm.Greys)
ax[0].set_title('ground truth')
ir = ax[1].imshow(0 * img[..., n2 // 2],
                  vmin=0,
                  vmax=1.3 * img.max(),
                  cmap=plt.cm.Greys)
ax[1].set_title('recon')
ib = ax[2].imshow(img[..., n2 // 2] - img[..., n2 // 2],
                  vmin=-0.2 * img.max(),
                  vmax=0.2 * img.max(),
                  cmap=plt.cm.bwr)
ax[2].set_title('bias')
fig.tight_layout()

#-----------------------------------------------------------------------------------------------
# callback functions to calculate likelihood and show recon updates


def update_img(x):
    ir.set_data(x[..., n2 // 2])
    ib.set_data(x[..., n2 // 2] - img[..., n2 // 2])
    plt.pause(1e-6)


def calc_cost(x):
    exp = ppp.pet_fwd_model(x, proj, attn_sino, sens_sino,
                            fwhm=fwhm) + contam_sino
    cost = (exp - em_sino * np.log(exp)).sum()

    return cost


def _cb(x, **kwargs):
    """ This function is called by the iterative recon algorithm after every iteration 
      where x is the current reconstructed image
  """
    it = kwargs.get('iteration', 0)
    update_img(x)
    if 'cost' in kwargs:
        kwargs['cost'][it - 1] = calc_cost(x)


#-----------------------------------------------------------------------------------------------
# run the actual reconstruction using listmode OSEM

cost_lmosem = np.zeros(niter)
cbk = {'cost': cost_lmosem}

recon = osem_lm(events,
                attn_list,
                sens_list,
                contam_list,
                proj,
                sens_img,
                niter,
                nsubsets,
                fwhm=fwhm,
                verbose=True,
                callback=_cb,
                callback_kwargs=cbk)

#-----------------------------------------------------------------------------------------------
# plot the cost function

fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
ax2.plot(np.arange(1, niter + 1), cost_lmosem, '.-')
ax2.set_xlabel('iteration')
ax2.set_ylabel('cost')

fig2.tight_layout()
fig2.show()
