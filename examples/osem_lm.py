# small demo for listmode TOF OSEM without subsets

import os
import matplotlib.pyplot as py
import pyparallelproj as ppp
import numpy as np
import argparse

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
random_subsets   = False

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
proj        = ppp.SinogramProjector(scanner, sino_params, img.shape, nsubsets = 1, 
                                    voxsize = voxsize, img_origin = img_origin, ngpus = ngpus,
                                    tof = True, sigma_tof = 60./2.35, n_sigmas = 3.)

img_fwd  = proj.fwd_project(img, subset = 0)

# generate sensitity sinogram (product from attenuation and normalization sinogram)
# this sinogram is usually non TOF! which results in a different shape
sens_sino = np.ones(img_fwd.shape[:-1], dtype = np.float32)

scale_fac = (counts / img_fwd.sum())
img_fwd  *= scale_fac 
img      *= scale_fac 

# contamination sinogram with scatter and randoms
# useful to avoid division by 0 in the ratio of data and exprected data
contam_sino = np.full(img_fwd.shape, 0.2*img_fwd.mean(), dtype = np.float32)

em_sino = np.random.poisson(sens_sino[...,np.newaxis]*img_fwd + contam_sino)


#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

# generate list mode events and the corresponting values in the contamination and sensitivity
# sinogram

events, multi_index = sino_params.sinogram_to_listmode(em_sino, return_multi_index = True)

contam_list = contam_sino[multi_index[:,0],multi_index[:,1],multi_index[:,2], multi_index[:,3]]
sens_list   = sens_sino[multi_index[:,0],multi_index[:,1],multi_index[:,2]]

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
#--- MLEM reconstruction

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


# calculate the sensitivity image, we have to repeat the sens sino since it is non TOF
# for the sensitivity image we have to back project all possible LORs (not only those
# appeared in the LM file. Henec, we use the sino projector to back project the sensitibity sinogram
sens_img = proj.back_project(sens_sino.repeat(proj.ntofbins).reshape(sens_sino.shape + 
                             (proj.ntofbins,)), subset = 0) 

# create a listmode projector for the LM MLEM iterations
lmproj = ppp.LMProjector(proj.scanner, proj.img_dim, voxsize = proj.voxsize, 
                         img_origin = proj.img_origin, ngpus = proj.ngpus,
                         tof = proj.tof, sigma_tof = proj.sigma_tof, tofbin_width = proj.tofbin_width,
                         n_sigmas = proj.nsigmas)

# generate an index array that enumerates all events
ind = np.arange(events.shape[0])

# run OSEM iterations
for it in range(niter):

  if random_subsets:
    np.random.shuffle(ind)

  for i in range(nsubsets):
    print(f'iteration {it + 1} subset {i+1}')

    # event numbers for the current subsets
    ind_sub = ind[i::nsubsets]

    exp_list = sens_list[ind_sub]*lmproj.fwd_project(recon, events[ind_sub,:]) + contam_list[ind_sub]
    recon   *= (nsubsets*lmproj.back_project(sens_list[ind_sub]/exp_list, events[ind_sub,:]) / sens_img) 

    ir.set_data(recon[...,n2//2])
    ax[1].set_title(f'itertation {it+1} subset {i+1}')
    ib.set_data(recon[...,n2//2] - img[...,n2//2])
    fig.canvas.draw()

  if track_likelihood:
    exp_sino = sens_sino[...,np.newaxis]*proj.fwd_project(recon, subset = 0) + contam_sino
    logL[it] = (exp_sino - em_sino*np.log(exp_sino)).sum()
    print(f'neg logL {logL[it]}')

if track_likelihood:
  fig2, ax2 = py.subplots(1,1, figsize = (4,4))
  ax2.plot(np.arange(niter) + 1, logL)
  fig2.tight_layout()
  fig2.show()
