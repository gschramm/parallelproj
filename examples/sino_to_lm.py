# a short demo on how to generate LM PET data 
# we first generate a sinogram, add Poisson noise and then convert it to LM data

import os
import pyparallelproj as ppp
import numpy as np

#---------------------------------------------------------------------------------

ngpus       = 0
nsubsets    = 1
subset      = 0 
counts      = 1e5

np.random.seed(1)

n_sigmas = 3.0

# setup a scanner
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
sino_params = ppp.PETSinogramParameters(scanner, ntofbins = 27, tofbin_width = 28.)
proj        = ppp.SinogramProjector(scanner, sino_params, img.shape, nsubsets = nsubsets, 
                                    voxsize = voxsize, img_origin = img_origin, ngpus = ngpus,
                                    tof = True, sigma_tof = 60./2.35, n_sigmas = n_sigmas)

img_fwd  = proj.fwd_project(img, subset = subset)

# scale sum of fwd image to counts
scale_fac = (counts / img_fwd.sum())
img_fwd  *= scale_fac 
img      *= scale_fac 

# generate a noise realization
noisy_sino = np.random.poisson(img_fwd)

# back project noisy sinogram as reference for listmode backprojection of events
back_img = proj.back_project(noisy_sino, subset = subset) 

# events is a list of all events
# each event if characterize by 5 integers: 
# [start_crystal_id_tr, start_crystal_id_ax, end_crystal_id_tr, end_crystal_id_ax, tofbin]
events = sino_params.sinogram_to_listmode(noisy_sino)

### create LM projector
lmproj = ppp.LMProjector(scanner, img.shape, voxsize = voxsize, img_origin = img_origin, ngpus = ngpus,
                         tof = True, sigma_tof = proj.sigma_tof, tofbin_width = proj.tofbin_width,
                         n_sigmas = n_sigmas)

fwd_img_lm  = lmproj.fwd_project(img, events)
back_img_lm = lmproj.back_project(np.ones(events.shape[0]), events)


### debug plots
r = ((back_img_lm - back_img)/back_img).squeeze()
r[back_img.squeeze() == 0] = 0

print(f' min rel diff of back projections {r.min()}')
print(f' min rel diff of back projections {r.max()}')

import matplotlib.pyplot as py
fig, ax = py.subplots(1, 2, figsize =(8,4))
ax[0].imshow(back_img[...,n2//2].squeeze())
ax[0].set_title('back projection of sinogram')
ax[1].imshow(back_img_lm[...,n2//2].squeeze())
ax[1].set_title('back projection of LM data')
fig.tight_layout()
fig.show()
