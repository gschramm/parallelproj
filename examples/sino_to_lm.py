import sys
import os

if not os.path.abspath('..') in sys.path: sys.path.append(os.path.abspath('..'))

import pyparallelproj as ppp
import numpy as np

#---------------------------------------------------------------------------------

ngpus       = 0
nsubsets    = 1
subset      = 0 

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
img = np.zeros((n0,n1,n2))
img[(n0//4):(3*n0//4),(n1//4):(3*n1//4),:] = 0.4
img_origin = (-(np.array(img.shape) / 2) +  0.5) * voxsize

# generate sinogram parameters and the projector
sino_params = ppp.PETSinogram(scanner, ntofbins = 27, tofbin_width = 28.)
proj        = ppp.SinogramProjector(scanner, sino_params, img.shape, nsubsets = nsubsets, 
                                    voxsize = voxsize, img_origin = img_origin, ngpus = ngpus,
                                    tof = True, sigma_tof = 60./2.35, n_sigmas = n_sigmas)

img_fwd  = proj.fwd_project(img, subset = subset)

# generate a noise realization
noisy_sino = np.random.poisson(img_fwd)

########################
#noisy_sino *= 0
#noisy_sino[178 + 10,1,0,13-1] = 1

# back project noisy sinogram as reference for listmode backprojection of events
back_img = proj.back_project(noisy_sino, subset = subset) 

# events is a list of all events
# each event if characterize by 5 integers: 
# [start_crystal_id_tr, start_crystal_id_ax, end_crystal_id_tr, end_crystal_id_ax, tofbin]

events  = np.zeros((noisy_sino.sum(),5), dtype = np.int16)
counter = 0

it = np.nditer(noisy_sino, flags=['multi_index'])

for x in it:
  if x > 0:
    events[counter:(counter+x),0:2] = proj.istart[it.multi_index[:-1]]
    events[counter:(counter+x):,2:4] = proj.iend[it.multi_index[:-1]]
    # for the LM projector, the central tofbin is 0, so we have to shift
    # the tof index of the sinogram bu ntofbins // 2
    events[counter:(counter+x):,4]   = it.multi_index[-1] - sino_params.ntofbins // 2

    counter += x

# shuffle the event order
tmp = np.arange(events.shape[0])
np.random.shuffle(tmp)
events = events[tmp,:]

### create LM projector
lmproj = ppp.LMProjector(scanner, img.shape, voxsize = voxsize, img_origin = img_origin, ngpus = ngpus,
                         tof = True, sigma_tof = proj.sigma_tof, tofbin_width = proj.tofbin_width,
                         n_sigmas = n_sigmas)

fwd_img_lm  = lmproj.fwd_project(img, events)
back_img_lm = lmproj.back_project(np.ones(events.shape[0]), events)


### debug plots
r = ((back_img_lm - back_img)/back_img).squeeze()
r[back_img.squeeze() == 0] = 0

print(r.min())
print(r.max())

import matplotlib.pyplot as py
py.imshow(back_img.squeeze())
py.show()
