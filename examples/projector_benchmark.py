# small demo for listmode TOF MLEM without subsets

import os
import matplotlib.pyplot as py
import pyparallelproj as ppp
import numpy as np
import argparse

from timeit import timeit

#---------------------------------------------------------------------------------
# parse the command line

parser = argparse.ArgumentParser()
parser.add_argument('--ngpus',  help = 'number of GPUs to use', default = 0,   type = int)
parser.add_argument('--counts', help = 'counts to simulate',    default = 1e5, type = float)
args = parser.parse_args()

#---------------------------------------------------------------------------------

ngpus     = args.ngpus
counts    = args.counts

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

scale_fac = (counts / img_fwd.sum())
img_fwd  *= scale_fac 
img      *= scale_fac 

# contamination sinogram with scatter and randoms
# useful to avoid division by 0 in the ratio of data and exprected data
em_sino     = np.random.poisson(img_fwd)

#-------------------------------------------------------------------------------------

# generate list mode events and the corresponting values in the contamination and sensitivity
# sinogram

events, multi_index = sino_params.sinogram_to_listmode(em_sino, return_multi_index = True)

# create a listmode projector for the LM MLEM iterations
lmproj = ppp.LMProjector(proj.scanner, proj.img_dim, voxsize = proj.voxsize, 
                         img_origin = proj.img_origin, ngpus = proj.ngpus,
                         tof = proj.tof, sigma_tof = proj.sigma_tof, tofbin_width = proj.tofbin_width,
                         n_sigmas = proj.nsigmas)

ones = np.ones(events.shape[0], dtype = np.float32)

#-------------------------------------------------------------------------------------
# time the fwd and back projection

from IPython import get_ipython
ipython = get_ipython()

# forward project 
print(f'timing LM fwd projection {events.shape[0]} events')
ipython.magic("timeit lm_fwd  = lmproj.fwd_project(img, events)")
# back project
print(f'timing LM back projection {events.shape[0]} events')
ipython.magic("timeit lm_back = lmproj.back_project(ones, events)")
