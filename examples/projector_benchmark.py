# small demo for listmode TOF MLEM without subsets

import os
import matplotlib.pyplot as py
import pyparallelproj as ppp
import numpy as np
import argparse

from time import time

#---------------------------------------------------------------------------------
# parse the command line

parser = argparse.ArgumentParser()
parser.add_argument('--ngpus',    help = 'number of GPUs to use', default = 0,   type = int)
parser.add_argument('--counts',   help = 'counts to simulate',    default = 4e6, type = float)
parser.add_argument('--nsubsets', help = 'number of subsets',     default = 28,  type = int)
parser.add_argument('--n',        help = 'number of averages',    default = 5,   type = int)
parser.add_argument('--tpb',      help = 'threads per block',     default = 64,  type = int)
parser.add_argument('--nontof',   help = 'non-TOF instead of TOF', action = 'store_true')
args = parser.parse_args()

#---------------------------------------------------------------------------------

ngpus     = args.ngpus
counts    = args.counts
nsubsets  = args.nsubsets
n         = args.n
tpb       = args.tpb
tof       = not args.nontof

if tof:
  ntofbins = 27
else:
  ntofbins = 1

np.random.seed(1)

#---------------------------------------------------------------------------------

# setup a scanner with one ring
scanner = ppp.RegularPolygonPETScanner(ncrystals_per_module = np.array([16,9]),
                                       nmodules             = np.array([28,5]))

# setup a test image
voxsize = np.array([2.,2.,2.])
n0      = 250
n1      = 250
n2      = max(1,int((scanner.xc2.max() - scanner.xc2.min()) / voxsize[2]))


# setup a random image
img = np.zeros((n0,n1,n2), dtype = np.float32)
img[(n0//6):(5*n0//6),(n1//6):(5*n1//6),:] = 1
img_origin = (-(np.array(img.shape) / 2) +  0.5) * voxsize

# generate sinogram parameters and the projector
sino_params = ppp.PETSinogramParameters(scanner, ntofbins = ntofbins, tofbin_width = 23.)
proj        = ppp.SinogramProjector(scanner, sino_params, img.shape, nsubsets = nsubsets, 
                                    voxsize = voxsize, img_origin = img_origin, ngpus = ngpus,
                                    tof = tof, sigma_tof = 60./2.35, n_sigmas = 3.,
                                    threadsperblock = tpb)

# contamination sinogram with scatter and randoms
# useful to avoid division by 0 in the ratio of data and exprected data
ones_sino = np.ones(proj.subset_sino_shapes[0], dtype = np.float32)

#-------------------------------------------------------------------------------------
# time sino fwd and back projection

t_sino_fwd  = np.zeros(n)
t_sino_back = np.zeros(n)

# forward project 
print(f'timing sinogram fwd projection for 1 out of {nsubsets} subsets')
for i in range(n):
  print(f'run {i+1} / {n}')
  t0 = time()
  img_fwd = proj.fwd_project(img, subset = 0)
  t1 = time()
  t_sino_fwd[i] = t1 - t0
print(f'{t_sino_fwd.mean():.4f} s (mean) +-  {t_sino_fwd.std():.4f} s (std)\n')
  
# back project
print(f'timing sinogram back projection for 1 out of {nsubsets} subsets')
for i in range(n):
  print(f'run {i+1} / {n}')
  t0 = time()
  sino_back = proj.back_project(ones_sino, subset = 0)
  t1 = time()
  t_sino_back[i] = t1 - t0
print(f'{t_sino_back.mean():.4f} s (mean) +-  {t_sino_back.std():.4f} s (std)\n')



#-------------------------------------------------------------------------------------
# time LM fwd and back projection

if counts > 0:
  print('generating LM data \n')
  # generate list mode events and the corresponting values in the contamination and sensitivity
  scale_fac = (counts / img_fwd.sum())
  img_fwd  *= scale_fac 
  
  em_sino = np.random.poisson(img_fwd)
  
  events, multi_index = sino_params.sinogram_to_listmode(em_sino, return_multi_index = True)
  
  # create a listmode projector for the LM MLEM iterations
  lmproj = ppp.LMProjector(proj.scanner, proj.img_dim, voxsize = proj.voxsize, 
                           img_origin = proj.img_origin, ngpus = proj.ngpus,
                           tof = proj.tof, sigma_tof = proj.sigma_tof, 
                           tofbin_width = proj.tofbin_width,
                           n_sigmas = proj.nsigmas,
                           threadsperblock = proj.threadsperblock)
  
  ones = np.ones(events.shape[0], dtype = np.float32)
  
  t_lm_fwd  = np.zeros(n)
  t_lm_back = np.zeros(n)
  
  # forward project 
  print(f'timing LM fwd projection {events.shape[0]:.1E} events')
  for i in range(n):
    print(f'run {i+1} / {n}')
    t0 = time()
    lm_fwd  = lmproj.fwd_project(img, events)
    t1 = time()
    t_lm_fwd[i] = t1 - t0
  print(f'{t_lm_fwd.mean():.4f} s (mean) +-  {t_lm_fwd.std():.4f} s (std)\n')
  
  
  # back project
  print(f'timing LM back projection {events.shape[0]:.1E} events')
  for i in range(n):
    print(f'run {i+1} / {n}')
    t0 = time()
    lm_back = lmproj.back_project(ones, events)
    t1 = time()
    t_lm_back[i] = t1 - t0
  print(f'{t_lm_back.mean():.4f} s (mean) +-  {t_lm_back.std():.4f} s (std)\n')
