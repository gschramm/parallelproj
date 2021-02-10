# small demo for listmode TOF MLEM without subsets

import os
import matplotlib.pyplot as py
import pyparallelproj as ppp
from pyparallelproj.wrapper import joseph3d_fwd, joseph3d_fwd_tof, joseph3d_back, joseph3d_back_tof
import numpy as np
import argparse
import ctypes

from time import time

#---------------------------------------------------------------------------------
# parse the command line

parser = argparse.ArgumentParser()
parser.add_argument('--ngpus',    help = 'number of GPUs to use', default = 0,   type = int)
parser.add_argument('--nsubsets', help = 'number of subsets',     default = 28,  type = int)
parser.add_argument('--tpb',      help = 'threads per block',     default = 64,  type = int)
parser.add_argument('--nontof',   help = 'non-TOF instead of TOF', action = 'store_true')
parser.add_argument('--img_mem_order', help = 'memory layout for image', default = 'C',
                                       choices = ['C','F'])
args = parser.parse_args()

#---------------------------------------------------------------------------------

ngpus     = args.ngpus
nsubsets  = args.nsubsets
tpb       = args.tpb
tof       = not args.nontof

img_mem_order = args.img_mem_order
subset        = 0

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
img = np.zeros((n0,n1,n2), dtype = np.float32, order = img_mem_order)
img[(n0//6):(5*n0//6),(n1//6):(5*n1//6),:] = 1
img_origin = (-(np.array(img.shape) / 2) +  0.5) * voxsize

# generate sinogram parameters and the projector
sd = np.array([[0,1,2],
               [0,2,1],
               [1,2,0],
               [1,0,2],
               [2,0,1],
               [2,1,0]])

for sdo in sd:
  sino_params = ppp.PETSinogramParameters(scanner, ntofbins = ntofbins, tofbin_width = 23.,
                                          spatial_dim_order = sdo)
  proj        = ppp.SinogramProjector(scanner, sino_params, img.shape, nsubsets = nsubsets,
                                      voxsize = voxsize, img_origin = img_origin, ngpus = ngpus,
                                      tof = tof, sigma_tof = 60./2.35, n_sigmas = 3.,
                                      threadsperblock = tpb)

 
  # do a forward / back projection of subset 0 - same as img_fwd = proj.fwd_project(img, 0)
  # we just write out the single steps to time the python overhead separately
 
  #img_fwd = proj.fwd_project(img, 0)
  #ones_sino = np.ones(img_fwd.shape, dtype = np.float32)
  #back = proj.back_project(ones_sino, 0)

  subset_slice = proj.subset_slices[subset]
 
  sigma_tof = np.full(proj.nLORs[subset], proj.sigma_tof, dtype = ctypes.c_float).ravel()
  tofcenter_offset = np.zeros(proj.nLORs[subset], dtype = ctypes.c_float).ravel()

  xstart = proj.xstart[subset_slice].ravel()
  xend   = proj.xend[subset_slice].ravel()
  img_ravel = img.ravel(order = img_mem_order)
  subset_nLORs     = proj.nLORs[subset]

  img_fwd = np.zeros(subset_nLORs*proj.ntofbins, dtype = ctypes.c_float)  

  back_img = np.zeros(proj.nvox, dtype = ctypes.c_float)  
  sino     = np.ones(subset_nLORs*proj.ntofbins, dtype = ctypes.c_float)  

  #--- time fwd projection
  t0 = time()
  if tof:
    ok = joseph3d_fwd_tof(xstart, xend, img_ravel, proj.img_origin, proj.voxsize, 
                          img_fwd, subset_nLORs, proj.img_dim,
                          proj.tofbin_width, sigma_tof, tofcenter_offset, 
                          proj.nsigmas, proj.ntofbins, 
                          threadsperblock = proj.threadsperblock, ngpus = proj.ngpus, lm = False) 
  else:
    ok = joseph3d_fwd(xstart, xend, img_ravel, proj.img_origin, proj.voxsize, 
                      img_fwd, subset_nLORs, proj.img_dim,
                      threadsperblock = proj.threadsperblock, ngpus = proj.ngpus, lm = False) 
    
  t1 = time()


  #--- time back projection
  t2 = time()
  if tof:
    ok = joseph3d_back_tof(xstart, xend, back_img, proj.img_origin, proj.voxsize, 
                           sino, subset_nLORs, proj.img_dim,
                           proj.tofbin_width, sigma_tof, tofcenter_offset, 
                           proj.nsigmas, proj.ntofbins, 
                           threadsperblock = proj.threadsperblock, ngpus = proj.ngpus, lm = False) 
  else:
    ok = joseph3d_back(xstart, xend, back_img, proj.img_origin, proj.voxsize, 
                       sino, subset_nLORs, proj.img_dim,
                       threadsperblock = proj.threadsperblock, ngpus = proj.ngpus, lm = False) 
  t3 = time()

  print(f'{sdo} {t1-t0} {t3-t2}')
