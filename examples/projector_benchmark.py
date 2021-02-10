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
parser.add_argument('--counts',   help = 'counts to simulate',    default = 4e6, type = float)
parser.add_argument('--nsubsets', help = 'number of subsets',     default = 28,  type = int)
parser.add_argument('--n',        help = 'number of averages',    default = 5,   type = int)
parser.add_argument('--tpb',      help = 'threads per block',     default = 64,  type = int)
parser.add_argument('--nontof',   help = 'non-TOF instead of TOF', action = 'store_true')
parser.add_argument('--img_mem_order', help = 'memory layout for image', default = 'C',
                                       choices = ['C','F'])
parser.add_argument('--sino_dim_order', help = 'axis order in sinogram', default = ['0','1','2'],
                     nargs = '+')

args = parser.parse_args()

#---------------------------------------------------------------------------------

ngpus         = args.ngpus
counts        = args.counts
nsubsets      = args.nsubsets
n             = args.n
tpb           = args.tpb
tof           = not args.nontof
img_mem_order = args.img_mem_order
subset        = 0

spatial_dim_order = np.array(args.sino_dim_order, dtype = np.int)

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
sino_params = ppp.PETSinogramParameters(scanner, ntofbins = ntofbins, tofbin_width = 23.,
                                        spatial_dim_order = spatial_dim_order)
proj        = ppp.SinogramProjector(scanner, sino_params, img.shape, nsubsets = nsubsets, 
                                    voxsize = voxsize, img_origin = img_origin, ngpus = ngpus,
                                    tof = tof, sigma_tof = 60./2.35, n_sigmas = 3.,
                                    threadsperblock = tpb)

# contamination sinogram with scatter and randoms
# useful to avoid division by 0 in the ratio of data and exprected data
ones_sino = np.ones(proj.subset_sino_shapes[0], dtype = np.float32)

#-------------------------------------------------------------------------------------
# time sino fwd and back projection
# do a forward / back projection of subset 0 - same as img_fwd = proj.fwd_project(img, 0)
# we just write out the single steps to time the python overhead separately

t_sino_fwd  = np.zeros(n)
t_sino_back = np.zeros(n)


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
for i in range(n+1):
  if i > 0: print(f'run {i} / {n}')
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
  if i > 0:
    t_sino_fwd[i-1]  = t1 - t0

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
                       threadsperblock = proj.threadsperblock, ngpus = proj.ngpus) 
  t3 = time()
  if i > 0:
    t_sino_back[i-1]  = t3 - t2

print(f'\nsino fwd  {t_sino_fwd[1:].mean():.4f} s (mean) +-  {t_sino_fwd[1:].std():.4f} s (std)')
print(f'sino back {t_sino_back[1:].mean():.4f} s (mean) +-  {t_sino_back[1:].std():.4f} s (std)')


#-------------------------------------------------------------------------------------
# time LM fwd and back projection

if counts > 0:
  print('\ngenerating LM data\n')
  # generate list mode events and the corresponting values in the contamination and sensitivity
  scale_fac = (counts / img_fwd.sum())
  img_fwd  *= scale_fac 
  
  em_sino = np.random.poisson(img_fwd.reshape(proj.subset_sino_shapes[subset]))
  
  events = sino_params.sinogram_to_listmode(em_sino, subset = subset, 
                                            nsubsets = nsubsets)
  
  # create a listmode projector for the LM MLEM iterations
  lmproj = ppp.LMProjector(proj.scanner, proj.img_dim, voxsize = proj.voxsize, 
                           img_origin = proj.img_origin, ngpus = proj.ngpus,
                           tof = proj.tof, sigma_tof = proj.sigma_tof, 
                           tofbin_width = proj.tofbin_width,
                           n_sigmas = proj.nsigmas,
                           threadsperblock = proj.threadsperblock)
  
  values = np.ones(events.shape[0], dtype = np.float32)
  
  t_lm_fwd  = np.zeros(n)
  t_lm_back = np.zeros(n)

  nevents = events.shape[0]

  img_fwd = np.zeros(nevents, dtype = ctypes.c_float)  

  xstart = lmproj.scanner.get_crystal_coordinates(events[:,0:2]).ravel()
  xend   = lmproj.scanner.get_crystal_coordinates(events[:,2:4]).ravel()
  tofbin = events[:,4].astype(ctypes.c_short)

  sigma_tof = np.full(nevents, lmproj.sigma_tof, dtype = ctypes.c_float)
  tofcenter_offset = np.zeros(nevents, dtype = ctypes.c_float)

  back_img = np.zeros(lmproj.nvox, dtype = ctypes.c_float)  

  # time LM fwd projection
  for i in range(n+1):
    if i > 0: print(f'run {i} / {n}')
    t0 = time()
    if tof:
      ok = joseph3d_fwd_tof(xstart, xend, 
                            img_ravel, lmproj.img_origin, lmproj.voxsize, 
                            img_fwd, nevents, lmproj.img_dim,
                            lmproj.tofbin_width, sigma_tof, tofcenter_offset, lmproj.nsigmas,
                            tofbin, threadsperblock = lmproj.threadsperblock, 
                            ngpus = lmproj.ngpus, lm = True) 
    else:
      ok = joseph3d_fwd(xstart, xend, 
                        img_ravel, lmproj.img_origin, lmproj.voxsize, 
                        img_fwd, nevents, lmproj.img_dim,
                        threadsperblock = lmproj.threadsperblock, 
                        ngpus = lmproj.ngpus) 
    t1 = time()
    if i > 0:
      t_lm_fwd[i-1] = t1 - t0


    t2 = time()
    if tof:
      ok = joseph3d_back_tof(xstart, xend, 
                             back_img, lmproj.img_origin, lmproj.voxsize, 
                             values, nevents, lmproj.img_dim,
                             lmproj.tofbin_width, sigma_tof, tofcenter_offset, lmproj.nsigmas, 
                             tofbin, threadsperblock = lmproj.threadsperblock, 
                             ngpus = lmproj.ngpus, lm = True) 
    else:
      ok = joseph3d_back(xstart, xend, 
                         back_img, lmproj.img_origin, lmproj.voxsize, 
                         values, nevents, lmproj.img_dim,
                         threadsperblock = lmproj.threadsperblock, 
                         ngpus = lmproj.ngpus) 
    t3 = time()
    if i > 0:
      t_lm_back[i-1] = t3 - t2
  
  print(f'\n{t_lm_fwd.mean():.4f} s (mean) +-  {t_lm_fwd.std():.4f} s (std)')
  print(f'{t_lm_back.mean():.4f} s (mean) +-  {t_lm_back.std():.4f} s (std)\n')
