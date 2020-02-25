import numpy as np
import math

import numpy.ctypeslib as npct
import ctypes

from setup_testdata   import setup_testdata
from time import time

from scipy.special import erf

#------------------------------------------------------------------------------------------------
#---- parse the command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--nv', type=int, default = 7, help='number of view to project', nargs='+')
parser.add_argument('--ngpus', type=int, default = -1, help='number of GPUs to use')
parser.add_argument('--tpb',   type=int, default = 64, help='threads per block')
args = parser.parse_args()

if isinstance(args.nv,int):
  nviews = [int(args.nv)]
else:
  nviews = [int(x) for x in args.nv]
ngpus           = args.ngpus
threadsperblock = args.tpb

###############################################################
# wrappers to call functions from compiled libs ###############

ar_1d_single = npct.ndpointer(dtype = ctypes.c_float, ndim = 1, flags = 'C')
ar_1d_uint   = npct.ndpointer(dtype = ctypes.c_uint,  ndim = 1, flags = 'C')

lib_parallelproj = npct.load_library('libparallelproj_cuda.so','../lib')

lib_parallelproj.joseph3d_fwd_tof_sino_cuda.restype  = None
lib_parallelproj.joseph3d_fwd_tof_sino_cuda.argtypes = [ar_1d_single,
                                                        ar_1d_single,
                                                        ar_1d_single,
                                                        ar_1d_single,
                                                        ar_1d_single,
                                                        ar_1d_single,
                                                        ctypes.c_longlong,
                                                        ar_1d_uint,        #
                                                        ctypes.c_int,      # n_tofbins
                                                        ctypes.c_float,    # tofbin_width 
                                                        ar_1d_single,      # sigma tof
                                                        ar_1d_single,      # tofcenter_offset
                                                        ctypes.c_uint,     # n_sigmas 
                                                        ctypes.c_uint,     # threads per block
                                                        ctypes.c_int]      # number of devices 

lib_parallelproj.joseph3d_back_tof_sino_cuda.restype  = None
lib_parallelproj.joseph3d_back_tof_sino_cuda.argtypes = [ar_1d_single,
                                                         ar_1d_single,
                                                         ar_1d_single,
                                                         ar_1d_single,
                                                         ar_1d_single,
                                                         ar_1d_single,
                                                         ctypes.c_longlong,
                                                         ar_1d_uint,        #
                                                         ctypes.c_int,      # n_tofbins
                                                         ctypes.c_float,    # tofbin_width 
                                                         ar_1d_single,      # sigma tof
                                                         ar_1d_single,      # tofcenter_offset
                                                         ctypes.c_uint,     # n_sigmas 
                                                         ctypes.c_uint,     # threads per block
                                                         ctypes.c_int]      # number of devices 

###############################################################
###############################################################

d_scanner    = 300.
n_tofbins    = 27
sig_tof      = (d_scanner/10)/2.35
tofbin_width = (d_scanner + 2*sig_tof) / n_tofbins
n_sigmas     = 3

#--------------------------------------------------------------------------------------
#---- set up phantom and dector coordindates

for nv in nviews:
  xstart, xend, img, img_origin, voxsize = setup_testdata(nviews = nv)
  
  # swap axes
  # it seems to be best to have (radial, angle, plane) in memory
  xstart     = np.swapaxes(np.swapaxes(xstart, 1, 3), 1, 2)
  xend       = np.swapaxes(np.swapaxes(xend,   1, 3), 1, 2)
  
  nLORs   = np.prod(xstart.shape[1:])
  
  sino_shape = (xstart.shape[1:] + (n_tofbins,))
  print(sino_shape)
  
  # flatten the sinogram coordinates
  xstart = xstart.reshape((3,nLORs)).transpose()
  xend   = xend.reshape((3,nLORs)).transpose()
  
  img_dim = np.array(img.shape, dtype = ctypes.c_uint)
  
  sigma_tof        = np.full(nLORs, sig_tof, dtype = ctypes.c_float)
  tofcenter_offset = np.full(nLORs, 0, dtype = ctypes.c_float)
  
  #---- forward projection
  img_fwd = np.zeros(nLORs*n_tofbins, dtype = ctypes.c_float)  
  
  t0 = time()
  ok = lib_parallelproj.joseph3d_fwd_tof_sino_cuda(xstart.flatten(), xend.flatten(), img.flatten(), 
                                                   img_origin, voxsize, img_fwd, nLORs, img_dim,
                                                   n_tofbins, tofbin_width, sigma_tof, tofcenter_offset, 
                                                   n_sigmas, threadsperblock, ngpus)
  
  fwd_tof_sino = img_fwd.reshape(sino_shape)
  t1 = time()
  t_fwd = t1 - t0
  
  fwd_nontof_sino = fwd_tof_sino.sum(3)
  
  #---- back projection
  ones     = np.ones(nLORs*n_tofbins, dtype = ctypes.c_float)  
  back_img = np.zeros(img_dim, dtype = ctypes.c_float).flatten()
  
  t2 = time()
  ok = lib_parallelproj.joseph3d_back_tof_sino_cuda(xstart.flatten(), xend.flatten(), back_img, 
                                                    img_origin, voxsize, ones, nLORs, img_dim,
                                                    n_tofbins, tofbin_width, sigma_tof, tofcenter_offset, 
                                                    n_sigmas, threadsperblock, ngpus)
  
  back_img = back_img.reshape(img.shape)
  t3 = time()
  t_back = t3 - t2
  
  #----
  # print results
  print(str(ngpus) + '-P100',nv,'fwd',t_fwd)
  print(str(ngpus) + '-P100',nv,'back',t_back)

# show results
#import pymirc.viewer as pv
#vi = pv.ThreeAxisViewer(fwd_nontof_sino[:,:,:88])
#vi = pv.ThreeAxisViewer(back_img)
