import numpy as np
import math

import numpy.ctypeslib as npct
import ctypes

from setup_testdata   import setup_testdata
from time import time

#---------------------------------------------------------------------------------------
#---- parse the command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ne' ,   type = float, default = [1e7], help = 'number of events', 
                               nargs = '+')
parser.add_argument('--ngpus', type = int,   default = -1,    help = 'number of GPUs to use')
parser.add_argument('--tpb',   type = int,   default = 64,    help = 'threads per block')
args = parser.parse_args()

ne              = [int(x) for x in args.ne]
ngpus           = args.ngpus
threadsperblock = args.tpb

###############################################################
# wrappers to call functions from compiled libs ###############

ar_1d_single = npct.ndpointer(dtype = np.float32, ndim = 1, flags = 'C')
ar_1d_double = npct.ndpointer(dtype = np.float64, ndim = 1, flags = 'C')

lib_cudaproj = npct.load_library('libparallelproj_cuda.so','../lib')

lib_cudaproj.joseph3d_lm_cuda.restype  = None
lib_cudaproj.joseph3d_lm_cuda.argtypes = [ar_1d_single,
                                          ar_1d_single,
                                          ar_1d_single,
                                          ar_1d_single,
                                          ar_1d_single,
                                          ar_1d_single,
                                          ctypes.c_ulonglong,
                                          ctypes.c_uint,
                                          ctypes.c_uint,
                                          ctypes.c_uint,
                                          ctypes.c_uint,
                                          ctypes.c_int]

lib_cudaproj.joseph3d_lm_back_cuda.restype  = None
lib_cudaproj.joseph3d_lm_back_cuda.argtypes = [ar_1d_single,
                                               ar_1d_single,
                                               ar_1d_single,
                                               ar_1d_single,
                                               ar_1d_single,
                                               ar_1d_single,
                                               ctypes.c_ulonglong,
                                               ctypes.c_uint,
                                               ctypes.c_uint,
                                               ctypes.c_uint,
                                               ctypes.c_uint,
                                               ctypes.c_int]



###############################################################
###############################################################

#----------------------------------------------------------------------------------
#---- set up phantom and dector coordindates

np.random.seed(1)

# load full sinogram coordinates
xstart, xend, img, img_origin, voxsize = setup_testdata(nviews = 244)

n_lors = np.prod(xstart.shape[1:])

# flatten the coordinates
xstart_sino = xstart.reshape((3,n_lors)).transpose()
xend_sino   = xend.reshape((3,n_lors)).transpose()

# shuffle the LORs to simulate LM behavior
r_inds = np.random.permutation(n_lors)

n0, n1, n2 = img.shape

for nevents in ne:
  inds   = r_inds[:nevents]
  xstart = xstart_sino[inds,:]
  xend   = xend_sino[inds,:]
  nLORs  = xstart.shape[0]

  # forward projection
  t0 = time()
  img_fwd = np.zeros(nLORs, np.float32)  
  
  ok = lib_cudaproj.joseph3d_lm_cuda(xstart.flatten(), xend.flatten(), img.flatten(), 
                                     img_origin, voxsize, img_fwd, nLORs, n0, n1, n2,
                                     threadsperblock, ngpus)
  t1 = time()
  t_fwd = t1 - t0
  
  # back projection
  ones     = np.ones(nLORs, np.float32)
  back_img = np.zeros(np.prod(img.shape), np.float32)
  
  t2 = time()
  ok = lib_cudaproj.joseph3d_lm_back_cuda(xstart.flatten(), xend.flatten(), back_img, 
                                          img_origin, voxsize, ones, nLORs, n0, n1, n2,
                                          threadsperblock, ngpus)
  back_img = back_img.reshape(img.shape)
  t3 = time()
  t_back = t3 - t2
   
   
  #----
  # print results
  print('')
  print('cuda','#nevents',f'{xstart.shape[0]:.1E}','fwd',t_fwd)
  print('cuda','#nevents',f'{xstart.shape[0]:.1E}','back',t_back)

## show results
#import pymirc.viewer as pv
#vi = pv.ThreeAxisViewer(img_fwd_sino[:,:,:88])
#vi = pv.ThreeAxisViewer(back_img)
