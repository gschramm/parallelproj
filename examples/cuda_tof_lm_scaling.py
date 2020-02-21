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
parser.add_argument('--ne' , type=float, default = 1e8, help='number of events', nargs='+')
parser.add_argument('--ngpus', type = int,   default = -1,    help = 'number of GPUs to use')
parser.add_argument('--tpb',   type = int,   default = 64,    help = 'threads per block')
args = parser.parse_args()

if isinstance(args.ne,float):
  ne = [int(args.ne)]
else:
  ne = [int(x) for x in args.ne]
ngpus           = args.ngpus
threadsperblock = args.tpb

###############################################################
# wrappers to call functions from compiled libs ###############

ar_1d_single = npct.ndpointer(dtype = ctypes.c_float, ndim = 1, flags = 'C')
ar_1d_uint   = npct.ndpointer(dtype = ctypes.c_uint,  ndim = 1, flags = 'C')
ar_1d_int    = npct.ndpointer(dtype = ctypes.c_int,   ndim = 1, flags = 'C')

lib_parallelproj = npct.load_library('libparallelproj_cuda.so','../lib')

lib_parallelproj.joseph3d_fwd_tof_lm_cuda.restype  = None
lib_parallelproj.joseph3d_fwd_tof_lm_cuda.argtypes = [ar_1d_single,
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
                                                      ar_1d_int,         # tof bin 
                                                      ar_1d_single,      # look up table for erf
                                                      ctypes.c_uint,
                                                      ctypes.c_int]

lib_parallelproj.joseph3d_back_tof_lm_cuda.restype  = None
lib_parallelproj.joseph3d_back_tof_lm_cuda.argtypes = [ar_1d_single,
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
                                                       ar_1d_int,         # tof bin 
                                                       ar_1d_single,      # look up table for erf
                                                       ctypes.c_uint,
                                                       ctypes.c_int]

###############################################################
###############################################################

d_scanner    = 300.
n_tofbins    = 351
sigma_tof    = (d_scanner/10)/2.35
tofbin_width = (d_scanner + 2*sigma_tof) / n_tofbins
n_sigmas     = 3

half_erf_lut = 0.5*erf(np.linspace(-3,3,6001), dtype = ctypes.c_float)


#------------------------------------------------------------------------------------
#---- set up phantom and dector coordindates

np.random.seed(1)

# load full sinogram coordinates
xstart, xend, img, img_origin, voxsize = setup_testdata(nviews = 224)

nLORs = np.prod(xstart.shape[1:])

# flatten the coordinates
xstart_sino = xstart.reshape((3,nLORs)).transpose()
xend_sino   = xend.reshape((3,nLORs)).transpose()

# shuffle the LORs to simulate LM behavior

r_inds = np.random.permutation(nLORs)

img_dim = np.array(img.shape, dtype = ctypes.c_uint)

sigma_tof        = np.full(nLORs, sigma_tof, dtype = ctypes.c_float)
tofcenter_offset = np.full(nLORs, 0, dtype = ctypes.c_float)

for nevents in ne:
  inds    = r_inds[:nevents]
  xstart  = xstart_sino[inds,:]
  xend    = xend_sino[inds,:]
  nLORs   = xstart.shape[0]
  tof_bin = np.random.randint(-n_tofbins//4, n_tofbins//4, size = nLORs, dtype = ctypes.c_int) 
 
  # forward projection
  img_fwd = np.zeros(nLORs, dtype = ctypes.c_float)  
  
  t0 = time()
  ok = lib_parallelproj.joseph3d_fwd_tof_lm_cuda(xstart.flatten(), xend.flatten(), img.flatten(), 
                                            img_origin, voxsize, img_fwd, nLORs, img_dim,
                                            n_tofbins, tofbin_width, sigma_tof, tofcenter_offset, 
                                            tof_bin, half_erf_lut, threadsperblock, ngpus)
  t1 = time()
  t_fwd = t1 - t0
  
  # back projection
  ones = np.ones(nLORs, dtype = ctypes.c_float)  
  back_img = np.zeros(img.shape, dtype = ctypes.c_float).flatten()
  t2 = time()
  ok = lib_parallelproj.joseph3d_back_tof_lm_cuda(xstart.flatten(), xend.flatten(), back_img, 
                                                  img_origin, voxsize, ones, nLORs, img_dim,
                                                  n_tofbins, tofbin_width, sigma_tof, tofcenter_offset, 
                                                  tof_bin, half_erf_lut, threadsperblock, ngpus)
  back_img = back_img.reshape(img_dim)
  t3 = time()
  t_back = t3 - t2

  #----
  # print results
  print('')
  print('openmp cpu','#nevents',f'{xstart.shape[0]:.1E}','fwd',t_fwd)
  print('openmp cpu','#nevents',f'{xstart.shape[0]:.1E}','back',t_back)
