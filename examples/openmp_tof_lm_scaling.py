import numpy as np
import math

import numpy.ctypeslib as npct
import ctypes

import os
import multiprocessing
import platform

from setup_testdata   import setup_testdata
from time import time

#------------------------------------------------------------------------------------------------
#---- parse the command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ne' , type=float, default = 1e8, help='number of events', nargs='+')
parser.add_argument('--nrep', type=int, default = 1, help='number of repetitions')
args = parser.parse_args()

if isinstance(args.ne,float):
  ne = [int(args.ne)]
else:
  ne = [int(x) for x in args.ne]

nrep = args.nrep
###############################################################
# wrappers to call functions from compiled libs ###############

ar_1d_single = npct.ndpointer(dtype = ctypes.c_float, ndim = 1, flags = 'C')
ar_1d_int    = npct.ndpointer(dtype = ctypes.c_int,   ndim = 1, flags = 'C')

lib_parallelproj = npct.load_library('libparallelproj.so','../lib')

lib_parallelproj.joseph3d_fwd_tof_lm.restype  = None
lib_parallelproj.joseph3d_fwd_tof_lm.argtypes = [ar_1d_single,
                                                 ar_1d_single,
                                                 ar_1d_single,
                                                 ar_1d_single,
                                                 ar_1d_single,
                                                 ar_1d_single,
                                                 ctypes.c_longlong,
                                                 ar_1d_int,        #
                                                 ctypes.c_float,    # tofbin_width 
                                                 ar_1d_single,      # sigma tof
                                                 ar_1d_single,      # tofcenter_offset
                                                 ar_1d_int]         # tof bin 

lib_parallelproj.joseph3d_back_tof_lm.restype  = None
lib_parallelproj.joseph3d_back_tof_lm.argtypes = [ar_1d_single,
                                                  ar_1d_single,
                                                  ar_1d_single,
                                                  ar_1d_single,
                                                  ar_1d_single,
                                                  ar_1d_single,
                                                  ctypes.c_longlong,
                                                  ar_1d_int,        #
                                                  ctypes.c_float,    # tofbin_width 
                                                  ar_1d_single,      # sigma tof
                                                  ar_1d_single,      # tofcenter_offset
                                                  ar_1d_int]         # tof bin 

lib_parallelproj.joseph3d_back_tof_lm_2.restype  = None
lib_parallelproj.joseph3d_back_tof_lm_2.argtypes = [ar_1d_single,
                                                    ar_1d_single,
                                                    ar_1d_single,
                                                    ar_1d_single,
                                                    ar_1d_single,
                                                    ar_1d_single,
                                                    ctypes.c_longlong,
                                                    ar_1d_int,        #
                                                    ctypes.c_float,    # tofbin_width 
                                                    ar_1d_single,      # sigma tof
                                                    ar_1d_single,      # tofcenter_offset
                                                    ar_1d_int]         # tof bin 

# get the number of available CPUs for OPENMP
if os.getenv('OMP_NUM_THREADS') is None:
  ncpus = multiprocessing.cpu_count()
else:
  ncpus = int(os.getenv('OMP_NUM_THREADS'))

###############################################################
###############################################################

d_scanner    = 300.
n_tofbins    = 351
sigma_tof    = (d_scanner/10)/2.35
tofbin_width = (d_scanner + 2*sigma_tof) / n_tofbins
n_sigmas     = 3

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

img_dim = np.array(img.shape, dtype = ctypes.c_int)

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
  
  for i in range(nrep):  
    t0 = time()
    ok = lib_parallelproj.joseph3d_fwd_tof_lm(xstart.flatten(), xend.flatten(), img.flatten(), 
                                              img_origin, voxsize, img_fwd, nLORs, img_dim,
                                              tofbin_width, sigma_tof, tofcenter_offset, 
                                              tof_bin)
    t1 = time()
    t_fwd = t1 - t0
    print(str(ncpus) + 'th-' + platform.node(), f'{xstart.shape[0]:.1E}','fwd',t_fwd)
  
  # back projection
  ones = np.ones(nLORs, dtype = ctypes.c_float)  
  for i in range(nrep):  
    back_img = np.zeros(img.shape, dtype = ctypes.c_float).flatten()
    t2 = time()
    ok = lib_parallelproj.joseph3d_back_tof_lm(xstart.flatten(), xend.flatten(), back_img, 
                                               img_origin, voxsize, ones, nLORs, img_dim,
                                               tofbin_width, sigma_tof, tofcenter_offset, 
                                               tof_bin)
    back_img = back_img.reshape(img_dim)
    t3 = time()
    t_back = t3 - t2
    print(str(ncpus) + 'th-' + platform.node(),f'{xstart.shape[0]:.1E}','back',t_back)

  #-----
  for i in range(nrep):  
    back_img2 = np.zeros(img.shape, dtype = ctypes.c_float).flatten()
    t4 = time()
    ok = lib_parallelproj.joseph3d_back_tof_lm_2(xstart.flatten(), xend.flatten(), back_img2, 
                                                 img_origin, voxsize, ones, nLORs, img_dim,
                                                 tofbin_width, sigma_tof, tofcenter_offset, 
                                                 tof_bin)
    back_img2 = back_img2.reshape(img_dim)
    t5 = time()
    t_back2 = t5 - t4
    print(str(ncpus) + 'th-' + platform.node(),f'{xstart.shape[0]:.1E}','back2',t_back2)
