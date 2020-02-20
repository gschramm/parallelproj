import numpy as np
import math

import numpy.ctypeslib as npct
import ctypes

from setup_testdata   import setup_testdata
from time import time

#------------------------------------------------------------------------------------------------
#---- parse the command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ne' , type=float, default = 1e8, help='number of events', nargs='+')
args = parser.parse_args()

if isinstance(args.ne,float):
  ne = [int(args.ne)]
else:
  ne = [int(x) for x in args.ne]

###############################################################
# wrappers to call functions from compiled libs ###############

ar_1d_single = npct.ndpointer(dtype = ctypes.c_float, ndim = 1, flags = 'C')
ar_1d_uint   = npct.ndpointer(dtype = ctypes.c_uint,  ndim = 1, flags = 'C')

lib_parallelproj = npct.load_library('libparallelproj.so','../lib')

lib_parallelproj.joseph3d_fwd.restype  = None
lib_parallelproj.joseph3d_fwd.argtypes = [ar_1d_single,
                                          ar_1d_single,
                                          ar_1d_single,
                                          ar_1d_single,
                                          ar_1d_single,
                                          ar_1d_single,
                                          ctypes.c_ulonglong,
                                          ar_1d_uint]

lib_parallelproj.joseph3d_back.restype  = None
lib_parallelproj.joseph3d_back.argtypes = [ar_1d_single,
                                           ar_1d_single,
                                           ar_1d_single,
                                           ar_1d_single,
                                           ar_1d_single,
                                           ar_1d_single,
                                           ctypes.c_ulonglong,
                                           ar_1d_uint]

lib_parallelproj.joseph3d_back_2.restype  = None
lib_parallelproj.joseph3d_back_2.argtypes = [ar_1d_single,
                                             ar_1d_single,
                                             ar_1d_single,
                                             ar_1d_single,
                                             ar_1d_single,
                                             ar_1d_single,
                                             ctypes.c_ulonglong,
                                             ar_1d_uint]
###############################################################
###############################################################

#------------------------------------------------------------------------------------
#---- set up phantom and dector coordindates

np.random.seed(1)

# load full sinogram coordinates
xstart, xend, img, img_origin, voxsize = setup_testdata(nviews = 224)

n_lors = np.prod(xstart.shape[1:])

# flatten the coordinates
xstart_sino = xstart.reshape((3,n_lors)).transpose()
xend_sino   = xend.reshape((3,n_lors)).transpose()

# shuffle the LORs to simulate LM behavior

r_inds = np.random.permutation(n_lors)

img_dim = np.array(img.shape, dtype = ctypes.c_uint)

for nevents in ne:
  inds   = r_inds[:nevents]
  xstart = xstart_sino[inds,:]
  xend   = xend_sino[inds,:]
  nLORs  = xstart.shape[0]
 
  # forward projection
  img_fwd = np.zeros(nLORs, dtype = ctypes.c_float)  
  
  t0 = time()
  ok = lib_parallelproj.joseph3d_fwd(xstart.flatten(), xend.flatten(), img.flatten(), 
                                     img_origin, voxsize, img_fwd, nLORs, img_dim)
  
  t1 = time()
  t_fwd = t1 - t0
  
  # back projection
  ones = np.ones(nLORs, dtype = ctypes.c_float)  
  back_img = np.zeros(img.shape, dtype = ctypes.c_float).flatten()
  
  t2 = time()
  ok = lib_parallelproj.joseph3d_back(xstart.flatten(), xend.flatten(), back_img, 
                                      img_origin, voxsize, ones, nLORs, img_dim)
  back_img = back_img.reshape(img_dim)
  t3 = time()
  t_back = t3 - t2

  #----
  back_img2= np.zeros(img.shape, dtype = ctypes.c_float).flatten()
  
  t4 = time()
  ok = lib_parallelproj.joseph3d_back_2(xstart.flatten(), xend.flatten(), back_img2, 
                                        img_origin, voxsize, ones, nLORs, img_dim)
  back_img2 = back_img2.reshape(img_dim)
  t5 = time()
  t_back2 = t5 - t4
    
  #----
  # print results
  print('')
  print('openmp cpu','#nevents',f'{xstart.shape[0]:.1E}','fwd',t_fwd)
  print('openmp cpu','#nevents',f'{xstart.shape[0]:.1E}','back',t_back)
  print('openmp cpu','#nevents',f'{xstart.shape[0]:.1E}','back2',t_back2)
