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

ar_1d_single = npct.ndpointer(dtype = np.float32, ndim = 1, flags = 'C')

lib_NCparallelproj = npct.load_library('libparallelproj.so','../lib')

lib_NCparallelproj.joseph3d_lm.restype  = None
lib_NCparallelproj.joseph3d_lm.argtypes = [ar_1d_single,
                                           ar_1d_single,
                                           ar_1d_single,
                                           ar_1d_single,
                                           ar_1d_single,
                                           ar_1d_single,
                                           ctypes.c_ulonglong,
                                           ctypes.c_uint,
                                           ctypes.c_uint,
                                           ctypes.c_uint]

lib_NCparallelproj.joseph3d_lm_back.restype  = None
lib_NCparallelproj.joseph3d_lm_back.argtypes = [ar_1d_single,
                                                ar_1d_single,
                                                ar_1d_single,
                                                ar_1d_single,
                                                ar_1d_single,
                                                ar_1d_single,
                                                ctypes.c_ulonglong,
                                                ctypes.c_uint,
                                                ctypes.c_uint,
                                                ctypes.c_uint]
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

n0, n1, n2 = img.shape

for nevents in ne:
  inds   = r_inds[:nevents]
  xstart = xstart_sino[inds,:]
  xend   = xend_sino[inds,:]
  nLORs  = xstart.shape[0]
 
  # forward projection
  img_fwd = np.zeros(nLORs, np.float32)  
  
  t0 = time()
  ok = lib_NCparallelproj.joseph3d_lm(xstart.flatten(), xend.flatten(), img.flatten(), 
                                      img_origin, voxsize, img_fwd, nLORs, n0, n1, n2)
  
  t1 = time()
  t_fwd = t1 - t0
  
  # back projection
  ones = np.ones(nLORs, dtype = np.float32)  
  back_img = np.zeros(img.shape, dtype = np.float32).flatten()
  
  t2 = time()
  ok = lib_NCparallelproj.joseph3d_lm_back(xstart.flatten(), xend.flatten(), back_img, 
                                           img_origin, voxsize, ones, nLORs, n0, n1, n2)
  back_img = back_img.reshape((n0,n1,n2))
  t3 = time()
  t_back = t3 - t2
    
  #----
  # print results
  print('')
  print('openmp cpu','#nevents',f'{xstart.shape[0]:.1E}','fwd',t_fwd)
  print('openmp cpu','#nevents',f'{xstart.shape[0]:.1E}','back',t_back)
