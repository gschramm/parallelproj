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
parser.add_argument('--nv', type=int, default = 7, help='number of view to project')
args = parser.parse_args()

nviews = args.nv

###############################################################
# wrappers to call functions from compiled libs ###############

ar_1d_single = npct.ndpointer(dtype = ctypes.c_float, ndim = 1, flags = 'C')
ar_1d_uint   = npct.ndpointer(dtype = ctypes.c_uint,  ndim = 1, flags = 'C')

lib_parallelproj = npct.load_library('libparallelproj.so','../lib')

lib_parallelproj.joseph3d_fwd_tof_sino.restype  = None
lib_parallelproj.joseph3d_fwd_tof_sino.argtypes = [ar_1d_single,
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
                                                   ar_1d_single]      # look up table for erf

lib_parallelproj.joseph3d_back_tof_sino.restype  = None
lib_parallelproj.joseph3d_back_tof_sino.argtypes = [ar_1d_single,
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
                                                    ar_1d_single]      # look up table for erf

lib_parallelproj.joseph3d_back_tof_sino_2.restype  = None
lib_parallelproj.joseph3d_back_tof_sino_2.argtypes = [ar_1d_single,
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
                                                      ar_1d_single]      # look up table for erf




###############################################################
###############################################################

d_scanner    = 300.
n_tofbins    = 27
sigma_tof    = (d_scanner/10)/2.35
tofbin_width = (d_scanner + 2*sigma_tof) / n_tofbins
n_sigmas     = 3

half_erf_lut = 0.5*erf(np.linspace(-3,3,6001), dtype = ctypes.c_float)

#--------------------------------------------------------------------------------------
#---- set up phantom and dector coordindates

xstart, xend, img, img_origin, voxsize = setup_testdata(nviews = nviews)

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

sigma_tof        = np.full(nLORs, sigma_tof, dtype = ctypes.c_float)
tofcenter_offset = np.full(nLORs, 0, dtype = ctypes.c_float)

#---- forward projection
img_fwd = np.zeros(nLORs*n_tofbins, dtype = ctypes.c_float)  

t0 = time()
ok = lib_parallelproj.joseph3d_fwd_tof_sino(xstart.flatten(), xend.flatten(), img.flatten(), 
                                            img_origin, voxsize, img_fwd, nLORs, img_dim,
                                            n_tofbins, tofbin_width, sigma_tof, tofcenter_offset, 
                                            n_sigmas, half_erf_lut)

fwd_tof_sino = img_fwd.reshape(sino_shape)
t1 = time()
t_fwd = t1 - t0

fwd_nontof_sino = fwd_tof_sino.sum(3)

#---- back projection with atomic add
ones     = np.ones(nLORs*n_tofbins, dtype = ctypes.c_float)  
back_img = np.zeros(img_dim, dtype = ctypes.c_float).flatten()

t2 = time()
ok = lib_parallelproj.joseph3d_back_tof_sino(xstart.flatten(), xend.flatten(), back_img, 
                                             img_origin, voxsize, ones, nLORs, img_dim,
                                             n_tofbins, tofbin_width, sigma_tof, tofcenter_offset, 
                                             n_sigmas, half_erf_lut)

back_img = back_img.reshape(img.shape)
t3 = time()
t_back1 = t3 - t2

#---- back projection in separate images followed by summing
back_img2 = np.zeros(img_dim, dtype = ctypes.c_float).flatten()

t4 = time()
ok = lib_parallelproj.joseph3d_back_tof_sino_2(xstart.flatten(), xend.flatten(), back_img2, 
                                               img_origin, voxsize, ones, nLORs, img_dim,
                                               n_tofbins, tofbin_width, sigma_tof, tofcenter_offset, 
                                               n_sigmas, half_erf_lut)

back_img2 = back_img2.reshape(img.shape)
t5 = time()
t_back2 = t5 - t4


#----
# print results
print('openmp cpu','#views',nviews,'fwd',t_fwd)
print('openmp cpu','#views',nviews,'back1',t_back1)
print('openmp cpu','#views',nviews,'back2',t_back2)

# show results
#import pymirc.viewer as pv
#vi = pv.ThreeAxisViewer(nontof_sino[:,:,:88])
#vi = pv.ThreeAxisViewer(back_img)
