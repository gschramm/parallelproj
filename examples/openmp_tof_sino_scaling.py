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
parser.add_argument('--nv', type=int, default = 1, help='number of view to project')
args = parser.parse_args()

nviews = args.nv

###############################################################
# wrappers to call functions from compiled libs ###############

ar_1d_single = npct.ndpointer(dtype = ctypes.c_float, ndim = 1, flags = 'C')
ar_1d_uint   = npct.ndpointer(dtype = ctypes.c_uint,  ndim = 1, flags = 'C')

lib_parallelproj = npct.load_library('libparallelproj.so','../lib')

lib_parallelproj.joseph3d_tof_sino.restype  = None
lib_parallelproj.joseph3d_tof_sino.argtypes = [ar_1d_single,
                                               ar_1d_single,
                                               ar_1d_single,
                                               ar_1d_single,
                                               ar_1d_single,
                                               ar_1d_single,
                                               ctypes.c_ulonglong,
                                               ar_1d_uint,        #
                                               ctypes.c_uint,     # n_tofbins
                                               ctypes.c_float,    # tofbin_width 
                                               ar_1d_single,      # sigma tof
                                               ar_1d_single,      # tofcenter_offset
                                               ctypes.c_uint]     # n_sigmas 

###############################################################
###############################################################

d_scanner    = 300.
n_tofbins    = 27
sigma_tof    = (d_scanner/10)/2.35
tofbin_width = (d_scanner + 2*sigma_tof) / n_tofbins
n_sigmas     = 3

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

# forward projection
img_dim = np.array(img.shape, dtype = ctypes.c_uint)

img_fwd          = np.zeros(nLORs*n_tofbins, dtype = ctypes.c_float)  
sigma_tof        = np.full(nLORs, sigma_tof, dtype = ctypes.c_float)
tofcenter_offset = np.full(nLORs, 0, dtype = ctypes.c_float)

t0 = time()
ok = lib_parallelproj.joseph3d_tof_sino(xstart.flatten(), xend.flatten(), img.flatten(), 
                                        img_origin, voxsize, img_fwd, nLORs, img_dim,
                                        n_tofbins, tofbin_width, sigma_tof, tofcenter_offset, n_sigmas)

img_fwd_sino = img_fwd.reshape(sino_shape)
t1 = time()
t_fwd = t1 - t0
#
##----
## print results
#print('openmp cpu','#views',nviews,'fwd',t_fwd)

## show results
#import pymirc.viewer as pv
#vi = pv.ThreeAxisViewer(img_fwd_sino[:,:,:88])
#vi = pv.ThreeAxisViewer(back_img)
