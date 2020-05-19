import numpy as np
import matplotlib.pyplot as py
import numpy.ctypeslib as npct
import ctypes

from pet_scanners import RegularPolygonPETScanner
from sinogram import PETSinogram

###############################################################
# wrappers to call functions from compiled libs ###############

ar_1d_single = npct.ndpointer(dtype = ctypes.c_float, ndim = 1, flags = 'C')
ar_1d_int    = npct.ndpointer(dtype = ctypes.c_int,   ndim = 1, flags = 'C')

lib_parallelproj = npct.load_library('libparallelproj.so','../lib')

lib_parallelproj.joseph3d_fwd.restype  = None
lib_parallelproj.joseph3d_fwd.argtypes = [ar_1d_single,
                                          ar_1d_single,
                                          ar_1d_single,
                                          ar_1d_single,
                                          ar_1d_single,
                                          ar_1d_single,
                                          ctypes.c_ulonglong,
                                          ar_1d_int]

lib_parallelproj.joseph3d_back.restype  = None
lib_parallelproj.joseph3d_back.argtypes = [ar_1d_single,
                                           ar_1d_single,
                                           ar_1d_single,
                                           ar_1d_single,
                                           ar_1d_single,
                                           ar_1d_single,
                                           ctypes.c_ulonglong,
                                           ar_1d_int]

lib_parallelproj.joseph3d_fwd_tof_sino.restype  = None
lib_parallelproj.joseph3d_fwd_tof_sino.argtypes = [ar_1d_single,
                                                   ar_1d_single,
                                                   ar_1d_single,
                                                   ar_1d_single,
                                                   ar_1d_single,
                                                   ar_1d_single,
                                                   ctypes.c_longlong,
                                                   ar_1d_int,         #
                                                   ctypes.c_int,      # n_tofbins
                                                   ctypes.c_float,    # tofbin_width 
                                                   ar_1d_single,      # sigma tof
                                                   ar_1d_single,      # tofcenter_offset
                                                   ctypes.c_int]      # n_sigmas 

lib_parallelproj.joseph3d_back_tof_sino.restype  = None
lib_parallelproj.joseph3d_back_tof_sino.argtypes = [ar_1d_single,
                                                    ar_1d_single,
                                                    ar_1d_single,
                                                    ar_1d_single,
                                                    ar_1d_single,
                                                    ar_1d_single,
                                                    ctypes.c_longlong,
                                                    ar_1d_int,         #
                                                    ctypes.c_int,      # n_tofbins
                                                    ctypes.c_float,    # tofbin_width 
                                                    ar_1d_single,      # sigma tof
                                                    ar_1d_single,      # tofcenter_offset
                                                    ctypes.c_int]      # n_sigmas 


###############################################################


scanner = RegularPolygonPETScanner(ncrystals_per_module = np.array([16,3]),
                                   nmodules             = np.array([28,1]))

# setup TOF params for sino
d_scanner    = 2*scanner.R
sig_tof      = 60
n_tofbins    = 27
tofbin_width = (2*scanner.R + 2*sig_tof) / n_tofbins
n_sigmas     = 3


### nontof forward projection

sino      = PETSinogram(scanner)
all_views = np.arange(sino.nviews)

# get the crystals IDs for all views
istart, iend = sino.get_view_crystal_indices(all_views)
# get the world coordiates for all view
xstart = scanner.get_crystal_coordinates(istart.reshape(-1,2)).reshape((sino.nrad,sino.nviews,sino.nplanes,3))
xend   = scanner.get_crystal_coordinates(iend.reshape(-1,2)).reshape((sino.nrad,sino.nviews,sino.nplanes,3))

n = 250
img = np.zeros((n,n,1), dtype = ctypes.c_float)  
img[(n//4):(3*n//4),(n//4):(3*n//4),0] = 1
voxsize = np.array([2.,2.,2.], dtype = ctypes.c_float)
img_origin = np.array([(-n//2 + 0.5)*voxsize[0],(-n//2 + 0.5)*voxsize[1], 0], dtype = ctypes.c_float)

# subset parameters
subset_slice = 4*[slice(None,None,None)]
subset_dir   = 1
nsubsets     = 28
subset       = 13
subset_slice[subset_dir] = slice(subset,None,nsubsets)
subset_slice = tuple(subset_slice)

nLORs   = xstart[subset_slice].reshape(-1,3).shape[0]
img_dim = np.array(img.shape, dtype = ctypes.c_int)

#img_fwd = np.zeros(sino.shape, dtype = ctypes.c_float)  
img_fwd = np.zeros(nLORs*sino.ntofbins, dtype = ctypes.c_float)  
ok = lib_parallelproj.joseph3d_fwd(xstart[subset_slice].flatten(), 
                                   xend[subset_slice].flatten(), 
                                   img.flatten(), img_origin, voxsize, 
                                   img_fwd, nLORs, img_dim) 

subset_shape = np.array(sino.shape)
subset_shape[subset_dir] = subset_shape[subset_dir] // nsubsets
subset_shape = tuple(subset_shape)

img_fwd_sino = img_fwd.reshape(subset_shape)

### tof forward projection

sino             = PETSinogram(scanner, ntofbins = n_tofbins, tofbin_width = tofbin_width)
sigma_tof        = np.full(nLORs, sig_tof, dtype = ctypes.c_float)
tofcenter_offset = np.full(nLORs, 0, dtype = ctypes.c_float)
img_fwd_tof      = np.zeros(nLORs*sino.ntofbins, dtype = ctypes.c_float)  

ok = lib_parallelproj.joseph3d_fwd_tof_sino(xstart[subset_slice].flatten(), 
                                            xend[subset_slice].flatten(), 
                                            img.flatten(), img_origin, voxsize, 
                                            img_fwd_tof, nLORs, img_dim,
                                            sino.ntofbins, sino.tofbin_width, 
                                            sigma_tof, tofcenter_offset, n_sigmas)

subset_shape = np.array(sino.shape)
subset_shape[subset_dir] = subset_shape[subset_dir] // nsubsets
subset_shape = tuple(subset_shape)

img_fwd_tof_sino = img_fwd_tof.reshape(subset_shape)                                                
