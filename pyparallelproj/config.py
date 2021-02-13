import os
import sys
import numpy.ctypeslib as npct
import ctypes
import platform

from glob import glob

ar_1d_single = npct.ndpointer(dtype = ctypes.c_float, ndim = 1, flags = 'C')
ar_1d_int    = npct.ndpointer(dtype = ctypes.c_int,   ndim = 1, flags = 'C')
ar_1d_short  = npct.ndpointer(dtype = ctypes.c_short, ndim = 1, flags = 'C')

#---- find the compiled C / CUDA libraries

plt = platform.system()

if plt == 'Linux':
  fname      = 'libparallelproj_c.so'
  fname_cuda = 'libparallelproj_cuda.so'
elif plt == 'Windows':
  fname      = 'parallelproj_c.dll'
  fname_cuda = 'parallelproj_cuda.dll'
else:
  raise SystemError(f'{platform.system()} not supported yet.')


lib_parallelproj_c_fnames = glob(os.path.abspath(os.path.join(os.path.dirname(__file__),'*',fname)))
if len(lib_parallelproj_c_fnames) > 0:
  lib_parallelproj_c_fname = lib_parallelproj_c_fnames[0]
else:
  lib_parallelproj_c_fname = None

lib_parallelproj_cuda_fnames = glob(os.path.abspath(os.path.join(os.path.dirname(__file__),'*',fname_cuda)))
if len(lib_parallelproj_cuda_fnames) > 0:
  lib_parallelproj_cuda_fname = lib_parallelproj_cuda_fnames[0]
else:
  lib_parallelproj_cuda_fname = None


#-------------------------------------------------------------------------------------------
# add the calling signature

lib_parallelproj_c    = None
lib_parallelproj_cuda = None

if lib_parallelproj_c_fname is not None:
  lib_parallelproj_c = npct.load_library(os.path.basename(lib_parallelproj_c_fname),
                                       os.path.dirname(lib_parallelproj_c_fname))
  lib_parallelproj_c.joseph3d_fwd.restype  = None
  lib_parallelproj_c.joseph3d_fwd.argtypes = [ar_1d_single,
                                            ar_1d_single,
                                            ar_1d_single,
                                            ar_1d_single,
                                            ar_1d_single,
                                            ar_1d_single,
                                            ctypes.c_ulonglong,
                                            ar_1d_int]
  
  lib_parallelproj_c.joseph3d_back.restype  = None
  lib_parallelproj_c.joseph3d_back.argtypes = [ar_1d_single,
                                             ar_1d_single,
                                             ar_1d_single,
                                             ar_1d_single,
                                             ar_1d_single,
                                             ar_1d_single,
                                             ctypes.c_ulonglong,
                                             ar_1d_int]
  
  lib_parallelproj_c.joseph3d_fwd_tof_sino.restype  = None
  lib_parallelproj_c.joseph3d_fwd_tof_sino.argtypes = [ar_1d_single,
                                                     ar_1d_single,
                                                     ar_1d_single,
                                                     ar_1d_single,
                                                     ar_1d_single,
                                                     ar_1d_single,
                                                     ctypes.c_longlong,
                                                     ar_1d_int,         #
                                                     ctypes.c_float,    # tofbin_width 
                                                     ar_1d_single,      # sigma tof
                                                     ar_1d_single,      # tofcenter_offset
                                                     ctypes.c_float,    # n_sigmas 
                                                     ctypes.c_short]    # n_tofbins
  
  lib_parallelproj_c.joseph3d_back_tof_sino.restype  = None
  lib_parallelproj_c.joseph3d_back_tof_sino.argtypes = [ar_1d_single,
                                                      ar_1d_single,
                                                      ar_1d_single,
                                                      ar_1d_single,
                                                      ar_1d_single,
                                                      ar_1d_single,
                                                      ctypes.c_longlong,
                                                      ar_1d_int,         #
                                                      ctypes.c_float,    # tofbin_width 
                                                      ar_1d_single,      # sigma tof
                                                      ar_1d_single,      # tofcenter_offset
                                                      ctypes.c_float,    # n_sigmas 
                                                      ctypes.c_short]    # n_tofbins

  lib_parallelproj_c.joseph3d_fwd_tof_lm.restype  = None
  lib_parallelproj_c.joseph3d_fwd_tof_lm.argtypes = [ar_1d_single,
                                                   ar_1d_single,
                                                   ar_1d_single,
                                                   ar_1d_single,
                                                   ar_1d_single,
                                                   ar_1d_single,
                                                   ctypes.c_longlong,
                                                   ar_1d_int,         #
                                                   ctypes.c_float,    # tofbin_width 
                                                   ar_1d_single,      # sigma tof
                                                   ar_1d_single,      # tofcenter_offset
                                                   ctypes.c_float,    # n_sigmas 
                                                   ar_1d_short]       # tof bin 
  
  lib_parallelproj_c.joseph3d_back_tof_lm.restype  = None
  lib_parallelproj_c.joseph3d_back_tof_lm.argtypes = [ar_1d_single,
                                                    ar_1d_single,
                                                    ar_1d_single,
                                                    ar_1d_single,
                                                    ar_1d_single,
                                                    ar_1d_single,
                                                    ctypes.c_longlong,
                                                    ar_1d_int,         #
                                                    ctypes.c_float,    # tofbin_width 
                                                    ar_1d_single,      # sigma tof
                                                    ar_1d_single,      # tofcenter_offset
                                                    ctypes.c_float,    # n_sigmas 
                                                    ar_1d_short]       # tof bin 
  

if lib_parallelproj_cuda_fname is not None:
  lib_parallelproj_cuda = npct.load_library(os.path.basename(lib_parallelproj_cuda_fname),
                                            os.path.dirname(lib_parallelproj_cuda_fname))
  lib_parallelproj_cuda.joseph3d_fwd_cuda.restype  = None
  lib_parallelproj_cuda.joseph3d_fwd_cuda.argtypes = [ar_1d_single,
                                                      ar_1d_single,
                                                      ar_1d_single,
                                                      ar_1d_single,
                                                      ar_1d_single,
                                                      ar_1d_single,
                                                      ctypes.c_longlong,
                                                      ar_1d_int,
                                                      ctypes.c_int,
                                                      ctypes.c_int]
  
  lib_parallelproj_cuda.joseph3d_back_cuda.restype  = None
  lib_parallelproj_cuda.joseph3d_back_cuda.argtypes = [ar_1d_single,
                                                       ar_1d_single,
                                                       ar_1d_single,
                                                       ar_1d_single,
                                                       ar_1d_single,
                                                       ar_1d_single,
                                                       ctypes.c_longlong,
                                                       ar_1d_int,
                                                       ctypes.c_int,
                                                       ctypes.c_int]

  lib_parallelproj_cuda.joseph3d_fwd_tof_sino_cuda.restype  = None
  lib_parallelproj_cuda.joseph3d_fwd_tof_sino_cuda.argtypes = [ar_1d_single,
                                                               ar_1d_single,
                                                               ar_1d_single,
                                                               ar_1d_single,
                                                               ar_1d_single,
                                                               ar_1d_single,
                                                               ctypes.c_longlong,
                                                               ar_1d_int,         #
                                                               ctypes.c_float,    # tofbin_width 
                                                               ar_1d_single,      # sigma tof
                                                               ar_1d_single,      # tofcenter_offset
                                                               ctypes.c_float,    # n_sigmas 
                                                               ctypes.c_short,    # n_tofbins
                                                               ctypes.c_int,      # threads per block
                                                               ctypes.c_int]      # number of devices 
  
  lib_parallelproj_cuda.joseph3d_back_tof_sino_cuda.restype  = None
  lib_parallelproj_cuda.joseph3d_back_tof_sino_cuda.argtypes = [ar_1d_single,
                                                                ar_1d_single,
                                                                ar_1d_single,
                                                                ar_1d_single,
                                                                ar_1d_single,
                                                                ar_1d_single,
                                                                ctypes.c_longlong,
                                                                ar_1d_int,         #
                                                                ctypes.c_float,    # tofbin_width 
                                                                ar_1d_single,      # sigma tof
                                                                ar_1d_single,      # tofcenter_offset
                                                                ctypes.c_float,    # n_sigmas 
                                                                ctypes.c_short,    # n_tofbins
                                                                ctypes.c_int,      # threads per block
                                                                ctypes.c_int]      # number of devices 

  lib_parallelproj_cuda.joseph3d_fwd_tof_lm_cuda.restype  = None
  lib_parallelproj_cuda.joseph3d_fwd_tof_lm_cuda.argtypes = [ar_1d_single,
                                                             ar_1d_single,
                                                             ar_1d_single,
                                                             ar_1d_single,
                                                             ar_1d_single,
                                                             ar_1d_single,
                                                             ctypes.c_longlong,
                                                             ar_1d_int,         #
                                                             ctypes.c_float,    # tofbin_width 
                                                             ar_1d_single,      # sigma tof
                                                             ar_1d_single,      # tofcenter_offset
                                                             ctypes.c_float,    # n_sigmas 
                                                             ar_1d_short,       # tof bin 
                                                             ctypes.c_int,
                                                             ctypes.c_int]
  
  lib_parallelproj_cuda.joseph3d_back_tof_lm_cuda.restype  = None
  lib_parallelproj_cuda.joseph3d_back_tof_lm_cuda.argtypes = [ar_1d_single,
                                                              ar_1d_single,
                                                              ar_1d_single,
                                                              ar_1d_single,
                                                              ar_1d_single,
                                                              ar_1d_single,
                                                              ctypes.c_longlong,
                                                              ar_1d_int,         #
                                                              ctypes.c_float,    # tofbin_width 
                                                              ar_1d_single,      # sigma tof
                                                              ar_1d_single,      # tofcenter_offset
                                                              ctypes.c_float,    # n_sigmas 
                                                              ar_1d_short,       # tof bin 
                                                              ctypes.c_int,
                                                              ctypes.c_int]
