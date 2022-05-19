import os
import sys
import re
import numpy.ctypeslib as npct
import ctypes
from ctypes.util import find_library
from ctypes import POINTER
import platform
import warnings

from glob import glob
from numba import cuda

try:
  import cupy as cp
  cupy_available = True
except:
  cupy_available = False

#---------------------------------------------------------------------------------------
# get the number of visible GPUs
try:
  n_visible_gpus = len(cuda.gpus)
except:
  n_visible_gpus = 0

#---------------------------------------------------------------------------------------

# load a kernel defined in a external file
if cupy_available:
  with open('../cuda/src/projector_kernels.cu','r') as f:
    lines = f.read()
    joseph3d_fwd_cuda_kernel  = cp.RawKernel(lines, 'joseph3d_fwd_cuda_kernel')
    joseph3d_back_cuda_kernel = cp.RawKernel(lines, 'joseph3d_back_cuda_kernel')
else:
  joseph3d_fwd_cuda_kernel  = None
  joseph3d_back_cuda_kernel = None

#---------------------------------------------------------------------------------------


ar_1d_single = npct.ndpointer(dtype = ctypes.c_float, ndim = 1, flags = 'C')
ar_1d_int    = npct.ndpointer(dtype = ctypes.c_int,   ndim = 1, flags = 'C')
ar_1d_short  = npct.ndpointer(dtype = ctypes.c_short, ndim = 1, flags = 'C')

#---- find the compiled C / CUDA libraries

lib_parallelproj_c_fname    = find_library('parallelproj_c')
lib_parallelproj_cuda_fname = find_library('parallelproj_cuda')

#-------------------------------------------------------------------------------------------
# add the calling signature

lib_parallelproj_c    = None
lib_parallelproj_cuda = None

if lib_parallelproj_c_fname is not None:
  lib_parallelproj_c = npct.load_library(os.path.basename(lib_parallelproj_c_fname),
                                       os.path.dirname(lib_parallelproj_c_fname))
  lib_parallelproj_c.__file__    = lib_parallelproj_c_fname

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
                                                     ctypes.c_short,    # n_tofbins
                                                     ctypes.c_ubyte,    # LOR dep. TOF sigma
                                                     ctypes.c_ubyte]    # LOR dep. TOF center offset
  
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
                                                      ctypes.c_short,    # n_tofbins
                                                      ctypes.c_ubyte,    # LOR dep. TOF sigma
                                                      ctypes.c_ubyte]    # LOR dep. TOF center offset

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
                                                   ar_1d_short,       # tof bin 
                                                   ctypes.c_ubyte,    # LOR dep. TOF sigma
                                                   ctypes.c_ubyte]    # LOR dep. TOF center offset
  
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
                                                    ar_1d_short,       # tof bin 
                                                    ctypes.c_ubyte,    # LOR dep. TOF sigma
                                                    ctypes.c_ubyte]    # LOR dep. TOF center offset
  

if lib_parallelproj_cuda_fname is not None:
  lib_parallelproj_cuda = npct.load_library(os.path.basename(lib_parallelproj_cuda_fname),
                                            os.path.dirname(lib_parallelproj_cuda_fname))
  lib_parallelproj_cuda.__file__    = lib_parallelproj_cuda_fname

  lib_parallelproj_cuda.joseph3d_fwd_cuda.restype  = None
  lib_parallelproj_cuda.joseph3d_fwd_cuda.argtypes = [ar_1d_single,
                                                      ar_1d_single,
                                                      POINTER(POINTER(ctypes.c_float)),
                                                      ar_1d_single,
                                                      ar_1d_single,
                                                      ar_1d_single,
                                                      ctypes.c_longlong,
                                                      ar_1d_int,
                                                      ctypes.c_int]
  
  lib_parallelproj_cuda.joseph3d_back_cuda.restype  = None
  lib_parallelproj_cuda.joseph3d_back_cuda.argtypes = [ar_1d_single,
                                                       ar_1d_single,
                                                       POINTER(POINTER(ctypes.c_float)),
                                                       ar_1d_single,
                                                       ar_1d_single,
                                                       ar_1d_single,
                                                       ctypes.c_longlong,
                                                       ar_1d_int,
                                                       ctypes.c_int]

  lib_parallelproj_cuda.joseph3d_fwd_tof_sino_cuda.restype  = None
  lib_parallelproj_cuda.joseph3d_fwd_tof_sino_cuda.argtypes = [ar_1d_single,
                                                               ar_1d_single,
                                                               POINTER(POINTER(ctypes.c_float)),
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
                                                               ctypes.c_ubyte,    # LOR dep. TOF sigma
                                                               ctypes.c_ubyte,    # LOR dep. TOF center offset
                                                               ctypes.c_int]      # threads per block
  
  lib_parallelproj_cuda.joseph3d_back_tof_sino_cuda.restype  = None
  lib_parallelproj_cuda.joseph3d_back_tof_sino_cuda.argtypes = [ar_1d_single,
                                                                ar_1d_single,
                                                                POINTER(POINTER(ctypes.c_float)),
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
                                                                ctypes.c_ubyte,    # LOR dep.TOF sigma
                                                                ctypes.c_ubyte,    # LOR dep.TOF center offset
                                                                ctypes.c_int]      # threads per block

  lib_parallelproj_cuda.joseph3d_fwd_tof_lm_cuda.restype  = None
  lib_parallelproj_cuda.joseph3d_fwd_tof_lm_cuda.argtypes = [ar_1d_single,
                                                             ar_1d_single,
                                                             POINTER(POINTER(ctypes.c_float)),
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
                                                             ctypes.c_ubyte,    # LOR dep. TOF sigma
                                                             ctypes.c_ubyte,    # LOR dep. TOF center offset
                                                             ctypes.c_int]      # threads per block
  
  lib_parallelproj_cuda.joseph3d_back_tof_lm_cuda.restype  = None
  lib_parallelproj_cuda.joseph3d_back_tof_lm_cuda.argtypes = [ar_1d_single,
                                                              ar_1d_single,
                                                              POINTER(POINTER(ctypes.c_float)),
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
                                                              ctypes.c_ubyte,    # LOR dep. TOF sigma
                                                              ctypes.c_ubyte,    # LOR dep. TOF center offset
                                                              ctypes.c_int]      # threads per block


  lib_parallelproj_cuda.copy_float_array_to_all_devices.restype  = POINTER(POINTER(ctypes.c_float))
  lib_parallelproj_cuda.copy_float_array_to_all_devices.argtypes = [ar_1d_single,
                                                                    ctypes.c_longlong]

  lib_parallelproj_cuda.free_float_array_on_all_devices.restype  = None
  lib_parallelproj_cuda.free_float_array_on_all_devices.argtypes = [POINTER(POINTER(ctypes.c_float))] 

  lib_parallelproj_cuda.sum_float_arrays_on_first_device.restype  = None
  lib_parallelproj_cuda.sum_float_arrays_on_first_device.argtypes = [POINTER(POINTER(ctypes.c_float)), ctypes.c_longlong]

  lib_parallelproj_cuda.get_float_array_from_device.restype  = None
  lib_parallelproj_cuda.get_float_array_from_device.argtypes = [POINTER(POINTER(ctypes.c_float)), ctypes.c_longlong, ctypes.c_int, ar_1d_single]
