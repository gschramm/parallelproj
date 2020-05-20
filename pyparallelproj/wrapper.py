import os
import numpy.ctypeslib as npct
import ctypes

use_gpu = False

ar_1d_single = npct.ndpointer(dtype = ctypes.c_float, ndim = 1, flags = 'C')
ar_1d_int    = npct.ndpointer(dtype = ctypes.c_int,   ndim = 1, flags = 'C')

if not use_gpu:
  lib_parallelproj = npct.load_library('libparallelproj.so', os.path.join('..','lib'))
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

  
  joseph3d_fwd = lib_parallelproj.joseph3d_fwd
  joseph3d_fwd_tof_sino = lib_parallelproj.joseph3d_fwd_tof_sino

  joseph3d_back = lib_parallelproj.joseph3d_back
  joseph3d_back_tof_sino = lib_parallelproj.joseph3d_back_tof_sino
