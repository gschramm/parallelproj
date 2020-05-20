import os
import numpy.ctypeslib as npct
import ctypes

ar_1d_single = npct.ndpointer(dtype = ctypes.c_float, ndim = 1, flags = 'C')
ar_1d_int    = npct.ndpointer(dtype = ctypes.c_int,   ndim = 1, flags = 'C')

lib_parallelproj_fname = os.path.join('..','lib','libparallelproj.so')
lib_parallelproj_cuda_fname = os.path.join('..','lib','libparallelproj_cuda.so')

if os.path.exists(lib_parallelproj_fname):
  lib_parallelproj = npct.load_library(os.path.basename(lib_parallelproj_fname),
                                       os.path.dirname(lib_parallelproj_fname))
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

if os.path.exists(lib_parallelproj_cuda_fname):
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
                                                             ar_1d_int,         # tof bin 
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
                                                              ar_1d_int,         # tof bin 
                                                              ctypes.c_int,
                                                              ctypes.c_int]
  
#--------------------------------------------------------------------------------------------------

# wrapper python function to allow same call for gpu and non-gpu projector functions
# the (*args, *kwargs) trick is needed to deal with the fact that the GPU function always take
# two extra arguments (threadsperblock, ngpus). We pass them as kwargs to the wrapper and
# just ignore them if we call the CPU function

def joseph3d_fwd(*args,**kwargs):
  if kwargs.setdefault('ngpus', 0) == 0:
    return lib_parallelproj.joseph3d_fwd(*args)
  else:
    return lib_parallelproj_cuda.joseph3d_fwd_cuda(*args, 
             kwargs.setdetault('threadsperblock',64), kwargs.setdefault('ngpus',-1))

def joseph3d_fwd_tof_sino(*args,**kwargs):
  if kwargs.setdefault('ngpus', 0) == 0:
    return lib_parallelproj.joseph3d_fwd_tof_sino(*args)
  else:
    return lib_parallelproj_cuda.joseph3d_fwd_tof_sino_cuda(*args, 
             kwargs.setdetault('threadsperblock',64), kwargs.setdefault('ngpus',-1))

def joseph3d_back(*args,**kwargs):
  if kwargs.setdefault('ngpus', 0) == 0:
    return lib_parallelproj.joseph3d_back(*args)
  else:
    return lib_parallelproj_cuda.joseph3d_back_cuda(*args, 
             kwargs.setdetault('threadsperblock',64), kwargs.setdefault('ngpus',-1))

def joseph3d_back_tof_sino(*args,**kwargs): 
  if kwargs.setdefault('ngpus', 0) == 0:
    return lib_parallelproj.joseph3d_back_tof_sino(*args)
  else:
    return lib_parallelproj_cuda.joseph3d_back_tof_sino_cuda(*args, 
             kwargs.setdetault('threadsperblock',64), kwargs.setdefault('ngpus',-1))
