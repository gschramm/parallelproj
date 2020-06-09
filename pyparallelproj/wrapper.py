import os
import sys
import numpy.ctypeslib as npct
import ctypes
import platform

ar_1d_single = npct.ndpointer(dtype = ctypes.c_float, ndim = 1, flags = 'C')
ar_1d_int    = npct.ndpointer(dtype = ctypes.c_int,   ndim = 1, flags = 'C')
ar_1d_short  = npct.ndpointer(dtype = ctypes.c_short, ndim = 1, flags = 'C')

#---- find the compiled C / CUDA libraries
# we first look into the relative ../lib dir which is needed to support users that
# work on the C / CUDA libs
# if they don't exist in the relative ../lib dir, we check the install dir which
# is relative to sys.prefix

plt = platform.system()

if plt == 'Linux':
  fname      = 'libparallelproj.so'
  fname_cuda = 'libparallelproj_cuda.so'
elif plt == 'Windows':
  fname      = 'parallelproj.dll'
  fname_cuda = 'parallelproj_cuda.dll'
else:
  raise SystemError(f'{platform.system()} not supprted yet.')

lib_parallelproj_fname = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','lib',fname))
if not os.path.exists(lib_parallelproj_fname):
  lib_parallelproj_fname = os.path.join(sys.prefix,'lib',fname)

lib_parallelproj_cuda_fname = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','lib',fname_cuda))
if not os.path.exists(lib_parallelproj_cuda_fname):
  lib_parallelproj_cuda_fname = os.path.join(sys.prefix,'lib',fname_cuda)
#-----

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
                                                     ctypes.c_float,    # tofbin_width 
                                                     ar_1d_single,      # sigma tof
                                                     ar_1d_single,      # tofcenter_offset
                                                     ctypes.c_float,    # n_sigmas 
                                                     ctypes.c_short]    # n_tofbins
  
  lib_parallelproj.joseph3d_back_tof_sino.restype  = None
  lib_parallelproj.joseph3d_back_tof_sino.argtypes = [ar_1d_single,
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

  lib_parallelproj.joseph3d_fwd_tof_lm.restype  = None
  lib_parallelproj.joseph3d_fwd_tof_lm.argtypes = [ar_1d_single,
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
  
  lib_parallelproj.joseph3d_back_tof_lm.restype  = None
  lib_parallelproj.joseph3d_back_tof_lm.argtypes = [ar_1d_single,
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
             kwargs.setdefault('threadsperblock',64), kwargs.setdefault('ngpus',-1))

#------------------

def joseph3d_fwd_tof(*args, lm = True, **kwargs):
  if lm:
    # TOF LM case
    if kwargs.setdefault('ngpus', 0) == 0:
      return lib_parallelproj.joseph3d_fwd_tof_lm(*args)
    else:
      return lib_parallelproj_cuda.joseph3d_fwd_tof_lm_cuda(*args,
               kwargs.setdefault('threadsperblock',64), kwargs.setdefault('ngpus',-1))
  else:
    # TOF sinogram case
    if kwargs.setdefault('ngpus', 0) == 0:
      return lib_parallelproj.joseph3d_fwd_tof_sino(*args)
    else:
      return lib_parallelproj_cuda.joseph3d_fwd_tof_sino_cuda(*args, 
               kwargs.setdefault('threadsperblock',64), kwargs.setdefault('ngpus',-1))

#------------------

def joseph3d_back(*args,**kwargs):
  if kwargs.setdefault('ngpus', 0) == 0:
    return lib_parallelproj.joseph3d_back(*args)
  else:
    return lib_parallelproj_cuda.joseph3d_back_cuda(*args, 
             kwargs.setdefault('threadsperblock',64), kwargs.setdefault('ngpus',-1))

#------------------

def joseph3d_back_tof(*args, lm = True, **kwargs): 
  if lm:
    # TOF LM case
    if kwargs.setdefault('ngpus', 0) == 0:
      return lib_parallelproj.joseph3d_back_tof_lm(*args)
    else:
      return lib_parallelproj_cuda.joseph3d_back_tof_lm_cuda(*args,
               kwargs.setdefault('threadsperblock',64), kwargs.setdefault('ngpus',-1))
  else:
    # TOF sinogram case
    if kwargs.setdefault('ngpus', 0) == 0:
      return lib_parallelproj.joseph3d_back_tof_sino(*args)
    else:
      return lib_parallelproj_cuda.joseph3d_back_tof_sino_cuda(*args,
               kwargs.setdefault('threadsperblock',64), kwargs.setdefault('ngpus',-1))
