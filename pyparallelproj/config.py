import os
import sys
import re
import numpy.ctypeslib as npct
import ctypes
import platform
import warnings

ar_1d_single = npct.ndpointer(dtype = ctypes.c_float, ndim = 1, flags = 'C')
ar_1d_int    = npct.ndpointer(dtype = ctypes.c_int,   ndim = 1, flags = 'C')
ar_1d_short  = npct.ndpointer(dtype = ctypes.c_short, ndim = 1, flags = 'C')

#---- find the compiled C / CUDA libraries

# this is the default lib install dir used by the python-cmake build script
lib_subdir = f'lib_{platform.system()}_{platform.architecture()[0]}' 

# get the version of the libs from the ConfigVersion file
configVersion_file = os.path.abspath(os.path.join(os.path.dirname(__file__), lib_subdir, 'cmake',
                                                  'parallelproj', 'parallelprojConfigVersion.cmake'))

version = None
if os.access(configVersion_file, os.R_OK):
  with open(configVersion_file,'r') as f:
    version = re.search("set\(PACKAGE_VERSION (.*)\)", f.read()).group(1).replace('"','')
else:
  warnings.warn("failed to read lib version from cmake config version file", UserWarning) 

libname_c    = 'parallelproj_c'
libname_cuda = 'parallelproj_cuda'

if platform.system() == 'Linux':
  libprefix = 'lib'
  libfext   = 'so' 
elif platform.system() == 'Windows':
  libprefix = ''
  libfext   = 'dll' 
else:
  raise SystemError(f'{platform.system()} not supported yet.')

lib_parallelproj_c_fname    = os.path.abspath(os.path.join(os.path.dirname(__file__),lib_subdir, 
                                              f'{libprefix}{libname_c}.{libfext}'))
lib_parallelproj_cuda_fname = os.path.abspath(os.path.join(os.path.dirname(__file__),lib_subdir,
                                              f'{libprefix}{libname_cuda}.{libfext}'))

#-------------------------------------------------------------------------------------------
# add the calling signature

lib_parallelproj_c    = None
lib_parallelproj_cuda = None

if os.access(lib_parallelproj_c_fname, os.R_OK):
  lib_parallelproj_c = npct.load_library(os.path.basename(lib_parallelproj_c_fname),
                                       os.path.dirname(lib_parallelproj_c_fname))
  lib_parallelproj_c.__version__ = version
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
  

if os.access(lib_parallelproj_cuda_fname, os.R_OK):
  lib_parallelproj_cuda = npct.load_library(os.path.basename(lib_parallelproj_cuda_fname),
                                            os.path.dirname(lib_parallelproj_cuda_fname))
  lib_parallelproj_cuda.__version__ = version
  lib_parallelproj_cuda.__file__    = lib_parallelproj_cuda_fname

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
