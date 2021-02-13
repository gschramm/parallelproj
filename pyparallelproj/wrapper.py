from pyparallelproj.config import lib_parallelproj_c, lib_parallelproj_cuda

# wrapper python function to allow same call for gpu and non-gpu projector functions
# the (*args, *kwargs) trick is needed to deal with the fact that the GPU function always take
# two extra arguments (threadsperblock, ngpus). We pass them as kwargs to the wrapper and
# just ignore them if we call the CPU function

def joseph3d_fwd(*args,**kwargs):
  if kwargs.setdefault('ngpus', 0) == 0:
    return lib_parallelproj_c.joseph3d_fwd(*args)
  else:
    return lib_parallelproj_cuda.joseph3d_fwd_cuda(*args, 
             kwargs.setdefault('threadsperblock',64), kwargs.setdefault('ngpus',-1))

#------------------

def joseph3d_fwd_tof(*args, lm = True, **kwargs):
  if lm:
    # TOF LM case
    if kwargs.setdefault('ngpus', 0) == 0:
      return lib_parallelproj_c.joseph3d_fwd_tof_lm(*args)
    else:
      return lib_parallelproj_cuda.joseph3d_fwd_tof_lm_cuda(*args,
               kwargs.setdefault('threadsperblock',64), kwargs.setdefault('ngpus',-1))
  else:
    # TOF sinogram case
    if kwargs.setdefault('ngpus', 0) == 0:
      return lib_parallelproj_c.joseph3d_fwd_tof_sino(*args)
    else:
      return lib_parallelproj_cuda.joseph3d_fwd_tof_sino_cuda(*args, 
               kwargs.setdefault('threadsperblock',64), kwargs.setdefault('ngpus',-1))

#------------------

def joseph3d_back(*args,**kwargs):
  if kwargs.setdefault('ngpus', 0) == 0:
    return lib_parallelproj_c.joseph3d_back(*args)
  else:
    return lib_parallelproj_cuda.joseph3d_back_cuda(*args, 
             kwargs.setdefault('threadsperblock',64), kwargs.setdefault('ngpus',-1))

#------------------

def joseph3d_back_tof(*args, lm = True, **kwargs): 
  if lm:
    # TOF LM case
    if kwargs.setdefault('ngpus', 0) == 0:
      return lib_parallelproj_c.joseph3d_back_tof_lm(*args)
    else:
      return lib_parallelproj_cuda.joseph3d_back_tof_lm_cuda(*args,
               kwargs.setdefault('threadsperblock',64), kwargs.setdefault('ngpus',-1))
  else:
    # TOF sinogram case
    if kwargs.setdefault('ngpus', 0) == 0:
      return lib_parallelproj_c.joseph3d_back_tof_sino(*args)
    else:
      return lib_parallelproj_cuda.joseph3d_back_tof_sino_cuda(*args,
               kwargs.setdefault('threadsperblock',64), kwargs.setdefault('ngpus',-1))
