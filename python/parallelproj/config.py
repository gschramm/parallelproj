"""package configurations"""
import os
import importlib
import GPUtil

import ctypes
from ctypes import POINTER
from ctypes.util import find_library

from pathlib import Path
from warnings import warn

import numpy.ctypeslib as npct

# number of available CUDA devices
num_available_cuda_devices = len(GPUtil.getGPUs())
cuda_enabled = (num_available_cuda_devices > 0)
# check if cupy is available
cupy_enabled = (importlib.util.find_spec('cupy') is not None)
if cuda_enabled:
    import cupy as cp

# numpy ctypes lib array definitions
ar_1d_single = npct.ndpointer(dtype=ctypes.c_float, ndim=1, flags='C')
ar_1d_int = npct.ndpointer(dtype=ctypes.c_int, ndim=1, flags='C')
ar_1d_short = npct.ndpointer(dtype=ctypes.c_short, ndim=1, flags='C')

#---------------------------------------------------------------------------------------
#---- find the compiled C / CUDA libraries

lib_parallelproj_c_fname = None
if 'PARALLELPROJ_C_LIB' in os.environ:
    lib_parallelproj_c_fname = os.environ['PARALLELPROJ_C_LIB']
else:
    lib_parallelproj_c_fname = find_library('parallelproj_c')

if lib_parallelproj_c_fname is None:
    raise ImportError(
        'Cannot find parallelproj c lib. Consider setting the environment variable PARALLELPROJ_C_LIB.'
    )
else:
    lib_parallelproj_c = npct.load_library(
        os.path.basename(lib_parallelproj_c_fname),
        os.path.dirname(lib_parallelproj_c_fname))
    lib_parallelproj_c.__file__ = lib_parallelproj_c_fname

    lib_parallelproj_c.joseph3d_fwd.restype = None
    lib_parallelproj_c.joseph3d_fwd.argtypes = [
        ar_1d_single,  # xstart
        ar_1d_single,  # xend
        ar_1d_single,  # img
        ar_1d_single,  # img_origin
        ar_1d_single,  # voxsize
        ar_1d_single,  # p
        ctypes.c_ulonglong,  # nlors
        ar_1d_int  # img_dim
    ]
    lib_parallelproj_c.joseph3d_fwd.__doc__ = """
    non-tof joseph forward projector
    
    Parameters
    ----------

    xstart : float array 
           array of shape [3*nlors] with the coordinates of the start points of the LORs.
           The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
           Units are the ones of voxsize.
    xend : float array   
           array of shape [3*nlors] with the coordinates of the end   points of the LORs.
           The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
           Units are the ones of voxsize.
    img : float array
          array of shape [n0*n1*n2] containing the 3D image to be projected.
          The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
    img_origin : float array  
                 [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
    voxsize : float array
              array [vs0, vs1, vs2] of the voxel sizes
    p : float array
        array of length nlors (output) used to store the projections
    nlors : int
            number of geomtrical LORs
    img_dim : int array
              array with dimensions of image [n0,n1,n2]
    """

    lib_parallelproj_c.joseph3d_back.restype = None
    lib_parallelproj_c.joseph3d_back.argtypes = [
        ar_1d_single,  # xstart
        ar_1d_single,  # xend
        ar_1d_single,  # img
        ar_1d_single,  # img_origin
        ar_1d_single,  # voxsize
        ar_1d_single,  # p            
        ctypes.c_ulonglong,  # nlors
        ar_1d_int  # img_dim
    ]
    lib_parallelproj_c.joseph3d_back.__doc__ = """
    3D non-tof joseph back projector

    All threads back project in one image using openmp's atomic add.
    
    Parameters
    ----------

    xstart : float array
           array of shape [3*nlors] with the coordinates of the start points of the LORs.
           The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2.
           Units are the ones of voxsize.
    xend : float array
           array of shape [3*nlors] with the coordinates of the end   points of the LORs.
           The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2.
           Units are the ones of voxsize.
    img : float array    
          array of shape [n0*n1*n2] containing the 3D image used for back projection (output).
          The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
          !! values are added to existing array !!
    img_origin : float array
                 [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
    voxsize : int array
              array [vs0, vs1, vs2] of the voxel sizes
    p : float array
        array of length nlors with the values to be back projected
    nlors : int
            number of geometrical LORs
    img_dim : int array
              array with dimensions of image [n0,n1,n2]
    """

    lib_parallelproj_c.joseph3d_fwd_tof_sino.restype = None
    lib_parallelproj_c.joseph3d_fwd_tof_sino.argtypes = [
        ar_1d_single,  #  xstart
        ar_1d_single,  #  xend
        ar_1d_single,  #  img
        ar_1d_single,  #  img_origin
        ar_1d_single,  #  voxsize
        ar_1d_single,  #  p         
        ctypes.c_longlong,  # nlors
        ar_1d_int,  # img_dim
        ctypes.c_float,  # tofbin_width
        ar_1d_single,  # sigma tof
        ar_1d_single,  # tofcenter_offset
        ctypes.c_float,  # n_sigmas
        ctypes.c_short,  # n_tofbins
        ctypes.c_ubyte,  # LOR dep. TOF sigma
        ctypes.c_ubyte  # LOR dep. TOF center offset
    ]
    lib_parallelproj_c.joseph3d_fwd_tof_sino.__doc__ = """
    3D sinogram tof joseph forward projector

    Parameters
    ----------

    xstart : float array
           array of shape [3*nlors] with the coordinates of the start points of the LORs.
           The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
           Units are the ones of voxsize.
    xend : float array
           array of shape [3*nlors] with the coordinates of the end   points of the LORs.
           The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
           Units are the ones of voxsize.
    img : float array
          array of shape [n0*n1*n2] containing the 3D image to be projected.
          The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
    img_origin : float array
                 [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
    voxsize : float array
              [vs0, vs1, vs2] of the voxel sizes
    p : float array
        array of length nlors*n_tofbins (output) used to store the projections
        the order of the array is
        [LOR0-TOFBIN-0, LOR0-TOFBIN-1, ... LOR0_TOFBIN-(n-1), 
         LOR1-TOFBIN-0, LOR1-TOFBIN-1, ... LOR1_TOFBIN-(n-1), 
         ...
         LOR(N-1)-TOFBIN-0, LOR(N-1)-TOFBIN-1, ... LOR(N-1)_TOFBIN-(n-1)] 
    nlors : int
            number of geomtrical LORs
    img_dim : int array
              array with dimensions of image [n0,n1,n2]
    tofbin_width : float
                   width of the TOF bins in spatial units (units of xstart and xend)
    sigma_tof : float array
                array of length 1 or nlors (depending on lor_dependent_sigma_tof)
                with the TOF resolution (sigma) for each LOR in
                spatial units (units of xstart and xend) 
    tofcenter_offset : float array
                       array of length 1 or nlors (depending on lor_dependent_tofcenter_offset)
                       with the offset of the central TOF bin from the 
                       midpoint of each LOR in spatial units (units of xstart and xend). 
                       A positive value means a shift towards the end point of the LOR.
    n_sigmas : float
               number of sigmas to consider for calculation of TOF kernel
    n_tofbins : int
                number of TOF bins
    lor_dependent_sigma_tof : unsigned char 0 or 1
                              1 means that the TOF sigmas are LOR dependent
                              any other value means that the first value in the sigma_tof
                              array is used for all LORs
    lor_dependent_tofcenter_offset : unsigned char 0 or 1
                                     1 means that the TOF center offsets are LOR dependent
                                     any other value means that the first value in the 
                                     tofcenter_offset array is used for all LORs
    """

    lib_parallelproj_c.joseph3d_back_tof_sino.restype = None
    lib_parallelproj_c.joseph3d_back_tof_sino.argtypes = [
        ar_1d_single,  # xstart
        ar_1d_single,  # xend
        ar_1d_single,  # img
        ar_1d_single,  # img_origin
        ar_1d_single,  # voxsize
        ar_1d_single,  # p         
        ctypes.c_longlong,  # nlors
        ar_1d_int,  # img_dim
        ctypes.c_float,  # tofbin_width
        ar_1d_single,  # sigma tof
        ar_1d_single,  # tofcenter_offset
        ctypes.c_float,  # n_sigmas
        ctypes.c_short,  # n_tofbins
        ctypes.c_ubyte,  # LOR dep. TOF sigma
        ctypes.c_ubyte  # LOR dep. TOF center offset
    ]
    lib_parallelproj_c.joseph3d_back_tof_sino.__doc__ = """
    3D sinogram tof joseph back projector

    reads back project in one image using openmp's atomic add.

    Parameters
    ----------
    
    xstart : float array
           array of shape [3*nlors] with the coordinates of the start points of the LORs.
           The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
           Units are the ones of voxsize.
    xend : float array
           array of shape [3*nlors] with the coordinates of the end   points of the LORs.
           The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
           Units are the ones of voxsize.
    img : float array
          array of shape [n0*n1*n2] containing the 3D image used for back projection (output).
          The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
          !! values are added to existing array !!
    img_origin : float array
                 array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
    voxsize : float array
              array [vs0, vs1, vs2] of the voxel sizes
    p : float array
        array of length nlors*n_tofbins with the values to be back projected
        he order of the array is 
        LOR0-TOFBIN-0, LOR0-TOFBIN-1, ... LOR0_TOFBIN-(n-1), 
        LOR1-TOFBIN-0, LOR1-TOFBIN-1, ... LOR1_TOFBIN-(n-1), 
        ...
        LOR(N-1)-TOFBIN-0, LOR(N-1)-TOFBIN-1, ... LOR(N-1)_TOFBIN-(n-1)] 
    nlors : int
            number of geometrical LORs
    img_dim : int array
              array with dimensions of image [n0,n1,n2]
    tofbin_width : float
                   width of the TOF bins in spatial units (units of xstart and xend)
    sigma_tof : float array
                array of length 1 or nlors (depending on lor_dependent_sigma_tof)
                with the TOF resolution (sigma) for each LOR in
                spatial units (units of xstart and xend) 
    tofcenter_offset : float array
                       array of length 1 or nlors (depending on lor_dependent_tofcenter_offset)
                       with the offset of the central TOF bin from the 
                       midpoint of each LOR in spatial units (units of xstart and xend). 
                       A positive value means a shift towards the end point of the LOR.
    n_sigmas : float
               number of sigmas to consider for calculation of TOF kernel
    n_tofbins: int
               number of TOF bins
    lor_dependent_sigma_tof : unsigned char 0 or 1
                              1 means that the TOF sigmas are LOR dependent
                              any other value means that the first value in the sigma_tof
                              array is used for all LORs
    lor_dependent_tofcenter_offset : unsigned char 0 or 1
                                     1 means that the TOF center offsets are LOR dependent
                                     any other value means that the first value in the 
                                     tofcenter_offset array is used for all LORs
    """

    lib_parallelproj_c.joseph3d_fwd_tof_lm.restype = None
    lib_parallelproj_c.joseph3d_fwd_tof_lm.argtypes = [
        ar_1d_single,  # xstart
        ar_1d_single,  # xend
        ar_1d_single,  # img
        ar_1d_single,  # img_origin
        ar_1d_single,  # voxsize
        ar_1d_single,  # p         
        ctypes.c_longlong,  # nlors
        ar_1d_int,  # img_dim
        ctypes.c_float,  # tofbin_width
        ar_1d_single,  # sigma tof
        ar_1d_single,  # tofcenter_offset
        ctypes.c_float,  # n_sigmas
        ar_1d_short,  # tof bin
        ctypes.c_ubyte,  # LOR dep. TOF sigma
        ctypes.c_ubyte  # LOR dep. TOF center offset
    ]
    lib_parallelproj_c.joseph3d_fwd_tof_lm.__doc__ = """
    3D listmode tof joseph forward projector

    Parameters
    ----------
    
    xstart : float array
             array of shape [3*nlors] with the coordinates of the start points of the LORs.
             The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
             Units are the ones of voxsize.
    xend : float array
           array of shape [3*nlors] with the coordinates of the end   points of the LORs.
           The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
           Units are the ones of voxsize.
    img : float array
          array of shape [n0*n1*n2] containing the 3D image to be projected.
          The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
    img_origin : float array
                 array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
    voxsize : float array
              array [vs0, vs1, vs2] of the voxel sizes
    p : float array
        array of length nlors (output) used to store the projections
    nlors : int
            number of geomtrical LORs
    img_dim : int array
              array with dimensions of image [n0,n1,n2]
    tofbin_width : float
                   width of the TOF bins in spatial units (units of xstart and xend)
    sigma_tof : float array
                array of length 1 or nlors (depending on lor_dependent_sigma_tof)
                with the TOF resolution (sigma) for each LOR in
                spatial units (units of xstart and xend) 
    tofcenter_offset : float array
                       array of length 1 or nlors (depending on lor_dependent_tofcenter_offset)
                       with the offset of the central TOF bin from the 
                       midpoint of each LOR in spatial units (units of xstart and xend). 
                       A positive value means a shift towards the end point of the LOR.
    n_sigmas : float         
               number of sigmas to consider for calculation of TOF kernel
    tof_bin : int array
              array containing the TOF bin of each event
    lor_dependent_sigma_tof: unsigned char 0 or 1
                             1 means that the TOF sigmas are LOR dependent
                             any other value means that the first value in the sigma_tof
                             array is used for all LORs
    lor_dependent_tofcenter_offset : unsigned char 0 or 1
                                     1 means that the TOF center offsets are LOR dependent
                                     any other value means that the first value in the 
                                     tofcenter_offset array is used for all LORs
    """

    lib_parallelproj_c.joseph3d_back_tof_lm.restype = None
    lib_parallelproj_c.joseph3d_back_tof_lm.argtypes = [
        ar_1d_single,  # xstart
        ar_1d_single,  # xend
        ar_1d_single,  # img
        ar_1d_single,  # img_origin
        ar_1d_single,  # voxsize
        ar_1d_single,  # p         
        ctypes.c_longlong,  # nlors
        ar_1d_int,  # img_dim
        ctypes.c_float,  # tofbin_width
        ar_1d_single,  # sigma tof
        ar_1d_single,  # tofcenter_offset
        ctypes.c_float,  # n_sigmas
        ar_1d_short,  # tof bin
        ctypes.c_ubyte,  # LOR dep. TOF sigma
        ctypes.c_ubyte  # LOR dep. TOF center offset
    ]
    lib_parallelproj_c.joseph3d_back_tof_lm.__doc__ = """
    listmode tof joseph back projector
 
    All threads back project in one image using openmp's atomic add.

    Parameters
    ----------
 
    xstart : float array
             array of shape [3*nlors] with the coordinates of the start points of the LORs.
             The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
             Units are the ones of voxsize.
    xend : float array
           array of shape [3*nlors] with the coordinates of the end   points of the LORs.
           The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
           Units are the ones of voxsize.
    img : float array
          array of shape [n0*n1*n2] containing the 3D image used for back projection (output).
          The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
          !! values are added to existing array !!
    img_origin : float array
                 array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
    voxsize : float array
              array [vs0, vs1, vs2] of the voxel sizes
    p : float array
        array of length nlors with the values to be back projected
    nlors : int
            number of geometrical LORs
    img_dim : int array
              array with dimensions of image [n0,n1,n2]
    tofbin_width : float
                   width of the TOF bins in spatial units (units of xstart and xend)
    sigma_tof : float array
                array of length 1 or nlors (depending on lor_dependent_sigma_tof)
                with the TOF resolution (sigma) for each LOR in
                spatial units (units of xstart and xend) 
    tofcenter_offset : float array
                       array of length 1 or nlors (depending on lor_dependent_tofcenter_offset)
                       with the offset of the central TOF bin from the 
                       midpoint of each LOR in spatial units (units of xstart and xend). 
                       A positive value means a shift towards the end point of the LOR.
    n_sigmas : float
               number of sigmas to consider for calculation of TOF kernel
    tof_bin : int array
              array containing the TOF bin of each event
    lor_dependent_sigma_tof : unsigned char 0 or 1
                              1 means that the TOF sigmas are LOR dependent
                              any other value means that the first value in the sigma_tof
                              array is used for all LORs
    lor_dependent_tofcenter_offset : unsigned char 0 or 1
                                     1 means that the TOF center offsets are LOR dependent
                                     any other value means that the first value in the 
                                     tofcenter_offset array is used for all LORs
    """

#---------------------------------------------------------------------------------------

lib_parallelproj_cuda_fname = None
lib_parallelproj_cuda = None

joseph3d_fwd_cuda_kernel = None
joseph3d_back_cuda_kernel = None
joseph3d_fwd_tof_sino_cuda_kernel = None
joseph3d_back_tof_sino_cuda_kernel = None
joseph3d_fwd_tof_lm_cuda_kernel = None
joseph3d_back_tof_lm_cuda_kernel = None

cuda_kernel_file = None

if cuda_enabled:
    if 'PARALLELPROJ_CUDA_LIB' in os.environ:
        lib_parallelproj_cuda_fname = os.environ['PARALLELPROJ_CUDA_LIB']
    else:
        lib_parallelproj_cuda_fname = find_library('parallelproj_cuda')

    if lib_parallelproj_cuda_fname is None:
        raise ImportError(
            'Cannot find parallelproj cuda lib. Consider settting the environment variable PARALLELPROJ_CUDA_LIB.'
        )
    else:
        lib_parallelproj_cuda = npct.load_library(
            os.path.basename(lib_parallelproj_cuda_fname),
            os.path.dirname(lib_parallelproj_cuda_fname))
        lib_parallelproj_cuda.__file__ = lib_parallelproj_cuda_fname

        lib_parallelproj_cuda.joseph3d_fwd_cuda.restype = None
        lib_parallelproj_cuda.joseph3d_fwd_cuda.argtypes = [
            ar_1d_single,  # h_xstart
            ar_1d_single,  # h_xend
            POINTER(POINTER(ctypes.c_float)),  # d_img
            ar_1d_single,  # h_img_origin
            ar_1d_single,  # h_voxsize
            ar_1d_single,  # h_p
            ctypes.c_longlong,  # nlors
            ar_1d_int,  # h_img_dim
            ctypes.c_int  # threadsperblock
        ]

        lib_parallelproj_cuda.joseph3d_back_cuda.restype = None
        lib_parallelproj_cuda.joseph3d_back_cuda.argtypes = [
            ar_1d_single,  # h_xstart
            ar_1d_single,  # h_xend
            POINTER(POINTER(ctypes.c_float)),  # d_img
            ar_1d_single,  # h_img_origin
            ar_1d_single,  # h_voxsize
            ar_1d_single,  # h_p
            ctypes.c_longlong,  # nlors
            ar_1d_int,  # h_img_dim
            ctypes.c_int  # threadsperblock
        ]

        lib_parallelproj_cuda.joseph3d_fwd_tof_sino_cuda.restype = None
        lib_parallelproj_cuda.joseph3d_fwd_tof_sino_cuda.argtypes = [
            ar_1d_single,  # h_xstart
            ar_1d_single,  # h_end
            POINTER(POINTER(ctypes.c_float)),  # d_img
            ar_1d_single,  # h_img_origin
            ar_1d_single,  # h_voxsize
            ar_1d_single,  # h_p
            ctypes.c_longlong,  # nlors
            ar_1d_int,  # h_img_dim
            ctypes.c_float,  # tofbin_width
            ar_1d_single,  # sigma tof
            ar_1d_single,  # tofcenter_offset
            ctypes.c_float,  # n_sigmas
            ctypes.c_short,  # n_tofbins
            ctypes.c_ubyte,  # LOR dep. TOF sigma
            ctypes.c_ubyte,  # LOR dep. TOF center offset
            ctypes.c_int  # threadsperblock
        ]

        lib_parallelproj_cuda.joseph3d_back_tof_sino_cuda.restype = None
        lib_parallelproj_cuda.joseph3d_back_tof_sino_cuda.argtypes = [
            ar_1d_single,  # h_xstart
            ar_1d_single,  # h_end
            POINTER(POINTER(ctypes.c_float)),  # d_img
            ar_1d_single,  # h_img_origin
            ar_1d_single,  # h_voxsize
            ar_1d_single,  # h_p
            ctypes.c_longlong,  # nlors
            ar_1d_int,  # h_img_dim
            ctypes.c_float,  # tofbin_width
            ar_1d_single,  # sigma tof
            ar_1d_single,  # tofcenter_offset
            ctypes.c_float,  # n_sigmas
            ctypes.c_short,  # n_tofbins
            ctypes.c_ubyte,  # LOR dep.TOF sigma
            ctypes.c_ubyte,  # LOR dep.TOF center offset
            ctypes.c_int
        ]  # threads per block

        lib_parallelproj_cuda.joseph3d_fwd_tof_lm_cuda.restype = None
        lib_parallelproj_cuda.joseph3d_fwd_tof_lm_cuda.argtypes = [
            ar_1d_single,  # h_xstart
            ar_1d_single,  # h_xend
            POINTER(POINTER(ctypes.c_float)),  # d_img
            ar_1d_single,  # h_img_origin
            ar_1d_single,  # h_voxsize
            ar_1d_single,  # h_p
            ctypes.c_longlong,  # nlors
            ar_1d_int,  # h_img_dim
            ctypes.c_float,  # tofbin_width
            ar_1d_single,  # sigma tof
            ar_1d_single,  # tofcenter_offset
            ctypes.c_float,  # n_sigmas
            ar_1d_short,  # tof bin
            ctypes.c_ubyte,  # LOR dep. TOF sigma
            ctypes.c_ubyte,  # LOR dep. TOF center offset
            ctypes.c_int
        ]  # threads per block

        lib_parallelproj_cuda.joseph3d_back_tof_lm_cuda.restype = None
        lib_parallelproj_cuda.joseph3d_back_tof_lm_cuda.argtypes = [
            ar_1d_single,  # h_xstart
            ar_1d_single,  # h_xend
            POINTER(POINTER(ctypes.c_float)),  # d_img
            ar_1d_single,  # h_img_origin
            ar_1d_single,  # h_voxsize
            ar_1d_single,  # h_p
            ctypes.c_longlong,  # nlors
            ar_1d_int,  # h_img_dim
            ctypes.c_float,  # tofbin_width
            ar_1d_single,  # sigma tof
            ar_1d_single,  # tofcenter_offset
            ctypes.c_float,  # n_sigmas
            ar_1d_short,  # tof bin
            ctypes.c_ubyte,  # LOR dep. TOF sigma
            ctypes.c_ubyte,  # LOR dep. TOF center offset
            ctypes.c_int
        ]  # threads per block

        lib_parallelproj_cuda.copy_float_array_to_all_devices.restype = POINTER(
            POINTER(ctypes.c_float))
        lib_parallelproj_cuda.copy_float_array_to_all_devices.argtypes = [
            ar_1d_single,  # h_array
            ctypes.c_longlong  # n
        ]

        lib_parallelproj_cuda.free_float_array_on_all_devices.restype = None
        lib_parallelproj_cuda.free_float_array_on_all_devices.argtypes = [
            POINTER(POINTER(ctypes.c_float))  # d_array
        ]

        lib_parallelproj_cuda.sum_float_arrays_on_first_device.restype = None
        lib_parallelproj_cuda.sum_float_arrays_on_first_device.argtypes = [
            POINTER(POINTER(ctypes.c_float)),  # d_array
            ctypes.c_longlong  # n
        ]

        lib_parallelproj_cuda.get_float_array_from_device.restype = None
        lib_parallelproj_cuda.get_float_array_from_device.argtypes = [
            POINTER(POINTER(ctypes.c_float)),  # d_array
            ctypes.c_longlong,  # n
            ctypes.c_int,  # i_dev
            ar_1d_single  # h_array
        ]

    #---------------------------------------------------------------------------------------
    if cupy_enabled:
        # find all cuda kernel files installed with the parallelproj libs
        cuda_kernel_files = sorted(
            list((Path(lib_parallelproj_cuda_fname).parents[1] /
                  'lib').glob('projector_kernels.cu.*')))
        if len(cuda_kernel_files) == 1:
            cuda_kernel_file = cuda_kernel_files[0]
        elif len(cuda_kernel_files) > 1:
            cuda_kernel_file = cuda_kernel_files[-1]
            warn('More than one kernel file available.')
        else:
            raise ImportError('No kernel file found.')

        if cuda_kernel_file is not None:
            # load a kernel defined in a external file
            with open(cuda_kernel_file, 'r') as f:
                lines = f.read()

            joseph3d_fwd_cuda_kernel = cp.RawKernel(
                lines, 'joseph3d_fwd_cuda_kernel')
            joseph3d_back_cuda_kernel = cp.RawKernel(
                lines, 'joseph3d_back_cuda_kernel')
            joseph3d_fwd_tof_sino_cuda_kernel = cp.RawKernel(
                lines, 'joseph3d_fwd_tof_sino_cuda_kernel')
            joseph3d_back_tof_sino_cuda_kernel = cp.RawKernel(
                lines, 'joseph3d_back_tof_sino_cuda_kernel')
            joseph3d_fwd_tof_lm_cuda_kernel = cp.RawKernel(
                lines, 'joseph3d_fwd_tof_lm_cuda_kernel')
            joseph3d_back_tof_lm_cuda_kernel = cp.RawKernel(
                lines, 'joseph3d_back_tof_lm_cuda_kernel')
        else:
            warn('cannot find cuda kernel file for cupy kernels')
