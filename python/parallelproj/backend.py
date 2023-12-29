from __future__ import annotations

import os
import importlib
import distutils
import math

import ctypes
from ctypes import POINTER
from ctypes.util import find_library

from pathlib import Path
from warnings import warn

import numpy as np
import numpy.ctypeslib as npct
from numpy.array_api._array_object import Array

import array_api_compat

# check if cuda is present
cuda_present = distutils.spawn.find_executable('nvidia-smi') is not None

# check if cupy is available
cupy_enabled = (importlib.util.find_spec('cupy') is not None)

# check if cupy is available
torch_enabled = (importlib.util.find_spec('torch') is not None)

# define type for cupy or numpy array
if cupy_enabled:
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

#---------------------------------------------------------------------------------------

num_visible_cuda_devices = 0

if cuda_present:
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

        # get the number of visible cuda devices
        lib_parallelproj_cuda.get_cuda_device_count.restype = np.int32
        num_visible_cuda_devices = lib_parallelproj_cuda.get_cuda_device_count(
        )

        if (num_visible_cuda_devices == 0) and cupy_enabled:
            cupy_enabled = False

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
            import cupy as cp

            # load a kernel defined in a external file
            with open(cuda_kernel_file, 'r', encoding='utf8') as f:
                lines = f.read()

            _joseph3d_fwd_cuda_kernel = cp.RawKernel(
                lines, 'joseph3d_fwd_cuda_kernel')
            _joseph3d_back_cuda_kernel = cp.RawKernel(
                lines, 'joseph3d_back_cuda_kernel')
            _joseph3d_fwd_tof_sino_cuda_kernel = cp.RawKernel(
                lines, 'joseph3d_fwd_tof_sino_cuda_kernel')
            _joseph3d_back_tof_sino_cuda_kernel = cp.RawKernel(
                lines, 'joseph3d_back_tof_sino_cuda_kernel')
            _joseph3d_fwd_tof_lm_cuda_kernel = cp.RawKernel(
                lines, 'joseph3d_fwd_tof_lm_cuda_kernel')
            _joseph3d_back_tof_lm_cuda_kernel = cp.RawKernel(
                lines, 'joseph3d_back_tof_lm_cuda_kernel')
        else:
            warn('cannot find cuda kernel file for cupy kernels')


def calc_chunks(nLORs: int | np.int64,
                num_chunks: int) -> list[int] | list[np.int64]:
    """ calculate indices to split an array of length nLORs into num_chunks chunks

        example: splitting an array of length 10 into 3 chunks returns [0,4,7,10]
    """
    rem = nLORs % num_chunks
    div = (nLORs // num_chunks)

    chunks = [0]

    for i in range(num_chunks):
        if i < rem:
            nLORs_chunck = div + 1
        else:
            nLORs_chunck = div

        chunks.append(chunks[i] + nLORs_chunck)

    return chunks


def is_cuda_array(x: Array) -> bool:
    """test whether an array is a cuda array

    Parameters
    ----------
    x : Array
        array to be tested

    Returns
    -------
    bool
    """    

    iscuda = False

    if 'cupy' in array_api_compat.get_namespace(x).__name__:
        iscuda = True
    elif 'torch' in array_api_compat.get_namespace(x).__name__:
        if array_api_compat.device(x).type == 'cuda':
            iscuda = True

    return iscuda


def joseph3d_fwd(xstart: Array,
                 xend: Array,
                 img: Array,
                 img_origin: Array,
                 voxsize: Array,
                 threadsperblock: int = 32,
                 num_chunks: int = 1) -> Array:
    """Non-TOF Joseph 3D forward projector

    Parameters
    ----------
    xstart : Array
        start world coordinates of the LORs, shape (nLORs, 3)
    xend : Array
        end world coordinates of the LORs, shape (nLORs, 3)
    img : Array
        containing the 3D image to be projected
    img_origin : Array
        containing the world coordinates of the image origin (voxel [0,0,0])
    voxsize : Array
        array containing the voxel size
    threadsperblock : int, optional
        by default 32
    num_chunks : int, optional
        break down the projection in hybrid mode into chunks to
        save memory on the GPU, by default 1
    """
    nLORs = np.int64(array_api_compat.size(xstart) // 3)
    xp = array_api_compat.get_namespace(img)

    if is_cuda_array(img):
        # projection of GPU array (cupy to torch GPU array) using the cupy raw kernel
        img_fwd = cp.zeros(xstart.shape[:-1], dtype=cp.float32)

        _joseph3d_fwd_cuda_kernel(
            (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
            (cp.asarray(xstart, dtype=cp.float32).ravel(),
             cp.asarray(xend, dtype=cp.float32).ravel(),
             cp.asarray(img, dtype=cp.float32).ravel(),
             cp.asarray(img_origin, dtype=cp.float32),
             cp.asarray(voxsize, dtype=cp.float32), img_fwd.ravel(), nLORs,
             cp.asarray(img.shape, dtype=cp.int32)))
        cp.cuda.Device().synchronize()
    else:
        img_fwd = np.zeros(xstart.shape[:-1], dtype=np.float32)

        # projection of CPU array (numpy to torch CPU array)
        if num_visible_cuda_devices > 0:
            # projection of numpy array using the cuda parallelproj lib
            num_voxel = ctypes.c_longlong(array_api_compat.size(img))

            # send image to all devices
            d_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(
                np.asarray(img, dtype=np.float32).ravel(), num_voxel)

            # split call to GPU lib into chunks (useful for systems with limited memory)
            ic = calc_chunks(nLORs, num_chunks)

            for i in range(num_chunks):
                lib_parallelproj_cuda.joseph3d_fwd_cuda(
                    np.asarray(xstart, dtype=np.float32).reshape(
                        -1, 3)[ic[i]:(ic[i + 1]), :].ravel(),
                    np.asarray(xend, dtype=np.float32).reshape(
                        -1, 3)[ic[i]:(ic[i + 1]), :].ravel(), d_img,
                    np.asarray(img_origin, dtype=np.float32),
                    np.asarray(voxsize, dtype=np.float32),
                    img_fwd.ravel()[ic[i]:ic[i + 1]], ic[i + 1] - ic[i],
                    np.asarray(img.shape, dtype=np.int32), threadsperblock)

            # free image device arrays
            lib_parallelproj_cuda.free_float_array_on_all_devices(d_img)
        else:
            # projection of numpy array using the openmp parallelproj lib
            lib_parallelproj_c.joseph3d_fwd(
                np.asarray(xstart, dtype=np.float32).ravel(),
                np.asarray(xend, dtype=np.float32).ravel(),
                np.asarray(img, dtype=np.float32).ravel(),
                np.asarray(img_origin, dtype=np.float32),
                np.asarray(voxsize, dtype=np.float32), img_fwd.ravel(), nLORs,
                np.asarray(img.shape, dtype=np.int32))

    return xp.asarray(img_fwd, device=array_api_compat.device(img))


def joseph3d_back(xstart: Array,
                  xend: Array,
                  img_shape: tuple[int, int, int],
                  img_origin: Array,
                  voxsize: Array,
                  img_fwd: Array,
                  threadsperblock: int = 32,
                  num_chunks: int = 1) -> Array:
    """Non-TOF Joseph 3D back projector

    Parameters
    ----------
    xstart : Array
        start world coordinates of the LORs, shape (nLORs, 3)
    xend : Array
        end world coordinates of the LORs, shape (nLORs, 3)
    img_shape : tuple[int, int, int]
        the shape of the back projected image
    img_origin : Array
        containing the world coordinates of the image origin (voxel [0,0,0])
    voxsize : Array
        array containing the voxel size
    img_fwd : Array
        array of length nLORs containing the values to be back projected
    threadsperblock : int, optional
        by default 32
    num_chunks : int, optional
        break down the back projection in hybrid mode into chunks to
        save memory on the GPU, by default 1
    """
    nLORs = np.int64(array_api_compat.size(xstart) // 3)
    xp = array_api_compat.get_namespace(img_fwd)

    if (is_cuda_array(img_fwd)):
        # back projection of cupy or torch GPU array using the cupy raw kernel
        back_img = cp.zeros(img_shape, dtype=cp.float32)

        _joseph3d_back_cuda_kernel(
            (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
            (cp.asarray(xstart, dtype=cp.float32).ravel(),
             cp.asarray(xend, dtype=cp.float32).ravel(), back_img.ravel(),
             cp.asarray(img_origin, dtype=cp.float32),
             cp.asarray(voxsize, dtype=cp.float32),
             cp.asarray(img_fwd, dtype=cp.float32).ravel(), nLORs,
             cp.asarray(back_img.shape, dtype=cp.int32)))
        cp.cuda.Device().synchronize()
    else:
        # back projection of numpy or torch CPU array
        back_img = np.zeros(img_shape, dtype=np.float32)

        if num_visible_cuda_devices > 0:
            # back projection of numpy array using the cuda parallelproj lib
            num_voxel = ctypes.c_longlong(array_api_compat.size(back_img))
            # send image to all devices
            d_back_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(
                back_img.ravel(), num_voxel)

            # split call to GPU lib into chunks (useful for systems with limited memory)
            ic = calc_chunks(nLORs, num_chunks)

            for i in range(num_chunks):
                lib_parallelproj_cuda.joseph3d_back_cuda(
                    np.asarray(xstart, dtype=np.float32).reshape(
                        -1, 3)[ic[i]:(ic[i + 1]), :].ravel(),
                    np.asarray(xend, dtype=np.float32).reshape(
                        -1, 3)[ic[i]:(ic[i + 1]), :].ravel(), d_back_img,
                    np.asarray(img_origin, dtype=np.float32),
                    np.asarray(voxsize, dtype=np.float32),
                    np.asarray(img_fwd,
                               dtype=np.float32).ravel()[ic[i]:ic[i + 1]],
                    ic[i + 1] - ic[i],
                    np.asarray(back_img.shape,
                               dtype=np.int32), threadsperblock)

            # sum all device arrays in the first device
            lib_parallelproj_cuda.sum_float_arrays_on_first_device(
                d_back_img, num_voxel)

            # copy summed image back from first device
            lib_parallelproj_cuda.get_float_array_from_device(
                d_back_img, num_voxel, 0, back_img.ravel())

            # free image device arrays
            lib_parallelproj_cuda.free_float_array_on_all_devices(d_back_img)
        else:
            # back projection of numpy array using the openmp parallelproj lib
            lib_parallelproj_c.joseph3d_back(
                np.asarray(xstart, dtype=np.float32).ravel(),
                np.asarray(xend, dtype=np.float32).ravel(), back_img.ravel(),
                np.asarray(img_origin, dtype=np.float32),
                np.asarray(voxsize, dtype=np.float32),
                np.asarray(img_fwd, dtype=np.float32).ravel(), nLORs,
                np.asarray(back_img.shape, dtype=np.int32))

    return xp.asarray(back_img, device=array_api_compat.device(img_fwd))


def joseph3d_fwd_tof_sino(xstart: Array,
                          xend: Array,
                          img: Array,
                          img_origin: Array,
                          voxsize: Array,
                          tofbin_width: float,
                          sigma_tof: Array,
                          tofcenter_offset: Array,
                          nsigmas: float,
                          ntofbins: int,
                          threadsperblock: int = 32,
                          num_chunks: int = 1) -> Array:
    """TOF Joseph 3D sinogram forward projector

    Parameters
    ----------
    xstart : Array
        start world coordinates of the LORs, shape (nLORs, 3)
    xend : Array
        end world coordinates of the LORs, shape (nLORs, 3)
    img : Array
        containing the 3D image to be projected
    img_origin : Array
        containing the world coordinates of the image origin (voxel [0,0,0])
    voxsize : Array
        array containing the voxel size
    tofbin_width : float
        width of the TOF bin in spatial units (same units as xstart)
    sigma_tof : Array
        sigma of Gaussian TOF kernel in spatial units (same units as xstart)
        can be an array of length 1 -> same sigma for all LORs
        or an array of length nLORs -> LOR dependent sigma
    tofcenter_offset: Array
        center offset of the central TOF bin in spatial units (same units as xstart)
        can be an array of length 1 -> same offset for all LORs
        or an array of length nLORs -> LOR dependent offset
    nsigmas: float
        number of sigmas to consider when Gaussian kernel is evaluated (truncated)
    ntofbins: int
        total number of TOF bins
    threadsperblock : int, optional
        by default 32
    num_chunks : int, optional
        break down the projection in hybrid mode into chunks to
        save memory on the GPU, by default 1

    Returns
    -------
    Array
    """

    nLORs = np.int64(array_api_compat.size(xstart) // 3)
    xp = array_api_compat.get_namespace(img)

    lor_dependent_sigma_tof = np.uint8(sigma_tof.shape[0] == nLORs)
    lor_dependent_tofcenter_offset = np.uint8(
        tofcenter_offset.shape[0] == nLORs)

    if (is_cuda_array(img)):
        # projection of cupy or torch GPU array using the cupy raw kernel
        img_fwd = cp.zeros(xstart.shape[:-1] + (ntofbins, ), dtype=cp.float32)

        _joseph3d_fwd_tof_sino_cuda_kernel(
            (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
            (cp.asarray(xstart, dtype=cp.float32).ravel(),
             cp.asarray(xend, dtype=cp.float32).ravel(),
             cp.asarray(img, dtype=cp.float32).ravel(),
             cp.asarray(img_origin, dtype=cp.float32),
             cp.asarray(voxsize, dtype=cp.float32), img_fwd.ravel(), nLORs,
             cp.asarray(img.shape, dtype=cp.int32), np.int16(ntofbins),
             np.float32(tofbin_width), cp.asarray(sigma_tof,
                                                  dtype=cp.float32).ravel(),
             cp.asarray(tofcenter_offset,
                        dtype=cp.float32).ravel(), np.float32(nsigmas),
             lor_dependent_sigma_tof, lor_dependent_tofcenter_offset))
        cp.cuda.Device().synchronize()
    else:
        # projection of numpy or torch CPU array
        img_fwd = np.zeros(xstart.shape[:-1] + (ntofbins, ), dtype=np.float32)

        if num_visible_cuda_devices > 0:
            # projection using libparallelproj_cuda
            num_voxel = ctypes.c_longlong(array_api_compat.size(img))

            # send image to all devices
            d_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(
                np.asarray(img, dtype=np.float32).ravel(), num_voxel)

            # split call to GPU lib into chunks (useful for systems with limited memory)
            ic = calc_chunks(nLORs, num_chunks)

            for i in range(num_chunks):
                if lor_dependent_sigma_tof:
                    isig0 = ic[i]
                    isig1 = ic[i + 1]
                else:
                    isig0 = 0
                    isig1 = 1

                if lor_dependent_tofcenter_offset:
                    ioff0 = ic[i]
                    ioff1 = ic[i + 1]
                else:
                    ioff0 = 0
                    ioff1 = 1

                lib_parallelproj_cuda.joseph3d_fwd_tof_sino_cuda(
                    np.asarray(xstart, dtype=np.float32).reshape(
                        -1, 3)[ic[i]:(ic[i + 1]), :].ravel(),
                    np.asarray(xend, dtype=np.float32).reshape(
                        -1, 3)[ic[i]:(ic[i + 1]), :].ravel(), d_img,
                    np.asarray(img_origin, dtype=np.float32),
                    np.asarray(voxsize, dtype=np.float32),
                    img_fwd.ravel()[ic[i]:(ic[i + 1])], ic[i + 1] - ic[i],
                    np.asarray(img.shape, dtype=np.int32),
                    np.float32(tofbin_width),
                    np.asarray(sigma_tof,
                               dtype=np.float32).ravel()[isig0:isig1],
                    np.asarray(tofcenter_offset,
                               dtype=np.float32).ravel()[ioff0:ioff1],
                    np.float32(nsigmas), np.int16(ntofbins),
                    lor_dependent_sigma_tof, lor_dependent_tofcenter_offset,
                    threadsperblock)

            # free image device arrays
            lib_parallelproj_cuda.free_float_array_on_all_devices(d_img)
        else:
            # projection the openmp libparallelproj_c
            lib_parallelproj_c.joseph3d_fwd_tof_sino(
                np.asarray(xstart, dtype=np.float32).ravel(),
                np.asarray(xend, dtype=np.float32).ravel(),
                np.asarray(img, dtype=np.float32).ravel(),
                np.asarray(img_origin, dtype=np.float32),
                np.asarray(voxsize, dtype=np.float32), img_fwd.ravel(),
                np.int64(nLORs), np.asarray(img.shape, dtype=np.int32),
                np.float32(tofbin_width),
                np.asarray(sigma_tof, dtype=np.float32).ravel(),
                np.asarray(tofcenter_offset, dtype=np.float32).ravel(),
                np.float32(nsigmas), np.int16(ntofbins),
                lor_dependent_sigma_tof, lor_dependent_tofcenter_offset)

    return xp.asarray(img_fwd, device=array_api_compat.device(img))


def joseph3d_back_tof_sino(xstart: Array,
                           xend: Array,
                           img_shape: tuple[int, int, int],
                           img_origin: Array,
                           voxsize: Array,
                           img_fwd: Array,
                           tofbin_width: float,
                           sigma_tof: Array,
                           tofcenter_offset: Array,
                           nsigmas: float,
                           ntofbins: int,
                           threadsperblock: int = 32,
                           num_chunks: int = 1) -> Array:
    """TOF Joseph 3D sinogram back projector

    Parameters
    ----------
    xstart : Array
        start world coordinates of the LORs, shape (nLORs, 3)
    xend : Array
        end world coordinates of the LORs, shape (nLORs, 3)
    img_shape : tuple[int, int, int]
        the shape of the back projected image
    img_origin : Array
        containing the world coordinates of the image origin (voxel [0,0,0])
    voxsize : Array
        array containing the voxel size
    img_fwd : Array
        array of size nLOR*ntofbins containing the values to be back projected
    tofbin_width : float
        width of the TOF bin in spatial units (same units as xstart)
    sigma_tof : Array
        sigma of Gaussian TOF kernel in spatial units (same units as xstart)
        can be an array of length 1 -> same sigma for all LORs
        or an array of length nLORs -> LOR dependent sigma
    tofcenter_offset: Array
        center offset of the central TOF bin in spatial units (same units as xstart)
        can be an array of length 1 -> same offset for all LORs
        or an array of length nLORs -> LOR dependent offset
    nsigmas: float
        number of sigmas to consider when Gaussian kernel is evaluated (truncated)
    ntofbins: int
        total number of TOF bins
    threadsperblock : int, optional
        by default 32
    num_chunks : int, optional
        break down the projection in hybrid mode into chunks to
        save memory on the GPU, by default 1

    Returns
    -------
    Array
    """

    nLORs = np.int64(array_api_compat.size(xstart) // 3)
    xp = array_api_compat.get_namespace(img_fwd)

    lor_dependent_sigma_tof = np.uint8(sigma_tof.shape[0] == nLORs)
    lor_dependent_tofcenter_offset = np.uint8(
        tofcenter_offset.shape[0] == nLORs)

    if (is_cuda_array(img_fwd)):
        # back projection of cupy or torch GPU array using the cupy raw kernel
        back_img = cp.zeros(img_shape, dtype=cp.float32)

        _joseph3d_back_tof_sino_cuda_kernel(
            (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
            (cp.asarray(xstart, dtype=cp.float32).ravel(),
             cp.asarray(xend, dtype=cp.float32).ravel(), back_img.ravel(),
             cp.asarray(img_origin, dtype=cp.float32),
             cp.asarray(voxsize, dtype=cp.float32),
             cp.asarray(img_fwd, dtype=cp.float32).ravel(), nLORs,
             cp.asarray(back_img.shape, dtype=cp.int32), np.int16(ntofbins),
             np.float32(tofbin_width), cp.asarray(sigma_tof,
                                                  dtype=cp.float32).ravel(),
             cp.asarray(tofcenter_offset,
                        dtype=cp.float32).ravel(), np.float32(nsigmas),
             lor_dependent_sigma_tof, lor_dependent_tofcenter_offset))
        cp.cuda.Device().synchronize()
    else:
        # back projection of numpy or torch CPU array
        back_img = np.zeros(img_shape, dtype=np.float32)

        if num_visible_cuda_devices > 0:
            # back projection of numpy array using the cuda parallelproj lib
            num_voxel = ctypes.c_longlong(array_api_compat.size(back_img))
            # send image to all devices
            d_back_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(
                back_img.ravel(), num_voxel)

            # split call to GPU lib into chunks (useful for systems with limited memory)
            ic = calc_chunks(nLORs, num_chunks)

            for i in range(num_chunks):
                if lor_dependent_sigma_tof:
                    isig0 = ic[i]
                    isig1 = ic[i + 1]
                else:
                    isig0 = 0
                    isig1 = 1

                if lor_dependent_tofcenter_offset:
                    ioff0 = ic[i]
                    ioff1 = ic[i + 1]
                else:
                    ioff0 = 0
                    ioff1 = 1

                lib_parallelproj_cuda.joseph3d_back_tof_sino_cuda(
                    np.asarray(xstart, dtype=np.float32).reshape(
                        -1, 3)[ic[i]:(ic[i + 1]), :].ravel(),
                    np.asarray(xend, dtype=np.float32).reshape(
                        -1, 3)[ic[i]:(ic[i + 1]), :].ravel(), d_back_img,
                    np.asarray(img_origin, dtype=np.float32),
                    np.asarray(voxsize, dtype=np.float32),
                    np.asarray(img_fwd,
                               dtype=np.float32).ravel()[ic[i]:ic[i + 1]],
                    ic[i + 1] - ic[i],
                    np.asarray(back_img.shape, dtype=np.int32),
                    np.float32(tofbin_width),
                    np.asarray(sigma_tof,
                               dtype=np.float32).ravel()[isig0:isig1],
                    np.asarray(tofcenter_offset,
                               dtype=np.float32).ravel()[ioff0:ioff1],
                    np.float32(nsigmas), np.int16(ntofbins),
                    lor_dependent_sigma_tof, lor_dependent_tofcenter_offset,
                    threadsperblock)

            # sum all device arrays in the first device
            lib_parallelproj_cuda.sum_float_arrays_on_first_device(
                d_back_img, num_voxel)

            # copy summed image back from first device
            lib_parallelproj_cuda.get_float_array_from_device(
                d_back_img, num_voxel, 0, back_img.ravel())

            # free image device arrays
            lib_parallelproj_cuda.free_float_array_on_all_devices(d_back_img)
        else:
            # back projection of numpy array using the openmp parallelproj lib
            lib_parallelproj_c.joseph3d_back_tof_sino(
                np.asarray(xstart, dtype=np.float32).ravel(),
                np.asarray(xend, dtype=np.float32).ravel(), back_img.ravel(),
                np.asarray(img_origin, dtype=np.float32),
                np.asarray(voxsize, dtype=np.float32),
                np.asarray(img_fwd, dtype=np.float32).ravel(), nLORs,
                np.asarray(back_img.shape, dtype=np.int32),
                np.float32(tofbin_width),
                np.asarray(sigma_tof, dtype=np.float32).ravel(),
                np.asarray(tofcenter_offset, dtype=np.float32).ravel(),
                np.float32(nsigmas), np.int16(ntofbins),
                lor_dependent_sigma_tof, lor_dependent_tofcenter_offset)

    return xp.asarray(back_img, device=array_api_compat.device(img_fwd))


def joseph3d_fwd_tof_lm(xstart: Array,
                        xend: Array,
                        img: Array,
                        img_origin: Array,
                        voxsize: Array,
                        tofbin_width: float,
                        sigma_tof: Array,
                        tofcenter_offset: Array,
                        nsigmas: float,
                        tofbin: Array,
                        threadsperblock: int = 32,
                        num_chunks: int = 1) -> Array:
    """TOF Joseph 3D listmode forward projector

    Parameters
    ----------
    xstart : Array
        start world coordinates of the event LORs, shape (num_events, 3)
    xend : Array
        end world coordinates of the event LORs, shape (num_events, 3)
    img : Array
        containing the 3D image to be projected
    img_origin : Array
        containing the world coordinates of the image origin (voxel [0,0,0])
    voxsize : Array
        array containing the voxel size
    tofbin_width : float
        width of the TOF bin in spatial units (same units as xstart)
    sigma_tof : Array
        sigma of Gaussian TOF kernel in spatial units (same units as xstart)
        can be an array of length 1 -> same sigma for all LORs
        or an array of length nLORs -> LOR dependent sigma
    tofcenter_offset: Array
        center offset of the central TOF bin in spatial units (same units as xstart)
        can be an array of length 1 -> same offset for all events
        or an array of length num_events -> event dependent offset
    nsigmas: float
        number of sigmas to consider when Gaussian kernel is evaluated (truncated)
    tofbin: Array
        array containing the tof bin of the events
    threadsperblock : int, optional
        by default 32
    num_chunks : int, optional
        break down the projection in hybrid mode into chunks to
        save memory on the GPU, by default 1

    Returns
    -------
    Array
    """

    nLORs = np.int64(xstart.shape[0])
    xp = array_api_compat.get_namespace(img)

    lor_dependent_sigma_tof = np.uint8(sigma_tof.shape[0] == nLORs)
    lor_dependent_tofcenter_offset = np.uint8(
        tofcenter_offset.shape[0] == nLORs)

    if (is_cuda_array(img)):
        # projection of cupy or torch GPU array using the cupy raw kernel
        img_fwd = cp.zeros(nLORs, dtype=cp.float32)

        _joseph3d_fwd_tof_lm_cuda_kernel(
            (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
            (cp.asarray(xstart, dtype=cp.float32).ravel(),
             cp.asarray(xend, dtype=cp.float32).ravel(),
             cp.asarray(img, dtype=cp.float32).ravel(),
             cp.asarray(img_origin, dtype=cp.float32),
             cp.asarray(voxsize, dtype=cp.float32), img_fwd, nLORs,
             cp.asarray(img.shape, dtype=cp.int32), np.float32(tofbin_width),
             cp.asarray(sigma_tof, dtype=cp.float32),
             cp.asarray(tofcenter_offset, dtype=cp.float32),
             np.float32(nsigmas), cp.asarray(tofbin, dtype=cp.int16),
             lor_dependent_sigma_tof, lor_dependent_tofcenter_offset))
        cp.cuda.Device().synchronize()
    else:
        # projection of numpy or torch CPU array
        img_fwd = np.zeros(nLORs, dtype=np.float32)

        if num_visible_cuda_devices > 0:
            # projection using libparallelproj_cuda
            num_voxel = ctypes.c_longlong(array_api_compat.size(img))

            # send image to all devices
            d_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(
                np.asarray(img, dtype=np.float32).ravel(), num_voxel)

            # split call to GPU lib into chunks (useful for systems with limited memory)
            ic = calc_chunks(nLORs, num_chunks)

            for i in range(num_chunks):
                if lor_dependent_sigma_tof:
                    isig0 = ic[i]
                    isig1 = ic[i + 1]
                else:
                    isig0 = 0
                    isig1 = 1

                if lor_dependent_tofcenter_offset:
                    ioff0 = ic[i]
                    ioff1 = ic[i + 1]
                else:
                    ioff0 = 0
                    ioff1 = 1

                lib_parallelproj_cuda.joseph3d_fwd_tof_lm_cuda(
                    np.asarray(xstart,
                               dtype=np.float32)[ic[i]:(ic[i + 1]), :].ravel(),
                    np.asarray(xend,
                               dtype=np.float32)[ic[i]:(ic[i + 1]), :].ravel(),
                    d_img, np.asarray(img_origin, dtype=np.float32),
                    np.asarray(voxsize, dtype=np.float32),
                    img_fwd[ic[i]:(ic[i + 1])], ic[i + 1] - ic[i],
                    np.asarray(img.shape, dtype=np.int32),
                    np.float32(tofbin_width),
                    np.asarray(sigma_tof, dtype=np.float32)[isig0:isig1],
                    np.asarray(tofcenter_offset,
                               dtype=np.float32)[ioff0:ioff1],
                    np.float32(nsigmas),
                    np.asarray(tofbin, dtype=np.int16)[ic[i]:(ic[i + 1])],
                    lor_dependent_sigma_tof, lor_dependent_tofcenter_offset,
                    threadsperblock)

            # free image device arrays
            lib_parallelproj_cuda.free_float_array_on_all_devices(d_img)
        else:
            # projection the openmp libparallelproj_c
            lib_parallelproj_c.joseph3d_fwd_tof_lm(
                np.asarray(xstart, dtype=np.float32).ravel(),
                np.asarray(xend, dtype=np.float32).ravel(),
                np.asarray(img, dtype=np.float32).ravel(),
                np.asarray(img_origin, dtype=np.float32),
                np.asarray(voxsize, dtype=np.float32), img_fwd,
                np.int64(nLORs), np.asarray(img.shape, dtype=np.int32),
                np.float32(tofbin_width),
                np.asarray(sigma_tof, dtype=np.float32),
                np.asarray(tofcenter_offset, dtype=np.float32),
                np.float32(nsigmas), np.asarray(tofbin, dtype=np.int16),
                lor_dependent_sigma_tof, lor_dependent_tofcenter_offset)

    return xp.asarray(img_fwd, device=array_api_compat.device(img))


def joseph3d_back_tof_lm(xstart: Array,
                         xend: Array,
                         img_shape: tuple[int, int, int],
                         img_origin: Array,
                         voxsize: Array,
                         img_fwd: Array,
                         tofbin_width: float,
                         sigma_tof: Array,
                         tofcenter_offset: Array,
                         nsigmas: float,
                         tofbin: Array,
                         threadsperblock: int = 32,
                         num_chunks: int = 1) -> Array:
    """TOF Joseph 3D listmode back projector

    Parameters
    ----------
    xstart : Array
        start world coordinates of the event LORs, shape (nLORs, 3)
    xend : Array
        end world coordinates of the event LORs, shape (nLORs, 3)
    img_shape : tuple[int, int, int]
        the shape of the back projected image
    img_origin : Array
        containing the world coordinates of the image origin (voxel [0,0,0])
    voxsize : Array
        array containing the voxel size
    img_fwd : Array
        array of size num_events containing the values to be back projected
    tofbin_width : float
        width of the TOF bin in spatial units (same units as xstart)
    sigma_tof : Array
        sigma of Gaussian TOF kernel in spatial units (same units as xstart)
        can be an array of length 1 -> same sigma for all LORs
        or an array of length num_events -> event dependent sigma
    tofcenter_offset: Array
        center offset of the central TOF bin in spatial units (same units as xstart)
        can be an array of length 1 -> same offset for all LORs
        or an array of length num_events -> event dependent offset
    nsigmas: float
        number of sigmas to consider when Gaussian kernel is evaluated (truncated)
    tofbin: Array
        array with the tofbin of the events
    threadsperblock : int, optional
        by default 32
    num_chunks : int, optional
        break down the projection in hybrid mode into chunks to
        save memory on the GPU, by default 1

    Returns
    -------
    Array
    """

    nLORs = np.int64(xstart.shape[0])
    xp = array_api_compat.get_namespace(img_fwd)

    lor_dependent_sigma_tof = np.uint8(sigma_tof.shape[0] == nLORs)
    lor_dependent_tofcenter_offset = np.uint8(
        tofcenter_offset.shape[0] == nLORs)

    if (is_cuda_array(img_fwd)):
        # back projection of cupy or torch GPU array using the cupy raw kernel
        back_img = cp.zeros(img_shape, dtype=cp.float32)

        _joseph3d_back_tof_lm_cuda_kernel(
            (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
            (cp.asarray(xstart, dtype=cp.float32).ravel(),
             cp.asarray(xend, dtype=cp.float32).ravel(), back_img.ravel(),
             cp.asarray(img_origin, dtype=cp.float32),
             cp.asarray(voxsize, dtype=cp.float32),
             cp.asarray(img_fwd, dtype=cp.float32), nLORs,
             cp.asarray(back_img.shape, dtype=cp.int32),
             np.float32(tofbin_width), cp.asarray(sigma_tof, dtype=cp.float32),
             cp.asarray(tofcenter_offset, dtype=cp.float32),
             np.float32(nsigmas), cp.asarray(tofbin, dtype=cp.int16),
             lor_dependent_sigma_tof, lor_dependent_tofcenter_offset))
        cp.cuda.Device().synchronize()
    else:
        # back projection of numpy or torch CPU array
        back_img = np.zeros(img_shape, dtype=np.float32)

        if num_visible_cuda_devices > 0:
            # back projection of numpy array using the cuda parallelproj lib
            num_voxel = ctypes.c_longlong(array_api_compat.size(back_img))
            # send image to all devices
            d_back_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(
                back_img.ravel(), num_voxel)

            # split call to GPU lib into chunks (useful for systems with limited memory)
            ic = calc_chunks(nLORs, num_chunks)

            for i in range(num_chunks):
                if lor_dependent_sigma_tof:
                    isig0 = ic[i]
                    isig1 = ic[i + 1]
                else:
                    isig0 = 0
                    isig1 = 1

                if lor_dependent_tofcenter_offset:
                    ioff0 = ic[i]
                    ioff1 = ic[i + 1]
                else:
                    ioff0 = 0
                    ioff1 = 1

                lib_parallelproj_cuda.joseph3d_back_tof_lm_cuda(
                    np.asarray(xstart,
                               dtype=np.float32)[ic[i]:(ic[i + 1]), :].ravel(),
                    np.asarray(xend,
                               dtype=np.float32)[ic[i]:(ic[i + 1]), :].ravel(),
                    d_back_img, np.asarray(img_origin, dtype=np.float32),
                    np.asarray(voxsize, dtype=np.float32),
                    np.asarray(img_fwd, dtype=np.float32)[ic[i]:ic[i + 1]],
                    ic[i + 1] - ic[i],
                    np.asarray(back_img.shape, dtype=np.int32),
                    np.float32(tofbin_width),
                    np.asarray(sigma_tof, dtype=np.float32)[isig0:isig1],
                    np.asarray(tofcenter_offset,
                               dtype=np.float32)[ioff0:ioff1],
                    np.float32(nsigmas),
                    np.asarray(tofbin, dtype=np.int16)[ic[i]:(ic[i + 1])],
                    lor_dependent_sigma_tof, lor_dependent_tofcenter_offset,
                    threadsperblock)

            # sum all device arrays in the first device
            lib_parallelproj_cuda.sum_float_arrays_on_first_device(
                d_back_img, num_voxel)

            # copy summed image back from first device
            lib_parallelproj_cuda.get_float_array_from_device(
                d_back_img, num_voxel, 0, back_img.ravel())

            # free image device arrays
            lib_parallelproj_cuda.free_float_array_on_all_devices(d_back_img)
        else:
            # back projection of numpy array using the openmp parallelproj lib
            lib_parallelproj_c.joseph3d_back_tof_lm(
                np.asarray(xstart, dtype=np.float32).ravel(),
                np.asarray(xend, dtype=np.float32).ravel(), back_img.ravel(),
                np.asarray(img_origin, dtype=np.float32),
                np.asarray(voxsize, dtype=np.float32),
                np.asarray(img_fwd, dtype=np.float32), nLORs,
                np.asarray(back_img.shape, dtype=np.int32),
                np.float32(tofbin_width),
                np.asarray(sigma_tof, dtype=np.float32),
                np.asarray(tofcenter_offset, dtype=np.float32),
                np.float32(nsigmas), np.asarray(tofbin, dtype=np.int16),
                lor_dependent_sigma_tof, lor_dependent_tofcenter_offset)

    return xp.asarray(back_img, device=array_api_compat.device(img_fwd))
