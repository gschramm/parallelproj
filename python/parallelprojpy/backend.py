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
import numpy.typing as npt

from typing import Union
from types import ModuleType

# check if cuda is present
cuda_present = distutils.spawn.find_executable('nvidia-smi') is not None

# check if cupy is available
cupy_enabled = (importlib.util.find_spec('cupy') is not None)

# define type for cupy or numpy array
if cupy_enabled:
    import cupy as cp
    import cupy.typing as cpt
    XPArray = Union[npt.NDArray, cpt.NDArray]
    XPFloat32Array = Union[npt.NDArray[np.float32], cpt.NDArray[np.float32]]
    XPShortArray = Union[npt.NDArray[np.int16], cpt.NDArray[np.int16]]
else:
    XPArray = npt.NDArray
    XPFloat32Array = npt.NDArray[np.float32]
    XPShortArray = npt.NDArray[np.int16]

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


def calc_chunks(nLORs: int, num_chunks: int) -> list[int]:
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


def joseph3d_fwd(xstart: XPFloat32Array,
                 xend: XPFloat32Array,
                 img: XPFloat32Array,
                 img_origin: XPFloat32Array,
                 voxsize: XPFloat32Array,
                 img_fwd: XPFloat32Array,
                 threadsperblock: int = 32,
                 num_chunks: int = 1) -> None:
    """Non-TOF Joseph 3D forward projector

    Parameters
    ----------
    xstart : XPFloat32Array (float32 numpy or cupy array)
        start world coordinates of the LORs, shape (nLORs, 3)
    xend : XPFloat32Array (float32 numpy or cupy array)
        end world coordinates of the LORs, shape (nLORs, 3)
    img : XPFloat32Array (float32 numpy or cupy array)
        containing the 3D image to be projected
    img_origin : XPFloat32Array (float32 numpy or cupy array)
        containing the world coordinates of the image origin (voxel [0,0,0])
    voxsize : XPFloat32Array (float32 numpy or cupy array)
        array containing the voxel size
    img_fwd : XPFloat32Array (float32 numpy or cupy array)
        output array of length nLORs for storing the forward projection 
    threadsperblock : int, optional
        by default 32
    num_chunks : int, optional
        break down the projection in hybrid mode into chunks to
        save memory on the GPU, by default 1
    """
    img_dim = np.array(img.shape, dtype=np.int32)
    nLORs = xstart.shape[0]

    # check whether the input image is a numpy or cupy array
    xp = get_array_module(img)

    if (xp.__name__ == 'cupy'):
        # projection of cupy GPU array using the cupy raw kernel
        _joseph3d_fwd_cuda_kernel(
            (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
            (xstart.ravel(), xend.ravel(), img.ravel(), xp.asarray(img_origin),
             xp.asarray(voxsize), img_fwd, np.int64(nLORs),
             xp.asarray(img_dim)))
        xp.cuda.Device().synchronize()
    else:
        if num_visible_cuda_devices > 0:
            # projection of numpy array using the cuda parallelproj lib
            num_voxel = ctypes.c_longlong(img_dim[0] * img_dim[1] * img_dim[2])

            # send image to all devices
            d_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(
                img.ravel(), num_voxel)

            # split call to GPU lib into chunks (useful for systems with limited memory)
            ic = calc_chunks(nLORs, num_chunks)

            for i in range(num_chunks):
                lib_parallelproj_cuda.joseph3d_fwd_cuda(
                    xstart[ic[i]:(ic[i + 1]), :].ravel(),
                    xend[ic[i]:(ic[i + 1]), :].ravel(), d_img, img_origin,
                    voxsize,
                    img_fwd.ravel()[ic[i]:ic[i + 1]], ic[i + 1] - ic[i],
                    img_dim, threadsperblock)

            # free image device arrays
            lib_parallelproj_cuda.free_float_array_on_all_devices(d_img)
        else:
            # projection of numpy array using the openmp parallelproj lib
            lib_parallelproj_c.joseph3d_fwd(xstart.ravel(), xend.ravel(),
                                            img.ravel(), img_origin, voxsize,
                                            img_fwd.ravel(), np.int64(nLORs),
                                            img_dim)


def joseph3d_back(xstart: XPFloat32Array,
                  xend: XPFloat32Array,
                  back_img: XPFloat32Array,
                  img_origin: XPFloat32Array,
                  voxsize: XPFloat32Array,
                  sino: XPFloat32Array,
                  threadsperblock: int = 32,
                  num_chunks: int = 1) -> None:
    """Non-TOF Joseph 3D forward projector

    Parameters
    ----------
    xstart : XPFloat32Array (float32 numpy or cupy array)
        start world coordinates of the LORs, shape (nLORs, 3)
    xend : XPFloat32Array (float32 numpy or cupy array)
        end world coordinates of the LORs, shape (nLORs, 3)
    back_img : XPFloat32Array (float32 numpy or cupy array)
        output array for the back projection
    img_origin : XPFloat32Array (float32 numpy or cupy array)
        containing the world coordinates of the image origin (voxel [0,0,0])
    voxsize : XPFloat32Array (float32 numpy or cupy array)
        array containing the voxel size
    sino : XPFloat32Array (float32 numpy or cupy array)
        array of length nLORs containing the values to be back projected
    threadsperblock : int, optional
        by default 32
    num_chunks : int, optional
        break down the back projection in hybrid mode into chunks to
        save memory on the GPU, by default 1
    """
    img_dim = np.array(back_img.shape, dtype=np.int32)
    nLORs = xstart.shape[0]

    # check whether the input image is a numpy or cupy array
    xp = get_array_module(sino)

    if (xp.__name__ == 'cupy'):
        # back projection of cupy GPU array using the cupy raw kernel
        _joseph3d_back_cuda_kernel(
            (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
            (xstart.ravel(), xend.ravel(), back_img.ravel(),
             xp.asarray(img_origin), xp.asarray(voxsize), sino.ravel(),
             np.int64(nLORs), xp.asarray(img_dim)))
        xp.cuda.Device().synchronize()
    else:
        if num_visible_cuda_devices > 0:
            # back projection of numpy array using the cuda parallelproj lib
            num_voxel = ctypes.c_longlong(img_dim[0] * img_dim[1] * img_dim[2])

            # send image to all devices
            d_back_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(
                back_img.ravel(), num_voxel)

            # split call to GPU lib into chunks (useful for systems with limited memory)
            ic = calc_chunks(nLORs, num_chunks)

            for i in range(num_chunks):
                lib_parallelproj_cuda.joseph3d_back_cuda(
                    xstart[ic[i]:(ic[i + 1]), :].ravel(),
                    xend[ic[i]:(ic[i + 1]), :].ravel(), d_back_img, img_origin,
                    voxsize,
                    sino.ravel()[ic[i]:ic[i + 1]], ic[i + 1] - ic[i], img_dim,
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
            lib_parallelproj_c.joseph3d_back(xstart.ravel(), xend.ravel(),
                                             back_img.ravel(), img_origin,
                                             voxsize, sino.ravel(),
                                             np.int64(nLORs), img_dim)


def joseph3d_fwd_tof_sino(xstart: XPFloat32Array,
                          xend: XPFloat32Array,
                          img: XPFloat32Array,
                          img_origin: XPFloat32Array,
                          voxsize: XPFloat32Array,
                          img_fwd: XPFloat32Array,
                          tofbin_width: float,
                          sigma_tof: XPFloat32Array,
                          tofcenter_offset: XPFloat32Array,
                          nsigmas: float,
                          ntofbins: int,
                          threadsperblock: int = 32,
                          num_chunks: int = 1) -> None:
    """TOF Joseph 3D sinogram forward projector

    Parameters
    ----------
    xstart : XPFloat32Array (float32 numpy or cupy array)
        start world coordinates of the LORs, shape (nLORs, 3)
    xend : XPFloat32Array (float32 numpy or cupy array)
        end world coordinates of the LORs, shape (nLORs, 3)
    img : XPFloat32Array (float32 numpy or cupy array)
        containing the 3D image to be projected
    img_origin : XPFloat32Array (float32 numpy or cupy array)
        containing the world coordinates of the image origin (voxel [0,0,0])
    voxsize : XPFloat32Array (float32 numpy or cupy array)
        array containing the voxel size
    img_fwd : XPFloat32Array (float32 numpy or cupy array)
        output array of size (nLORs*ntofbins) for storing the forward projection 
    tofbin_width : float
        width of the TOF bin in spatial units (same units as xstart)
    sigma_tof : XPFloat32Array (float32 numpy or cupy array)
        sigma of Gaussian TOF kernel in spatial units (same units as xstart)
        can be an array of length 1 -> same sigma for all LORs
        or an array of length nLORs -> LOR dependent sigma
    tofcenter_offset: XPFloat32Array (float32 numpy or cupy array)
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
    """
    img_dim = np.array(img.shape, dtype=np.int32)
    nLORs = xstart.shape[0]

    lor_dependent_sigma_tof = np.uint8(sigma_tof.shape[0] == nLORs)
    lor_dependent_tofcenter_offset = np.uint8(
        tofcenter_offset.shape[0] == nLORs)

    # check whether the input image is a numpy or cupy array
    xp = get_array_module(img)

    if (xp.__name__ == 'cupy'):
        # projection of cupy GPU array using the cupy raw kernel
        _joseph3d_fwd_tof_sino_cuda_kernel(
            (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
            (xstart.ravel(), xend.ravel(), img.ravel(), xp.asarray(img_origin),
             xp.asarray(voxsize), img_fwd.ravel(), np.int64(nLORs),
             xp.asarray(img_dim), xp.int16(ntofbins), np.float32(tofbin_width),
             xp.asarray(sigma_tof.ravel()), xp.asarray(
                 tofcenter_offset.ravel()), xp.float32(nsigmas),
             lor_dependent_sigma_tof, lor_dependent_tofcenter_offset))
        xp.cuda.Device().synchronize()
    else:
        if num_visible_cuda_devices > 0:
            # back projection of numpy array using the cuda parallelproj lib
            num_voxel = ctypes.c_longlong(img_dim[0] * img_dim[1] * img_dim[2])

            # send image to all devices
            d_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(
                img.ravel(), num_voxel)

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
                    xstart[ic[i]:(ic[i + 1]), :].ravel(),
                    xend[ic[i]:(ic[i + 1]), :].ravel(), d_img, img_origin,
                    voxsize, img_fwd[ic[i]:(ic[i + 1]), :].ravel(),
                    np.int64(ic[i + 1] - ic[i]), img_dim,
                    np.float32(tofbin_width),
                    sigma_tof.ravel()[isig0:isig1],
                    tofcenter_offset.ravel()[ioff0:ioff1], np.float32(nsigmas),
                    np.int16(ntofbins), lor_dependent_sigma_tof,
                    lor_dependent_tofcenter_offset, threadsperblock)

            # free image device arrays
            lib_parallelproj_cuda.free_float_array_on_all_devices(d_img)
        else:
            # back projection of numpy array using the openmp parallelproj lib
            lib_parallelproj_c.joseph3d_fwd_tof_sino(
                xstart.ravel(), xend.ravel(), img.ravel(), img_origin, voxsize,
                img_fwd.ravel(), np.int64(nLORs), img_dim, tofbin_width,
                sigma_tof.ravel(), tofcenter_offset.ravel(), nsigmas, ntofbins,
                lor_dependent_sigma_tof, lor_dependent_tofcenter_offset)


def joseph3d_back_tof_sino(xstart: XPFloat32Array,
                           xend: XPFloat32Array,
                           back_img: XPFloat32Array,
                           img_origin: XPFloat32Array,
                           voxsize: XPFloat32Array,
                           sino: XPFloat32Array,
                           tofbin_width: float,
                           sigma_tof: XPFloat32Array,
                           tofcenter_offset: XPFloat32Array,
                           nsigmas: float,
                           ntofbins: int,
                           threadsperblock: int = 32,
                           num_chunks: int = 1) -> None:
    """TOF Joseph 3D sinogram forward projector

    Parameters
    ----------
    xstart : XPFloat32Array (float32 numpy or cupy array)
        start world coordinates of the LORs, shape (nLORs, 3)
    xend : XPFloat32Array (float32 numpy or cupy array)
        end world coordinates of the LORs, shape (nLORs, 3)
    back_img : XPFloat32Array (float32 numpy or cupy array)
        output array to be back projected
    img_origin : XPFloat32Array (float32 numpy or cupy array)
        containing the world coordinates of the image origin (voxel [0,0,0])
    voxsize : XPFloat32Array (float32 numpy or cupy array)
        array containing the voxel size
    sino : XPFloat32Array (float32 numpy or cupy array)
        TOF sinogram of size (nLORs*ntofbins) to be back projected 
    threadsperblock : int, optional
        by default 32
    sigma_tof : XPFloat32Array (float32 numpy or cupy array)
        sigma of Gaussian TOF kernel in spatial units (same units as xstart)
        can be an array of length 1 -> same sigma for all LORs
        or an array of length nLORs -> LOR dependent sigma
    tofcenter_offset: XPFloat32Array (float32 numpy or cupy array)
        center offset of the central TOF bin in spatial units (same units as xstart)
        can be an array of length 1 -> same offset for all LORs
        or an array of length nLORs -> LOR dependent offset
    nsigmas: float
        number of sigmas to consider when Gaussian kernel is evaluated (truncated)
    ntofbins: int
        total number of TOF bins
    num_chunks : int, optional
        break down the projection in hybrid mode into chunks to
        save memory on the GPU, by default 1
    tofbin_width : float
        width of the TOF bin in spatial units (same units as xstart)
    """

    img_dim = np.array(back_img.shape, dtype=np.int32)
    nLORs = xstart.shape[0]

    lor_dependent_sigma_tof = np.uint8(sigma_tof.shape[0] == nLORs)
    lor_dependent_tofcenter_offset = np.uint8(
        tofcenter_offset.shape[0] == nLORs)

    # check whether the input image is a numpy or cupy array
    xp = get_array_module(sino)

    if (xp.__name__ == 'cupy'):
        # projection of cupy GPU array using the cupy raw kernel
        _joseph3d_back_tof_sino_cuda_kernel(
            (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
            (xstart.ravel(), xend.ravel(), back_img, xp.asarray(img_origin),
             xp.asarray(voxsize), sino.ravel(), np.int64(nLORs),
             xp.asarray(img_dim), np.int16(ntofbins), np.float32(tofbin_width),
             xp.asarray(sigma_tof).ravel(),
             xp.asarray(tofcenter_offset).ravel(), np.float32(nsigmas),
             lor_dependent_sigma_tof, lor_dependent_tofcenter_offset))
        xp.cuda.Device().synchronize()
    else:
        if num_visible_cuda_devices > 0:
            # back projection of numpy array using the cuda parallelproj lib
            num_voxel = ctypes.c_longlong(img_dim[0] * img_dim[1] * img_dim[2])

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
                    xstart[ic[i]:(ic[i + 1]), :].ravel(),
                    xend[ic[i]:(ic[i + 1]), :].ravel(), d_back_img, img_origin,
                    voxsize, sino[ic[i]:(ic[i + 1]), :].ravel(),
                    ic[i + 1] - ic[i], img_dim, tofbin_width,
                    sigma_tof.ravel()[isig0:isig1],
                    tofcenter_offset.ravel()[ioff0:ioff1], nsigmas, ntofbins,
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
                xstart.ravel(), xend.ravel(), back_img.ravel(), img_origin,
                voxsize, sino.ravel(), np.int64(nLORs), img_dim, tofbin_width,
                sigma_tof.ravel(), tofcenter_offset.ravel(), nsigmas, ntofbins,
                lor_dependent_sigma_tof, lor_dependent_tofcenter_offset)


def joseph3d_fwd_tof_lm(xstart: XPFloat32Array,
                        xend: XPFloat32Array,
                        img: XPFloat32Array,
                        img_origin: XPFloat32Array,
                        voxsize: XPFloat32Array,
                        img_fwd: XPFloat32Array,
                        tofbin_width: float,
                        sigma_tof: XPFloat32Array,
                        tofcenter_offset: XPFloat32Array,
                        nsigmas: float,
                        tofbin: XPShortArray,
                        threadsperblock: int = 32,
                        num_chunks: int = 1) -> None:
    """TOF Joseph 3D listmode forward projector

    Parameters
    ----------
    xstart : XPFloat32Array (float32 numpy or cupy array)
        start world coordinates of the LORs, shape (nLORs, 3)
    xend : XPFloat32Array (float32 numpy or cupy array)
        end world coordinates of the LORs, shape (nLORs, 3)
    img : XPFloat32Array (float32 numpy or cupy array)
        containing the 3D image to be projected
    img_origin : XPFloat32Array (float32 numpy or cupy array)
        containing the world coordinates of the image origin (voxel [0,0,0])
    voxsize : XPFloat32Array (float32 numpy or cupy array)
        array containing the voxel size
    img_fwd : XPFloat32Array (float32 numpy or cupy array)
        output array of size (xstart.shape[0] / nevents) for storing the forward projection 
    tofbin_width : float
        width of the TOF bin in spatial units (same units as xstart)
    sigma_tof : XPFloat32Array (float32 numpy or cupy array)
        sigma of Gaussian TOF kernel in spatial units (same units as xstart)
        can be an array of length 1 -> same sigma for all LORs
        or an array of length nLORs -> LOR dependent sigma
    tofcenter_offset: XPFloat32Array (float32 numpy or cupy array)
        center offset of the central TOF bin in spatial units (same units as xstart)
        can be an array of length 1 -> same offset for all LORs
        or an array of length nLORs -> LOR dependent offset
    nsigmas: float
        number of sigmas to consider when Gaussian kernel is evaluated (truncated)
    tofbin: XPShortArray
        array containing the tof bin of the events
    threadsperblock : int, optional
        by default 32
    num_chunks : int, optional
        break down the projection in hybrid mode into chunks to
        save memory on the GPU, by default 1
    """

    img_dim = np.array(img.shape, dtype=np.int32)
    nLORs = xstart.shape[0]

    lor_dependent_sigma_tof = int(sigma_tof.shape[0] == nLORs)
    lor_dependent_tofcenter_offset = int(tofcenter_offset.shape[0] == nLORs)

    # check whether the input image is a numpy or cupy array
    xp = get_array_module(img)

    if (xp.__name__ == 'cupy'):
        # projection of cupy GPU array using the cupy raw kernel
        _joseph3d_fwd_tof_lm_cuda_kernel(
            (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
            (xstart.ravel(), xend.ravel(), img.ravel(), xp.asarray(img_origin),
             xp.asarray(voxsize), img_fwd, np.int64(nLORs),
             xp.asarray(img_dim), np.float32(tofbin_width),
             xp.asarray(sigma_tof), xp.asarray(tofcenter_offset),
             np.float32(nsigmas), tofbin, lor_dependent_sigma_tof,
             lor_dependent_tofcenter_offset))
        xp.cuda.Device().synchronize()
    else:
        if num_visible_cuda_devices > 0:
            # projection of numpy array using the cuda parallelproj lib
            num_voxel = ctypes.c_longlong(img_dim[0] * img_dim[1] * img_dim[2])

            # send image to all devices
            d_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(
                img.ravel(), num_voxel)

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
                    xstart.ravel()[(3 * ic[i]):(3 * ic[i + 1])],
                    xend.ravel()[(3 * ic[i]):(3 * ic[i + 1])], d_img,
                    img_origin, voxsize, img_fwd[ic[i]:ic[i + 1]],
                    ic[i + 1] - ic[i], img_dim, tofbin_width,
                    sigma_tof[isig0:isig1], tofcenter_offset[ioff0:ioff1],
                    nsigmas, tofbin[ic[i]:ic[i + 1]], lor_dependent_sigma_tof,
                    lor_dependent_tofcenter_offset, threadsperblock)

            # free image device arrays
            lib_parallelproj_cuda.free_float_array_on_all_devices(
                d_img, num_voxel)
        else:
            # projection of numpy array using the openmp parallelproj lib
            lib_parallelproj_c.joseph3d_fwd_tof_lm(
                xstart.ravel(), xend.ravel(), img.ravel(), img_origin, voxsize,
                img_fwd, np.int64(nLORs), img_dim, tofbin_width, sigma_tof,
                tofcenter_offset, nsigmas, tofbin, lor_dependent_sigma_tof,
                lor_dependent_tofcenter_offset)


def joseph3d_back_tof_lm(xstart: XPFloat32Array,
                         xend: XPFloat32Array,
                         back_img: XPFloat32Array,
                         img_origin: XPFloat32Array,
                         voxsize: XPFloat32Array,
                         lst: XPFloat32Array,
                         tofbin_width: float,
                         sigma_tof: XPFloat32Array,
                         tofcenter_offset: XPFloat32Array,
                         nsigmas: float,
                         tofbin: XPShortArray,
                         threadsperblock: int = 32,
                         num_chunks: int = 1) -> None:
    """TOF Joseph 3D listmode back projector

    Parameters
    ----------
    xstart : XPFloat32Array (float32 numpy or cupy array)
        start world coordinates of the LORs, shape (nLORs, 3)
    xend : XPFloat32Array (float32 numpy or cupy array)
        end world coordinates of the LORs, shape (nLORs, 3)
    back_img : XPFloat32Array (float32 numpy or cupy array)
        output array to be back projected
    img_origin : XPFloat32Array (float32 numpy or cupy array)
        containing the world coordinates of the image origin (voxel [0,0,0])
    voxsize : XPFloat32Array (float32 numpy or cupy array)
        array containing the voxel size
    lst : XPFloat32Array (float32 numpy or cupy array)
        "list" of values to be projected
    threadsperblock : int, optional
        by default 32
    sigma_tof : XPFloat32Array (float32 numpy or cupy array)
        sigma of Gaussian TOF kernel in spatial units (same units as xstart)
        can be an array of length 1 -> same sigma for all LORs
        or an array of length nLORs -> LOR dependent sigma
    tofcenter_offset: XPFloat32Array (float32 numpy or cupy array)
        center offset of the central TOF bin in spatial units (same units as xstart)
        can be an array of length 1 -> same offset for all LORs
        or an array of length nLORs -> LOR dependent offset
    nsigmas: float
        number of sigmas to consider when Gaussian kernel is evaluated (truncated)
    tofbin: XPShortArray
        array containing the tof bin of the events
    num_chunks : int, optional
        break down the projection in hybrid mode into chunks to
        save memory on the GPU, by default 1
    tofbin_width : float
        width of the TOF bin in spatial units (same units as xstart)
    """

    img_dim = np.array(back_img.shape, dtype=np.int32)
    nLORs = xstart.shape[0]

    lor_dependent_sigma_tof = int(sigma_tof.shape[0] == nLORs)
    lor_dependent_tofcenter_offset = int(tofcenter_offset.shape[0] == nLORs)

    # check whether the input is a numpy or cupy array
    xp = get_array_module(lst)

    if (xp.__name__ == 'cupy'):
        # projection of cupy GPU array using the cupy raw kernel
        _joseph3d_back_tof_lm_cuda_kernel(
            (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
            (xstart.ravel(), xend.ravel(), back_img.ravel(),
             xp.asarray(img_origin), xp.asarray(voxsize), lst, np.int64(nLORs),
             xp.asarray(img_dim), np.float32(tofbin_width),
             xp.asarray(sigma_tof), xp.asarray(tofcenter_offset),
             np.float32(nsigmas), tofbin, lor_dependent_sigma_tof,
             lor_dependent_tofcenter_offset))
        xp.cuda.Device().synchronize()
    else:
        if num_visible_cuda_devices > 0:
            # back projection of numpy array using the cuda parallelproj lib
            num_voxel = ctypes.c_longlong(img_dim[0] * img_dim[1] * img_dim[2])

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
                    xstart.ravel()[(3 * ic[i]):(3 * ic[i + 1])],
                    xend.ravel()[(3 * ic[i]):(3 * ic[i + 1])], d_back_img,
                    img_origin, voxsize, lst[ic[i]:ic[i + 1]],
                    ic[i + 1] - ic[i], img_dim, tofbin_width,
                    sigma_tof[isig0:isig1], tofcenter_offset[ioff0:ioff1],
                    nsigmas, tofbin[ic[i]:ic[i + 1]], lor_dependent_sigma_tof,
                    lor_dependent_tofcenter_offset, threadsperblock)

            # sum all device arrays in the first device
            lib_parallelproj_cuda.sum_float_arrays_on_first_device(
                d_back_img, num_voxel)

            # copy summed image back from first device
            lib_parallelproj_cuda.get_float_array_from_device(
                d_back_img, num_voxel, 0, back_img.ravel())

            # free image device arrays
            lib_parallelproj_cuda.free_float_array_on_all_devices(
                d_back_img, num_voxel)
        else:
            # back projection of numpy array using the openmp parallelproj lib
            lib_parallelproj_c.joseph3d_back_tof_lm(
                xstart.ravel(),
                xend.ravel(), back_img.ravel(), img_origin, voxsize, lst,
                np.int64(nLORs), img_dim, tofbin_width, sigma_tof,
                tofcenter_offset, nsigmas, tofbin, lor_dependent_sigma_tof,
                lor_dependent_tofcenter_offset)


#-----------------------------------------------------------------------------


def get_array_module(array) -> ModuleType:
    """return module of a cupy or numpy array

    Parameters
    ----------
    array : cupy or numpy array

    Returns
    -------
    cupy or numpy module
    """
    if cupy_enabled:
        return cp.get_array_module(array)
    else:
        return np
