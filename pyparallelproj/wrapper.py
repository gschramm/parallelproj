import ctypes
from .config import lib_parallelproj_c, lib_parallelproj_cuda, n_visible_gpus
from .config import joseph3d_fwd_cuda_kernel, joseph3d_back_cuda_kernel, joseph3d_fwd_tof_sino_cuda_kernel, joseph3d_back_tof_sino_cuda_kernel, joseph3d_fwd_tof_lm_cuda_kernel, joseph3d_back_tof_lm_cuda_kernel
import math

import numpy as np

try:
    import cupy as cp
except:
    import numpy as np


def calc_chunks(nLORs, n_chunks):
    """ calculate indices to split an array of length nLORs into n_chunks chunks

        example: splitting an array of length 10 into 3 chunks returns [0,4,7,10]
    """
    rem = nLORs % n_chunks
    div = (nLORs // n_chunks)

    chunks = [0]

    for i in range(n_chunks):
        if i < rem:
            nLORs_chunck = div + 1
        else:
            nLORs_chunck = div

        chunks.append(chunks[i] + nLORs_chunck)

    return chunks


#------------------


def joseph3d_fwd(xstart,
                 xend,
                 img,
                 img_origin,
                 voxsize,
                 img_fwd,
                 nLORs,
                 img_dim,
                 threadsperblock=64,
                 n_chunks=1):

    if n_visible_gpus > 0:
        if isinstance(img, cp.ndarray):
            ok = joseph3d_fwd_cuda_kernel(
                (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
                (xstart.ravel(), xend.ravel(), img.ravel(),
                 cp.asarray(img_origin), cp.asarray(voxsize), img_fwd,
                 np.int64(nLORs), cp.asarray(img_dim)))
        else:
            nvox = ctypes.c_longlong(img_dim[0] * img_dim[1] * img_dim[2])

            # send image to all devices
            d_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(
                img.ravel(), nvox)

            # split call to GPU lib into chunks (useful for systems with limited memory)
            ic = calc_chunks(nLORs, n_chunks)

            for i in range(n_chunks):
                ok = lib_parallelproj_cuda.joseph3d_fwd_cuda(
                    xstart[(3 * ic[i]):(3 * ic[i + 1])],
                    xend[(3 * ic[i]):(3 * ic[i + 1])], d_img, img_origin,
                    voxsize, img_fwd[ic[i]:ic[i + 1]], ic[i + 1] - ic[i],
                    img_dim, threadsperblock)

            # free image device arrays
            lib_parallelproj_cuda.free_float_array_on_all_devices(d_img, nvox)
    else:
        ok = lib_parallelproj_c.joseph3d_fwd(xstart, xend, img, img_origin,
                                             voxsize, img_fwd, nLORs, img_dim)

    return ok


#------------------


def joseph3d_back(xstart,
                  xend,
                  back_img,
                  img_origin,
                  voxsize,
                  sino,
                  nLORs,
                  img_dim,
                  threadsperblock=64,
                  n_chunks=1):

    if n_visible_gpus > 0:
        if isinstance(sino, cp.ndarray):
            ok = joseph3d_back_cuda_kernel(
                (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
                (xstart.ravel(), xend.ravel(), back_img,
                 cp.asarray(img_origin), cp.asarray(voxsize), sino.ravel(),
                 np.int64(nLORs), cp.asarray(img_dim)))
        else:
            nvox = ctypes.c_longlong(img_dim[0] * img_dim[1] * img_dim[2])

            # send image to all devices
            d_back_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(
                back_img, nvox)

            # split call to GPU lib into chunks (useful for systems with limited memory)
            ic = calc_chunks(nLORs, n_chunks)

            for i in range(n_chunks):
                ok = lib_parallelproj_cuda.joseph3d_back_cuda(
                    xstart[(3 * ic[i]):(3 * ic[i + 1])],
                    xend[(3 * ic[i]):(3 * ic[i + 1])], d_back_img, img_origin,
                    voxsize, sino[ic[i]:ic[i + 1]], ic[i + 1] - ic[i], img_dim,
                    threadsperblock)

            # sum all device arrays in the first device
            lib_parallelproj_cuda.sum_float_arrays_on_first_device(
                d_back_img, nvox)

            # copy summed image back from first device
            lib_parallelproj_cuda.get_float_array_from_device(
                d_back_img, nvox, 0, back_img)

            # free image device arrays
            lib_parallelproj_cuda.free_float_array_on_all_devices(
                d_back_img, nvox)
    else:
        ok = lib_parallelproj_c.joseph3d_back(xstart, xend, back_img,
                                              img_origin, voxsize, sino, nLORs,
                                              img_dim)

    return ok


#------------------


def joseph3d_fwd_tof_sino(xstart,
                          xend,
                          img,
                          img_origin,
                          voxsize,
                          img_fwd,
                          nLORs,
                          img_dim,
                          tofbin_width,
                          sigma_tof,
                          tofcenter_offset,
                          nsigmas,
                          ntofbins,
                          threadsperblock=64,
                          n_chunks=1):

    lor_dependent_sigma_tof = np.uint8(sigma_tof.shape[0] == nLORs)
    lor_dependent_tofcenter_offset = np.uint8(
        tofcenter_offset.shape[0] == nLORs)

    if n_visible_gpus > 0:
        if isinstance(img, cp.ndarray):
            ok = joseph3d_fwd_tof_sino_cuda_kernel(
                (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
                (xstart.ravel(), xend.ravel(), img.ravel(),
                 cp.asarray(img_origin), cp.asarray(voxsize), img_fwd,
                 np.int64(nLORs), cp.asarray(img_dim), np.int16(ntofbins),
                 np.float32(tofbin_width), cp.asarray(sigma_tof).ravel(),
                 cp.asarray(tofcenter_offset).ravel(), np.float32(nsigmas),
                 lor_dependent_sigma_tof, lor_dependent_tofcenter_offset))
        else:
            nvox = ctypes.c_longlong(img_dim[0] * img_dim[1] * img_dim[2])

            # send image to all devices
            d_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(
                img.ravel(), nvox)

            # split call to GPU lib into chunks (useful for systems with limited memory)
            ic = calc_chunks(nLORs, n_chunks)

            for i in range(n_chunks):
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

                ok = lib_parallelproj_cuda.joseph3d_fwd_tof_sino_cuda(
                    xstart[(3 * ic[i]):(3 * ic[i + 1])],
                    xend[(3 * ic[i]):(3 * ic[i + 1])], d_img, img_origin,
                    voxsize,
                    img_fwd[(ntofbins * ic[i]):(ntofbins * ic[i + 1])],
                    ic[i + 1] - ic[i], img_dim, tofbin_width,
                    sigma_tof[isig0:isig1], tofcenter_offset[ioff0:ioff1],
                    nsigmas, ntofbins, lor_dependent_sigma_tof,
                    lor_dependent_tofcenter_offset, threadsperblock)

            # free image device arrays
            lib_parallelproj_cuda.free_float_array_on_all_devices(d_img, nvox)

    else:
        ok = lib_parallelproj_c.joseph3d_fwd_tof_sino(
            xstart, xend, img, img_origin, voxsize, img_fwd, nLORs, img_dim,
            tofbin_width, sigma_tof, tofcenter_offset, nsigmas, ntofbins,
            lor_dependent_sigma_tof, lor_dependent_tofcenter_offset)

    return ok


#------------------


def joseph3d_back_tof_sino(xstart,
                           xend,
                           back_img,
                           img_origin,
                           voxsize,
                           sino,
                           nLORs,
                           img_dim,
                           tofbin_width,
                           sigma_tof,
                           tofcenter_offset,
                           nsigmas,
                           ntofbins,
                           threadsperblock=64,
                           n_chunks=1):

    lor_dependent_sigma_tof = np.uint8(sigma_tof.shape[0] == nLORs)
    lor_dependent_tofcenter_offset = np.uint8(
        tofcenter_offset.shape[0] == nLORs)

    if n_visible_gpus > 0:
        if isinstance(sino, cp.ndarray):
            ok = joseph3d_back_tof_sino_cuda_kernel(
                (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
                (xstart.ravel(), xend.ravel(), back_img,
                 cp.asarray(img_origin), cp.asarray(voxsize), sino.ravel(),
                 np.int64(nLORs), cp.asarray(img_dim), np.int16(ntofbins),
                 np.float32(tofbin_width), cp.asarray(sigma_tof).ravel(),
                 cp.asarray(tofcenter_offset).ravel(), np.float32(nsigmas),
                 lor_dependent_sigma_tof, lor_dependent_tofcenter_offset))
        else:
            nvox = ctypes.c_longlong(img_dim[0] * img_dim[1] * img_dim[2])

            # send image to all devices
            d_back_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(
                back_img, nvox)

            # split call to GPU lib into chunks (useful for systems with limited memory)
            ic = calc_chunks(nLORs, n_chunks)

            for i in range(n_chunks):
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

                ok = lib_parallelproj_cuda.joseph3d_back_tof_sino_cuda(
                    xstart[(3 * ic[i]):(3 * ic[i + 1])],
                    xend[(3 * ic[i]):(3 * ic[i + 1])], d_back_img, img_origin,
                    voxsize, sino[(ntofbins * ic[i]):(ntofbins * ic[i + 1])],
                    ic[i + 1] - ic[i], img_dim, tofbin_width,
                    sigma_tof[isig0:isig1], tofcenter_offset[ioff0:ioff1],
                    nsigmas, ntofbins, lor_dependent_sigma_tof,
                    lor_dependent_tofcenter_offset, threadsperblock)

            # sum all device arrays in the first device
            lib_parallelproj_cuda.sum_float_arrays_on_first_device(
                d_back_img, nvox)

            # copy summed image back from first device
            lib_parallelproj_cuda.get_float_array_from_device(
                d_back_img, nvox, 0, back_img)

            # free image device arrays
            lib_parallelproj_cuda.free_float_array_on_all_devices(
                d_back_img, nvox)
    else:
        ok = lib_parallelproj_c.joseph3d_back_tof_sino(
            xstart, xend, back_img, img_origin, voxsize, sino, nLORs, img_dim,
            tofbin_width, sigma_tof, tofcenter_offset, nsigmas, ntofbins,
            lor_dependent_sigma_tof, lor_dependent_tofcenter_offset)

    return ok


#------------------


def joseph3d_fwd_tof_lm(xstart,
                        xend,
                        img,
                        img_origin,
                        voxsize,
                        img_fwd,
                        nLORs,
                        img_dim,
                        tofbin_width,
                        sigma_tof,
                        tofcenter_offset,
                        nsigmas,
                        tofbin,
                        threadsperblock=64,
                        n_chunks=1):

    lor_dependent_sigma_tof = int(sigma_tof.shape[0] == nLORs)
    lor_dependent_tofcenter_offset = int(tofcenter_offset.shape[0] == nLORs)

    if n_visible_gpus > 0:
        if isinstance(img, cp.ndarray):
            ok = joseph3d_fwd_tof_lm_cuda_kernel(
                (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
                (xstart.ravel(), xend.ravel(), img.ravel(),
                 cp.asarray(img_origin), cp.asarray(voxsize), img_fwd,
                 np.int64(nLORs), cp.asarray(img_dim),
                 np.float32(tofbin_width), cp.asarray(sigma_tof).ravel(),
                 cp.asarray(tofcenter_offset).ravel(), np.float32(nsigmas),
                 tofbin, lor_dependent_sigma_tof,
                 lor_dependent_tofcenter_offset))
        else:
            nvox = ctypes.c_longlong(img_dim[0] * img_dim[1] * img_dim[2])

            # send image to all devices
            d_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(
                img.ravel(), nvox)

            # split call to GPU lib into chunks (useful for systems with limited memory)
            ic = calc_chunks(nLORs, n_chunks)

            for i in range(n_chunks):
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

                ok = lib_parallelproj_cuda.joseph3d_fwd_tof_lm_cuda(
                    xstart[(3 * ic[i]):(3 * ic[i + 1])],
                    xend[(3 * ic[i]):(3 * ic[i + 1])], d_img, img_origin,
                    voxsize, img_fwd[ic[i]:ic[i + 1]], ic[i + 1] - ic[i],
                    img_dim, tofbin_width, sigma_tof[isig0:isig1],
                    tofcenter_offset[ioff0:ioff1], nsigmas,
                    tofbin[ic[i]:ic[i + 1]], lor_dependent_sigma_tof,
                    lor_dependent_tofcenter_offset, threadsperblock)

            # free image device arrays
            lib_parallelproj_cuda.free_float_array_on_all_devices(d_img, nvox)
    else:
        ok = lib_parallelproj_c.joseph3d_fwd_tof_lm(
            xstart, xend, img, img_origin, voxsize, img_fwd, nLORs, img_dim,
            tofbin_width, sigma_tof, tofcenter_offset, nsigmas, tofbin,
            lor_dependent_sigma_tof, lor_dependent_tofcenter_offset)

    return ok


#------------------


def joseph3d_back_tof_lm(xstart,
                         xend,
                         back_img,
                         img_origin,
                         voxsize,
                         lst,
                         nLORs,
                         img_dim,
                         tofbin_width,
                         sigma_tof,
                         tofcenter_offset,
                         nsigmas,
                         tofbin,
                         threadsperblock=64,
                         n_chunks=1):

    lor_dependent_sigma_tof = int(sigma_tof.shape[0] == nLORs)
    lor_dependent_tofcenter_offset = int(tofcenter_offset.shape[0] == nLORs)

    if n_visible_gpus > 0:
        if isinstance(lst, cp.ndarray):
            lor_dependent_sigma_tof = np.uint8(sigma_tof.shape[0] == nLORs)
            lor_dependent_tofcenter_offset = np.uint8(
                tofcenter_offset.shape[0] == nLORs)

            ok = joseph3d_back_tof_lm_cuda_kernel(
                (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
                (xstart.ravel(), xend.ravel(), back_img,
                 cp.asarray(img_origin), cp.asarray(voxsize), lst,
                 np.int64(nLORs), cp.asarray(img_dim),
                 np.float32(tofbin_width), cp.asarray(sigma_tof).ravel(),
                 cp.asarray(tofcenter_offset).ravel(), np.float32(nsigmas),
                 tofbin, lor_dependent_sigma_tof,
                 lor_dependent_tofcenter_offset))
        else:
            nvox = ctypes.c_longlong(img_dim[0] * img_dim[1] * img_dim[2])

            # send image to all devices
            d_back_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(
                back_img, nvox)

            # split call to GPU lib into chunks (useful for systems with limited memory)
            ic = calc_chunks(nLORs, n_chunks)

            for i in range(n_chunks):
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

                ok = lib_parallelproj_cuda.joseph3d_back_tof_lm_cuda(
                    xstart[(3 * ic[i]):(3 * ic[i + 1])],
                    xend[(3 * ic[i]):(3 * ic[i + 1])], d_back_img, img_origin,
                    voxsize, lst[ic[i]:ic[i + 1]], ic[i + 1] - ic[i], img_dim,
                    tofbin_width, sigma_tof[isig0:isig1],
                    tofcenter_offset[ioff0:ioff1], nsigmas,
                    tofbin[ic[i]:ic[i + 1]], lor_dependent_sigma_tof,
                    lor_dependent_tofcenter_offset, threadsperblock)

            # sum all device arrays in the first device
            lib_parallelproj_cuda.sum_float_arrays_on_first_device(
                d_back_img, nvox)

            # copy summed image back from first device
            lib_parallelproj_cuda.get_float_array_from_device(
                d_back_img, nvox, 0, back_img)

            # free image device arrays
            lib_parallelproj_cuda.free_float_array_on_all_devices(
                d_back_img, nvox)
    else:
        ok = lib_parallelproj_c.joseph3d_back_tof_lm(
            xstart, xend, back_img, img_origin, voxsize, lst, nLORs, img_dim,
            tofbin_width, sigma_tof, tofcenter_offset, nsigmas, tofbin,
            lor_dependent_sigma_tof, lor_dependent_tofcenter_offset)

    return ok
