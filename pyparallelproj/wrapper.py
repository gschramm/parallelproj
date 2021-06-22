import ctypes
from pyparallelproj.config import lib_parallelproj_c, lib_parallelproj_cuda, n_visible_gpus

def joseph3d_fwd(xstart, xend, img, img_origin, voxsize, img_fwd, nLORs, img_dim, threadsperblock = 64): 

  if n_visible_gpus > 0:
    nvox = ctypes.c_longlong(img_dim[0]*img_dim[1]*img_dim[2])

    # send image to all devices
    d_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(img.ravel(), nvox)

    ok = lib_parallelproj_cuda.joseph3d_fwd_cuda(xstart, xend, d_img, img_origin, voxsize, img_fwd, nLORs, 
                                                 img_dim, threadsperblock)

    # free image device arrays
    lib_parallelproj_cuda.free_float_array_on_all_devices(d_img, nvox)
  else:
    ok = lib_parallelproj_c.joseph3d_fwd(xstart, xend, img, img_origin, voxsize, img_fwd, nLORs, img_dim)

  return ok

#------------------

def joseph3d_back(xstart, xend, back_img, img_origin, voxsize, sino, nLORs, img_dim, threadsperblock = 64):

  if n_visible_gpus > 0:
    nvox = ctypes.c_longlong(img_dim[0]*img_dim[1]*img_dim[2])

    # send image to all devices
    d_back_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(back_img, nvox)

    ok = lib_parallelproj_cuda.joseph3d_back_cuda(xstart, xend, d_back_img, img_origin, voxsize, sino, 
                                                  nLORs, img_dim, threadsperblock) 

    # sum all device arrays in the first device
    lib_parallelproj_cuda.sum_float_arrays_on_first_device(d_back_img, nvox) 

    # copy summed image back from first device
    lib_parallelproj_cuda.get_float_array_from_device(d_back_img, nvox, 0, back_img)

    # free image device arrays
    lib_parallelproj_cuda.free_float_array_on_all_devices(d_back_img, nvox)
  else:
    ok = lib_parallelproj_c.joseph3d_back(xstart, xend, back_img, img_origin, voxsize, sino, nLORs, img_dim)

  return ok

#------------------

def joseph3d_fwd_tof_sino(xstart, xend, img, img_origin, voxsize, img_fwd, nLORs, img_dim,
                          tofbin_width, sigma_tof, tofcenter_offset, nsigmas, ntofbins, threadsperblock = 64):

  if n_visible_gpus > 0:

    nvox = ctypes.c_longlong(img_dim[0]*img_dim[1]*img_dim[2])

    # send image to all devices
    d_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(img.ravel(), nvox)

    ok = lib_parallelproj_cuda.joseph3d_fwd_tof_sino_cuda(xstart, xend, d_img, img_origin, voxsize, 
                                                          img_fwd, nLORs, img_dim,
                                                          tofbin_width, sigma_tof, tofcenter_offset, 
                                                          nsigmas, ntofbins, threadsperblock) 

    # free image device arrays
    lib_parallelproj_cuda.free_float_array_on_all_devices(d_img, nvox)


  else:
    ok = lib_parallelproj.joseph3d_fwd_tof_sino(xstart, xend, img, img_origin, voxsize, 
                                                img_fwd, nLORs, img_dim,
                                                tofbin_width, sigma_tof, tofcenter_offset, 
                                                nsigmas, ntofbins) 

  return ok

#------------------

def joseph3d_back_tof_sino(xstart, xend, back_img, img_origin, voxsize, sino, nLORs, img_dim,
                           tofbin_width, sigma_tof, tofcenter_offset, nsigmas, ntofbins, threadsperblock = 64):

  if n_visible_gpus > 0:

    nvox = ctypes.c_longlong(img_dim[0]*img_dim[1]*img_dim[2])

    # send image to all devices
    d_back_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(back_img, nvox)

    ok = lib_parallelproj_cuda.joseph3d_back_tof_sino_cuda(xstart, xend, d_back_img, img_origin, voxsize, 
                                                           sino, nLORs, img_dim,
                                                           tofbin_width, sigma_tof, tofcenter_offset, 
                                                           nsigmas, ntofbins, threadsperblock) 

    # sum all device arrays in the first device
    lib_parallelproj_cuda.sum_float_arrays_on_first_device(d_back_img, nvox) 

    # copy summed image back from first device
    lib_parallelproj_cuda.get_float_array_from_device(d_back_img, nvox, 0, back_img)

    # free image device arrays
    lib_parallelproj_cuda.free_float_array_on_all_devices(d_back_img, nvox)
  else:
    ok = lib_parallelproj.joseph3d_back_tof_sino(xstart, xend, back_img, img_origin, voxsize, 
                                                 sino, nLORs, img_dim,
                                                 tofbin_width, sigma_tof, tofcenter_offset, 
                                                 nsigmas, ntofbins)

  return ok 


#------------------

def joseph3d_fwd_tof_lm(xstart, xend, img, img_origin, voxsize, img_fwd, nLORs, img_dim,
                        tofbin_width, sigma_tof, tofcenter_offset, nsigmas, tofbin, threadsperblock = 64):

  if n_visible_gpus > 0:

    nvox = ctypes.c_longlong(img_dim[0]*img_dim[1]*img_dim[2])

    # send image to all devices
    d_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(img.ravel(), nvox)

    ok = lib_parallelproj_cuda.joseph3d_fwd_tof_lm_cuda(xstart, xend, d_img, img_origin, voxsize, 
                                                        img_fwd, nLORs, img_dim,
                                                        tofbin_width, sigma_tof, tofcenter_offset, 
                                                        nsigmas, tofbin, threadsperblock) 

    # free image device arrays
    lib_parallelproj_cuda.free_float_array_on_all_devices(d_img, nvox)


  else:
    ok = lib_parallelproj.joseph3d_fwd_tof_lm(xstart, xend, img, img_origin, voxsize, 
                                              img_fwd, nLORs, img_dim,
                                              tofbin_width, sigma_tof, tofcenter_offset, 
                                              nsigmas, tofbin) 

  return ok

#------------------

def joseph3d_back_tof_lm(xstart, xend, back_img, img_origin, voxsize, lst, nLORs, img_dim,
                         tofbin_width, sigma_tof, tofcenter_offset, nsigmas, tofbin, threadsperblock = 64):

  if n_visible_gpus > 0:

    nvox = ctypes.c_longlong(img_dim[0]*img_dim[1]*img_dim[2])

    # send image to all devices
    d_back_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(back_img, nvox)

    ok = lib_parallelproj_cuda.joseph3d_back_tof_lm_cuda(xstart, xend, d_back_img, img_origin, voxsize, 
                                                         lst, nLORs, img_dim,
                                                         tofbin_width, sigma_tof, tofcenter_offset, 
                                                         nsigmas, tofbin, threadsperblock) 

    # sum all device arrays in the first device
    lib_parallelproj_cuda.sum_float_arrays_on_first_device(d_back_img, nvox) 

    # copy summed image back from first device
    lib_parallelproj_cuda.get_float_array_from_device(d_back_img, nvox, 0, back_img)

    # free image device arrays
    lib_parallelproj_cuda.free_float_array_on_all_devices(d_back_img, nvox)
  else:
    ok = lib_parallelproj.joseph3d_back_tof_lm(xstart, xend, back_img, img_origin, voxsize, 
                                               lst, nLORs, img_dim,
                                               tofbin_width, sigma_tof, tofcenter_offset, 
                                               nsigmas, tofbin)

  return ok 


