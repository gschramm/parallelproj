import ctypes
from pyparallelproj.config import lib_parallelproj_c, lib_parallelproj_cuda, n_visible_gpus

def calc_chunks(nLORs, n_chunks):
  """ calculate indices to split an array of length nLORs into n_chunks chunks

      example: splitting an array of length 10 into 3 chunks returns [0,4,7,10]
  """
  rem  = nLORs % n_chunks
  div  = (nLORs // n_chunks)

  chunks = [0]

  for i in range(n_chunks):
    if i < rem: 
      nLORs_chunck = div + 1
    else:
      nLORs_chunck = div

    chunks.append(chunks[i] + nLORs_chunck) 

  return chunks

#------------------

def joseph3d_fwd(xstart, xend, img, img_origin, voxsize, img_fwd, nLORs, img_dim, 
                 threadsperblock = 64, n_chunks = 1): 

  if n_visible_gpus > 0:
    nvox = ctypes.c_longlong(img_dim[0]*img_dim[1]*img_dim[2])

    # send image to all devices
    d_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(img.ravel(), nvox)

    # split call to GPU lib into chunks (useful for systems with limited memory)
    ic = calc_chunks(nLORs, n_chunks)

    for i in range(n_chunks):
      ok = lib_parallelproj_cuda.joseph3d_fwd_cuda(xstart[(3*ic[i]):(3*ic[i+1])], xend[(3*ic[i]):(3*ic[i+1])], 
                                                   d_img, img_origin, voxsize, 
                                                   img_fwd[ic[i]:ic[i+1]], ic[i+1] - ic[i], 
                                                   img_dim, threadsperblock)

    # free image device arrays
    lib_parallelproj_cuda.free_float_array_on_all_devices(d_img, nvox)
  else:
    ok = lib_parallelproj_c.joseph3d_fwd(xstart, xend, img, img_origin, voxsize, img_fwd, nLORs, img_dim)

  return ok

#------------------

def joseph3d_back(xstart, xend, back_img, img_origin, voxsize, sino, nLORs, img_dim,
                 threadsperblock = 64, n_chunks = 1): 

  if n_visible_gpus > 0:
    nvox = ctypes.c_longlong(img_dim[0]*img_dim[1]*img_dim[2])

    # send image to all devices
    d_back_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(back_img, nvox)


    # split call to GPU lib into chunks (useful for systems with limited memory)
    ic = calc_chunks(nLORs, n_chunks)

    for i in range(n_chunks):
      ok = lib_parallelproj_cuda.joseph3d_back_cuda(xstart[(3*ic[i]):(3*ic[i+1])], xend[(3*ic[i]):(3*ic[i+1])],
                                                    d_back_img, img_origin, voxsize, 
                                                    sino[ic[i]:ic[i+1]], ic[i+1] - ic[i],
                                                    img_dim, threadsperblock) 

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
                          tofbin_width, sigma_tof, tofcenter_offset, nsigmas, ntofbins,
                          threadsperblock = 64, n_chunks = 1): 

  if n_visible_gpus > 0:

    nvox = ctypes.c_longlong(img_dim[0]*img_dim[1]*img_dim[2])

    # send image to all devices
    d_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(img.ravel(), nvox)

    # split call to GPU lib into chunks (useful for systems with limited memory)
    ic = calc_chunks(nLORs, n_chunks)

    for i in range(n_chunks):
      ok = lib_parallelproj_cuda.joseph3d_fwd_tof_sino_cuda(xstart[(3*ic[i]):(3*ic[i+1])], 
                                                            xend[(3*ic[i]):(3*ic[i+1])], 
                                                            d_img, img_origin, voxsize, 
                                                            img_fwd[(ntofbins*ic[i]):(ntofbins*ic[i+1])], 
                                                            ic[i+1] - ic[i], img_dim,
                                                            tofbin_width, sigma_tof[ic[i]:ic[i+1]], 
                                                            tofcenter_offset[ic[i]:ic[i+1]], 
                                                            nsigmas, ntofbins, threadsperblock) 

    # free image device arrays
    lib_parallelproj_cuda.free_float_array_on_all_devices(d_img, nvox)


  else:
    ok = lib_parallelproj_c.joseph3d_fwd_tof_sino(xstart, xend, img, img_origin, voxsize, 
                                                img_fwd, nLORs, img_dim,
                                                tofbin_width, sigma_tof, tofcenter_offset, 
                                                nsigmas, ntofbins) 

  return ok

#------------------

def joseph3d_back_tof_sino(xstart, xend, back_img, img_origin, voxsize, sino, nLORs, img_dim,
                           tofbin_width, sigma_tof, tofcenter_offset, nsigmas, ntofbins,
                           threadsperblock = 64, n_chunks = 1): 

  if n_visible_gpus > 0:

    nvox = ctypes.c_longlong(img_dim[0]*img_dim[1]*img_dim[2])

    # send image to all devices
    d_back_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(back_img, nvox)

    # split call to GPU lib into chunks (useful for systems with limited memory)
    ic = calc_chunks(nLORs, n_chunks)

    for i in range(n_chunks):
      ok = lib_parallelproj_cuda.joseph3d_back_tof_sino_cuda(xstart[(3*ic[i]):(3*ic[i+1])], 
                                                             xend[(3*ic[i]):(3*ic[i+1])], 
                                                             d_back_img, img_origin, voxsize, 
                                                             sino[(ntofbins*ic[i]):(ntofbins*ic[i+1])], 
                                                             ic[i+1] - ic[i], img_dim,
                                                             tofbin_width, sigma_tof[ic[i]:ic[i+1]], 
                                                             tofcenter_offset[ic[i]:ic[i+1]], 
                                                             nsigmas, ntofbins, threadsperblock) 

    # sum all device arrays in the first device
    lib_parallelproj_cuda.sum_float_arrays_on_first_device(d_back_img, nvox) 

    # copy summed image back from first device
    lib_parallelproj_cuda.get_float_array_from_device(d_back_img, nvox, 0, back_img)

    # free image device arrays
    lib_parallelproj_cuda.free_float_array_on_all_devices(d_back_img, nvox)
  else:
    ok = lib_parallelproj_c.joseph3d_back_tof_sino(xstart, xend, back_img, img_origin, voxsize, 
                                                 sino, nLORs, img_dim,
                                                 tofbin_width, sigma_tof, tofcenter_offset, 
                                                 nsigmas, ntofbins)

  return ok 


#------------------

def joseph3d_fwd_tof_lm(xstart, xend, img, img_origin, voxsize, img_fwd, nLORs, img_dim,
                        tofbin_width, sigma_tof, tofcenter_offset, nsigmas, tofbin,
                        threadsperblock = 64, n_chunks = 1): 

  if n_visible_gpus > 0:

    nvox = ctypes.c_longlong(img_dim[0]*img_dim[1]*img_dim[2])

    # send image to all devices
    d_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(img.ravel(), nvox)

    # split call to GPU lib into chunks (useful for systems with limited memory)
    ic = calc_chunks(nLORs, n_chunks)

    for i in range(n_chunks):
      ok = lib_parallelproj_cuda.joseph3d_fwd_tof_lm_cuda(xstart[(3*ic[i]):(3*ic[i+1])], 
                                                          xend[(3*ic[i]):(3*ic[i+1])], 
                                                          d_img, img_origin, voxsize, 
                                                          img_fwd[ic[i]:ic[i+1]], ic[i+1] - ic[i], img_dim,
                                                          tofbin_width, sigma_tof[ic[i]:ic[i+1]], 
                                                          tofcenter_offset[ic[i]:ic[i+1]], 
                                                          nsigmas, tofbin[ic[i]:ic[i+1]], threadsperblock) 

    # free image device arrays
    lib_parallelproj_cuda.free_float_array_on_all_devices(d_img, nvox)


  else:
    ok = lib_parallelproj_c.joseph3d_fwd_tof_lm(xstart, xend, img, img_origin, voxsize, 
                                              img_fwd, nLORs, img_dim,
                                              tofbin_width, sigma_tof, tofcenter_offset, 
                                              nsigmas, tofbin) 

  return ok

#------------------

def joseph3d_back_tof_lm(xstart, xend, back_img, img_origin, voxsize, lst, nLORs, img_dim,
                         tofbin_width, sigma_tof, tofcenter_offset, nsigmas, tofbin,
                         threadsperblock = 64, n_chunks = 1): 

  if n_visible_gpus > 0:

    nvox = ctypes.c_longlong(img_dim[0]*img_dim[1]*img_dim[2])

    # send image to all devices
    d_back_img = lib_parallelproj_cuda.copy_float_array_to_all_devices(back_img, nvox)

    # split call to GPU lib into chunks (useful for systems with limited memory)
    ic = calc_chunks(nLORs, n_chunks)

    for i in range(n_chunks):
      ok = lib_parallelproj_cuda.joseph3d_back_tof_lm_cuda(xstart[(3*ic[i]):(3*ic[i+1])], 
                                                           xend[(3*ic[i]):(3*ic[i+1])], 
                                                           d_back_img, img_origin, voxsize, 
                                                           lst[ic[i]:ic[i+1]], ic[i+1] - ic[i], img_dim,
                                                           tofbin_width, sigma_tof[ic[i]:ic[i+1]], 
                                                           tofcenter_offset[ic[i]:ic[i+1]], 
                                                           nsigmas, tofbin[ic[i]:ic[i+1]], threadsperblock) 

    # sum all device arrays in the first device
    lib_parallelproj_cuda.sum_float_arrays_on_first_device(d_back_img, nvox) 

    # copy summed image back from first device
    lib_parallelproj_cuda.get_float_array_from_device(d_back_img, nvox, 0, back_img)

    # free image device arrays
    lib_parallelproj_cuda.free_float_array_on_all_devices(d_back_img, nvox)
  else:
    ok = lib_parallelproj_c.joseph3d_back_tof_lm(xstart, xend, back_img, img_origin, voxsize, 
                                               lst, nLORs, img_dim,
                                               tofbin_width, sigma_tof, tofcenter_offset, 
                                               nsigmas, tofbin)

  return ok 
