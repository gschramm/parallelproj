/**
 * @file parallelproj_cuda.h
 */

#ifndef __PARALLELPROJ_CUDA_H__
#define __PARALLELPROJ_CUDA_H__

/** @brief 3D non-tof joseph back projector CUDA wrapper
 *
 *  @param h_xstart array of shape [3*nlors] with the coordinates of the start points of the LORs.
 *                  The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                  Units are the ones of voxsize.
 *  @param h_xend   array of shape [3*nlors] with the coordinates of the end   points of the LORs.
 *                  The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                  Units are the ones of voxsize.
 *  @param d_img    Pointer to device arrays of shape [n0*n1*n2] containing the 3D image to 
 *                  used to store the back projections.
 *                  The pixel [i,j,k] is stored at [n1*n2*i + n2*j + k].
 *                  The backprojector adds existing values.
 *  @param h_img_origin  array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
 *  @param h_voxsize     array [vs0, vs1, vs2] of the voxel sizes
 *  @param h_p           array of length nlors containg the values to be back projected
 *  @param nlors          number of projections (length of p array)
 *  @param h_img_dim      array with dimensions of image [n0,n1,n2]
 *  @param threadsperblock number of threads per block
 */
extern "C" void joseph3d_back_cuda(const float *h_xstart, 
                                   const float *h_xend, 
                                   float **d_img,
                                   const float *h_img_origin, 
                                   const float *h_voxsize, 
                                   const float *h_p,
                                   long long nlors, 
                                   const int *h_img_dim, 
                                   int threadsperblock);



/** @brief 3D listmode tof joseph back projector CUDA wrapper
 *
 *  @param h_xstart array of shape [3*nlors] with the coordinates of the start points of the LORs.
 *                  The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                  Units are the ones of voxsize.
 *  @param h_xend   array of shape [3*nlors] with the coordinates of the end   points of the LORs.
 *                  The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                  Units are the ones of voxsize.
 *  @param d_img    Pointer to device arrays of shape [n0*n1*n2] containing the 3D image to 
 *                  used to store the back projections.
 *                  The pixel [i,j,k] is stored at [n1*n2*i + n2*j + k].
 *                  The backprojector adds existing values.
 *  @param h_img_origin  array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
 *  @param h_voxsize     array [vs0, vs1, vs2] of the voxel sizes
 *  @param h_p           array of length nlors containg the values to be back projected
 *  @param nlors         number of projections (length of p array)
 *  @param h_img_dim     array with dimensions of image [n0,n1,n2]
 *  @param tofbin_width     width of the TOF bins in spatial units (units of xstart and xend)
 *  @param h_sigma_tof      array of length nlors with the TOF resolution (sigma) for each LOR in
 *                          spatial units (units of xstart and xend) 
 *  @param h_tofcenter_offset  array of length nlors with the offset of the central TOF bin from the 
 *                             midpoint of each LOR in spatial units (units of xstart and xend). 
 *                             A positive value means a shift towards the end point of the LOR.
 *  @param n_sigmas        number of sigmas to consider for calculation of TOF kernel
 *  @param h_tof_bin       array containing the TOF bin of each event
 *  @param threadsperblock number of threads per block
 */
extern "C" void joseph3d_back_tof_lm_cuda(const float *h_xstart,
                                          const float *h_xend,
                                          float **d_img,
                                          const float *h_img_origin,
                                          const float *h_voxsize,
                                          const float *h_p,
                                          long long nlors,
                                          const int *h_img_dim,
                                          float tofbin_width,
                                          const float *h_sigma_tof,
                                          const float *h_tofcenter_offset,
                                          float n_sigmas,
                                          const short *h_tof_bin,
                                          int threadsperblock);



/** @brief 3D sinogram tof joseph back projector CUDA wrapper
 *
 *  @param h_xstart array of shape [3*nlors] with the coordinates of the start points of the LORs.
 *                  The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                  Units are the ones of voxsize.
 *  @param h_xend   array of shape [3*nlors] with the coordinates of the end   points of the LORs.
 *                  The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                  Units are the ones of voxsize.
 *  @param d_img    Pointer to device arrays of shape [n0*n1*n2] containing the 3D image to 
 *                  used to store the back projections.
 *                  The pixel [i,j,k] is stored at [n1*n2*i + n2*j + k].
 *                  The backprojector adds existing values.
 *  @param h_img_origin  array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
 *  @param h_voxsize     array [vs0, vs1, vs2] of the voxel sizes
 *  @param h_p           array of length (nlors*n_tofbins) containg the values to be back projected
 *  @param nlors          number of gemeometrical projections
 *  @param h_img_dim      array with dimensions of image [n0,n1,n2]
 *  @param tofbin_width       width of the TOF bins in spatial units (units of xstart and xend)
 *  @param h_sigma_tof        array of length nlors with the TOF resolution (sigma) for each LOR in
 *                            spatial units (units of xstart and xend) 
 *  @param h_tofcenter_offset  array of length nlors with the offset of the central TOF bin from the 
 *                             midpoint of each LOR in spatial units (units of xstart and xend). 
 *                             A positive value means a shift towards the end point of the LOR.
 *  @param n_sigmas           number of sigmas to consider for calculation of TOF kernel
 *  @param n_tofbins          number of TOF bins
 *  @param threadsperblock number of threads per block
 */
extern "C" void joseph3d_back_tof_sino_cuda(const float *h_xstart, 
                                            const float *h_xend, 
                                            float **d_img,
                                            const float *h_img_origin, 
                                            const float *h_voxsize, 
                                            const float *h_p,
                                            long long nlors, 
                                            const int *h_img_dim, 
                                            float tofbin_width,
                                            const float *h_sigma_tof,
                                            const float *h_tofcenter_offset,
                                            float n_sigmas,
                                            short n_tofbins,
                                            int threadsperblock);



/** @brief 3D non-tof joseph forward projector CUDA wrapper
 *
 *  @param h_xstart array of shape [3*nlors] with the coordinates of the start points of the LORs.
 *                  The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                  Units are the ones of voxsize.
 *  @param h_xend   array of shape [3*nlors] with the coordinates of the end   points of the LORs.
 *                  The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                  Units are the ones of voxsize.
 *  @param d_img    Pointer to device arrays of shape [n0*n1*n2] containing the 3D image to 
 *                  be projected.
 *                  The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
 *  @param h_img_origin  array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
 *  @param h_voxsize     array [vs0, vs1, vs2] of the voxel sizes
 *  @param h_p           array of length nlors (output) used to store the projections
 *  @param nlors           number of projections (length of p array)
 *  @param h_img_dim       array with dimensions of image [n0,n1,n2]
 *  @param threadsperblock number of threads per block
 */
extern "C" void joseph3d_fwd_cuda(const float *h_xstart, 
                                  const float *h_xend, 
                                  float **d_img,
                                  const float *h_img_origin, 
                                  const float *h_voxsize, 
                                  float *h_p,
                                  long long nlors, 
                                  const int *h_img_dim,
                                  int threadsperblock);



/** @brief 3D listmode tof joseph forward projector CUDA wrapper
 *
 *  @param h_xstart array of shape [3*nlors] with the coordinates of the start points of the LORs.
 *                  The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                  Units are the ones of voxsize.
 *  @param h_xend   array of shape [3*nlors] with the coordinates of the end   points of the LORs.
 *                  The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                  Units are the ones of voxsize.
 *  @param d_img    Pointer to device arrays of shape [n0*n1*n2] containing the 3D image to 
 *                  used to store the back projections.
 *                  The pixel [i,j,k] is stored at [n1*n2*i + n2*j + k].
 *  @param h_img_origin  array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
 *  @param h_voxsize     array [vs0, vs1, vs2] of the voxel sizes
 *  @param h_p           array of length nlors (output) used to store the projections
 *  @param nlors         number of projections (length of p array)
 *  @param h_img_dim     array with dimensions of image [n0,n1,n2]
 *  @param tofbin_width     width of the TOF bins in spatial units (units of xstart and xend)
 *  @param h_sigma_tof      array of length nlors with the TOF resolution (sigma) for each LOR in
 *                          spatial units (units of xstart and xend) 
 *  @param h_tofcenter_offset  array of length nlors with the offset of the central TOF bin from the 
 *                             midpoint of each LOR in spatial units (units of xstart and xend). 
 *                             A positive value means a shift towards the end point of the LOR.
 *  @param n_sigmas           number of sigmas to consider for calculation of TOF kernel
 *  @param h_tof_bin          array of length nlors with the tofbin of every event 
 *  @param threadsperblock    number of threads per block
 */
extern "C" void joseph3d_fwd_tof_lm_cuda(const float *h_xstart, 
                                         const float *h_xend, 
                                         float **d_img,
                                         const float *h_img_origin, 
                                         const float *h_voxsize, 
                                         float *h_p,
                                         long long nlors, 
                                         const int *h_img_dim,
                                         float tofbin_width,
                                         const float *h_sigma_tof,
                                         const float *h_tofcenter_offset,
                                         float n_sigmas,
                                         const short *h_tof_bin,
                                         int threadsperblock);



/** @brief 3D sinogram tof joseph forward projector CUDA wrapper
 *
 *  @param h_xstart array of shape [3*nlors] with the coordinates of the start points of the LORs.
 *                  The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                  Units are the ones of voxsize.
 *  @param h_xend   array of shape [3*nlors] with the coordinates of the end   points of the LORs.
 *                  The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                  Units are the ones of voxsize.
 *  @param d_img    Pointer to device arrays of shape [n0*n1*n2] containing the 3D image to 
 *                  used to store the back projections.
 *                  The pixel [i,j,k] is stored at [n1*n2*i + n2*j + k].
 *  @param h_img_origin  array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
 *  @param h_voxsize     array [vs0, vs1, vs2] of the voxel sizes
 *  @param h_p           array of length nlors*n_tofbins (output) used to store the projections
 *  @param nlors         number of geometrical LORs
 *  @param h_img_dim     array with dimensions of image [n0,n1,n2]
 *  @param tofbin_width     width of the TOF bins in spatial units (units of xstart and xend)
 *  @param h_sigma_tof      array of length nlors with the TOF resolution (sigma) for each LOR in
 *                          spatial units (units of xstart and xend) 
 *  @param h_tofcenter_offset  array of length nlors with the offset of the central TOF bin from the 
 *                             midpoint of each LOR in spatial units (units of xstart and xend) 
 *                             A positive value means a shift towards the end point of the LOR.
 *  @param n_sigmas           number of sigmas to consider for calculation of TOF kernel
 *  @param n_tofbins          number of TOF bins
 *  @param threadsperblock    number of threads per block
 */
extern "C" void joseph3d_fwd_tof_sino_cuda(const float *h_xstart, 
                                           const float *h_xend, 
                                           float **d_img,
                                           const float *h_img_origin, 
                                           const float *h_voxsize, 
                                           float *h_p,
                                           long long nlors, 
                                           const int *h_img_dim,
                                           float tofbin_width,
                                           const float *h_sigma_tof,
                                           const float *h_tofcenter_offset,
                                           float n_sigmas,
                                           short n_tofbins,
                                           int threadsperblock);



/** @brief copy a float array to all visible cuda devices
 *
 *  The number of visible cuda devices is determined automatically via the CUDA API
 *
 *  @param    h_array   array of shape [n] on the host
 *  @param    n         number of array elements
 *  @return   a pointer to all devices arrays
 */
extern "C" float** copy_float_array_to_all_devices(const float *h_array, long long n);



/** @brief free device array on all visible cuda devices
 *
 *  The number of visible cuda devices is determined automatically via the CUDA API
 *
 *  @param d_array a pointer to all devices arrays
 */
extern "C" void free_float_array_on_all_devices(float **d_array);



/** @brief sum multiple versions of an array on different devices on first device
 *
 *  The number of visible cuda devices is determined automatically via the CUDA API
 *  This becomes usefule when multiple devices backproject into separate images.
 *
 *  @param d_array a pointer to all devices arrays
 *  @param    n         number of array elements
 */
extern "C" void sum_float_arrays_on_first_device(float **d_array, long long n);



/** @brief copy a (summed) float array from first device back to host
 *
 *  The number of visible cuda devices is determined automatically via the CUDA API
 *
 *  @param  d_array   a pointer to all devices arrays of shape [n]
 *  @param  n         number of array elements
 *  @param  i_dev     device number
 *  @param  h_array   array of shape [n] on the host used for output
 */
extern "C" void get_float_array_from_device(float **d_array, long long n, int i_dev, float *h_array);

#endif
