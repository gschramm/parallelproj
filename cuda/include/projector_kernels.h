#ifndef __PARALLELPROJ_CUDA_PROJECTOR_KERNELS_H__
#define __PARALLELPROJ_CUDA_PROJECTOR_KERNELS_H__

/** @brief 3D non-tof joseph back projector CUDA kernel
 *
 *  @param xstart array of shape [3*nlors] with the coordinates of the start points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                Units are the ones of voxsize.
 *  @param xend   array of shape [3*nlors] with the coordinates of the end   points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                Units are the ones of voxsize.
 *  @param img    array of shape [n0*n1*n2] for the back projection image (output).
 *                The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
 *  @param img_origin  array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
 *  @param voxsize     array [vs0, vs1, vs2] of the voxel sizes
 *  @param p           array of length nlors containg the values to be back projected
 *  @param nlors       number of projections (length of p array)
 *  @param img_dim     array with dimensions of image [n0,n1,n2]
 */
extern "C" __global__ void joseph3d_back_cuda_kernel(float *xstart, 
                                                     float *xend, 
                                                     float *img,
                                                     float *img_origin, 
                                                     float *voxsize, 
                                                     float *p,              
                                                     long long nlors,
                                                     int *img_dim);



/** @brief 3D non-tof joseph forward projector CUDA kernel
 *
 *  @param xstart array of shape [3*nlors] with the coordinates of the start points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                Units are the ones of voxsize.
 *  @param xend   array of shape [3*nlors] with the coordinates of the end   points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                Units are the ones of voxsize.
 *  @param img    array of shape [n0*n1*n2] containing the 3D image to be projected.
 *                The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
 *  @param img_origin  array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
 *  @param voxsize     array [vs0, vs1, vs2] of the voxel sizes
 *  @param p           array of length nlors (output) used to store the projections
 *  @param nlors       number of projections (length of p array)
 *  @param img_dim     array with dimensions of image [n0,n1,n2]
 */
extern "C" __global__ void joseph3d_fwd_cuda_kernel(float *xstart, 
                                                    float *xend, 
                                                    float *img,
                                                    float *img_origin, 
                                                    float *voxsize, 
                                                    float *p,
                                                    long long nlors, 
                                                    int *img_dim);




/** @brief 3D sinogram tof cuda joseph back projector kernel
 *
 *  @param xstart array of shape [3*nlors] with the coordinates of the start points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                Units are the ones of voxsize.
 *  @param xend   array of shape [3*nlors] with the coordinates of the end   points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                Units are the ones of voxsize.
 *  @param img    array of shape [n0*n1*n2] containing the 3D image used for back projection (output).
 *                The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
 *  @param img_origin  array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
 *  @param voxsize     array [vs0, vs1, vs2] of the voxel sizes
 *  @param p           array of length nlors with the values to be back projected
 *  @param nlors       number of geometrical LORs
 *  @param img_dim     array with dimensions of image [n0,n1,n2]
 *  @param n_tofbins        number of TOF bins
 *  @param tofbin_width     width of the TOF bins in spatial units (units of xstart and xend)
 *  @param sigma_tof        array of length 1 or nlors (depending on lor_dependent_sigma_tof)
 *                          with the TOF resolution (sigma) for each LOR in
 *                          spatial units (units of xstart and xend) 
 *  @param tofcenter_offset array of length 1 or nlors (depending on lor_dependent_tofcenter_offset)
 *                          with the offset of the central TOF bin from the 
 *                          midpoint of each LOR in spatial units (units of xstart and xend). 
 *                          A positive value means a shift towards the end point of the LOR.
 *  @param n_sigmas         number of sigmas to consider for calculation of TOF kernel
 *  @params lor_dependent_sigma_tof unsigned char 0 or 1
 *                                  1 means that the TOF sigmas are LOR dependent
 *                                  any other value means that the first value in the sigma_tof
 *                                  array is used for all LORs
 *  @params lor_dependent_tofcenter_offset unsigned char 0 or 1
 *                                         1 means that the TOF center offsets are LOR dependent
 *                                         any other value means that the first value in the tofcenter_offset
 *                                         array is used for all LORs
 */
extern "C" __global__ void joseph3d_back_tof_sino_cuda_kernel(float *xstart, 
                                                              float *xend, 
                                                              float *img,
                                                              float *img_origin, 
                                                              float *voxsize,
                                                              float *p, 
                                                              long long nlors, 
                                                              int *img_dim,
                                                              short n_tofbins,
                                                              float tofbin_width,
                                                              float *sigma_tof,
                                                              float *tofcenter_offset,
                                                              float n_sigmas,
                                                              unsigned char lor_dependent_sigma_tof,
                                                              unsigned char lor_dependent_tofcenter_offset);




/** @brief 3D sinogram tof joseph forward projector kernel
 *
 *  @param xstart array of shape [3*nlors] with the coordinates of the start points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                Units are the ones of voxsize.
 *  @param xend   array of shape [3*nlors] with the coordinates of the end   points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                Units are the ones of voxsize.
 *  @param img    array of shape [n0*n1*n2] containing the 3D image to be projected.
 *                The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
 *  @param img_origin  array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
 *  @param voxsize     array [vs0, vs1, vs2] of the voxel sizes
 *  @param p           array of length nlors*n_tofbins (output) used to store the projections
 *  @param nlors       number of geomtrical LORs
 *  @param img_dim     array with dimensions of image [n0,n1,n2]
 *  @param n_tofbins        number of TOF bins
 *  @param tofbin_width     width of the TOF bins in spatial units (units of xstart and xend)
 *  @param sigma_tof        array of length 1 or nlors (depending on lor_dependent_sigma_tof)
 *                          with the TOF resolution (sigma) for each LOR in
 *                          spatial units (units of xstart and xend) 
 *  @param tofcenter_offset array of length 1 or nlors (depending on lor_dependent_tofcenter_offset)
 *                          with the offset of the central TOF bin from the 
 *                          midpoint of each LOR in spatial units (units of xstart and xend). 
 *                          A positive value means a shift towards the end point of the LOR.
 *  @param n_sigmas         number of sigmas to consider for calculation of TOF kernel
 *  @params lor_dependent_sigma_tof unsigned char 0 or 1
 *                                  1 means that the TOF sigmas are LOR dependent
 *                                  any other value means that the first value in the sigma_tof
 *                                  array is used for all LORs
 *  @params lor_dependent_tofcenter_offset unsigned char 0 or 1
 *                                         1 means that the TOF center offsets are LOR dependent
 *                                         any other value means that the first value in the tofcenter_offset
 *                                         array is used for all LORs
 */
extern "C" __global__ void joseph3d_fwd_tof_sino_cuda_kernel(float *xstart, 
                                                             float *xend, 
                                                             float *img,
                                                             float *img_origin, 
                                                             float *voxsize, 
                                                             float *p,
                                                             long long nlors, 
                                                             int *img_dim,
                                                             short n_tofbins,
                                                             float tofbin_width,
                                                             float *sigma_tof,
                                                             float *tofcenter_offset,
                                                             float n_sigmas,
                                                             unsigned char lor_dependent_sigma_tof,
                                                             unsigned char lor_dependent_tofcenter_offset);




/** @brief 3D listmode tof cuda joseph back projector kernel
 *
 *  @param xstart array of shape [3*nlors] with the coordinates of the start points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                Units are the ones of voxsize.
 *  @param xend   array of shape [3*nlors] with the coordinates of the end   points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                Units are the ones of voxsize.
 *  @param img    array of shape [n0*n1*n2] containing the 3D image used for back projection (output).
 *                The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
 *  @param img_origin  array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
 *  @param voxsize     array [vs0, vs1, vs2] of the voxel sizes
 *  @param p           array of length nlors with the values to be back projected
 *  @param nlors       number of geometrical LORs
 *  @param img_dim     array with dimensions of image [n0,n1,n2]
 *  @param tofbin_width     width of the TOF bins in spatial units (units of xstart and xend)
 *  @param sigma_tof        array of length 1 or nlors (depending on lor_dependent_sigma_tof)
 *                          with the TOF resolution (sigma) for each LOR in
 *                          spatial units (units of xstart and xend) 
 *  @param tofcenter_offset array of length 1 or nlors (depending on lor_dependent_tofcenter_offset)
 *                          with the offset of the central TOF bin from the 
 *                          midpoint of each LOR in spatial units (units of xstart and xend). 
 *                          A positive value means a shift towards the end point of the LOR.
 *  @param n_sigmas         number of sigmas to consider for calculation of TOF kernel
 *  @param tof_bin          array containing the TOF bin of each event
 *  @params lor_dependent_sigma_tof unsigned char 0 or 1
 *                                  1 means that the TOF sigmas are LOR dependent
 *                                  any other value means that the first value in the sigma_tof
 *                                  array is used for all LORs
 *  @params lor_dependent_tofcenter_offset unsigned char 0 or 1
 *                                         1 means that the TOF center offsets are LOR dependent
 *                                         any other value means that the first value in the tofcenter_offset
 *                                         array is used for all LORs
 */
extern "C" __global__ void joseph3d_back_tof_lm_cuda_kernel(float *xstart, 
                                                            float *xend, 
                                                            float *img,
                                                            float *img_origin, 
                                                            float *voxsize,
                                                            float *p, 
                                                            long long nlors, 
                                                            int *img_dim,
                                                            float tofbin_width,
                                                            float *sigma_tof,
                                                            float *tofcenter_offset,
                                                            float n_sigmas,
                                                            short *tof_bin,
                                                            unsigned char lor_dependent_sigma_tof,
                                                            unsigned char lor_dependent_tofcenter_offset);



/** @brief 3D listmode tof joseph forward projector kernel
 *
 *  @param xstart array of shape [3*nlors] with the coordinates of the start points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                Units are the ones of voxsize.
 *  @param xend   array of shape [3*nlors] with the coordinates of the end   points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                Units are the ones of voxsize.
 *  @param img    array of shape [n0*n1*n2] containing the 3D image to be projected.
 *                The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
 *  @param img_origin  array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
 *  @param voxsize     array [vs0, vs1, vs2] of the voxel sizes
 *  @param p           array of length nlors (output) used to store the projections
 *  @param nlors       number of geomtrical LORs
 *  @param img_dim     array with dimensions of image [n0,n1,n2]
 *  @param tofbin_width     width of the TOF bins in spatial units (units of xstart and xend)
 *  @param sigma_tof        array of length 1 or nlors (depending on lor_dependent_sigma_tof)
 *                          with the TOF resolution (sigma) for each LOR in
 *                          spatial units (units of xstart and xend) 
 *  @param tofcenter_offset array of length 1 or nlors (depending on lor_dependent_tofcenter_offset)
 *                          with the offset of the central TOF bin from the 
 *                          midpoint of each LOR in spatial units (units of xstart and xend). 
 *                          A positive value means a shift towards the end point of the LOR.
 *  @param n_sigmas         number of sigmas to consider for calculation of TOF kernel
 *  @param tof_bin          array containing the TOF bin of each event
 *  @params lor_dependent_sigma_tof unsigned char 0 or 1
 *                                  1 means that the TOF sigmas are LOR dependent
 *                                  any other value means that the first value in the sigma_tof
 *                                  array is used for all LORs
 *  @params lor_dependent_tofcenter_offset unsigned char 0 or 1
 *                                         1 means that the TOF center offsets are LOR dependent
 *                                         any other value means that the first value in the tofcenter_offset
 *                                         array is used for all LORs
 */
extern "C" __global__ void joseph3d_fwd_tof_lm_cuda_kernel(float *xstart, 
                                                           float *xend, 
                                                           float *img,
                                                           float *img_origin, 
                                                           float *voxsize, 
                                                           float *p,
                                                           long long nlors, 
                                                           int *img_dim,
                                                           float tofbin_width,
                                                           float *sigma_tof,
                                                           float *tofcenter_offset,
                                                           float n_sigmas,
                                                           short *tof_bin,
                                                           unsigned char lor_dependent_sigma_tof,
                                                           unsigned char lor_dependent_tofcenter_offset);
#endif
