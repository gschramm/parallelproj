#ifndef __PARALLELPROJ_CUDA_PROJECTOR_KERNELS_H__
#define __PARALLELPROJ_CUDA_PROJECTOR_KERNELS_H__

/** @brief Calculate whether a ray and a cube intersect and the possible intersection points
 *
 *  The ray is given by origin + t*rdir (vector notation)
 *  if the ray intersects the cube, the two t values t1 and t2
 *  for the intersection points are computed.
 *
 *  This algorithm assumes that the IEEE floating point arithmetic standard 754 is followed 
 *  which handles divions by 0 and -0 correctly.
 *
 *  @param orig0       ...  0 cordinate of the ray origin
 *  @param orig1       ...  1 cordinate of the ray origin
 *  @param orig2       ...  2 cordinate of the ray origin
 *  @param bounds0_min ...  0 cordinate of the start of the cube bounding box
 *  @param bounds1_min ...  1 cordinate of the start of the cube bounding box
 *  @param bounds2_min ...  2 cordinate of the start of the cube bounding box
 *  @param bounds0_max ...  0 cordinate of the end   of the cube bounding box
 *  @param bounds1_max ...  1 cordinate of the end   of the cube bounding box
 *  @param bounds2_max ...  2 cordinate of the end   of the cube bounding box
 *  @param rdir0       ...  0 cordinate of the ray directional vector
 *  @param rdir1       ...  1 cordinate of the ray directional vector
 *  @param rdir2       ...  2 cordinate of the ray directional vector
 *  @param t1          ...  (output) ray parameter of 1st intersection point
 *  @param t2          ...  (output) ray parameter of 2nd intersection point
 *
 *  @return            ...  unsigned char (0 or 1) whether ray intersects cube 
 */
extern "C" __device__ unsigned char ray_cube_intersection_cuda(float orig0,
                                                               float orig1,
                                                               float orig2,
                                                               float bounds0_min,
                                                               float bounds1_min,
                                                               float bounds2_min,
                                                               float bounds0_max,
                                                               float bounds1_max,
                                                               float bounds2_max,
                                                               float rdir0,
                                                               float rdir1,
                                                               float rdir2,
                                                               float* t1,
                                                               float* t2);



/**
 * @brief Calculate the TOF bins along an LOR to which a voxel contributes
 *
 * @param x_m0   0-component of center of LOR
 * @param x_m1   1-component of center of LOR
 * @param x_m2   2-component of center of LOR
 * @param x_v0   0-component of voxel
 * @param x_v1   1-component of voxel
 * @param x_v2   2-component of voxel
 * @param u0     0-component of unit vector that points from start to end of LOR 
 * @param u1     1-component of unit vector that points from start to end of LOR 
 * @param u2     2-component of unit vector that points from start to end of LOR 
 * @param tofbin_width      width of the TOF bins in spatial units
 * @param tofcenter_offset  offset of the central tofbin from the midpoint of the LOR in spatial units
 * @param sigma_tof         TOF resolution in spatial coordinates
 * @param n_sigmas          number of sigmas considered to be relevant
 * @param n_half            n_tofbins // 2
 * @param it1 (output)      lower relevant tof bin
 * @param it2 (output)      upper relevant tof bin
 */
extern "C" __device__ void relevant_tof_bins_cuda(float x_m0,
                                                  float x_m1, 
                                                  float x_m2, 
                                                  float x_v0, 
                                                  float x_v1, 
                                                  float x_v2, 
                                                  float u0,
                                                  float u1,
                                                  float u2,
                                                  float tofbin_width,
                                                  float tofcenter_offset,
                                                  float sigma_tof,
                                                  float n_sigmas,
                                                  int n_half,
                                                  int *it1,
                                                  int *it2);



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
 *  @param lor_dependent_sigma_tof unsigned char 0 or 1
 *                                 1 means that the TOF sigmas are LOR dependent
 *                                 any other value means that the first value in the sigma_tof
 *                                 array is used for all LORs
 *  @param lor_dependent_tofcenter_offset unsigned char 0 or 1
 *                                        1 means that the TOF center offsets are LOR dependent
 *                                        any other value means that the first value in the 
 *                                        tofcenter_offset array is used for all LORs
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
 *  @param lor_dependent_sigma_tof unsigned char 0 or 1
 *                                 1 means that the TOF sigmas are LOR dependent
 *                                 any other value means that the first value in the sigma_tof
 *                                 array is used for all LORs
 *  @param lor_dependent_tofcenter_offset unsigned char 0 or 1
 *                                        1 means that the TOF center offsets are LOR dependent
 *                                        any other value means that the first value in the 
 *                                        tofcenter_offset array is used for all LORs
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
 *  @param tof_bin          signed integer array with the tofbin of the events
 *                          the center of TOF bin 0 is assumed to be at the center of the LOR
 *                          (shifted by the tofcenter_offset)
 *  @param lor_dependent_sigma_tof unsigned char 0 or 1
 *                                 1 means that the TOF sigmas are LOR dependent
 *                                 any other value means that the first value in the sigma_tof
 *                                 array is used for all LORs
 *  @param lor_dependent_tofcenter_offset unsigned char 0 or 1
 *                                        1 means that the TOF center offsets are LOR dependent
 *                                        any other value means that the first value in the 
 *                                        tofcenter_offset array is used for all LORs
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
 *  @param tof_bin          signed integer array with the tofbin of the events
 *                          the center of TOF bin 0 is assumed to be at the center of the LOR
 *                          (shifted by the tofcenter_offset)
 *  @param lor_dependent_sigma_tof unsigned char 0 or 1
 *                                 1 means that the TOF sigmas are LOR dependent
 *                                 any other value means that the first value in the sigma_tof
 *                                 array is used for all LORs
 *  @param lor_dependent_tofcenter_offset unsigned char 0 or 1
 *                                        1 means that the TOF center offsets are LOR dependent
 *                                        any other value means that the first value in the 
 *                                        tofcenter_offset array is used for all LORs
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
