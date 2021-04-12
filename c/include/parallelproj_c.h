/**
 * @file parallelproj_c.h
 */

#ifndef __PARALLELPROJ_C_H__
#define __PARALLELPROJ_C_H__

#ifdef  __cplusplus
extern "C" {
#endif

/** @brief 3D non-tof joseph back projector
 *
 *  All threads back project in one image using openmp's atomic add.
 *
 *  @param xstart array of shape [3*nlors] with the coordinates of the start points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2.
 *                Units are the ones of voxsize.
 *  @param xend   array of shape [3*nlors] with the coordinates of the end   points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2.
 *                Units are the ones of voxsize.
 *  @param img    array of shape [n0*n1*n2] containing the 3D image used for back projection (output).
 *                The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
 *                !! values are added to existing array !!
 *  @param img_origin  array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
 *  @param voxsize     array [vs0, vs1, vs2] of the voxel sizes
 *  @param p           array of length nlors with the values to be back projected
 *  @param nlors       number of geometrical LORs
 *  @param img_dim     array with dimensions of image [n0,n1,n2]
 */
void joseph3d_back_2(const float *xstart, 
                     const float *xend, 
                     float *img,
                     const float *img_origin, 
                     const float *voxsize,
                     const float *p, 
                     long long nlors, 
                     const int *img_dim);

/** @brief 3D non-tof joseph back projector
 *
 *  Each thread projects into a separate image. At the end all images are summed.
 *
 *  @param xstart array of shape [3*nlors] with the coordinates of the start points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                Units are the ones of voxsize.
 *  @param xend   array of shape [3*nlors] with the coordinates of the end   points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                Units are the ones of voxsize.
 *  @param img    array of shape [n0*n1*n2] containing the 3D image used for back projection (output).
 *                The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
 *                !! values are added to existing array !!
 *  @param img_origin  array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
 *  @param voxsize     array [vs0, vs1, vs2] of the voxel sizes
 *  @param p           array of length nlors with the values to be back projected
 *  @param nlors       number of geometrical LORs
 *  @param img_dim     array with dimensions of image [n0,n1,n2]
 */
void joseph3d_back(const float *xstart, 
                   const float *xend, 
                   float *img,
                   const float *img_origin, 
                   const float *voxsize,
                   const float *p, 
                   long long nlors, 
                   const int *img_dim);

/** @brief 3D listmode tof joseph back projector
 *
 *  All threads back project in one image using openmp's atomic add.
 *
 *  @param xstart array of shape [3*nlors] with the coordinates of the start points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                Units are the ones of voxsize.
 *  @param xend   array of shape [3*nlors] with the coordinates of the end   points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                Units are the ones of voxsize.
 *  @param img    array of shape [n0*n1*n2] containing the 3D image used for back projection (output).
 *                The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
 *                !! values are added to existing array !!
 *  @param img_origin  array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
 *  @param voxsize     array [vs0, vs1, vs2] of the voxel sizes
 *  @param p           array of length nlors with the values to be back projected
 *  @param nlors       number of geometrical LORs
 *  @param img_dim     array with dimensions of image [n0,n1,n2]
 *  @param tofbin_width     width of the TOF bins in spatial units (units of xstart and xend)
 *  @param sigma_tof        array of length nlors with the TOF resolution (sigma) for each LOR in
 *                          spatial units (units of xstart and xend) 
 *  @param tofcenter_offset array of length nlors with the offset of the central TOF bin from the 
 *                          midpoint of each LOR in spatial units (units of xstart and xend). 
 *                          A positive value means a shift towards the end point of the LOR.
 *  @param n_sigmas         number of sigmas to consider for calculation of TOF kernel
 *  @param tof_bin          array containing the TOF bin of each event
 */
void joseph3d_back_tof_lm_2(const float *xstart, 
                            const float *xend, 
                            float *img,
                            const float *img_origin, 
                            const float *voxsize,
                            const float *p, 
                            long long nlors, 
                            const int *img_dim,
                            float tofbin_width,
                            const float *sigma_tof,
                            const float *tofcenter_offset,
                            float n_sigmas,
                            const short *tof_bin);

/** @brief 3D listmode tof joseph back projector
 *
 *  Each thread projects into a separate image. At the end all images are summed.
 *
 *  @param xstart array of shape [3*nlors] with the coordinates of the start points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                Units are the ones of voxsize.
 *  @param xend   array of shape [3*nlors] with the coordinates of the end   points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                Units are the ones of voxsize.
 *  @param img    array of shape [n0*n1*n2] containing the 3D image used for back projection (output).
 *                The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
 *                !! values are added to existing array !!
 *  @param img_origin  array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
 *  @param voxsize     array [vs0, vs1, vs2] of the voxel sizes
 *  @param p           array of length nlors with the values to be back projected
 *  @param nlors       number of geometrical LORs
 *  @param img_dim     array with dimensions of image [n0,n1,n2]
 *  @param tofbin_width     width of the TOF bins in spatial units (units of xstart and xend)
 *  @param sigma_tof        array of length nlors with the TOF resolution (sigma) for each LOR in
 *                          spatial units (units of xstart and xend) 
 *  @param tofcenter_offset array of length nlors with the offset of the central TOF bin from the 
 *                          midpoint of each LOR in spatial units (units of xstart and xend). 
 *                          A positive value means a shift towards the end point of the LOR.
 *  @param n_sigmas         number of sigmas to consider for calculation of TOF kernel
 *  @param tof_bin          array containing the TOF bin of each event
 */
void joseph3d_back_tof_lm(const float *xstart, 
                         const float *xend, 
                         float *img,
                         const float *img_origin, 
                         const float *voxsize,
                         const float *p, 
                         long long nlors, 
                         const int *img_dim,
                         float tofbin_width,
                         const float *sigma_tof,
                         const float *tofcenter_offset,
                         float n_sigmas,
                         const short *tof_bin);

/** @brief 3D sinogram tof joseph back projector
 *
 *  All threads back project in one image using openmp's atomic add.
 *
 *  @param xstart array of shape [3*nlors] with the coordinates of the start points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                Units are the ones of voxsize.
 *  @param xend   array of shape [3*nlors] with the coordinates of the end   points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                Units are the ones of voxsize.
 *  @param img    array of shape [n0*n1*n2] containing the 3D image used for back projection (output).
 *                The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
 *                !! values are added to existing array !!
 *  @param img_origin  array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
 *  @param voxsize     array [vs0, vs1, vs2] of the voxel sizes
 *  @param p           array of length nlors with the values to be back projected
 *  @param nlors       number of geometrical LORs
 *  @param img_dim     array with dimensions of image [n0,n1,n2]
 *  @param tofbin_width     width of the TOF bins in spatial units (units of xstart and xend)
 *  @param sigma_tof        array of length nlors with the TOF resolution (sigma) for each LOR in
 *                          spatial units (units of xstart and xend) 
 *  @param tofcenter_offset array of length nlors with the offset of the central TOF bin from the 
 *                          midpoint of each LOR in spatial units (units of xstart and xend). 
 *                          A positive value means a shift towards the end point of the LOR.
 *  @param n_sigmas         number of sigmas to consider for calculation of TOF kernel
 *  @param n_tofbins        number of TOF bins
 */
void joseph3d_back_tof_sino_2(const float *xstart, 
                              const float *xend, 
                              float *img,
                              const float *img_origin, 
                              const float *voxsize,
                              const float *p, 
                              long long nlors, 
                              const int *img_dim,
                              float tofbin_width,
                              const float *sigma_tof,
                              const float *tofcenter_offset,
                              float n_sigmas,
                              short n_tofbins);

/** @brief 3D sinogram tof joseph back projector
 *
 *  Each thread projects into a separate image. At the end all images are summed.
 *
 *  @param xstart array of shape [3*nlors] with the coordinates of the start points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                Units are the ones of voxsize.
 *  @param xend   array of shape [3*nlors] with the coordinates of the end   points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2. 
 *                Units are the ones of voxsize.
 *  @param img    array of shape [n0*n1*n2] containing the 3D image used for back projection (output).
 *                The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
 *                !! values are added to existing array !!
 *  @param img_origin  array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
 *  @param voxsize     array [vs0, vs1, vs2] of the voxel sizes
 *  @param p           array of length nlors*n_tofbins with the values to be back projected
 *  @param nlors       number of geometrical LORs
 *  @param img_dim     array with dimensions of image [n0,n1,n2]
 *  @param tofbin_width     width of the TOF bins in spatial units (units of xstart and xend)
 *  @param sigma_tof        array of length nlors with the TOF resolution (sigma) for each LOR in
 *                          spatial units (units of xstart and xend) 
 *  @param tofcenter_offset array of length nlors with the offset of the central TOF bin from the 
 *                          midpoint of each LOR in spatial units (units of xstart and xend). 
 *                          A positive value means a shift towards the end point of the LOR.
 *  @param n_sigmas         number of sigmas to consider for calculation of TOF kernel
 *  @param n_tofbins        number of TOF bins
 */
void joseph3d_back_tof_sino(const float *xstart, 
                            const float *xend, 
                            float *img,
                            const float *img_origin, 
                            const float *voxsize,
                            const float *p, 
                            long long nlors, 
                            const int *img_dim,
                            float tofbin_width,
                            const float *sigma_tof,
                            const float *tofcenter_offset,
                            float n_sigmas,
                            short n_tofbins);

/** @brief 3D non-tof joseph forward projector
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
 */
void joseph3d_fwd(const float *xstart, 
                  const float *xend, 
                  const float *img,
                  const float *img_origin, 
                  const float *voxsize, 
                  float *p,
                  long long nlors, 
                  const int *img_dim);

/** @brief 3D listmode tof joseph forward projector
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
 *  @param sigma_tof        array of length nlors with the TOF resolution (sigma) for each LOR in
 *                          spatial units (units of xstart and xend) 
 *  @param tofcenter_offset array of length nlors with the offset of the central TOF bin from the 
 *                          midpoint of each LOR in spatial units (units of xstart and xend).
 *                          A positive value means a shift towards the end point of the LOR.
 *  @param n_sigmas         number of sigmas to consider for calculation of TOF kernel
 *  @param tof_bin          array containing the TOF bin of each event
 */
void joseph3d_fwd_tof_lm(const float *xstart, 
                        const float *xend, 
                        const float *img,
                        const float *img_origin, 
                        const float *voxsize, 
                        float *p,
                        long long nlors, 
                        const int *img_dim,
                        float tofbin_width,
                        const float *sigma_tof,
                        const float *tofcenter_offset,
                        float n_sigmas,
                        const short *tof_bin);

/** @brief 3D sinogram tof joseph forward projector
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
 *  @param tofbin_width     width of the TOF bins in spatial units (units of xstart and xend)
 *  @param sigma_tof        array of length nlors with the TOF resolution (sigma) for each LOR in
 *                          spatial units (units of xstart and xend) 
 *  @param tofcenter_offset array of length nlors with the offset of the central TOF bin from the 
 *                          midpoint of each LOR in spatial units (units of xstart and xend).
 *                          A positive value means a shift towards the end point of the LOR.
 *  @param n_sigmas         number of sigmas to consider for calculation of TOF kernel
 *  @param n_tofbins        number of TOF bins
 */
void joseph3d_fwd_tof_sino(const float *xstart, 
                           const float *xend, 
                           const float *img,
                           const float *img_origin, 
                           const float *voxsize, 
                           float *p,
                           long long nlors, 
                           const int *img_dim,
                           float tofbin_width,
                           const float *sigma_tof,
                           const float *tofcenter_offset,
                           float n_sigmas,
                           short n_tofbins);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif
