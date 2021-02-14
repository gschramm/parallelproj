#ifndef __PARALLELPROJ_H__
#define __PARALLELPROJ_H__

#ifdef  __cplusplus
extern "C" {
#endif

void joseph3d_back_2(const float *xstart, 
                     const float *xend, 
                     float *img,
                     const float *img_origin, 
                     const float *voxsize,
                     const float *p, 
                     long long nlors, 
                     const int *img_dim);

void joseph3d_back(const float *xstart, 
                   const float *xend, 
                   float *img,
                   const float *img_origin, 
                   const float *voxsize,
                   const float *p, 
                   long long nlors, 
                   const int *img_dim);

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

void joseph3d_fwd(const float *xstart, 
                  const float *xend, 
                  const float *img,
                  const float *img_origin, 
                  const float *voxsize, 
                  float *p,
                  long long nlors, 
                  const int *img_dim);

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

#ifdef  __cplusplus
} // end of extern "C"
#endif

#endif
