#ifndef JOSEPH3D_FWD_CUDA
#define JOSEPH3D_FWD_CUDA

extern "C" void joseph3d_fwd_cuda(const float *h_xstart, 
                                  const float *h_xend, 
                                  const float *h_img,
                                  const float *h_img_origin, 
                                  const float *h_voxsize, 
                                  float *h_p,
                                  long long nlors, 
                                  const int *h_img_dim,
                                  int threadsperblock,
                                  int num_devices);

#endif
