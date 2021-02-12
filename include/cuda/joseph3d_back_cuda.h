#ifndef JOSEPH3D_BACK_CUDA 
#define JOSEPH3D_BACK_CUDA

extern "C" void joseph3d_back_cuda(const float *h_xstart, 
                                   const float *h_xend, 
                                   float *h_img,
                                   const float *h_img_origin, 
                                   const float *h_voxsize, 
                                   const float *h_p,
                                   long long nlors, 
                                   const int *h_img_dim, 
                                   int threadsperblock,
                                   int num_devices);

#endif
