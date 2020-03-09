#ifndef __TOF_UTILS_CUDA_H__
#define __TOF_UTILS_CUDA_H__

extern "C" __device__ float erff_as_cuda(float x);

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
		                                   unsigned int n_sigmas,
		                                   unsigned int n_half,
		                                   int *it1,
		                                   int *it2);
#endif
