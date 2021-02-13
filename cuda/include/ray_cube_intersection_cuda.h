#ifndef __PARALLELPROJ_CUDA_RAY_CUBE_INTERSECTION_CUDA_H__
#define __PARALLELPROJ_CUDA_RAY_CUBE_INTERSECTION_CUDA_H__

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
#endif
