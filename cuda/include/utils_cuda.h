#ifndef __PARALLELPROJ_CUDA_UTILS_CUDA_H__
#define __PARALLELPROJ_CUDA_UTILS_CUDA_H__

/** @brief CUDA kernel to add array b to array a
 * 
 *  @param a first array of length n
 *  @param b first array of length n
 *  @param n length of vectors
 *
*/
extern "C" __global__ void add_to_first_kernel(float* a, float* b, unsigned long long n);

#endif

