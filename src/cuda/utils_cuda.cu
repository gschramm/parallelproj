/**
 * @file utils_cuda.cu
 */

#include<stdio.h>
#include<stdlib.h>

/** @brief CUDA kernel to add array b to array a
 * 
 *  @param a first array of length n
 *  @param b first array of length n
 *  @param n length of vectors
 *
*/ 
extern "C" __global__ void add_to_first_kernel(float* a, float* b, unsigned long long n)
{
// add a vector b onto a vector a both of length n

  unsigned long long i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i < n)
  {
    a[i] += b[i];
  }
}



