/**
 * @file utils_cuda.cu
 */

#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

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

extern "C" __global__ void print_int_device_array(int* a)
{
  unsigned long long i = blockDim.x * blockIdx.x + threadIdx.x;
  printf("%lld %d\n", i, a[i]);
}

extern "C" __global__ void print_float_device_array(float* a)
{
  unsigned long long i = blockDim.x * blockIdx.x + threadIdx.x;
  printf("%lld %f\n", i, a[i]);
}

//////////////////////////////////////////////////////////////////////////////////////////

extern "C" void copy_float_array_to_all_devices(const float *h_array,
                                                long long n,
                                                float **d_array)
{
  cudaError_t error;  

  // get number of visible devices
  int num_devices;
  cudaGetDeviceCount(&num_devices);  

  long long img_bytes = (n)*sizeof(float);

  # pragma omp parallel for schedule(static)
  for (int i_dev = 0; i_dev < num_devices; i_dev++) 
  {
    cudaSetDevice(i_dev);

    error = cudaMalloc(&d_array[i_dev], img_bytes);
    if (error != cudaSuccess)
    {
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), 
               error, __LINE__);
        exit(EXIT_FAILURE);
    }
    cudaMemcpyAsync(d_array[i_dev], h_array, img_bytes, cudaMemcpyHostToDevice);
  }

  // make sure that all devices are done before leaving
  cudaDeviceSynchronize();
}

////////////////////////////////////////////////////////////////////////////////////////////

extern "C" void free_float_array_on_all_devices(float **d_array)
{

  // get number of avilable CUDA devices specified as <=0 in input
  int num_devices;
  cudaGetDeviceCount(&num_devices);

  # pragma omp parallel for schedule(static)
  for (int i_dev = 0; i_dev < num_devices; i_dev++) 
  {
    cudaFree(d_array[i_dev]);
  }

  // make sure that all devices are done before leaving
  cudaDeviceSynchronize();
}
