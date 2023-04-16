/**
 * @file utils_cuda.cu
 */

#include <stdio.h>
#include <omp.h>

extern "C" __global__ void add_to_first_kernel(float *a, float *b, unsigned long long n)
{
  // add a vector b onto a vector a both of length n

  unsigned long long i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n)
  {
    a[i] += b[i];
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void print_int_device_array(int *a)
{
  unsigned long long i = blockDim.x * blockIdx.x + threadIdx.x;
  printf("%lld %d\n", i, a[i]);
}

//////////////////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void print_float_device_array(float *a)
{
  unsigned long long i = blockDim.x * blockIdx.x + threadIdx.x;
  printf("%lld %f\n", i, a[i]);
}

//////////////////////////////////////////////////////////////////////////////////////////
extern "C" float **copy_float_array_to_all_devices(const float *h_array, long long n)
{
  cudaError_t error;

  // get number of visible devices
  int num_devices;
  cudaGetDeviceCount(&num_devices);

  // create pointer to device arrays
  float **d_array = new float *[num_devices];

  long long array_bytes = n * sizeof(float);

#pragma omp parallel for schedule(static)
  for (int i_dev = 0; i_dev < num_devices; i_dev++)
  {
    cudaSetDevice(i_dev);

    error = cudaMalloc(&d_array[i_dev], array_bytes);
    if (error != cudaSuccess)
    {
      printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error),
             error, __LINE__);
      exit(EXIT_FAILURE);
    }
    cudaMemcpyAsync(d_array[i_dev], h_array, array_bytes, cudaMemcpyHostToDevice);
  }

  // make sure that all devices are done before leaving
  cudaDeviceSynchronize();

  return d_array;
}

////////////////////////////////////////////////////////////////////////////////////////////
extern "C" void free_float_array_on_all_devices(float **d_array)
{

  // get number of avilable CUDA devices specified as <=0 in input
  int num_devices;
  cudaGetDeviceCount(&num_devices);

#pragma omp parallel for schedule(static)
  for (int i_dev = 0; i_dev < num_devices; i_dev++)
  {
    cudaFree(d_array[i_dev]);
  }

  // make sure that all devices are done before leaving
  cudaDeviceSynchronize();
}

////////////////////////////////////////////////////////////////////////////////////////////
extern "C" void sum_float_arrays_on_first_device(float **d_array, long long n)
{

  cudaError_t error;
  int threadsperblock = 32;
  dim3 block(threadsperblock);
  int blockspergrid = (int)ceil((float)n / threadsperblock);
  dim3 grid(blockspergrid);

  int num_devices;
  cudaGetDeviceCount(&num_devices);

  float *d_array2;

  long long array_bytes = n * sizeof(float);

  if (num_devices > 1)
  {
    cudaSetDevice(0);

    for (int i_dev = 0; i_dev < num_devices; i_dev++)
    {
      if (i_dev == 0)
      {
        // allocate memory for aux array to sum arrays on device 0
        error = cudaMalloc(&d_array2, array_bytes);
        if (error != cudaSuccess)
        {
          printf("cudaMalloc returned error %s (code %d), line(%d)\n",
                 cudaGetErrorString(error), error, __LINE__);
          exit(EXIT_FAILURE);
        }
      }

      else
      {
        // copy array from device i_dev to device 0
        cudaMemcpyPeer(d_array2, 0, d_array[i_dev], i_dev, array_bytes);

        // call summation kernel to add d_array2 to d_array on device 0
        add_to_first_kernel<<<grid, block>>>(d_array[0], d_array2, n);
      }

      cudaDeviceSynchronize();
    }

    cudaFree(d_array2);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////
extern "C" void get_float_array_from_device(float **d_array, long long n, int i_dev, float *h_array)
{
  cudaSetDevice(i_dev);
  cudaMemcpy(h_array, d_array[i_dev], n * sizeof(float), cudaMemcpyDeviceToHost);
}

////////////////////////////////////////////////////////////////////////////////////////////
extern "C" int get_cuda_device_count()
{
  int num_devices = 0;
  cudaError_t err = cudaGetDeviceCount(&num_devices);

  if (err != cudaSuccess)
    num_devices = 0;

  return num_devices;
}