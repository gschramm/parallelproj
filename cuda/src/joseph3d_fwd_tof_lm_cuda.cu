/**
 * @file joseph3d_fwd_tof_lm_cuda.cu
 */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "projector_kernels.h"

extern "C" void joseph3d_fwd_tof_lm_cuda(const float *h_xstart, 
                                         const float *h_xend, 
                                         float **d_img,
                                         const float *h_img_origin, 
                                         const float *h_voxsize, 
                                         float *h_p,
                                         long long nlors, 
                                         const int *h_img_dim,
                                         float tofbin_width,
                                         const float *h_sigma_tof,
                                         const float *h_tofcenter_offset,
                                         float n_sigmas,
                                         const short *h_tof_bin,
                                         unsigned char lor_dependent_sigma_tof,
                                         unsigned char lor_dependent_tofcenter_offset,
                                         int threadsperblock)
{
  // get number of avilable CUDA devices specified as <=0 in input
  int num_devices;
  cudaGetDeviceCount(&num_devices);

  // init the dynamic arrays of device arrays
  float **d_p              = new float * [num_devices];
  float **d_xstart         = new float * [num_devices];
  float **d_xend           = new float * [num_devices];
  float **d_img_origin     = new float * [num_devices];
  float **d_voxsize        = new float * [num_devices];
  int   **d_img_dim        = new int * [num_devices];

  // init the dynamic arrays of TOF device arrays
  float **d_sigma_tof        = new float * [num_devices];
  float **d_tofcenter_offset = new float * [num_devices];
  short **d_tof_bin          = new short * [num_devices];

  // we split the projections across all CUDA devices
  # pragma omp parallel for schedule(static)
  for (int i_dev = 0; i_dev < num_devices; i_dev++) 
  {
    cudaError_t error;  
    int blockspergrid;

    dim3 block(threadsperblock);

    // offset for chunk of projections passed to a device 
    long long dev_offset;
    // number of projections to be calculated on a device
    long long dev_nlors;

    long long proj_bytes_dev;

    cudaSetDevice(i_dev);
    // () are important in integer division!
    dev_offset = i_dev*(nlors/num_devices);
 
    // calculate the number of projections for a device (last chunck can be a bit bigger) 
    dev_nlors = i_dev == (num_devices - 1) ? (nlors - dev_offset) : (nlors/num_devices);

    // calculate the number of bytes for the projection array on the device
    proj_bytes_dev = dev_nlors*sizeof(float);

    // calculate the number of blocks needed for every device (chunk)
    blockspergrid = (int)ceil((float)dev_nlors / threadsperblock);
    dim3 grid(blockspergrid);

    // allocate the memory for the array containing the projection on the device
    error = cudaMalloc(&d_p[i_dev], proj_bytes_dev);
    if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);}
    cudaMemsetAsync(d_p[i_dev], 0, proj_bytes_dev);

    error = cudaMalloc(&d_xstart[i_dev], 3*proj_bytes_dev);
    if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);}
    cudaMemcpyAsync(d_xstart[i_dev], h_xstart + 3*dev_offset, 3*proj_bytes_dev, 
                    cudaMemcpyHostToDevice);

    error = cudaMalloc(&d_xend[i_dev], 3*proj_bytes_dev);
    if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);}
    cudaMemcpyAsync(d_xend[i_dev], h_xend + 3*dev_offset, 3*proj_bytes_dev, 
                    cudaMemcpyHostToDevice);
   
    error = cudaMalloc(&d_img_origin[i_dev], 3*sizeof(float));
    if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);}
    cudaMemcpyAsync(d_img_origin[i_dev], h_img_origin, 3*sizeof(float), 
                    cudaMemcpyHostToDevice);

    error = cudaMalloc(&d_voxsize[i_dev], 3*sizeof(float));
    if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);}
    cudaMemcpyAsync(d_voxsize[i_dev], h_voxsize, 3*sizeof(float), cudaMemcpyHostToDevice);

    error = cudaMalloc(&d_img_dim[i_dev], 3*sizeof(int));
    if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);}
    cudaMemcpyAsync(d_img_dim[i_dev], h_img_dim, 3*sizeof(int), cudaMemcpyHostToDevice);

    // send TOF arrays to device

    if (lor_dependent_sigma_tof == 1){
      error = cudaMalloc(&d_sigma_tof[i_dev], proj_bytes_dev);
      if (error != cudaSuccess){
          printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
          exit(EXIT_FAILURE);}
      cudaMemcpyAsync(d_sigma_tof[i_dev], h_sigma_tof + dev_offset, proj_bytes_dev, cudaMemcpyHostToDevice);
    }
    else{
      error = cudaMalloc(&d_sigma_tof[i_dev], sizeof(float));
      if (error != cudaSuccess){
          printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
          exit(EXIT_FAILURE);}
      cudaMemcpyAsync(d_sigma_tof[i_dev], h_sigma_tof, sizeof(float), cudaMemcpyHostToDevice);
    }

    if (lor_dependent_tofcenter_offset == 1){
      error = cudaMalloc(&d_tofcenter_offset[i_dev], proj_bytes_dev);
      if (error != cudaSuccess){
          printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
          exit(EXIT_FAILURE);}
      cudaMemcpyAsync(d_tofcenter_offset[i_dev], h_tofcenter_offset + dev_offset, proj_bytes_dev, cudaMemcpyHostToDevice);
    }
    else{
      error = cudaMalloc(&d_tofcenter_offset[i_dev], sizeof(float));
      if (error != cudaSuccess){
          printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
          exit(EXIT_FAILURE);}
      cudaMemcpyAsync(d_tofcenter_offset[i_dev], h_tofcenter_offset, sizeof(float), cudaMemcpyHostToDevice);
    }


    error = cudaMalloc(&d_tof_bin[i_dev], dev_nlors*sizeof(short));
    if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);}
    cudaMemcpyAsync(d_tof_bin[i_dev], h_tof_bin + dev_offset, dev_nlors*sizeof(short), cudaMemcpyHostToDevice);

    // call the kernel
    joseph3d_fwd_tof_lm_cuda_kernel<<<grid,block>>>(d_xstart[i_dev], d_xend[i_dev], d_img[i_dev], 
                                                    d_img_origin[i_dev], d_voxsize[i_dev], 
                                                    d_p[i_dev], dev_nlors, d_img_dim[i_dev],
                                                    tofbin_width, d_sigma_tof[i_dev],
                                                    d_tofcenter_offset[i_dev], n_sigmas, d_tof_bin[i_dev], 
                                                    lor_dependent_sigma_tof, lor_dependent_tofcenter_offset);

    // use pinned memory for projetion host array which speeds up memory transfer
    cudaHostRegister(h_p + dev_offset, proj_bytes_dev, cudaHostRegisterDefault); 

    // copy projection back from device to host
    cudaMemcpyAsync(h_p + dev_offset, d_p[i_dev], proj_bytes_dev, cudaMemcpyDeviceToHost);

    // unpin memory
    cudaHostUnregister(h_p + dev_offset); 

    // deallocate memory on device
    cudaFree(d_p[i_dev]);
    cudaFree(d_xstart[i_dev]);
    cudaFree(d_xend[i_dev]);
    cudaFree(d_img_origin[i_dev]);
    cudaFree(d_img_dim[i_dev]);
    cudaFree(d_voxsize[i_dev]);

    // deallocate TOF memory on device
    cudaFree(d_sigma_tof[i_dev]);
    cudaFree(d_tofcenter_offset[i_dev]);
    cudaFree(d_tof_bin[i_dev]);
  }

  // make sure that all devices are done before leaving
  cudaDeviceSynchronize();
}
