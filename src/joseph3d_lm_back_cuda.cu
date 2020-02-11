#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<cuda_runtime.h>

__global__ void add_to_first_kernel(float* a, float* b, unsigned long long n)
{
// add a vector b onto a vector a both of length n

  unsigned long long i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i < n)
  {
    a[i] += b[i];
  }
}


//-----------------------------------------------------------------------
__global__ void joseph3d_lm_back_cuda_kernel(float *xstart, 
                                             float *xend, 
                                             float *img,
                                             float *img_origin, 
                                             float *voxsize, 
                                             float *p,              
                                             unsigned long long np,
                                             unsigned int n0, 
                                             unsigned int n1, 
                                             unsigned int n2)
{
  //  3D listmode non-tof joseph back projector cuda kernel
  //
  //  Parameters
  //  ----------
  //  xstart, xend : 1d float device arrays of shape [3*nlors]
  //    with the coordinates of the start / end points of the LORs.
  //    The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2 
  //    The end   coordinates of the n-th LOR are at xend[n*3 + i]   with i = 0,1,2 
  //
  //  img : 1d float device array [n0*n1*n2] used for the back projection (output)
  //    The pixel [i,j,k] ist stored at [n1*n2+i + n2*k + j].
  //
  //  img_origin : 1d float device array [x0_0,x0_1,x0_2]
  //    coordinates of the center of the [0,0,0] voxel
  //
  //  voxsize : 1d float device array [vs0, vs1, vs2]
  //    the voxel size
  //
  //  p : 1d float device array of length np
  //    containing the values to be back projected
  //
  //  np : unsigned long long
  //    number of projections (length of p array)
  //
  //  n0, n1, n2 : unsigned int
  //    dimension of input img array

  unsigned long long i = blockDim.x * blockIdx.x + threadIdx.x;

  float d0, d1, d2, d0_sq, d1_sq, d2_sq;
  float cs0, cs1, cs2, corfac; 
  float lsq, cos0_sq, cos1_sq, cos2_sq;
  unsigned short direction; 
  unsigned int i0, i1, i2;
  int i0_floor, i1_floor, i2_floor;
  int i0_ceil, i1_ceil, i2_ceil;
  float x_pr0, x_pr1, x_pr2;
  float tmp_0, tmp_1, tmp_2;
   

  if(i < np)
  {
    if(p[i] != 0)
    {
   
      // test whether the ray between the two detectors is most parallel
      // with the 0, 1, or 2 axis
      d0    = xend[i*3 + 0] - xstart[i*3 + 0];
      d1    = xend[i*3 + 1] - xstart[i*3 + 1];
      d2    = xend[i*3 + 2] - xstart[i*3 + 2];

      d0_sq = d0*d0; 
      d1_sq = d1*d1;
      d2_sq = d2*d2;
      
      lsq = d0_sq + d1_sq + d2_sq;
      
      cos0_sq = d0_sq / lsq;
      cos1_sq = d1_sq / lsq;
      cos2_sq = d2_sq / lsq;

      cs0 = sqrt(cos0_sq); 
      cs1 = sqrt(cos1_sq); 
      cs2 = sqrt(cos2_sq); 
      
      direction = 0;
      if ((cos1_sq >= cos0_sq) && (cos1_sq >= cos2_sq))
      {
        direction = 1;
      }
      if ((cos2_sq >= cos0_sq) && (cos2_sq >= cos1_sq))
      {
        direction = 2;
      }

      if(direction == 0)
      {
        // case where ray is most parallel to the 0 axis
        // we step through the volume along the 0 direction

        // factor for correctiong voxel size and |cos(theta)|
        corfac = voxsize[direction]/cs0;

        for(i0 = 0; i0 < n0; i0++)
        {
          // get the indices where the ray intersects the image plane
          x_pr1 = xstart[i*3 + 1] + (img_origin[direction] + i0*voxsize[direction] - 
                                     xstart[i*3 + direction])*d1 / d0;
          x_pr2 = xstart[i*3 + 2] + (img_origin[direction] + i0*voxsize[direction] - 
                                     xstart[i*3 + direction])*d2 / d0;
  
          i1_floor = (int)floor((x_pr1 - img_origin[1])/voxsize[1]);
          i1_ceil  = i1_floor + 1; 
  
          i2_floor = (int)floor((x_pr2 - img_origin[2])/voxsize[2]);
          i2_ceil  = i2_floor + 1; 
  
          // calculate the distances to the floor normalized to [0,1]
          // for the bilinear interpolation
          tmp_1 = (x_pr1 - (i1_floor*voxsize[1] + img_origin[1])) / voxsize[1];
          tmp_2 = (x_pr2 - (i2_floor*voxsize[2] + img_origin[2])) / voxsize[2];
  
          if ((i1_floor >= 0) && (i1_floor < n1) && (i2_floor >= 0) && (i2_floor < n2))
          {
            atomicAdd(img + n1*n2*i0 + n2*i1_floor + i2_floor, 
                      p[i] * (1 - tmp_1) * (1 - tmp_2) * corfac);
          }
          if ((i1_ceil >= 0) && (i1_ceil < n1) && (i2_floor >= 0) && (i2_floor < n2))
          {
            atomicAdd(img + n1*n2*i0 + n2*i1_ceil + i2_floor, 
                      p[i] * tmp_1 * (1 - tmp_2) * corfac);
          }
          if ((i1_floor >= 0) && (i1_floor < n1) && (i2_ceil >= 0) && (i2_ceil < n2))
          {
            atomicAdd(img + n1*n2*i0 + n2*i1_floor + i2_ceil, 
                      p[i] * (1 - tmp_1) * tmp_2*corfac);
          }
          if ((i1_ceil >= 0) && (i1_ceil < n1) && (i2_ceil >= 0) && (i2_ceil < n2))
          {
            atomicAdd(img + n1*n2*i0 + n2*i1_ceil + i2_ceil, p[i] * tmp_1 * tmp_2 * corfac);
          }
        }
      }  
      // --------------------------------------------------------------------------------- 
      if(direction == 1)
      {
        // case where ray is most parallel to the 1 axis
        // we step through the volume along the 1 direction
  
        // factor for correctiong voxel size and |cos(theta)|
        corfac = voxsize[direction]/cs1;

        for(i1 = 0; i1 < n1; i1++)
        {
          // get the indices where the ray intersects the image plane
          x_pr0 = xstart[i*3 + 0] + (img_origin[direction] + i1*voxsize[direction] - 
                                     xstart[i*3 + direction])*d0 / d1;
          x_pr2 = xstart[i*3 + 2] + (img_origin[direction] + i1*voxsize[direction] - 
                                     xstart[i*3 + direction])*d2 / d1;
  
          i0_floor = (int)floor((x_pr0 - img_origin[0])/voxsize[0]);
          i0_ceil  = i0_floor + 1; 
  
          i2_floor = (int)floor((x_pr2 - img_origin[2])/voxsize[2]);
          i2_ceil  = i2_floor + 1; 
  
          // calculate the distances to the floor normalized to [0,1]
          // for the bilinear interpolation
          tmp_0 = (x_pr0 - (i0_floor*voxsize[0] + img_origin[0])) / voxsize[0];
          tmp_2 = (x_pr2 - (i2_floor*voxsize[2] + img_origin[2])) / voxsize[2];
  
          if ((i0_floor >= 0) && (i0_floor < n0) && (i2_floor >= 0) && (i2_floor < n2)) 
          {
            atomicAdd(img + n1*n2*i0_floor + n2*i1 + i2_floor, 
                      p[i] * (1 - tmp_0) * (1 - tmp_2) * corfac);
          }
          if ((i0_ceil >= 0) && (i0_ceil < n0) && (i2_floor >= 0) && (i2_floor < n2))
          {
            atomicAdd(img + n1*n2*i0_ceil + n2*i1 + i2_floor, 
                      p[i] * tmp_0 * (1 - tmp_2) * corfac);
          }
          if ((i0_floor >= 0) && (i0_floor < n0) && (i2_ceil >= 0) && (i2_ceil < n2))
          {
            atomicAdd(img + n1*n2*i0_floor + n2*i1 + i2_ceil, 
                      p[i] * (1 - tmp_0) * tmp_2 * corfac);
          }
          if((i0_ceil >= 0) && (i0_ceil < n0) && (i2_ceil >= 0) && (i2_ceil < n2))
          {
            atomicAdd(img + n1*n2*i0_ceil + n2*i1 + i2_ceil, 
                      p[i] * tmp_0 * tmp_2 * corfac);
          }
        }
      }
      //--------------------------------------------------------------------------------- 
      if (direction == 2)
      {
        // case where ray is most parallel to the 2 axis
        // we step through the volume along the 2 direction
  
        // factor for correctiong voxel size and |cos(theta)|
        corfac = voxsize[direction]/cs2;
  
        for(i2 = 0; i2 < n2; i2++)
        {
          // get the indices where the ray intersects the image plane
          x_pr0 = xstart[i*3 + 0] + (img_origin[direction] + i2*voxsize[direction] - 
                                     xstart[i*3 + direction])*d0 / d2;
          x_pr1 = xstart[i*3 + 1] + (img_origin[direction] + i2*voxsize[direction] - 
                                     xstart[i*3 + direction])*d1 / d2;
  
          i0_floor = (int)floor((x_pr0 - img_origin[0])/voxsize[0]);
          i0_ceil  = i0_floor + 1; 
  
          i1_floor = (int)floor((x_pr1 - img_origin[1])/voxsize[1]);
          i1_ceil  = i1_floor + 1; 
  
          // calculate the distances to the floor normalized to [0,1]
          // for the bilinear interpolation
          tmp_0 = (x_pr0 - (i0_floor*voxsize[0] + img_origin[0])) / voxsize[0];
          tmp_1 = (x_pr1 - (i1_floor*voxsize[1] + img_origin[1])) / voxsize[1];
  
          if ((i0_floor >= 0) && (i0_floor < n0) && (i1_floor >= 0) && (i1_floor < n1))
          {
            atomicAdd(img + n1*n2*i0_floor +  n2*i1_floor + i2, 
                      p[i] * (1 - tmp_0) * (1 - tmp_1) * corfac);
          }
          if ((i0_ceil >= 0) && (i0_ceil < n0) && (i1_floor >= 0) && (i1_floor < n1))
          {
            atomicAdd(img + n1*n2*i0_ceil + n2*i1_floor + i2, 
                      p[i] * tmp_0 * (1 - tmp_1) * corfac);
          }
          if ((i0_floor >= 0) && (i0_floor < n0) && (i1_ceil >= 0) && (i1_ceil < n1))
          {
            atomicAdd(img + n1*n2*i0_floor + n2*i1_ceil + i2, 
                      p[i] * (1 - tmp_0) * tmp_1 * corfac);
          }
          if ((i0_ceil >= 0) && (i0_ceil < n0) && (i1_ceil >= 0) && (i1_ceil < n1))
          {
            atomicAdd(img + n1*n2*i0_ceil + n2*i1_ceil + i2, 
                      p[i] * tmp_0 * tmp_1 * corfac);
          }
        }
      }
    }
  }
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------

extern "C" void joseph3d_lm_back_cuda(float *h_xstart, 
                                      float *h_xend, 
                                      float *h_img,
                                      float *h_img_origin, 
                                      float *h_voxsize, 
                                      float *h_p,
                                      unsigned long long np, 
                                      unsigned int n0, 
                                      unsigned int n1, 
                                      unsigned int n2,
                                      unsigned int threadsperblock,
                                      int num_devices)
{

  //  3D listmode non-tof joseph back projector cuda kernel
  //
  //  Parameters
  //  ----------
  //  h_xstart, h_xend : 1d float arrays of shape [3*nlors]
  //    with the coordinates of the start / end points of the LORs.
  //    The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2 
  //    The end   coordinates of the n-th LOR are at xend[n*3 + i]   with i = 0,1,2 
  //
  //  h_img : 1d float array [n0*n1*n2] used for the back projection (output)
  //    The pixel [i,j,k] ist stored at [n1*n2+i + n2*k + j].
  //
  //  h_img_origin : 1d float array [x0_0,x0_1,x0_2]
  //    coordinates of the center of the [0,0,0] voxel
  //
  //  h_voxsize : 1d float array [vs0, vs1, vs2]
  //    the voxel size
  //
  //  h_p : 1d float array of length np
  //    containing the values to be back projected
  //
  //  np : unsigned long long
  //    number of projections (length of p array)
  //
  //  n0, n1, n2 : unsigned int
  //    dimension of input img array
  //
  //  threadsperblock : unsigned int
  //    number of threads per block
  //
  //  num_devices : int
  //    number of CUDA devices to use
  //    if this is < 0 than cudaGetDeviceCount() is used to determine
  //    the number of devices
  //
  //  Note
  //  ----
  //  The projectiona array to be back projected is split accross the CUDA devices.
  //  Each device backprojects in its own image. At the end the all images are
  //  transfered to device 0 and summed there. It is therefore assumed that all devices
  //  are interconnected.

  unsigned int blockspergrid;

  dim3 block(threadsperblock);

  // offset for chunk of projections passed to a device 
  unsigned long long dev_offset;
  // number of projections to be calculated on a device
  unsigned long long dev_nproj;

  unsigned long long nimg_vox  = n0*n1*n2;
  unsigned long long img_bytes = nimg_vox*sizeof(float);
  unsigned long long proj_bytes_dev;

  // get number of avilable CUDA devices specified as <=0 in input
  if(num_devices <= 0){cudaGetDeviceCount(&num_devices);}  

  // init the dynamic array of device arrays
  float **d_p          = new float * [num_devices];
  float **d_xstart     = new float * [num_devices];
  float **d_xend       = new float * [num_devices];
  float **d_img        = new float * [num_devices];
  float **d_img_origin = new float * [num_devices];
  float **d_voxsize    = new float * [num_devices];

  // auxiallary image array needed to sum all back projections on device 0
  float *d_img2;

  printf("\n # CUDA devices: %d \n", num_devices);

  // we split the projections across all CUDA devices
  for (unsigned int i_dev = 0; i_dev < num_devices; i_dev++) 
  {
    cudaSetDevice(i_dev);
    // () are important in integer division!
    dev_offset = i_dev*(np/num_devices);
 
    // calculate the number of projections for a device (last chunck can be a bit bigger) 
    dev_nproj = i_dev == (num_devices - 1) ? (np - dev_offset) : (np/num_devices);

    // calculate the number of bytes for the projection array on the device
    proj_bytes_dev = dev_nproj*sizeof(float);

    // calculate the number of blocks needed for every device (chunk)
    blockspergrid = (unsigned int)ceil((float)dev_nproj / threadsperblock);
    dim3 grid(blockspergrid);

    // allocate the memory for the array containing the projection on the device
    cudaMalloc(&d_p[i_dev], proj_bytes_dev);
    cudaMemcpyAsync(d_p[i_dev], h_p + dev_offset, proj_bytes_dev, cudaMemcpyHostToDevice);

    cudaMalloc(&d_xstart[i_dev], 3*proj_bytes_dev);
    cudaMemcpyAsync(d_xstart[i_dev], h_xstart + 3*dev_offset, 3*proj_bytes_dev, 
                    cudaMemcpyHostToDevice);

    cudaMalloc(&d_xend[i_dev], 3*proj_bytes_dev);
    cudaMemcpyAsync(d_xend[i_dev], h_xend + 3*dev_offset, 3*proj_bytes_dev, 
                    cudaMemcpyHostToDevice);
  
    // initialize device image for back projection with 0s execpt for the last device 
    // we sent the input image to the last device to make sure that the back-projector
    // adds to it
    cudaMalloc(&d_img[i_dev], img_bytes);
    if(i_dev == (num_devices - 1)){
      cudaMemcpyAsync(d_img[i_dev], h_img, img_bytes,cudaMemcpyHostToDevice);
    }
    else{
      cudaMemsetAsync(d_img[i_dev], 0, img_bytes);
    }

    cudaMalloc(&d_img_origin[i_dev], 3*sizeof(float));
    cudaMemcpyAsync(d_img_origin[i_dev], h_img_origin, 3*sizeof(float), 
                    cudaMemcpyHostToDevice);

    cudaMalloc(&d_voxsize[i_dev], 3*sizeof(float));
    cudaMemcpyAsync(d_voxsize[i_dev], h_voxsize, 3*sizeof(float), cudaMemcpyHostToDevice);

    // call the kernel
    joseph3d_lm_back_cuda_kernel<<<grid,block>>>(d_xstart[i_dev], d_xend[i_dev], 
                                                 d_img[i_dev], d_img_origin[i_dev], 
                                                 d_voxsize[i_dev], d_p[i_dev], 
                                                 dev_nproj, n0, n1, n2); 

  }

  // sum the backprojection images from all devices on device 0
  for (unsigned int i_dev = 0; i_dev < num_devices; i_dev++) 
  {
    cudaSetDevice(i_dev);
    cudaDeviceSynchronize();
 
    if(i_dev == 0){
      // allocate memory for aux array to sum back projections on device 0
      // in case we have multiple devices
      if(num_devices > 1){cudaMalloc(&d_img2, img_bytes);}
    }
    else{
      // copy backprojection image from device i_dev to device 0
      cudaMemcpyPeer(d_img2, 0, d_img[i_dev], i_dev, img_bytes);

      cudaSetDevice(0);
      // call summation kernel here to add d_img2 to d_img2 on device 0
      blockspergrid = (unsigned int)ceil((float)nimg_vox / threadsperblock);
      dim3 grid(blockspergrid);
      add_to_first_kernel<<<grid,block>>>(d_img[0], d_img2, nimg_vox);
      cudaDeviceSynchronize();

      cudaSetDevice(i_dev);
      cudaFree(d_img[i_dev]);
    }

    // deallocate memory on device
    cudaFree(d_p[i_dev]);
    cudaFree(d_xstart[i_dev]);
    cudaFree(d_xend[i_dev]);
    cudaFree(d_img_origin);
    cudaFree(d_voxsize);
  }

  // copy everything back to host 
  cudaSetDevice(0);
  cudaMemcpy(h_img, d_img[0], img_bytes, cudaMemcpyDeviceToHost);

  // deallocate device image array on device 0
  cudaFree(d_img[0]);
  if(num_devices > 1){cudaFree(d_img2);}

  cudaDeviceSynchronize();
}
