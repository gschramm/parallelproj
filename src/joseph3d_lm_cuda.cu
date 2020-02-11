#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<cuda_runtime.h>

__global__ void joseph3d_lm_cuda_kernel(float *xstart, 
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
  //  3D listmode non-tof joseph forward projector cuda kernel
  //
  //  Parameters
  //  ----------
  //  xstart, xend : 1d float device arrays of shape [3*nlors]
  //    with the coordinates of the start / end points of the LORs.
  //    The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2 
  //    The end   coordinates of the n-th LOR are at xend[n*3 + i]   with i = 0,1,2 
  //
  //  img : 1d float device array [n0*n1*n2]
  //    containing the 3D image to be projected.
  //    The pixel [i,j,k] ist stored at [n1*n2+i + n2*k + j].
  //
  //  img_origin : 1d float device array [x0_0,x0_1,x0_2]
  //    coordinates of the center of the [0,0,0] voxel
  //
  //  voxsize : 1d float device array [vs0, vs1, vs2]
  //    the voxel size
  //
  //  p : 1d float device array of length np (output)
  //    used to store the projections
  //
  //  np : unsigned long long
  //    number of projections (length of p array)
  //
  //  n0, n1, n2 : unsigned int
  //    dimension of input img array

  unsigned long long i = blockDim.x * blockIdx.x + threadIdx.x;

  float d0, d1, d2, d0_sq, d1_sq, d2_sq; 
  float lsq, cos0_sq, cos1_sq, cos2_sq;
  unsigned short direction; 
  unsigned int i0, i1, i2;
  int i0_floor, i1_floor, i2_floor;
  int i0_ceil, i1_ceil, i2_ceil;
  float x_pr0, x_pr1, x_pr2;
  float tmp_0, tmp_1, tmp_2;

  if(i < np)
  {
    
    // initialize projected value to 0 
    p[i] = 0;

    // test whether the ray between the two detectors is most parallel
    // with the 0, 1, or 2 axis
    d0 = xend[i*3 + 0] - xstart[i*3 + 0];
    d1 = xend[i*3 + 1] - xstart[i*3 + 1];
    d2 = xend[i*3 + 2] - xstart[i*3 + 2];

    d0_sq = d0*d0;
    d1_sq = d1*d1;
    d2_sq = d2*d2;

    lsq = d0_sq + d1_sq + d2_sq;

    cos0_sq = d0_sq / lsq;
    cos1_sq = d1_sq / lsq;
    cos2_sq = d2_sq / lsq;

    direction = 0;
    if ((cos1_sq >= cos0_sq) && (cos1_sq >= cos2_sq))
    {
      direction = 1;
    }
    else
    {
      if ((cos2_sq >= cos0_sq) && (cos2_sq >= cos1_sq))
      {
        direction = 2;
      }
    }
 
    if (direction == 0)
    {
      // case where ray is most parallel to the 0 axis
      // we step through the volume along the 0 direction
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

        // do bilinear interpolation 
        if ((i1_floor >= 0) && (i1_floor < n1) && (i2_floor >= 0) && (i2_floor < n2))
        {
          p[i] += img[n1*n2*i0 + n2*i1_floor + i2_floor] * (1 - tmp_1) * (1 - tmp_2);
        }
        if ((i1_ceil >= 0) && (i1_ceil < n1) && (i2_floor >= 0) && (i2_floor < n2))
        {
          p[i] += img[n1*n2*i0 + n2*i1_ceil + i2_floor] * tmp_1 * (1 - tmp_2);
        }
        if ((i1_floor >= 0) && (i1_floor < n1) && (i2_ceil >= 0) && (i2_ceil < n2))
        {
          p[i] += img[n1*n2*i0 + n2*i1_floor + i2_ceil] * (1 - tmp_1) * tmp_2;
        }
        if ((i1_ceil >= 0) && (i1_ceil < n1) && (i2_ceil >= 0) && (i2_ceil < n2))
        {
          p[i] += img[n1*n2*i0 + n2*i1_ceil + i2_ceil] * tmp_1 * tmp_2;
        }
      }
      // correct for |cos(theta)| 
      p[i] /= sqrt(cos0_sq);
    }

    //--------------------------------------------------------------------------------- 
    if (direction == 1)
    {
      // case where ray is most parallel to the 1 axis
      // we step through the volume along the 1 direction
      for (i1 = 0; i1 < n1; i1++)
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
          p[i] += img[n1*n2*i0_floor +  n2*i1 + i2_floor] * (1 - tmp_0) * (1 - tmp_2);
        }
        if ((i0_ceil >= 0) && (i0_ceil < n0) && (i2_floor >= 0) && (i2_floor < n2))
        {
          p[i] += img[n1*n2*i0_ceil + n2*i1 + i2_floor] * tmp_0 * (1 - tmp_2);
        }
        if ((i0_floor >= 0) && (i0_floor < n0) && (i2_ceil >= 0) && (i2_ceil < n2))
        {
          p[i] += img[n1*n2*i0_floor + n2*i1 + i2_ceil] * (1 - tmp_0) * tmp_2;
        }
        if ((i0_ceil >= 0) && (i0_ceil < n0) && (i2_ceil >= 0) && (i2_ceil < n2))
        {
          p[i] += img[n1*n2*i0_ceil + n2*i1 + i2_ceil] * tmp_0 * tmp_2;
        }
      }
      // correct for |cos(theta)| 
      p[i] /= sqrt(cos1_sq);
    }

    //--------------------------------------------------------------------------------- 
    if (direction == 2)
    {
      // case where ray is most parallel to the 2 axis
      // we step through the volume along the 2 direction

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
          p[i] += img[n1*n2*i0_floor + n2*i1_floor + i2] * (1 - tmp_0) * (1 - tmp_1);
        }
        if ((i0_ceil >= 0) && (i0_ceil < n0) && (i1_floor >= 0) && (i1_floor < n1))
        {
          p[i] += img[n1*n2*i0_ceil + n2*i1_floor + i2] * tmp_0 * (1 - tmp_1);
        }
        if ((i0_floor >= 0) && (i0_floor < n0) && (i1_ceil >= 0) & (i1_ceil < n1))
        {
          p[i] += img[n1*n2*i0_floor + n2*i1_ceil + i2] * (1 - tmp_0) * tmp_1;
        }
        if ((i0_ceil >= 0) && (i0_ceil < n0) && (i1_ceil >= 0) && (i1_ceil < n1))
        {
          p[i] += img[n1*n2*i0_ceil + n2*i1_ceil + i2] * tmp_0 * tmp_1;
        }
      }
      // correct for |cos(theta)| 
      p[i] /= sqrt(cos2_sq);
    }
    // correct for the voxsize
    p[i] *= voxsize[direction];
  }
}


//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------

extern "C" void joseph3d_lm_cuda(float *h_xstart, 
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
  //  3D listmode non-tof joseph forward projector cuda wrapper
  //
  //  Parameters
  //  ----------
  //  h_xstart, h_xend : 1d float arrays of shape [3*nlors]
  //    with the coordinates of the start / end points of the LORs.
  //    The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2 
  //    The end   coordinates of the n-th LOR are at xend[n*3 + i]   with i = 0,1,2 
  //
  //  h_img : 1d float array [n0*n1*n2]
  //    containing the 3D image to be projected.
  //    The pixel [i,j,k] ist stored at [n1*n2+i + n2*k + j].
  //
  //  h_img_origin : 1d float array [x0_0,x0_1,x0_2]
  //    coordinates of the center of the [0,0,0] voxel
  //
  //  h_voxsize : 1d float  array [vs0, vs1, vs2]
  //    the voxel size
  //
  //  h_p : 1d float array of length np (output)
  //    used to store the projections
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

  unsigned int blockspergrid;

  dim3 block(threadsperblock);

  // offset for chunk of projections passed to a device 
  unsigned long long dev_offset;
  // number of projections to be calculated on a device
  unsigned long long dev_nproj;

  unsigned long long img_bytes = (n0*n1*n2)*sizeof(float);
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
    cudaMemsetAsync(d_p[i_dev], 0, proj_bytes_dev);

    cudaMalloc(&d_xstart[i_dev], 3*proj_bytes_dev);
    cudaMemcpyAsync(d_xstart[i_dev], h_xstart + 3*dev_offset, 3*proj_bytes_dev, 
                    cudaMemcpyHostToDevice);

    cudaMalloc(&d_xend[i_dev], 3*proj_bytes_dev);
    cudaMemcpyAsync(d_xend[i_dev], h_xend + 3*dev_offset, 3*proj_bytes_dev, 
                    cudaMemcpyHostToDevice);
   
    cudaMalloc(&d_img[i_dev], img_bytes);
    cudaMemcpyAsync(d_img[i_dev], h_img, img_bytes, cudaMemcpyHostToDevice);

    cudaMalloc(&d_img_origin[i_dev], 3*sizeof(float));
    cudaMemcpyAsync(d_img_origin[i_dev], h_img_origin, 3*sizeof(float), 
                    cudaMemcpyHostToDevice);

    cudaMalloc(&d_voxsize[i_dev], 3*sizeof(float));
    cudaMemcpyAsync(d_voxsize[i_dev], h_voxsize, 3*sizeof(float), cudaMemcpyHostToDevice);


    // call the kernel
    joseph3d_lm_cuda_kernel<<<grid,block>>>(d_xstart[i_dev], d_xend[i_dev], d_img[i_dev], 
                                            d_img_origin[i_dev], d_voxsize[i_dev], 
                                            d_p[i_dev], dev_nproj, n0, n1, n2); 

    // copy projection back from device to host
    cudaMemcpyAsync(h_p + dev_offset, d_p[i_dev], proj_bytes_dev, cudaMemcpyDeviceToHost);

    // deallocate memory on device
    cudaFree(d_p[i_dev]);
    cudaFree(d_xstart[i_dev]);
    cudaFree(d_xend[i_dev]);
    cudaFree(d_img);
    cudaFree(d_img_origin);
    cudaFree(d_voxsize);
  }

  // make sure that all devices are done before leaving
  for (unsigned int i_dev = 0; i_dev < num_devices; i_dev++){cudaDeviceSynchronize();}
}
