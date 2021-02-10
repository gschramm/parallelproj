/**
 * @file joseph3d_back_tof_lm_cuda.cu
 */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#include "utils_cuda.h"
#include "ray_cube_intersection_cuda.h"

/** @brief 3D listmode tof cuda joseph back projector kernel
 *
 *  @param xstart array of shape [3*nlors] with the coordinates of the start points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2 
 *  @param xend   array of shape [3*nlors] with the coordinates of the end   points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2 
 *  @param img    array of shape [n0*n1*n2] containing the 3D image used for back projection (output).
 *                The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
 *  @param img_origin  array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
 *  @param voxsize     array [vs0, vs1, vs2] of the voxel sizes
 *  @param p           array of length nlors with the values to be back projected
 *  @param nlors       number of geometrical LORs
 *  @param img_dim     array with dimensions of image [n0,n1,n2]
 *  @param tofbin_width     width of the TOF bins in spatial units (units of xstart and xend)
 *  @param sigma_tof        array of length nlors with the TOF resolution (sigma) for each LOR in
 *                          spatial units (units of xstart and xend) 
 *  @param tofcenter_offset array of length nlors with the offset of the central TOF bin from the 
 *                          midpoint of each LOR in spatial units (units of xstart and xend) 
 *  @param n_sigmas         number of sigmas to consider for calculation of TOF kernel
 *  @param tof_bin          array containing the TOF bin of each event
 */
__global__ void joseph3d_back_tof_lm_cuda_kernel(float *xstart, 
                                                 float *xend, 
                                                 float *img,
                                                 float *img_origin, 
                                                 float *voxsize,
                                                 float *p, 
                                                 long long nlors, 
                                                 int *img_dim,
                                                 float tofbin_width,
                                                 float *sigma_tof,
                                                 float *tofcenter_offset,
                                                 float n_sigmas,
                                                 short *tof_bin)
{
  long long i = blockDim.x * blockIdx.x + threadIdx.x;
  //long long i = blockIdx.x + threadIdx.x * gridDim.x;

  int n0 = img_dim[0];
  int n1 = img_dim[1];
  int n2 = img_dim[2];

  if(i < nlors)
  {
    float d0, d1, d2, d0_sq, d1_sq, d2_sq;
    float cs0, cs1, cs2, cf; 
    float lsq, cos0_sq, cos1_sq, cos2_sq;
    unsigned short direction; 
    int i0, i1, i2;
    int i0_floor, i1_floor, i2_floor;
    int i0_ceil, i1_ceil, i2_ceil;
    float x_pr0, x_pr1, x_pr2;
    float tmp_0, tmp_1, tmp_2;
   
    float u0, u1, u2, d_norm;
    float x_m0, x_m1, x_m2;    
    float x_v0, x_v1, x_v2;    

    int it = tof_bin[i];
    float dtof, tw;

    float sig_tof   = sigma_tof[i];
    float tc_offset = tofcenter_offset[i];

    float xstart0 = xstart[i*3 + 0];
    float xstart1 = xstart[i*3 + 1];
    float xstart2 = xstart[i*3 + 2];

    float xend0 = xend[i*3 + 0];
    float xend1 = xend[i*3 + 1];
    float xend2 = xend[i*3 + 2];

    float voxsize0 = voxsize[0];
    float voxsize1 = voxsize[1];
    float voxsize2 = voxsize[2];

    float img_origin0 = img_origin[0];
    float img_origin1 = img_origin[1];
    float img_origin2 = img_origin[2];

    unsigned char intersec;
    float t1, t2;
    float istart_f, iend_f, tmp;
    int   istart, iend;

    float istart_tof_f, iend_tof_f;
    int   istart_tof, iend_tof;

    // test whether the ray between the two detectors is most parallel
    // with the 0, 1, or 2 axis
    d0    = xend0 - xstart0;
    d1    = xend1 - xstart1;
    d2    = xend2 - xstart2;
  
    //-----------
    //--- test whether ray and cube intersect
    intersec = ray_cube_intersection_cuda(xstart0, xstart1, xstart2, 
                                          img_origin0 - 1*voxsize0, img_origin1 - 1*voxsize1, img_origin2 - 1*voxsize2,
                                          img_origin0 + n0*voxsize0, img_origin1 + n1*voxsize1, img_origin2 + n2*voxsize2,
                                          d0, d1, d2, &t1, &t2);

    if (intersec == 1)
    {
      d0_sq = d0*d0; 
      d1_sq = d1*d1;
      d2_sq = d2*d2;
      
      lsq = d0_sq + d1_sq + d2_sq;
      
      cos0_sq = d0_sq / lsq;
      cos1_sq = d1_sq / lsq;
      cos2_sq = d2_sq / lsq;

      cs0 = sqrtf(cos0_sq); 
      cs1 = sqrtf(cos1_sq); 
      cs2 = sqrtf(cos2_sq); 
      
      direction = 0;
      if ((cos1_sq >= cos0_sq) && (cos1_sq >= cos2_sq))
      {
        direction = 1;
      }
      if ((cos2_sq >= cos0_sq) && (cos2_sq >= cos1_sq))
      {
        direction = 2;
      }

      //---------------------------------------------------------
      //--- calculate TOF related quantities
      
      // unit vector (u0,u1,u2) that points from xstart to end
      d_norm = sqrtf(lsq);
      u0 = d0 / d_norm; 
      u1 = d1 / d_norm; 
      u2 = d2 / d_norm; 

      // calculate mid point of LOR
      x_m0 = 0.5f*(xstart0 + xend0);
      x_m1 = 0.5f*(xstart1 + xend1);
      x_m2 = 0.5f*(xstart2 + xend2);

      //---------------------------------------------------------


      if(direction == 0)
      {
        // case where ray is most parallel to the 0 axis
        // we step through the volume along the 0 direction

        // factor for correctiong voxel size and |cos(theta)|
        cf = voxsize0/cs0;

        //--- check where ray enters / leaves cube
        istart_f = (xstart0 + t1*d0 - img_origin0) / voxsize0;
        iend_f   = (xstart0 + t2*d0 - img_origin0) / voxsize0;

        if (istart_f > iend_f){
          tmp      = iend_f;
          iend_f   = istart_f;
          istart_f = tmp;
        }
    
        istart = (int)floor(istart_f);
        iend   = (int)ceil(iend_f);

        //-- check where we should start and stop according to the TOF kernel
        //-- the tof weights outside +- 3 sigma will be close to 0 so we can
        //-- ignore them         
        istart_tof_f = (x_m0 + (it*tofbin_width - n_sigmas*sig_tof)*u0 - img_origin0) / voxsize0;
        iend_tof_f   = (x_m0 + (it*tofbin_width + n_sigmas*sig_tof)*u0 - img_origin0) / voxsize0;
        
        if (istart_tof_f > iend_tof_f){
          tmp        = iend_tof_f;
          iend_tof_f = istart_tof_f;
          istart_tof_f = tmp;
        }

        istart_tof = (int)floor(istart_tof_f);
        iend_tof   = (int)ceil(iend_tof_f);

        if(istart_tof > istart){istart = istart_tof;}
        if(iend_tof   < iend){iend = iend_tof;}
        //-----------

        if (istart < 0){istart = 0;}
        if (iend >= n0){iend = n0;}
        //---

        for(i0 = istart; i0 < iend; i0++)
        {
          // get the indices where the ray intersects the image plane
          x_pr1 = xstart1 + (img_origin0 + i0*voxsize0 - xstart0)*d1 / d0;
          x_pr2 = xstart2 + (img_origin0 + i0*voxsize0 - xstart0)*d2 / d0;
  
          i1_floor = (int)floor((x_pr1 - img_origin1)/voxsize1);
          i1_ceil  = i1_floor + 1; 
  
          i2_floor = (int)floor((x_pr2 - img_origin2)/voxsize2);
          i2_ceil  = i2_floor + 1; 
  
          // calculate the distances to the floor normalized to [0,1]
          // for the bilinear interpolation
          tmp_1 = (x_pr1 - (i1_floor*voxsize1 + img_origin1)) / voxsize1;
          tmp_2 = (x_pr2 - (i2_floor*voxsize2 + img_origin2)) / voxsize2;

          //--------- TOF related quantities
          // calculate the voxel center needed for TOF weights
          x_v0 = img_origin0 + i0*voxsize0;
          x_v1 = x_pr1;
          x_v2 = x_pr2;

          if(p[i] != 0){
            // calculate distance of voxel to tof bin center
            dtof = sqrtf(powf((x_m0 + (it*tofbin_width + tc_offset)*u0 - x_v0), 2) + 
                         powf((x_m1 + (it*tofbin_width + tc_offset)*u1 - x_v1), 2) + 
                         powf((x_m2 + (it*tofbin_width + tc_offset)*u2 - x_v2), 2));

            //calculate the TOF weight
            tw = 0.5f*(erff((dtof + 0.5f*tofbin_width)/(sqrtf(2)*sig_tof)) - 
                      erff((dtof - 0.5f*tofbin_width)/(sqrtf(2)*sig_tof)));

            if ((i1_floor >= 0) && (i1_floor < n1) && (i2_floor >= 0) && (i2_floor < n2))
            {
              atomicAdd(img + n1*n2*i0 + n2*i1_floor + i2_floor, 
                        (tw * p[i] * (1 - tmp_1) * (1 - tmp_2) * cf));
            }
            if ((i1_ceil >= 0) && (i1_ceil < n1) && (i2_floor >= 0) && (i2_floor < n2))
            {
              atomicAdd(img + n1*n2*i0 + n2*i1_ceil + i2_floor, 
                        (tw * p[i] * tmp_1 * (1 - tmp_2) * cf));
            }
            if ((i1_floor >= 0) && (i1_floor < n1) && (i2_ceil >= 0) && (i2_ceil < n2))
            {
              atomicAdd(img + n1*n2*i0 + n2*i1_floor + i2_ceil, 
                        (tw * p[i] * (1 - tmp_1) * tmp_2*cf));
            }
            if ((i1_ceil >= 0) && (i1_ceil < n1) && (i2_ceil >= 0) && (i2_ceil < n2))
            {
              atomicAdd(img + n1*n2*i0 + n2*i1_ceil + i2_ceil, 
                        (tw * p[i] * tmp_1 * tmp_2 * cf));
            }
          }
        }
      }  
      // --------------------------------------------------------------------------------- 
      if(direction == 1)
      {
        // case where ray is most parallel to the 1 axis
        // we step through the volume along the 1 direction
  
        // factor for correctiong voxel size and |cos(theta)|
        cf = voxsize1/cs1;

        //--- check where ray enters / leaves cube
        istart_f = (xstart1 + t1*d1 - img_origin1) / voxsize1;
        iend_f   = (xstart1 + t2*d1 - img_origin1) / voxsize1;

        if (istart_f > iend_f){
          tmp      = iend_f;
          iend_f   = istart_f;
          istart_f = tmp;
        }
    
        istart = (int)floor(istart_f);
        iend   = (int)ceil(iend_f);

        //-- check where we should start and stop according to the TOF kernel
        //-- the tof weights outside +- 3 sigma will be close to 0 so we can
        //-- ignore them         
        istart_tof_f = (x_m1 + (it*tofbin_width - n_sigmas*sig_tof)*u1 - img_origin1) / voxsize1;
        iend_tof_f   = (x_m1 + (it*tofbin_width + n_sigmas*sig_tof)*u1 - img_origin1) / voxsize1;
        
        if (istart_tof_f > iend_tof_f){
          tmp        = iend_tof_f;
          iend_tof_f = istart_tof_f;
          istart_tof_f = tmp;
        }

        istart_tof = (int)floor(istart_tof_f);
        iend_tof   = (int)ceil(iend_tof_f);

        if(istart_tof > istart){istart = istart_tof;}
        if(iend_tof   < iend){iend = iend_tof;}
        //-----------


        if (istart < 0){istart = 0;}
        if (iend >= n1){iend = n1;}
        //---

        for(i1 = istart; i1 < iend; i1++)
        {
          // get the indices where the ray intersects the image plane
          x_pr0 = xstart0 + (img_origin1 + i1*voxsize1 - xstart1)*d0 / d1;
          x_pr2 = xstart2 + (img_origin1 + i1*voxsize1 - xstart1)*d2 / d1;
  
          i0_floor = (int)floor((x_pr0 - img_origin0)/voxsize0);
          i0_ceil  = i0_floor + 1; 
  
          i2_floor = (int)floor((x_pr2 - img_origin2)/voxsize2);
          i2_ceil  = i2_floor + 1; 
  
          // calculate the distances to the floor normalized to [0,1]
          // for the bilinear interpolation
          tmp_0 = (x_pr0 - (i0_floor*voxsize0 + img_origin0)) / voxsize0;
          tmp_2 = (x_pr2 - (i2_floor*voxsize2 + img_origin2)) / voxsize2;
  

          //--------- TOF related quantities
          // calculate the voxel center needed for TOF weights
          x_v0 = x_pr0;
          x_v1 = img_origin1 + i1*voxsize1;
          x_v2 = x_pr2;

          if(p[i] != 0){
            // calculate distance of voxel to tof bin center
            dtof = sqrtf(powf((x_m0 + (it*tofbin_width + tc_offset)*u0 - x_v0), 2) + 
                         powf((x_m1 + (it*tofbin_width + tc_offset)*u1 - x_v1), 2) + 
                         powf((x_m2 + (it*tofbin_width + tc_offset)*u2 - x_v2), 2));

            //calculate the TOF weight
            tw = 0.5f*(erff((dtof + 0.5f*tofbin_width)/(sqrtf(2)*sig_tof)) - 
                      erff((dtof - 0.5f*tofbin_width)/(sqrtf(2)*sig_tof)));

            if ((i0_floor >= 0) && (i0_floor < n0) && (i2_floor >= 0) && (i2_floor < n2)) 
            {
              atomicAdd(img + n1*n2*i0_floor + n2*i1 + i2_floor, 
                        (tw * p[i] * (1 - tmp_0) * (1 - tmp_2) * cf));
            }
            if ((i0_ceil >= 0) && (i0_ceil < n0) && (i2_floor >= 0) && (i2_floor < n2))
            {
              atomicAdd(img + n1*n2*i0_ceil + n2*i1 + i2_floor, 
                        (tw * p[i] * tmp_0 * (1 - tmp_2) * cf));
            }
            if ((i0_floor >= 0) && (i0_floor < n0) && (i2_ceil >= 0) && (i2_ceil < n2))
            {
              atomicAdd(img + n1*n2*i0_floor + n2*i1 + i2_ceil, 
                        (tw * p[i] * (1 - tmp_0) * tmp_2 * cf));
            }
            if((i0_ceil >= 0) && (i0_ceil < n0) && (i2_ceil >= 0) && (i2_ceil < n2))
            {
              atomicAdd(img + n1*n2*i0_ceil + n2*i1 + i2_ceil, 
                        (tw * p[i] * tmp_0 * tmp_2 * cf));
            }
          }
        }
      }
      //--------------------------------------------------------------------------------- 
      if (direction == 2)
      {
        // case where ray is most parallel to the 2 axis
        // we step through the volume along the 2 direction
  
        // factor for correctiong voxel size and |cos(theta)|
        cf = voxsize2/cs2;
  
        //--- check where ray enters / leaves cube
        istart_f = (xstart2 + t1*d2 - img_origin2) / voxsize2;
        iend_f   = (xstart2 + t2*d2 - img_origin2) / voxsize2;

        if (istart_f > iend_f){
          tmp      = iend_f;
          iend_f   = istart_f;
          istart_f = tmp;
        }
    
        istart = (int)floor(istart_f);
        iend   = (int)ceil(iend_f);

        //-- check where we should start and stop according to the TOF kernel
        //-- the tof weights outside +- 3 sigma will be close to 0 so we can
        //-- ignore them         
        istart_tof_f = (x_m2 + (it*tofbin_width - n_sigmas*sig_tof)*u2 - img_origin2) / voxsize2;
        iend_tof_f   = (x_m2 + (it*tofbin_width + n_sigmas*sig_tof)*u2 - img_origin2) / voxsize2;
        
        if (istart_tof_f > iend_tof_f){
          tmp        = iend_tof_f;
          iend_tof_f = istart_tof_f;
          istart_tof_f = tmp;
        }

        istart_tof = (int)floor(istart_tof_f);
        iend_tof   = (int)ceil(iend_tof_f);

        if(istart_tof > istart){istart = istart_tof;}
        if(iend_tof   < iend){iend = iend_tof;}
        //-----------

        if (istart < 0){istart = 0;}
        if (iend >= n2){iend = n2;}
        //---

        for(i2 = istart; i2 < iend; i2++)
        {
          // get the indices where the ray intersects the image plane
          x_pr0 = xstart0 + (img_origin2 + i2*voxsize2 - xstart2)*d0 / d2;
          x_pr1 = xstart1 + (img_origin2 + i2*voxsize2 - xstart2)*d1 / d2;
  
          i0_floor = (int)floor((x_pr0 - img_origin0)/voxsize0);
          i0_ceil  = i0_floor + 1; 
  
          i1_floor = (int)floor((x_pr1 - img_origin1)/voxsize1);
          i1_ceil  = i1_floor + 1; 
  
          // calculate the distances to the floor normalized to [0,1]
          // for the bilinear interpolation
          tmp_0 = (x_pr0 - (i0_floor*voxsize0 + img_origin0)) / voxsize0;
          tmp_1 = (x_pr1 - (i1_floor*voxsize1 + img_origin1)) / voxsize1;
  

          //--------- TOF related quantities
          // calculate the voxel center needed for TOF weights
          x_v0 = x_pr0;
          x_v1 = x_pr1;
          x_v2 = img_origin2 + i2*voxsize2;

          if(p[i] != 0){
            // calculate distance of voxel to tof bin center
            dtof = sqrtf(powf((x_m0 + (it*tofbin_width + tc_offset)*u0 - x_v0), 2) + 
                         powf((x_m1 + (it*tofbin_width + tc_offset)*u1 - x_v1), 2) + 
                         powf((x_m2 + (it*tofbin_width + tc_offset)*u2 - x_v2), 2));

            //calculate the TOF weight
            tw = 0.5f*(erff((dtof + 0.5f*tofbin_width)/(sqrtf(2)*sig_tof)) - 
                      erff((dtof - 0.5f*tofbin_width)/(sqrtf(2)*sig_tof)));

            if ((i0_floor >= 0) && (i0_floor < n0) && (i1_floor >= 0) && (i1_floor < n1))
            {
              atomicAdd(img + n1*n2*i0_floor +  n2*i1_floor + i2, 
                        (tw * p[i] * (1 - tmp_0) * (1 - tmp_1) * cf));
            }
            if ((i0_ceil >= 0) && (i0_ceil < n0) && (i1_floor >= 0) && (i1_floor < n1))
            {
              atomicAdd(img + n1*n2*i0_ceil + n2*i1_floor + i2, 
                        (tw * p[i] * tmp_0 * (1 - tmp_1) * cf));
            }
            if ((i0_floor >= 0) && (i0_floor < n0) && (i1_ceil >= 0) && (i1_ceil < n1))
            {
              atomicAdd(img + n1*n2*i0_floor + n2*i1_ceil + i2, 
                        (tw * p[i] * (1 - tmp_0) * tmp_1 * cf));
            }
            if ((i0_ceil >= 0) && (i0_ceil < n0) && (i1_ceil >= 0) && (i1_ceil < n1))
            {
              atomicAdd(img + n1*n2*i0_ceil + n2*i1_ceil + i2, 
                        (tw * p[i] * tmp_0 * tmp_1 * cf));
            }
          }
        }
      }
    }
  }
}


//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------

/** @brief 3D listmode tof joseph back projector CUDA wrapper
 *
 *  The array to be back projected is split accross all CUDA devices.
 *  Each device backprojects in its own image. At the end all images are
 *  transfered to device 0 and summed there. It is therefore assumed that all devices used
 *  are interconnected.
 *
 *  @param h_xstart array of shape [3*nlors] with the coordinates of the start points of the LORs.
 *                  The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2 
 *  @param h_xend   array of shape [3*nlors] with the coordinates of the end   points of the LORs.
 *                  The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2 
 *  @param h_img    array of shape [n0*n1*n2] for the back projection image (output).
 *                  The pixel [i,j,k] ist stored at [n1*n2*i + n2*j + k].
 *  @param h_img_origin  array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
 *  @param h_voxsize     array [vs0, vs1, vs2] of the voxel sizes
 *  @param h_p           array of length nlors containg the values to be back projected
 *  @param nlors         number of projections (length of p array)
 *  @param h_img_dim     array with dimensions of image [n0,n1,n2]
 *  @param tofbin_width     width of the TOF bins in spatial units (units of xstart and xend)
 *  @param h_sigma_tof      array of length nlors with the TOF resolution (sigma) for each LOR in
 *                          spatial units (units of xstart and xend) 
 *  @param h_tofcenter_offset  array of length nlors with the offset of the central TOF bin from the 
 *                             midpoint of each LOR in spatial units (units of xstart and xend) 
 *  @param n_sigmas        number of sigmas to consider for calculation of TOF kernel
 *  @param h_tof_bin       array containing the TOF bin of each event
 *  @param threadsperblock number of threads per block
 *  @param num_devices     number of CUDA devices to use. if set to -1 cudaGetDeviceCount() is used
 */
extern "C" void joseph3d_back_tof_lm_cuda(const float *h_xstart,
                                          const float *h_xend,
                                          float *h_img,
                                          const float *h_img_origin,
                                          const float *h_voxsize,
                                          const float *h_p,
                                          long long nlors,
                                          const int *h_img_dim,
                                          float tofbin_width,
                                          const float *h_sigma_tof,
                                          const float *h_tofcenter_offset,
                                          float n_sigmas,
                                          const short *h_tof_bin,
                                          int threadsperblock,
                                          int num_devices)
{
  cudaError_t error;  
  int blockspergrid;

  dim3 block(threadsperblock);

  // offset for chunk of projections passed to a device 
  long long dev_offset;
  // number of projections to be calculated on a device
  long long dev_nlors;

  int n0 = h_img_dim[0];
  int n1 = h_img_dim[1];
  int n2 = h_img_dim[2];

  long long nimg_vox  = n0*n1*n2;
  long long img_bytes = nimg_vox*sizeof(float);
  long long proj_bytes_dev;

  // get number of avilable CUDA devices specified as <=0 in input
  if(num_devices <= 0){cudaGetDeviceCount(&num_devices);}  

  // init the dynamic array of device arrays
  float **d_p              = new float * [num_devices];
  float **d_xstart         = new float * [num_devices];
  float **d_xend           = new float * [num_devices];
  float **d_img            = new float * [num_devices];
  float **d_img_origin     = new float * [num_devices];
  float **d_voxsize        = new float * [num_devices];
  int   **d_img_dim        = new int * [num_devices];

  // init the dynamic arrays of TOF device arrays
  float **d_sigma_tof        = new float * [num_devices];
  float **d_tofcenter_offset = new float * [num_devices];
  short **d_tof_bin          = new short * [num_devices];

  // auxiallary image array needed to sum all back projections on device 0
  float *d_img2;

  // we split the projections across all CUDA devices
  for (int i_dev = 0; i_dev < num_devices; i_dev++) 
  {
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
    cudaMemcpyAsync(d_p[i_dev], h_p + dev_offset, proj_bytes_dev, cudaMemcpyHostToDevice);

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
  
    // initialize device image for back projection with 0s execpt for the last device 
    // we sent the input image to the last device to make sure that the back-projector
    // adds to it
    error = cudaMalloc(&d_img[i_dev], img_bytes);
    if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);}
    if(i_dev == (num_devices - 1)){
      cudaMemcpyAsync(d_img[i_dev], h_img, img_bytes,cudaMemcpyHostToDevice);
    }
    else{
      cudaMemsetAsync(d_img[i_dev], 0, img_bytes);
    }

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
    error = cudaMalloc(&d_sigma_tof[i_dev], proj_bytes_dev);
    if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);}
    cudaMemcpyAsync(d_sigma_tof[i_dev], h_sigma_tof + dev_offset, proj_bytes_dev, cudaMemcpyHostToDevice);

    error = cudaMalloc(&d_tofcenter_offset[i_dev], proj_bytes_dev);
    if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);}
    cudaMemcpyAsync(d_tofcenter_offset[i_dev], h_tofcenter_offset + dev_offset, proj_bytes_dev, 
                    cudaMemcpyHostToDevice);

    error = cudaMalloc(&d_tof_bin[i_dev], dev_nlors*sizeof(short));
    if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);}
    cudaMemcpyAsync(d_tof_bin[i_dev], h_tof_bin + dev_offset, dev_nlors*sizeof(short), cudaMemcpyHostToDevice);
    // call the kernel
    joseph3d_back_tof_lm_cuda_kernel<<<grid,block>>>(d_xstart[i_dev], d_xend[i_dev], d_img[i_dev],
                                                     d_img_origin[i_dev], d_voxsize[i_dev], 
                                                     d_p[i_dev], dev_nlors, d_img_dim[i_dev],
                                                     tofbin_width, d_sigma_tof[i_dev],
                                                     d_tofcenter_offset[i_dev], n_sigmas,
                                                     d_tof_bin[i_dev]);

  }

  // sum the backprojection images from all devices on device 0
  for (int i_dev = 0; i_dev < num_devices; i_dev++) 
  {
    cudaSetDevice(i_dev);
    cudaDeviceSynchronize();
 
    if(i_dev == 0){
      // allocate memory for aux array to sum back projections on device 0
      // in case we have multiple devices
      if(num_devices > 1){
        error = cudaMalloc(&d_img2, img_bytes);
        if (error != cudaSuccess){
          printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
          exit(EXIT_FAILURE);}
      }
    }
    else{
      // copy backprojection image from device i_dev to device 0
      cudaMemcpyPeer(d_img2, 0, d_img[i_dev], i_dev, img_bytes);

      cudaSetDevice(0);
      // call summation kernel here to add d_img2 to d_img2 on device 0
      blockspergrid = (int)ceil((float)nimg_vox / threadsperblock);
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
    cudaFree(d_img_origin[i_dev]);
    cudaFree(d_img_dim[i_dev]);
    cudaFree(d_voxsize[i_dev]);

    cudaFree(d_sigma_tof[i_dev]);
    cudaFree(d_tofcenter_offset[i_dev]);
    cudaFree(d_tof_bin[i_dev]);
  }

  // copy everything back to host 
  cudaSetDevice(0);
  cudaMemcpy(h_img, d_img[0], img_bytes, cudaMemcpyDeviceToHost);

  // deallocate device image array on device 0
  cudaFree(d_img[0]);
  if(num_devices > 1){cudaFree(d_img2);}

  cudaDeviceSynchronize();
}
