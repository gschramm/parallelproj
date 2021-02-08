/**
 * @file joseph3d_back_2.c
 */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<omp.h>

#include "ray_cube_intersection.h"

/** @brief 3D non-tof joseph back projector
 *
 *  All threads back project in one image using openmp's atomic add.
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
 */
void joseph3d_back_2(float *xstart, 
                     float *xend, 
                     float *img,
                     float *img_origin, 
                     float *voxsize,
                     float *p, 
                     long long nlors, 
                     int *img_dim)
{
  long long i;

  int n0 = img_dim[0];
  int n1 = img_dim[1];
  int n2 = img_dim[2];

  # pragma omp parallel for schedule(static)
  for(i = 0; i < nlors; i++)
  {
    if(p[i] != 0)
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

      // test whether the ray between the two detectors is most parallel
      // with the 0, 1, or 2 axis
      d0    = xend0 - xstart0;
      d1    = xend1 - xstart1;
      d2    = xend2 - xstart2;
  
      //-----------
      //--- test whether ray and cube intersect
      intersec = ray_cube_intersection(xstart0, xstart1, xstart2, 
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
  
            if ((i1_floor >= 0) && (i1_floor < n1) && (i2_floor >= 0) && (i2_floor < n2))
            {
              #pragma omp atomic
              img[n1*n2*i0 + n2*i1_floor + i2_floor] += (p[i] * (1 - tmp_1) * (1 - tmp_2) * cf);
            }
            if ((i1_ceil >= 0) && (i1_ceil < n1) && (i2_floor >= 0) && (i2_floor < n2))
            {
              #pragma omp atomic
              img[n1*n2*i0 + n2*i1_ceil + i2_floor] += (p[i] * tmp_1 * (1 - tmp_2) * cf);
            }
            if ((i1_floor >= 0) && (i1_floor < n1) && (i2_ceil >= 0) && (i2_ceil < n2))
            {
              #pragma omp atomic
              img[n1*n2*i0 + n2*i1_floor + i2_ceil] += (p[i] * (1 - tmp_1) * tmp_2 * cf);
            }
            if ((i1_ceil >= 0) && (i1_ceil < n1) && (i2_ceil >= 0) && (i2_ceil < n2))
            {
              #pragma omp atomic
              img[n1*n2*i0 + n2*i1_ceil + i2_ceil] += (p[i] * tmp_1 * tmp_2 * cf);
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
  
            if ((i0_floor >= 0) && (i0_floor < n0) && (i2_floor >= 0) && (i2_floor < n2)) 
            {
              #pragma omp atomic
              img[n1*n2*i0_floor + n2*i1 + i2_floor] += (p[i] * (1 - tmp_0) * (1 - tmp_2) * cf);
            }
            if ((i0_ceil >= 0) && (i0_ceil < n0) && (i2_floor >= 0) && (i2_floor < n2))
            {
              #pragma omp atomic
              img[n1*n2*i0_ceil + n2*i1 + i2_floor] += (p[i] * tmp_0 * (1 - tmp_2) * cf);
            }
            if ((i0_floor >= 0) && (i0_floor < n0) && (i2_ceil >= 0) && (i2_ceil < n2))
            {
              #pragma omp atomic
              img[n1*n2*i0_floor + n2*i1 + i2_ceil] += (p[i] * (1 - tmp_0) * tmp_2 * cf);
            }
            if((i0_ceil >= 0) && (i0_ceil < n0) && (i2_ceil >= 0) && (i2_ceil < n2))
            {
              #pragma omp atomic
              img[n1*n2*i0_ceil + n2*i1 + i2_ceil] += (p[i] * tmp_0 * tmp_2 * cf);
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
  
            if ((i0_floor >= 0) && (i0_floor < n0) && (i1_floor >= 0) && (i1_floor < n1))
            {
              #pragma omp atomic
              img[n1*n2*i0_floor +  n2*i1_floor + i2] += (p[i] * (1 - tmp_0) * (1 - tmp_1) * cf);
            }
            if ((i0_ceil >= 0) && (i0_ceil < n0) && (i1_floor >= 0) && (i1_floor < n1))
            {
              #pragma omp atomic
              img[n1*n2*i0_ceil + n2*i1_floor + i2] += (p[i] * tmp_0 * (1 - tmp_1) * cf);
            }
            if ((i0_floor >= 0) && (i0_floor < n0) && (i1_ceil >= 0) && (i1_ceil < n1))
            {
              #pragma omp atomic
              img[n1*n2*i0_floor + n2*i1_ceil + i2] += (p[i] * (1 - tmp_0) * tmp_1 * cf);
            }
            if ((i0_ceil >= 0) && (i0_ceil < n0) && (i1_ceil >= 0) && (i1_ceil < n1))
            {
              #pragma omp atomic
              img[n1*n2*i0_ceil + n2*i1_ceil + i2] += (p[i] * tmp_0 * tmp_1 * cf);
            }
          }
        }
      }
    }
  }
}
