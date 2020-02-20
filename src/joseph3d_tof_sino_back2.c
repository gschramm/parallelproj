/**
 * @file joseph3d_tof_sino_back2.c
 */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<omp.h>

#include "tof_utils.h"

/** @brief 3D sinogram tof joseph back projector
 *
 *  @param xstart array of shape [3*nlors] with the coordinates of the start points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2 
 *  @param xend   array of shape [3*nlors] with the coordinates of the end   points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2 
 *  @param img    array of shape [n0*n1*n2] containing the 3D image used for back projection (output).
 *                The pixel [i,j,k] ist stored at [n1*n2+i + n2*k + j].
 *  @param img_origin  array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
 *  @param voxsize     array [vs0, vs1, vs2] of the voxel sizes
 *  @param p           array of length nlors with the values to be back projected
 *  @param nlors       number of geometrical LORs
 *  @param img_dim     array with dimensions of image [n0,n1,n2]
 *  @param n_tofbins        number of TOF bins
 *  @param tofbin_width     width of the TOF bins in spatial units (units of xstart and xend)
 *  @param sigma_tof        array of length nlors with the TOF resolution (sigma) for each LOR in
 *                          spatial units (units of xstart and xend) 
 *  @param tofcenter_offset array of length nlors with the offset of the central TOF bin from the 
 *                          midpoint of each LOR in spatial units (units of xstart and xend) 
 *  @param n_sigmas         number of sigmas to consider for calculation of TOF kernel
 *  @param half_erf_lut     look up table length 6001 for half erf between -3 and 3. 
 *                          The i-th element contains 0.5*erf(-3 + 0.001*i)
 */
void joseph3d_tof_sino_back2(float *xstart, 
                             float *xend, 
                             float *img,
                             float *img_origin, 
                             float *voxsize,
                             float *p, 
                             long long nlors, 
                             unsigned int *img_dim,
		                         int n_tofbins,
		                         float tofbin_width,
		                         float *sigma_tof,
		                         float *tofcenter_offset,
		                         unsigned int n_sigmas,
                             float *half_erf_lut)
{
  long long i;

  unsigned int n0 = img_dim[0];
  unsigned int n1 = img_dim[1];
  unsigned int n2 = img_dim[2];

  int n_half = n_tofbins/2;

  # pragma omp parallel for schedule(static)
  for(i = 0; i < nlors; i++)
  {
    float d0, d1, d2, d0_sq, d1_sq, d2_sq;
    float cs0, cs1, cs2, cf; 
    float lsq, cos0_sq, cos1_sq, cos2_sq;
    unsigned short direction; 
    unsigned int i0, i1, i2;
    int i0_floor, i1_floor, i2_floor;
    int i0_ceil, i1_ceil, i2_ceil;
    float x_pr0, x_pr1, x_pr2;
    float tmp_0, tmp_1, tmp_2;
   
    float u0, u1, u2, d_norm;
    float x_m0, x_m1, x_m2;    
    float x_v0, x_v1, x_v2;    

    int it, it1, it2;
    float tw;

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

    //---------------------------------------------------------
    //--- calculate TOF related quantities
    
    // unit vector (u0,u1,u2) that points from xstart to end
    d_norm = sqrt(lsq);
    u0 = d0 / d_norm; 
    u1 = d1 / d_norm; 
    u2 = d2 / d_norm; 

    // calculate mid point of LOR
    x_m0 = 0.5*(xstart[i*3 + 0] + xend[i*3 + 0]);
    x_m1 = 0.5*(xstart[i*3 + 1] + xend[i*3 + 1]);
    x_m2 = 0.5*(xstart[i*3 + 2] + xend[i*3 + 2]);

    //---------------------------------------------------------


    if(direction == 0)
    {
      // case where ray is most parallel to the 0 axis
      // we step through the volume along the 0 direction

      // factor for correctiong voxel size and |cos(theta)|
      cf = voxsize[direction]/cs0;

      for(i0 = 0; i0 < n0; i0++)
      {
        // get the indices where the ray intersects the image plane
        x_pr1 = xstart[i*3 + 1] + (img_origin[direction] + i0*voxsize[direction] - xstart[i*3 + direction])*d1 / d0;
        x_pr2 = xstart[i*3 + 2] + (img_origin[direction] + i0*voxsize[direction] - xstart[i*3 + direction])*d2 / d0;
  
        i1_floor = (int)floor((x_pr1 - img_origin[1])/voxsize[1]);
        i1_ceil  = i1_floor + 1; 
  
        i2_floor = (int)floor((x_pr2 - img_origin[2])/voxsize[2]);
        i2_ceil  = i2_floor + 1; 
  
        // calculate the distances to the floor normalized to [0,1]
        // for the bilinear interpolation
        tmp_1 = (x_pr1 - (i1_floor*voxsize[1] + img_origin[1])) / voxsize[1];
        tmp_2 = (x_pr2 - (i2_floor*voxsize[2] + img_origin[2])) / voxsize[2];

        //--------- TOF related quantities
        // calculate the voxel center needed for TOF weights
        x_v0 = img_origin[0] + i0*voxsize[0];
        x_v1 = x_pr1;
        x_v2 = x_pr2;

	      it1 = -n_half;
	      it2 =  n_half;

        // get the relevant tof bins (the TOF bins where the TOF weight is not close to 0)
        relevant_tof_bins(x_m0, x_m1, x_m2, x_v0, x_v1, x_v2, u0, u1, u2, 
			                    tofbin_width, tofcenter_offset[i], sigma_tof[i], n_sigmas, n_half,
		                      &it1, &it2);
        
        for(it = it1; it <= it2; it++){
          if(p[i*n_tofbins + it + n_half] != 0){
            tw = tof_weight(x_m0, x_m1, x_m2, x_v0, x_v1, x_v2, u0, u1, u2, it, 
		                       tofbin_width, tofcenter_offset[i], sigma_tof[i], half_erf_lut);

            if ((i1_floor >= 0) && (i1_floor < n1) && (i2_floor >= 0) && (i2_floor < n2))
            {
              #pragma omp atomic
              img[n1*n2*i0 + n2*i1_floor + i2_floor] += (tw * p[i*n_tofbins + it + n_half] * (1 - tmp_1) * 
                                                         (1 - tmp_2) * cf);
            }
            if ((i1_ceil >= 0) && (i1_ceil < n1) && (i2_floor >= 0) && (i2_floor < n2))
            {
              #pragma omp atomic
              img[n1*n2*i0 + n2*i1_ceil + i2_floor] += (tw * p[i*n_tofbins + it + n_half] * tmp_1 * 
                                                        (1 - tmp_2) * cf);
            }
            if ((i1_floor >= 0) && (i1_floor < n1) && (i2_ceil >= 0) && (i2_ceil < n2))
            {
              #pragma omp atomic
              img[n1*n2*i0 + n2*i1_floor + i2_ceil] += (tw * p[i*n_tofbins + it + n_half] * (1 - tmp_1) * 
                                                        tmp_2*cf);
            }
            if ((i1_ceil >= 0) && (i1_ceil < n1) && (i2_ceil >= 0) && (i2_ceil < n2))
            {
              #pragma omp atomic
              img[n1*n2*i0 + n2*i1_ceil + i2_ceil] += (tw * p[i*n_tofbins + it + n_half] * tmp_1 * 
                                                       tmp_2 * cf);
            }
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
      cf = voxsize[direction]/cs1;

      for(i1 = 0; i1 < n1; i1++)
      {
        // get the indices where the ray intersects the image plane
        x_pr0 = xstart[i*3 + 0] + (img_origin[direction] + i1*voxsize[direction] - xstart[i*3 + direction])*d0 / d1;
        x_pr2 = xstart[i*3 + 2] + (img_origin[direction] + i1*voxsize[direction] - xstart[i*3 + direction])*d2 / d1;
  
        i0_floor = (int)floor((x_pr0 - img_origin[0])/voxsize[0]);
        i0_ceil  = i0_floor + 1; 
  
        i2_floor = (int)floor((x_pr2 - img_origin[2])/voxsize[2]);
        i2_ceil  = i2_floor + 1; 
  
        // calculate the distances to the floor normalized to [0,1]
        // for the bilinear interpolation
        tmp_0 = (x_pr0 - (i0_floor*voxsize[0] + img_origin[0])) / voxsize[0];
        tmp_2 = (x_pr2 - (i2_floor*voxsize[2] + img_origin[2])) / voxsize[2];
  

        //--------- TOF related quantities
        // calculate the voxel center needed for TOF weights
        x_v0 = x_pr0;
        x_v1 = img_origin[1] + i1*voxsize[1];
        x_v2 = x_pr2;

	      it1 = -n_half;
	      it2 =  n_half;

        // get the relevant tof bins (the TOF bins where the TOF weight is not close to 0)
        relevant_tof_bins(x_m0, x_m1, x_m2, x_v0, x_v1, x_v2, u0, u1, u2, 
			                    tofbin_width, tofcenter_offset[i], sigma_tof[i], n_sigmas, n_half,
		                      &it1, &it2);

        for(it = it1; it <= it2; it++){
          if(p[i*n_tofbins + it + n_half] != 0){
            tw = tof_weight(x_m0, x_m1, x_m2, x_v0, x_v1, x_v2, u0, u1, u2, it, 
		                       tofbin_width, tofcenter_offset[i], sigma_tof[i], half_erf_lut);

            if ((i0_floor >= 0) && (i0_floor < n0) && (i2_floor >= 0) && (i2_floor < n2)) 
            {
              #pragma omp atomic
              img[n1*n2*i0_floor + n2*i1 + i2_floor] += (tw * p[i*n_tofbins + it + n_half] * (1 - tmp_0) * 
                                                         (1 - tmp_2) * cf);
            }
            if ((i0_ceil >= 0) && (i0_ceil < n0) && (i2_floor >= 0) && (i2_floor < n2))
            {
              #pragma omp atomic
              img[n1*n2*i0_ceil + n2*i1 + i2_floor] += (tw * p[i*n_tofbins + it + n_half] * tmp_0 * 
                                                        (1 - tmp_2) * cf);
            }
            if ((i0_floor >= 0) && (i0_floor < n0) && (i2_ceil >= 0) && (i2_ceil < n2))
            {
              #pragma omp atomic
              img[n1*n2*i0_floor + n2*i1 + i2_ceil] += (tw * p[i*n_tofbins + it + n_half] * (1 - tmp_0) * 
                                                        tmp_2 * cf);
            }
            if((i0_ceil >= 0) && (i0_ceil < n0) && (i2_ceil >= 0) && (i2_ceil < n2))
            {
              #pragma omp atomic
              img[n1*n2*i0_ceil + n2*i1 + i2_ceil] += (tw * p[i*n_tofbins + it + n_half] * tmp_0 * 
                                                       tmp_2 * cf);
            }
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
      cf = voxsize[direction]/cs2;
  
      for(i2 = 0; i2 < n2; i2++)
      {
        // get the indices where the ray intersects the image plane
        x_pr0 = xstart[i*3 + 0] + (img_origin[direction] + i2*voxsize[direction] - xstart[i*3 + direction])*d0 / d2;
        x_pr1 = xstart[i*3 + 1] + (img_origin[direction] + i2*voxsize[direction] - xstart[i*3 + direction])*d1 / d2;
  
        i0_floor = (int)floor((x_pr0 - img_origin[0])/voxsize[0]);
        i0_ceil  = i0_floor + 1; 
  
        i1_floor = (int)floor((x_pr1 - img_origin[1])/voxsize[1]);
        i1_ceil  = i1_floor + 1; 
  
        // calculate the distances to the floor normalized to [0,1]
        // for the bilinear interpolation
        tmp_0 = (x_pr0 - (i0_floor*voxsize[0] + img_origin[0])) / voxsize[0];
        tmp_1 = (x_pr1 - (i1_floor*voxsize[1] + img_origin[1])) / voxsize[1];
  

        //--------- TOF related quantities
        // calculate the voxel center needed for TOF weights
        x_v0 = x_pr0;
        x_v1 = x_pr1;
        x_v2 = img_origin[2] + i2*voxsize[2];

	      it1 = -n_half;
	      it2 =  n_half;

        // get the relevant tof bins (the TOF bins where the TOF weight is not close to 0)
        relevant_tof_bins(x_m0, x_m1, x_m2, x_v0, x_v1, x_v2, u0, u1, u2, 
			                    tofbin_width, tofcenter_offset[i], sigma_tof[i], n_sigmas, n_half,
		                      &it1, &it2);

        for(it = it1; it <= it2; it++){
          if(p[i*n_tofbins + it + n_half] != 0){
            tw = tof_weight(x_m0, x_m1, x_m2, x_v0, x_v1, x_v2, u0, u1, u2, it, 
		                       tofbin_width, tofcenter_offset[i], sigma_tof[i], half_erf_lut);

            if ((i0_floor >= 0) && (i0_floor < n0) && (i1_floor >= 0) && (i1_floor < n1))
            {
              #pragma omp atomic
              img[n1*n2*i0_floor +  n2*i1_floor + i2] += (tw * p[i*n_tofbins + it + n_half] * (1 - tmp_0) * 
                                                          (1 - tmp_1) * cf);
            }
            if ((i0_ceil >= 0) && (i0_ceil < n0) && (i1_floor >= 0) && (i1_floor < n1))
            {
              #pragma omp atomic
              img[n1*n2*i0_ceil + n2*i1_floor + i2] += (tw * p[i*n_tofbins + it + n_half] * tmp_0 * 
                                                        (1 - tmp_1) * cf);
            }
            if ((i0_floor >= 0) && (i0_floor < n0) && (i1_ceil >= 0) && (i1_ceil < n1))
            {
              #pragma omp atomic
              img[n1*n2*i0_floor + n2*i1_ceil + i2] += (tw * p[i*n_tofbins + it + n_half] * (1 - tmp_0) * 
                                                        tmp_1 * cf);
            }
            if ((i0_ceil >= 0) && (i0_ceil < n0) && (i1_ceil >= 0) && (i1_ceil < n1))
            {
              #pragma omp atomic
              img[n1*n2*i0_ceil + n2*i1_ceil + i2] += (tw * p[i*n_tofbins + it + n_half] * tmp_0 * 
                                                       tmp_1 * cf);
            }
          }
        }
      }
    }
  }
}
