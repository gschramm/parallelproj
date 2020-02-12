/**
 * @file joseph3d_lm.c
 */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<omp.h>

/** @brief 3D listmode non-tof joseph forward projector
 *
 *  @param xstart array of shape [3*nlors] with the coordinates of the start points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2 
 *  @param xend   array of shape [3*nlors] with the coordinates of the end   points of the LORs.
 *                The start coordinates of the n-th LOR are at xstart[n*3 + i] with i = 0,1,2 
 *  @param img    array of shape [n0*n1*n2] containing the 3D image to be projected.
 *                The pixel [i,j,k] ist stored at [n1*n2+i + n2*k + j].
 *  @param img_origin  array [x0_0,x0_1,x0_2] of coordinates of the center of the [0,0,0] voxel
 *  @param voxsize     array [vs0, vs1, vs2] of the voxel sizes
 *  @param p           array of length np (output) used to store the projections
 *  @param img_dim     array with dimensions of image [n0,n1,n2]
 */
void joseph3d_lm(float *xstart, 
                 float *xend, 
                 float *img,
                 float *img_origin, 
                 float *voxsize, 
                 float *p,
                 unsigned long long np, 
                 unsigned int *img_dim)
{
  unsigned long long i;

  # pragma omp parallel for schedule(static)
  for(i = 0; i < np; i++)
  {
    float d0, d1, d2, d0_sq, d1_sq, d2_sq; 
    float lsq, cos0_sq, cos1_sq, cos2_sq;
    unsigned short direction; 
    unsigned int i0, i1, i2;
    int i0_floor, i1_floor, i2_floor;
    int i0_ceil, i1_ceil, i2_ceil;
    float x_pr0, x_pr1, x_pr2;
    float tmp_0, tmp_1, tmp_2;
    
    unsigned int n0 = img_dim[0];
    unsigned int n1 = img_dim[1];
    unsigned int n2 = img_dim[2];
    
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
