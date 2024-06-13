/**
 * @file joseph3d_fwd_tof_sino.c
 */

#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<math.h>
#include<omp.h>

#include "tof_utils.h"
#include "ray_cube_intersection.h"


void joseph3d_fwd_tof_sino(const float *xstart, 
                           const float *xend, 
                           const float *img,
                           const float *img_origin, 
                           const float *voxsize, 
                           float *p,
                           long long nlors, 
                           const int *img_dim,
                           float tofbin_width,
                           const float *sigma_tof,
                           const float *tofcenter_offset,
                           float n_sigmas,
                           short n_tofbins,
                           unsigned char lor_dependent_sigma_tof,
                           unsigned char lor_dependent_tofcenter_offset)
{
  long long i;

  int n0 = img_dim[0];
  int n1 = img_dim[1];
  int n2 = img_dim[2];

  float voxsize0 = voxsize[0];
  float voxsize1 = voxsize[1];
  float voxsize2 = voxsize[2];

  float img_origin0 = img_origin[0];
  float img_origin1 = img_origin[1];
  float img_origin2 = img_origin[2];

  int n_half = n_tofbins/2;

  # pragma omp parallel for schedule(static)
  for(i = 0; i < nlors; i++)
  {
    float d0, d1, d2, d0_sq, d1_sq, d2_sq; 
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

    int it, it1, it2;
    float dtof, tw;

    // correction factor for cos(theta) and voxsize
    float cf;
    float toAdd;

    float sig_tof   = (lor_dependent_sigma_tof == 1) ? sigma_tof[i] : sigma_tof[0];
    float tc_offset = (lor_dependent_tofcenter_offset == 1) ? tofcenter_offset[i] : tofcenter_offset[0];

    float xstart0 = xstart[i*3 + 0];
    float xstart1 = xstart[i*3 + 1];
    float xstart2 = xstart[i*3 + 2];

    float xend0 = xend[i*3 + 0];
    float xend1 = xend[i*3 + 1];
    float xend2 = xend[i*3 + 2];

    unsigned char intersec;
    float t1, t2;
    float istart_f, iend_f, tmp;
    int   istart, iend;

    float istart_tof_f, iend_tof_f;
    int istart_tof, iend_tof;

    // calculate the effective sigma (standard deviation of the TOF Gaussian convolved with the tofbin width)
    // we need to to device which TOF bins a certain voxel along an LOR
    float sig_eff = sqrtf(sig_tof*sig_tof + tofbin_width*tofbin_width/12.0f);

    // initialize all TOF bins in projection along current LOR with 0
    for(it = 0; it < n_tofbins; it++){
      p[i*n_tofbins + it] = 0;
    }

    // test whether the ray between the two detectors is most parallel
    // with the 0, 1, or 2 axis
    d0 = xend0 - xstart0;
    d1 = xend1 - xstart1;
    d2 = xend2 - xstart2;

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

      if (direction == 0)
      {
        cf = voxsize0 / sqrtf(cos0_sq);

        // case where ray is most parallel to the 0 axis
        // we step through the volume along the 0 direction

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

        // check in which "plane" the start and end points are
        // we have to do this to avoid that we include voxels
        // that are "outside" the line segment bewteen xstart and xend
        
        // !! for these calculations we overwrite the istart_f and iend_f variables !!
        istart_f = (xstart0 - img_origin0) / voxsize0;
        iend_f   = (xend0   - img_origin0) / voxsize0;

        if (istart_f > iend_f){
          tmp      = iend_f;
          iend_f   = istart_f;
          istart_f = tmp;
        }

        if (istart < (int)floor(istart_f)){istart = (int)floor(istart_f);}
        if (iend >= (int)ceil(iend_f)){iend = (int)ceil(iend_f);}
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

          toAdd = 0;

          if ((i1_floor >= 0) && (i1_floor < n1) && (i2_floor >= 0) && (i2_floor < n2))
          {
            toAdd += img[n1*n2*i0 + n2*i1_floor + i2_floor] * (1 - tmp_1) * (1 - tmp_2);
          }
          if ((i1_ceil >= 0) && (i1_ceil < n1) && (i2_floor >= 0) && (i2_floor < n2))
          {
            toAdd += img[n1*n2*i0 + n2*i1_ceil + i2_floor] * tmp_1 * (1 - tmp_2);
          }
          if ((i1_floor >= 0) && (i1_floor < n1) && (i2_ceil >= 0) && (i2_ceil < n2))
          {
            toAdd += img[n1*n2*i0 + n2*i1_floor + i2_ceil] * (1 - tmp_1) * tmp_2;
          }
          if ((i1_ceil >= 0) && (i1_ceil < n1) && (i2_ceil >= 0) && (i2_ceil < n2))
          {
            toAdd += img[n1*n2*i0 + n2*i1_ceil + i2_ceil] * tmp_1 * tmp_2;
          }

          //--------- TOF related quantities
          // calculate the voxel center needed for TOF weights
          x_v0 = img_origin0 + i0*voxsize0;
          x_v1 = x_pr1;
          x_v2 = x_pr2;

          it1 = -n_half;
          it2 =  n_half;

          // get the relevant tof bins (the TOF bins where the TOF weight is not close to 0)
          relevant_tof_bins(x_m0, x_m1, x_m2, x_v0, x_v1, x_v2, u0, u1, u2, 
                            tofbin_width, tc_offset, sig_eff, n_sigmas, n_half,
                            &it1, &it2);

          if(toAdd != 0){
            for(it = it1; it <= it2; it++){
              //--- add extra check to be compatible with behavior of LM projector
              istart_tof_f = (x_m0 + (it*tofbin_width - n_sigmas*sig_eff)*u0 - img_origin0) / voxsize0;
              iend_tof_f   = (x_m0 + (it*tofbin_width + n_sigmas*sig_eff)*u0 - img_origin0) / voxsize0;
        
              if (istart_tof_f > iend_tof_f){
                tmp        = iend_tof_f;
                iend_tof_f = istart_tof_f;
                istart_tof_f = tmp;
              }

              istart_tof = (int)floor(istart_tof_f);
              iend_tof   = (int)ceil(iend_tof_f);
              //---

              if ((i0 >= istart_tof) && (i0 < iend_tof)){
                // calculate distance of voxel to tof bin center
                dtof = sqrtf(powf((x_m0 + (it*tofbin_width + tc_offset)*u0 - x_v0), 2) + 
                             powf((x_m1 + (it*tofbin_width + tc_offset)*u1 - x_v1), 2) + 
                             powf((x_m2 + (it*tofbin_width + tc_offset)*u2 - x_v2), 2));

                //calculate the TOF weight
                tw = 0.5f*(erff((dtof + 0.5f*tofbin_width)/(sqrtf(2)*sig_tof)) - 
                          erff((dtof - 0.5f*tofbin_width)/(sqrtf(2)*sig_tof)));

                p[i*n_tofbins + it + n_half] += (tw * cf * toAdd);
              }
            }
          }
        }
      }

      //--------------------------------------------------------------------------------- 
      if (direction == 1)
      {
        cf = voxsize1 / sqrtf(cos1_sq);

        // case where ray is most parallel to the 1 axis

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

        // check in which "plane" the start and end points are
        // we have to do this to avoid that we include voxels
        // that are "outside" the line segment bewteen xstart and xend
        
        // !! for these calculations we overwrite the istart_f and iend_f variables !!
        istart_f = (xstart1 - img_origin1) / voxsize1;
        iend_f   = (xend1   - img_origin1) / voxsize1;

        if (istart_f > iend_f){
          tmp      = iend_f;
          iend_f   = istart_f;
          istart_f = tmp;
        }

        if (istart < (int)floor(istart_f)){istart = (int)floor(istart_f);}
        if (iend >= (int)ceil(iend_f)){iend = (int)ceil(iend_f);}
        //---

        // we step through the volume along the 1 direction
        for (i1 = istart; i1 < iend; i1++)
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
 
          toAdd = 0;

          if ((i0_floor >= 0) && (i0_floor < n0) && (i2_floor >= 0) && (i2_floor < n2))
          {
            toAdd += img[n1*n2*i0_floor + n2*i1 + i2_floor] * (1 - tmp_0) * (1 - tmp_2);
          }
          if ((i0_ceil >= 0) && (i0_ceil < n0) && (i2_floor >= 0) && (i2_floor < n2))
          {
            toAdd += img[n1*n2*i0_ceil + n2*i1 + i2_floor] * tmp_0 * (1 - tmp_2);
          }
          if ((i0_floor >= 0) && (i0_floor < n0) && (i2_ceil >= 0) && (i2_ceil < n2))
          {
            toAdd += img[n1*n2*i0_floor + n2*i1 + i2_ceil] * (1 - tmp_0) * tmp_2;
          }
          if ((i0_ceil >= 0) && (i0_ceil < n0) && (i2_ceil >= 0) && (i2_ceil < n2))
          {
            toAdd += img[n1*n2*i0_ceil + n2*i1 + i2_ceil] * tmp_0 * tmp_2;
          }

          //--------- TOF related quantities
          // calculate the voxel center needed for TOF weights
          x_v0 = x_pr0;
          x_v1 = img_origin1 + i1*voxsize1;
          x_v2 = x_pr2;

          it1 = -n_half;
          it2 =  n_half;

          // get the relevant tof bins (the TOF bins where the TOF weight is not close to 0)
          relevant_tof_bins(x_m0, x_m1, x_m2, x_v0, x_v1, x_v2, u0, u1, u2, 
                            tofbin_width, tc_offset, sig_eff, n_sigmas, n_half,
                            &it1, &it2);

          if(toAdd != 0){
            for(it = it1; it <= it2; it++){
              //--- add extra check to be compatible with behavior of LM projector
              istart_tof_f = (x_m1 + (it*tofbin_width - n_sigmas*sig_eff)*u1 - img_origin1) / voxsize1;
              iend_tof_f   = (x_m1 + (it*tofbin_width + n_sigmas*sig_eff)*u1 - img_origin1) / voxsize1;
        
              if (istart_tof_f > iend_tof_f){
                tmp        = iend_tof_f;
                iend_tof_f = istart_tof_f;
                istart_tof_f = tmp;
              }

              istart_tof = (int)floor(istart_tof_f);
              iend_tof   = (int)ceil(iend_tof_f);
              //---

              if ((i1 >= istart_tof) && (i1 < iend_tof)){
                // calculate distance of voxel to tof bin center
                dtof = sqrtf(powf((x_m0 + (it*tofbin_width + tc_offset)*u0 - x_v0), 2) + 
                             powf((x_m1 + (it*tofbin_width + tc_offset)*u1 - x_v1), 2) + 
                             powf((x_m2 + (it*tofbin_width + tc_offset)*u2 - x_v2), 2));

                //calculate the TOF weight
                tw = 0.5f*(erff((dtof + 0.5f*tofbin_width)/(sqrtf(2)*sig_tof)) - 
                          erff((dtof - 0.5f*tofbin_width)/(sqrtf(2)*sig_tof)));


                p[i*n_tofbins + it + n_half] += (tw * cf * toAdd);
              }
            }
          }
        }
      }

      //--------------------------------------------------------------------------------- 
      if (direction == 2)
      {
        cf = voxsize2 / sqrtf(cos2_sq);

        // case where ray is most parallel to the 2 axis
        // we step through the volume along the 2 direction

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

        // check in which "plane" the start and end points are
        // we have to do this to avoid that we include voxels
        // that are "outside" the line segment bewteen xstart and xend
        
        // !! for these calculations we overwrite the istart_f and iend_f variables !!
        istart_f = (xstart2 - img_origin2) / voxsize2;
        iend_f   = (xend2   - img_origin2) / voxsize2;

        if (istart_f > iend_f){
          tmp      = iend_f;
          iend_f   = istart_f;
          istart_f = tmp;
        }

        if (istart < (int)floor(istart_f)){istart = (int)floor(istart_f);}
        if (iend >= (int)ceil(iend_f)){iend = (int)ceil(iend_f);}
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

          toAdd = 0;

          if ((i0_floor >= 0) && (i0_floor < n0) && (i1_floor >= 0) && (i1_floor < n1))
          {
            toAdd += img[n1*n2*i0_floor + n2*i1_floor + i2] * (1 - tmp_0) * (1 - tmp_1);
          }
          if ((i0_ceil >= 0) && (i0_ceil < n0) && (i1_floor >= 0) && (i1_floor < n1))
          {
            toAdd += img[n1*n2*i0_ceil + n2*i1_floor + i2] * tmp_0 * (1 - tmp_1);
          }
          if ((i0_floor >= 0) && (i0_floor < n0) && (i1_ceil >= 0) & (i1_ceil < n1))
          {
            toAdd += img[n1*n2*i0_floor + n2*i1_ceil + i2] * (1 - tmp_0) * tmp_1;
          }
          if ((i0_ceil >= 0) && (i0_ceil < n0) && (i1_ceil >= 0) && (i1_ceil < n1))
          {
            toAdd += img[n1*n2*i0_ceil + n2*i1_ceil + i2] * tmp_0 * tmp_1;
          }

          //--------- TOF related quantities
          // calculate the voxel center needed for TOF weights
          x_v0 = x_pr0;
          x_v1 = x_pr1;
          x_v2 = img_origin2 + i2*voxsize2;

          it1 = -n_half;
          it2 =  n_half;

          // get the relevant tof bins (the TOF bins where the TOF weight is not close to 0)
          relevant_tof_bins(x_m0, x_m1, x_m2, x_v0, x_v1, x_v2, u0, u1, u2, 
                            tofbin_width, tc_offset, sig_eff, n_sigmas, n_half,
                            &it1, &it2);

          if(toAdd != 0){
            for(it = it1; it <= it2; it++){
              //--- add extra check to be compatible with behavior of LM projector
              istart_tof_f = (x_m2 + (it*tofbin_width - n_sigmas*sig_eff)*u2 - img_origin2) / voxsize2;
              iend_tof_f   = (x_m2 + (it*tofbin_width + n_sigmas*sig_eff)*u2 - img_origin2) / voxsize2;
        
              if (istart_tof_f > iend_tof_f){
                tmp        = iend_tof_f;
                iend_tof_f = istart_tof_f;
                istart_tof_f = tmp;
              }

              istart_tof = (int)floor(istart_tof_f);
              iend_tof   = (int)ceil(iend_tof_f);
              //---

              if ((i2 >= istart_tof) && (i2 < iend_tof)){
                // calculate distance of voxel to tof bin center
                dtof = sqrtf(powf((x_m0 + (it*tofbin_width + tc_offset)*u0 - x_v0), 2) + 
                             powf((x_m1 + (it*tofbin_width + tc_offset)*u1 - x_v1), 2) + 
                             powf((x_m2 + (it*tofbin_width + tc_offset)*u2 - x_v2), 2));

                //calculate the TOF weight
                tw = 0.5f*(erff((dtof + 0.5f*tofbin_width)/(sqrtf(2)*sig_tof)) - 
                          erff((dtof - 0.5f*tofbin_width)/(sqrtf(2)*sig_tof)));

                p[i*n_tofbins + it + n_half] += (tw * cf * toAdd);
              }
            }
          }
        }
      }
    }
  }
}
