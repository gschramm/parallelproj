/**
 * @file tof_utils.c
 */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>

/**
 * @brief Calculate the tof weight between a voxel and a TOF bin on an LOR 
 *
 * @param x_m0   0-component of center of LOR
 * @param x_m1   1-component of center of LOR
 * @param x_m2   2-component of center of LOR
 * @param x_v0   0-component of voxel
 * @param x_v1   1-component of voxel
 * @param x_v2   2-component of voxel
 * @param u0     0-component of unit vector that points from start to end of LOR 
 * @param u1     1-component of unit vector that points from start to end of LOR 
 * @param u2     2-component of unit vector that points from start to end of LOR 
 * @param it     index of TOF bin (ranges from -n_tofbins//2 ... 0 ... n_tofbins//2)
 * @param tofbin_            width width of the TOF bins in spatial units
 * @param tofcenter_offset   offset of the central tofbin from the midpoint of the LOR in spatial units
 * @param sigma_tof          TOF resolution in spatial coordinates
 * @param erf_lut            look up table length 6001 for erf between -3 and 3. 
 *                           The i-th element contains erf(-3 + 0.001*i)
 */
float tof_weight(float x_m0, 
		             float x_m1, 
		             float x_m2, 
		             float x_v0, 
		             float x_v1, 
		             float x_v2, 
		             float u0,
		             float u1,
		             float u2,
		             int it,
		             float tofbin_width,
		             float tofcenter_offset,
		             float sigma_tof,
                 float *erf_lut)
{
  float x_c0, x_c1, x_c2;
  float dtof, dtof_far, dtof_near, tw;

  int ilut_near, ilut_far;

  // calculate center of the tofbin
  x_c0 = x_m0 + (it*tofbin_width + tofcenter_offset)*u0;
  x_c1 = x_m1 + (it*tofbin_width + tofcenter_offset)*u1;
  x_c2 = x_m2 + (it*tofbin_width + tofcenter_offset)*u2;

  // calculate distance to tof bin center
  dtof = sqrt((x_c0 - x_v0)*(x_c0 - x_v0) + (x_c1 - x_v1)*(x_c1 - x_v1) + (x_c2 - x_v2)*(x_c2 - x_v2));
  dtof_far  = dtof + 0.5*tofbin_width;
  dtof_near = dtof - 0.5*tofbin_width;

  // calculate the TOF weight
  //tw = 0.5*(erff(dtof_far/(sqrt(2)*sigma_tof)) - erff(dtof_near/(sqrt(2)*sigma_tof)));

  ilut_near = (int)(dtof_near/(sqrt(2)*sigma_tof) + 3)/0.001;
  ilut_far  = (int)(dtof_far /(sqrt(2)*sigma_tof) + 3)/0.001;

  if(ilut_near < 0){ilut_near = 0;}
  if(ilut_far  < 0){ilut_far  = 0;}
  if(ilut_near > 6000){ilut_near = 6000;}
  if(ilut_far  > 6000){ilut_far  = 6000;}

  tw = 0.5*(erf_lut[ilut_far] - erf_lut[ilut_near]);

  return(tw);
}

/**
 * @brief Calculate the TOF bins along an LOR to which a voxel contributes
 *
 * @param x_m0   0-component of center of LOR
 * @param x_m1   1-component of center of LOR
 * @param x_m2   2-component of center of LOR
 * @param x_v0   0-component of voxel
 * @param x_v1   1-component of voxel
 * @param x_v2   2-component of voxel
 * @param u0     0-component of unit vector that points from start to end of LOR 
 * @param u1     1-component of unit vector that points from start to end of LOR 
 * @param u2     2-component of unit vector that points from start to end of LOR 
 * @param tofbin_width      width of the TOF bins in spatial units
 * @param tofcenter_offset  offset of the central tofbin from the midpoint of the LOR in spatial units
 * @param sigma_tof         TOF resolution in spatial coordinates
 * @param n_sigmas          number of sigmas considered to be relevant
 * @param n_half            n_tofbins // 2
 * @param it1 (output)      lower relevant tof bin
 * @param it2 (output)      upper relevant tof bin
 */
void relevant_tof_bins(float x_m0,
		       float x_m1, 
		       float x_m2, 
		       float x_v0, 
		       float x_v1, 
		       float x_v2, 
		       float u0,
		       float u1,
		       float u2,
		       float tofbin_width,
		       float tofcenter_offset,
		       float sigma_tof,
		       unsigned int n_sigmas,
		       int n_half,
		       int *it1,
		       int *it2)
{
  float b1, b2;

  // calculate which TOF bins it1 and it2 are within +- n_sigmas
  // the tof bin numbers run from -n_tofbins//2 ... 0 ... n_tofbins//2
  b1 = (((x_v0 - x_m0)*u0 + (x_v1 - x_m1)*u1 + (x_v2 - x_m2)*u2) - 
            tofcenter_offset + n_sigmas*sigma_tof) / tofbin_width;
  b2 = (((x_v0 - x_m0)*u0 + (x_v1 - x_m1)*u1 + (x_v2 - x_m2)*u2) - 
            tofcenter_offset - n_sigmas*sigma_tof) / tofbin_width;

  if(b1 <= b2){
    *it1 = (int)floor(b1);
    *it2 = (int)ceil(b2);
  }
  else{
    *it1 = (int)floor(b2);
    *it2 = (int)ceil(b1);
  }

  if(*it1 < -n_half){*it1 = -n_half;}
  if(*it2 < -n_half){*it2 = -n_half;}
  if(*it1 > n_half){*it1 = n_half;}
  if(*it2 > n_half){*it2 = n_half;}
}
