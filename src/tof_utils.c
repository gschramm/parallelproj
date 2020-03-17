/**
 * @file tof_utils.c
 */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
/**
 * @brief approximation of error function with 4 order polynomial (max error 5e-4) according to
 * Abramowitz and Stegun "Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables"
 * supposed to be faster than the standard erff implementation
 *
 * @param x   argument of error function
 * @return    approxmiation of erf(x)
 */ 
float erff_as(float x)
{
  float res;
  float xa = fabsf(x);

  float d = 1.f + 0.278393f*xa + 0.230389f*(xa*xa) + 0.000972f*(xa*xa*xa) + 0.078108f*(xa*xa*xa*xa);

  res = (1.f - (1.f/(d*d*d*d)));

  if (x < 0){res *= -1;}

  return(res);
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
