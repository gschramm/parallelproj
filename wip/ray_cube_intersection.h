// demo script that checks whether a ray intersects a cube
// the ray is given by origin + t*rdir (vector notation)
// if the ray intersects the cube, the two t values t1 and t2
// for the intersection points are computed
// algorithm is taken from 
// https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
//
// this algorithm assume that the IEEE floating point arith standard 754 is followed which handles
// divions by 0 correctly


#include <stdio.h> 

unsigned char ray_cube_intersection(float orig0,
                                    float orig1,
                                    float orig2,
                                    float bounds0_min,
                                    float bounds1_min,
                                    float bounds2_min,
                                    float bounds0_max,
                                    float bounds1_max,
                                    float bounds2_max,
                                    float rdir0,
                                    float rdir1,
                                    float rdir2,
                                    float* t1,
                                    float* t2){
  // the inverse of the directional vector
  // using the inverse of the directional vector and IEEE floating point arith standard 754
  // makes sure that 0's in the directional vector are handled correctly 
  float invdir0 = 1./rdir0;
  float invdir1 = 1./rdir1;
  float invdir2 = 1./rdir2;
  
  unsigned char intersec = 1;
  
  float t11, t12, t21, t22; 

  if (invdir0 >= 0){
    *t1  = (bounds0_min - orig0) * invdir0;
    *t2  = (bounds0_max - orig0) * invdir0; 
  }
  else{
    *t1  = (bounds0_max - orig0) * invdir0;
    *t2  = (bounds0_min - orig0) * invdir0;
  }
  
  if (invdir1 >= 0){
    t11 = (bounds1_min - orig1) * invdir1; 
    t12 = (bounds1_max - orig1) * invdir1; 
  }
  else{
    t11 = (bounds1_max - orig1) * invdir1;
    t12 = (bounds1_min - orig1) * invdir1; 
  }
  
  if ((*t1 > t12) || (t11 > *t2)){intersec = 0;}
  if (t11 > *t1){*t1 = t11;}
  if (t12 < *t2){*t2 = t12;}
  
  if (invdir2 >= 0){
    t21 = (bounds2_min - orig2) * invdir2; 
    t22 = (bounds2_max - orig2) * invdir2;
  } 
  else{
    t21 = (bounds2_max - orig2) * invdir2; 
    t22 = (bounds2_min - orig2) * invdir2;
  } 
  
  if ((*t1 > t22) || (t21 > *t2)){intersec = 0;}
  if (t21 > *t1){*t1 = t21;}
  if (t22 < *t2){*t2 = t22;} 

  return(intersec);
}
