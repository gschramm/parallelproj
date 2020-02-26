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
#include <ray_cube_intersection.h> 

int main(){
  // the origin of the ray
  float orig0 = -25;
  float orig1 = -25;
  float orig2 = 80;

  // bounding box values of the cube in all three dimension
  float bounds0_min = -40;
  float bounds1_min = -42;
  float bounds2_min = -44;

  float bounds0_max = 70;
  float bounds1_max = 72;
  float bounds2_max = 74;

  // the three components of the directional vector of the ray
  // in vector notation the ray would by orign + t*rdir
  float rdir0 = 0;
  float rdir1 = 0;
  float rdir2 = 1;

  unsigned char intersec;
  float t1, t2;

  intersec = ray_cube_intersection(orig0, orig1, orig2, bounds0_min, bounds1_min, bounds2_min,
                                   bounds0_max, bounds1_max, bounds2_max, rdir0, rdir1, rdir2, &t1, &t2);
  if (intersec){
  printf("intersection\n");
  printf("%f %f %f",orig0 + t1*rdir0, orig1 + t1*rdir1, orig2 + t1*rdir2);
  printf("\n %f %f %f",orig0 + t2*rdir0, orig1 + t2*rdir1, orig2 + t2*rdir2);
  printf("\n\n");
  }
}
