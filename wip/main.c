// gcc -O3 -I . main.c
//

// demo script that checks whether a ray intersects a cube
// the ray is given by origin + t*rdir (vector notation)
// if the ray intersects the cube, the two t values t1 and t2
// for the intersection points are computed
// algorithm is taken from 
// https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
//
// this algorithm assume that the IEEE floating point arith standard 754 is followed which handles
// divions by 0 correctly

#define RAYCUBEINTERSECTION(orig0, orig1, orig2, bounds0_min, bounds1_min, bounds2_min, bounds0_max, bounds1_max, bounds2_max, rdir0, rdir1, rdir2, t1, t2) \
  invdir0 = 1. / rdir0; \
  invdir1 = 1. / rdir1; \
  invdir2 = 1. / rdir2; \
                        \
  if (invdir0 >= 0){ \
    t1  = (bounds0_min - orig0) * invdir0; \
    t2  = (bounds0_max - orig0) * invdir0; \
  } \
  else{ \
    t1  = (bounds0_max - orig0) * invdir0; \
    t2  = (bounds0_min - orig0) * invdir0; \
  } \
    \
  if (invdir1 >= 0){ \
    t11 = (bounds1_min - orig1) * invdir1; \
    t12 = (bounds1_max - orig1) * invdir1; \
  } \
  else{ \
    t11 = (bounds1_max - orig1) * invdir1; \
    t12 = (bounds1_min - orig1) * invdir1; \
  } \
    \
  if ((t1 > t12) || (t11 > t2)){intersec = 0;} \
  if (t11 > t1){t1 = t11;} \
  if (t12 < t2){t2 = t12;} \
                             \
  if (invdir1 >= 0){ \
    t21 = (bounds2_min - orig2) * invdir2; \
    t22 = (bounds2_max - orig2) * invdir2; \
  } \
  else{ \
    t21 = (bounds2_max - orig2) * invdir2; \
    t22 = (bounds2_min - orig2) * invdir2; \
  } \
    \
  if ((t1 > t22) || (t21 > t2)){intersec = 0;} \
  if (t21 > t1){t1 = t21;} \
  if (t22 < t2){t2 = t22;} \

//-----------------------------------------------------------------------------------------------------------

#include <stdio.h> 
#include <time.h>
#include <ray_cube_intersection.h> 

int main(int argc, char **argv){

  int method = 0;
  if(argc > 1){method = 1;}

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

  unsigned char intersec = 1;
  float t1, t2;

  clock_t start, end;

  if(method == 0){ 
    printf("method: call function\n");
    start = clock();
    for(long i = 0; i < 1000000; i++){
      intersec = ray_cube_intersection(orig0, orig1, orig2, bounds0_min, bounds1_min, bounds2_min,
                                     bounds0_max, bounds1_max, bounds2_max, rdir0, rdir1, rdir2, &t1, &t2);
    }
    end = clock();
  }

  if(method == 1){ 
    printf("method: MACRO\n");
    start = clock();
    float invdir0;
    float invdir1;
    float invdir2;
    float t11; 
    float t12; 
    float t21;
    float t22; 
    for(long i = 0; i < 1000000; i++){
      RAYCUBEINTERSECTION(orig0, orig1, orig2, bounds0_min, bounds1_min, bounds2_min,
                          bounds0_max, bounds1_max, bounds2_max, rdir0, rdir1, rdir2, t1, t2);
    }
    end = clock();
  }

  double dt = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("dt: %f s\n", dt);

  if (intersec){
    printf("intersection\n");
    printf("%f %f %f",orig0 + t1*rdir0, orig1 + t1*rdir1, orig2 + t1*rdir2);
    printf("\n %f %f %f",orig0 + t2*rdir0, orig1 + t2*rdir1, orig2 + t2*rdir2);
    printf("\n\n");
  }

  return(0);
}
