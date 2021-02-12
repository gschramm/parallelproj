#ifndef RAY_CUBE_INTERSECTION_H
#define RAY_CUBE_INTERSECTION_H

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
                                    float* t2);
#endif
