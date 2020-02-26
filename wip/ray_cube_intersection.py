# demo script that checks whether a ray intersects a cube
# the ray is given by origin + t*rdir (vector notation)
# if the ray intersects the cube, the two t values t1 and t2
# for the intersection points are computed
# algorithm is taken from 
# https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
#
# this algorithm assume that the IEEE floating point arith standard 754 is followed which handles
# divions by 0 correctly

import random
random.seed(1)

# the origin of the ray
orig0 = -50
orig1 = -55
orig2 = -65

# bounding box values of the cube in all three dimension
bounds0_min = -40
bounds1_min = -42
bounds2_min = -44

bounds0_max = 70
bounds1_max = 72
bounds2_max = 74

for i in range(100):
  # the three components of the directional vector of the ray
  # in vector notation the ray would by orign + t*rdir
  rdir0 = 2*random.random() - 1
  rdir1 = 2*random.random() - 1
  rdir2 = 2*random.random() - 1
  #-------------------
 
  # the inverse of the directional vector
  # using the inverse of the directional vector and IEEE floating point arith standard 754
  # makes sure that 0's in the directional vector are handled correctly 
  invdir0 = 1./rdir0
  invdir1 = 1./rdir1
  invdir2 = 1./rdir2
  
  intersec = True
  
  if invdir0 >= 0:
    t1  = (bounds0_min - orig0) * invdir0
    t2  = (bounds0_max - orig0) * invdir0 
  else:
    t1  = (bounds0_max - orig0) * invdir0
    t2  = (bounds0_min - orig0) * invdir0 
  
  if invdir1 >= 0:
    t11 = (bounds1_min - orig1) * invdir1 
    t12 = (bounds1_max - orig1) * invdir1 
  else:
    t11 = (bounds1_max - orig1) * invdir1 
    t12 = (bounds1_min - orig1) * invdir1 
  
  
  if ((t1 > t12) or (t11 > t2)): 
    intersec = False
  if (t11 > t1): 
    t1 = t11 
  if (t12 < t2): 
    t2 = t12
  
  if invdir1 >= 0:
    t21 = (bounds2_min - orig2) * invdir2 
    t22 = (bounds2_max - orig2) * invdir2 
  else:
    t21 = (bounds2_max - orig2) * invdir2 
    t22 = (bounds2_min - orig2) * invdir2 
  
  
  if ((t1 > t22) or (t21 > t2)):
    intersec = False 
  if (t21 > t1): 
    t1 = t21 
  if (t22 < t2):
    t2 = t22 
  
  #------------
  if intersec:
    x_intersec_1 = [orig0 + t1*rdir0, orig1 + t1*rdir1, orig2 + t1*rdir2]
    x_intersec_2 = [orig0 + t2*rdir0, orig1 + t2*rdir1, orig2 + t2*rdir2]
  
    print(x_intersec_1)
    print(x_intersec_2)
    print('')
