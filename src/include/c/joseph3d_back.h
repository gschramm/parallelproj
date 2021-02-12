#ifndef JOSEPH3D_BACK
#define JOSEPH3D_BACK

void joseph3d_back(const float *xstart, 
                   const float *xend, 
                   float *img,
                   const float *img_origin, 
                   const float *voxsize,
                   const float *p, 
                   long long nlors, 
                   const int *img_dim);

#endif
