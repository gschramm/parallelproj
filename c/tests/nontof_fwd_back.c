#include <stdio.h>
#include <string.h>
#include "parallelproj_c.h"

int main()
{
    // const float img[] = {1, 0, 3,
    //                      0, 3, 0,
    //                      1, -1, 2};

    // const int img_dim[] = {1, 3, 3};
    // const float voxsize[] = {2, 2, 2};
    // const float img_origin[] = {0, -2, -2};

    // long long nlors = 5;

    // const float xstart[] = {0, 4, -2, 0, 4, -2, 0, 4, -2, 0, 4, 0, 0, 4, 1};
    // const float xend[] = {0, -4, -2, 0, -4, -2, 0, -4, -2, 0, -4, 0, 0, -4, 1};

    // float p[nlors];
    // memset(p, 0, sizeof p);

    //// forward projection test
    // joseph3d_fwd(xstart, xend, img, img_origin, voxsize, p, nlors, img_dim);

    // printf("%.1f %.1f %.1f %.1f %.1f\n", p[0], p[1], p[2], p[3], p[4]);

    //// back projection test
    // float bimg[] = {0, 0, 0,
    //                 0, 0, 0,
    //                 0, 0, 0};

    // float ones[nlors];
    // for (int i = 0; i < nlors; i++)
    //{
    //     ones[i] = 1;
    // }

    // joseph3d_back(xstart, xend, bimg, img_origin, voxsize, ones, nlors, img_dim);

    // printf("\n%.1f %.1f %.1f\n", bimg[0], bimg[1], bimg[2]);
    // printf("%.1f %.1f %.1f\n", bimg[3], bimg[4], bimg[5]);
    // printf("%.1f %.1f %.1f\n", bimg[6], bimg[7], bimg[8]);

    return 0;
}