#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "parallelproj_c.h"

// test nontof projectors using a simple 2D image along one direction
int main()
{
    int retval = 0;

    float eps = 1e-7;

    const int img_dim[] = {2, 3, 4};
    const float voxsize[] = {4, 3, 2};
    //--------------------------------------------------------------------------
    //--------------------------------------------------------------------------
    //--------------------------------------------------------------------------

    int n0 = img_dim[0];
    int n1 = img_dim[1];
    int n2 = img_dim[2];

    const float img_origin[] = {(-(float)n0 / 2 + 0.5) * voxsize[0],
                                (-(float)n1 / 2 + 0.5) * voxsize[1],
                                (-(float)n2 / 2 + 0.5) * voxsize[2]};

    printf("\nimage dimension: ");
    printf("%d %d %d\n", n0, n1, n2);

    printf("\nvoxel size: ");
    printf("%.1f %.1f %.1f\n", voxsize[0], voxsize[1], voxsize[2]);

    printf("\nimage origin: ");
    printf("%.1f %.1f %.1f\n", img_origin[0], img_origin[1], img_origin[2]);

    float *img = (float *)calloc(n0 * n1 * n2, sizeof(float));

    printf("\nimage:\n");
    for (int i0 = 0; i0 < n0; i0++)
    {
        for (int i1 = 0; i1 < n1; i1++)
        {
            for (int i2 = 0; i2 < n2; i2++)
            {
                img[n1 * n2 * i0 + n2 * i1 + i2] = (n1 * n2 * i0 + n2 * i1 + i2 + 1);
                printf("%.1f ", img[n1 * n2 * i0 + n2 * i1 + i2]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // setup the start and end coordinates of a few test rays in voxel coordinates
    long long nlors = 6;

    float vstart[] = {0, -1, 0,   // 0
                      0, -1, 0,   // 1
                      0, -1, 1,   // 2
                      0, -1, 0.5, // 3
                      0, 0, -1,   // 4
                      -1, 0, 0};  // 5

    float vend[] = {0, n1, 0,   // 0
                    0, n1, 0,   // 1
                    0, n1, 1,   // 2
                    0, n1, 0.5, // 3
                    0, 0, n2,   // 4
                    n0, 0, 0};  // 5

    for (int ir = 0; ir < nlors; ir++)
    {
        printf("test ray %d\n", ir);
        printf("start .: %.1f %.1f %.1f\n", vstart[ir * 3 + 0], vstart[ir * 3 + 1], vstart[ir * 3 + 2]);
        printf("end   .: %.1f %.1f %.1f\n", vend[ir * 3 + 0], vend[ir * 3 + 1], vend[ir * 3 + 2]);
    }

    // calculate the start and end coordinates in world coordinates
    float *xstart = (float *)calloc(3 * nlors, sizeof(float));
    float *xend = (float *)calloc(3 * nlors, sizeof(float));

    for (int ir = 0; ir < nlors; ir++)
    {
        for (int j = 0; j < 3; j++)
        {
            xstart[ir * 3 + j] = img_origin[j] + vstart[ir * 3 + j] * voxsize[j];
            xend[ir * 3 + j] = img_origin[j] + vend[ir * 3 + j] * voxsize[j];
        }
    }

    // allocate memory for the forward projection
    float *p = (float *)calloc(nlors, sizeof(float));

    // forward projection test
    joseph3d_fwd(xstart, xend, img, img_origin, voxsize, p, nlors, img_dim);
    // calculate the expected value of rays0/1 from [0,-1,0] to [0,last+1,0]
    float *expected_fwd_vals = (float *)calloc(nlors, sizeof(float));
    for (int i1 = 0; i1 < n1; i1++)
    {
        expected_fwd_vals[0] += img[0 * n1 * n2 + i1 * n2 + 0] * voxsize[1];
    }

    expected_fwd_vals[1] = expected_fwd_vals[0];

    // calculate the expected value of ray2 from [0,-1,1] to [0,last+1,1]
    for (int i1 = 0; i1 < n1; i1++)
    {
        expected_fwd_vals[2] += img[0 * n1 * n2 + i1 * n2 + 1] * voxsize[1];
    }

    // calculate the expected value of ray3 from [0,-1,0.5] to [0,last+1,0.5]
    expected_fwd_vals[3] = 0.5 * (expected_fwd_vals[0] + expected_fwd_vals[2]);

    // calculate the expected value of ray4 from [0,0,-1] to [0,0,last+1]
    for (int i2 = 0; i2 < n2; i2++)
    {
        expected_fwd_vals[4] += img[0 * n1 * n2 + 0 * n2 + i2] * voxsize[2];
    }

    // calculate the expected value of ray5 from [-1,0,0] to [last+1,0,0]
    for (int i0 = 0; i0 < n0; i0++)
    {
        expected_fwd_vals[5] += img[i0 * n1 * n2 + 0 * n2 + 0] * voxsize[0];
    }

    // check if we got the expected results
    float fwd_diff = 0;
    for (int ir = 0; ir < nlors; ir++)
    {
        printf("test ray %d: fwd projected: %.7e expected: %.7e\n", ir, p[ir], expected_fwd_vals[ir]);

        fwd_diff = fabs(p[ir] - expected_fwd_vals[ir]);
        if (fwd_diff > eps)
        {
            printf("\n################################################################################");
            printf("\nabs(fwd projected - expected value) = %.2e for ray%d above tolerance %.2e", fwd_diff, ir, eps);
            printf("\n################################################################################");
            retval = 1;
        }
    }

    //// back projection test
    // float bimg[] = {0, 0, 0,
    //                 0, 0, 0,
    //                 0, 0, 0,

    //                0, 0, 0,
    //                0, 0, 0,
    //                0, 0, 0,

    //                0, 0, 0,
    //                0, 0, 0,
    //                0, 0, 0};

    // float *ones = (float *)calloc(nlors, sizeof(float));
    // for (int i = 0; i < nlors; i++)
    //{
    //     ones[i] = 1;
    // }

    // joseph3d_back(xstart, xend, bimg, img_origin, voxsize, ones, nlors, img_dim);

    // printf("\n%.1f %.1f %.1f\n", bimg[0 + 9], bimg[1 + 9], bimg[2 + 9]);
    // printf("%.1f %.1f %.1f\n", bimg[3 + 9], bimg[4 + 9], bimg[5 + 9]);
    // printf("%.1f %.1f %.1f\n", bimg[6 + 9], bimg[7 + 9], bimg[8 + 9]);

    // if (bimg[0 + 9] != 6)
    //     retval = 1;
    // if (bimg[1 + 9] != 3)
    //     retval = 1;
    // if (bimg[2 + 9] != 1)
    //     retval = 1;
    // if (bimg[3 + 9] != 6)
    //     retval = 1;
    // if (bimg[4 + 9] != 3)
    //     retval = 1;
    // if (bimg[5 + 9] != 1)
    //     retval = 1;
    // if (bimg[6 + 9] != 6)
    //     retval = 1;
    // if (bimg[7 + 9] != 3)
    //     retval = 1;
    // if (bimg[8 + 9] != 1)
    //     retval = 1;

    return retval;
}
