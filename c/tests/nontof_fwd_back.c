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
    long long nlors = 10;

    float vstart[] = {
        0, -1, 0,           // 0
        0, -1, 0,           // 1
        0, -1, 1,           // 2
        0, -1, 0.5,         // 3
        0, 0, -1,           // 4
        -1, 0, 0,           // 5
        n0 - 1, -1, 0,      // 6 - (shifted 1)
        n0 - 1, -1, n2 - 1, // 7 - (shifted 6)
        n0 - 1, 0, -1,      // 8 - (shifted 4)
        n0 - 1, n1 - 1, -1, // 9 - (shifted 8)
    };

    float vend[] = {
        0, n1, 0,           // 0
        0, n1, 0,           // 1
        0, n1, 1,           // 2
        0, n1, 0.5,         // 3
        0, 0, n2,           // 4
        n0, 0, 0,           // 5
        n0 - 1, n1, 0,      // 6 - (shifted 1)
        n0 - 1, n1, n2 - 1, // 7 - (shifted 6)
        n0 - 1, 0, n2,      // 8 - (shifted 4)
        n0 - 1, n1 - 1, n2, // 9 - (shifted 8)
    };

    for (int ir = 0; ir < nlors; ir++)
    {
        printf("test ray %d\n", ir);
        printf("start voxel num .: %.1f %.1f %.1f\n", vstart[ir * 3 + 0], vstart[ir * 3 + 1], vstart[ir * 3 + 2]);
        printf("end   voxel num .: %.1f %.1f %.1f\n", vend[ir * 3 + 0], vend[ir * 3 + 1], vend[ir * 3 + 2]);
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

    // calculate the expected value of rays6 from [n0-1,-1,0] to [n0-1,last+1,0]
    for (int i1 = 0; i1 < n1; i1++)
    {
        expected_fwd_vals[6] += img[(n0 - 1) * n1 * n2 + i1 * n2 + 0] * voxsize[1];
    }

    // calculate the expected value of rays7 from [n0-1,-1,n2-1] to [n0-1,last+1,n2-1]
    for (int i1 = 0; i1 < n1; i1++)
    {
        expected_fwd_vals[7] += img[(n0 - 1) * n1 * n2 + i1 * n2 + (n2 - 1)] * voxsize[1];
    }

    // calculate the expected value of ray4 from [n0-1,0,-1] to [n0-1,0,last+1]
    for (int i2 = 0; i2 < n2; i2++)
    {
        expected_fwd_vals[8] += img[(n0 - 1) * n1 * n2 + 0 * n2 + i2] * voxsize[2];
    }

    // calculate the expected value of ray4 from [n0-1,0,-1] to [n0-1,0,last+1]
    for (int i2 = 0; i2 < n2; i2++)
    {
        expected_fwd_vals[9] += img[(n0 - 1) * n1 * n2 + (n1 - 1) * n2 + i2] * voxsize[2];
    }

    // check if we got the expected results
    float fwd_diff = 0;
    printf("\nforward projection test\n");
    for (int ir = 0; ir < nlors; ir++)
    {
        printf("test ray %d: fwd projected: %.7e expected: %.7e\n", ir, p[ir], expected_fwd_vals[ir]);

        fwd_diff = fabs(p[ir] - expected_fwd_vals[ir]);
        if (fwd_diff > eps)
        {
            printf("\n################################################################################");
            printf("\nabs(fwd projected - expected value) = %.2e for ray%d above tolerance %.2e", fwd_diff, ir, eps);
            printf("\n################################################################################\n");
            retval = 1;
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // backprojection of ones test
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    float *bimg = (float *)calloc(n0 * n1 * n2, sizeof(float));

    float *ones = (float *)calloc(nlors, sizeof(float));
    for (int i = 0; i < nlors; i++)
    {
        ones[i] = 1;
    }

    joseph3d_back(xstart, xend, bimg, img_origin, voxsize, ones, nlors, img_dim);

    printf("\nback projection of ones along all rays:\n");
    for (int i0 = 0; i0 < n0; i0++)
    {
        for (int i1 = 0; i1 < n1; i1++)
        {
            for (int i2 = 0; i2 < n2; i2++)
            {
                printf("%.1f ", bimg[n1 * n2 * i0 + n2 * i1 + i2]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // To test whether the back projection is correct, we test if the back projector is the adjoint
    // of the forward projector. This is more practical than checking a lot of single voxels in the
    // back projected image.

    float inner_product1 = 0;
    float inner_product2 = 0;

    for (int i = 0; i < (n0 * n1 * n2); i++)
    {
        inner_product1 += (img[i] * bimg[i]);
    }

    for (int ir = 0; ir < nlors; ir++)
    {
        inner_product2 += (p[ir] * ones[ir]);
    }

    float ip_diff = fabs(inner_product1 - inner_product2);

    if (ip_diff > eps)
    {
        printf("\n#########################################################################");
        printf("\nback projection test failed. back projection seems not to be the adjoint.");
        printf("\n %.7e", ip_diff);
        printf("\n#########################################################################\n");
        retval = 1;
    }

    return retval;
}
