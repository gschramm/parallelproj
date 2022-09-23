#include <stdio.h>
#include <string.h>
#include "parallelproj_cuda.h"

int main()
{
    int retval = 0;
    int tpb = 64;

    const float img[] = {1, 0, 3,
                         0, 3, 0,
                         1, -1, 2};

    const int img_dim[] = {1, 3, 3};
    const float voxsize[] = {2, 2, 2};
    const float img_origin[] = {0, -2, -2};

    long long nlors = 5;

    const float xstart[] = {0, 4, -2, 0, 4, -2, 0, 4, -2, 0, 4, 0, 0, 4, 1};
    const float xend[] = {0, -4, -2, 0, -4, -2, 0, -4, -2, 0, -4, 0, 0, -4, 1};

    float p[nlors];
    memset(p, 0, sizeof p);

    // forward projection test
    float** d_img = copy_float_array_to_all_devices(img, 9);
    joseph3d_fwd_cuda(xstart, xend, d_img, img_origin, voxsize, p, nlors, img_dim, tpb);
    free_float_array_on_all_devices(d_img);

    printf("%.1f %.1f %.1f %.1f %.1f\n", p[0], p[1], p[2], p[3], p[4]);

    // check if we got the expected results
    if (p[0] != 4)
        retval = 1;
    if (p[1] != 4)
        retval = 1;
    if (p[2] != 4)
        retval = 1;
    if (p[3] != 4)
        retval = 1;
    if (p[4] != 7)
        retval = 1;

    // back projection test
    float bimg[] = {0, 0, 0,
                    0, 0, 0,
                    0, 0, 0};

    float ones[nlors];
    for (int i = 0; i < nlors; i++)
    {
        ones[i] = 1;
    }

    float** d_bimg = copy_float_array_to_all_devices(bimg, 9);
    joseph3d_back_cuda(xstart, xend, d_bimg, img_origin, voxsize, ones, nlors, img_dim, tpb);
    sum_float_arrays_on_first_device(d_bimg, 9);
    get_float_array_from_device(d_bimg, 9, 0, bimg);
    free_float_array_on_all_devices(d_bimg);

    printf("\n%.1f %.1f %.1f\n", bimg[0], bimg[1], bimg[2]);
    printf("%.1f %.1f %.1f\n", bimg[3], bimg[4], bimg[5]);
    printf("%.1f %.1f %.1f\n", bimg[6], bimg[7], bimg[8]);

    if (bimg[0] != 6)
        retval = 1;
    if (bimg[1] != 3)
        retval = 1;
    if (bimg[2] != 1)
        retval = 1;
    if (bimg[3] != 6)
        retval = 1;
    if (bimg[4] != 3)
        retval = 1;
    if (bimg[5] != 1)
        retval = 1;
    if (bimg[6] != 6)
        retval = 1;
    if (bimg[7] != 3)
        retval = 1;
    if (bimg[8] != 1)
        retval = 1;

    return retval;
}
