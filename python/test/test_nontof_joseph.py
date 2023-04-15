import unittest
import parallelproj
import numpy as np

from types import ModuleType


def fwd_test(xp: ModuleType, verbose=True) -> bool:
    n0, n1, n2 = (2, 3, 4)

    img_dim = xp.array([n0, n1, n2])
    voxel_size = xp.array([4., 3., 2.], dtype=xp.float32)
    img_origin = ((-img_dim / 2 + 0.5) * voxel_size).astype(xp.float32)
    img = xp.arange(n0 * n1 * n2, dtype=xp.float32).reshape((n0, n1, n2))

    # LOR start points in voxel coordinates
    vstart = xp.array([
        [0, -1, 0],  # 
        [0, -1, 0],  #
        [0, -1, 1],  #
        [0, -1, 0.5],  #
        [0, 0, -1],  #
        [-1, 0, 0],  #
        [n0 - 1, -1, 0],  # 
        [n0 - 1, -1, n2 - 1],  #
        [n0 - 1, 0, -1],  #
        [n0 - 1, n1 - 1, -1]
    ])

    vend = xp.array([
        [0, n1, 0],  #           
        [0, n1, 0],  #           
        [0, n1, 1],  #          
        [0, n1, 0.5],  #         
        [0, 0, n2],  #          
        [n0, 0, 0],  #          
        [n0 - 1, n1, 0],  #      
        [n0 - 1, n1, n2 - 1],  # 
        [n0 - 1, 0, n2],  #     
        [n0 - 1, n1 - 1, n2]
    ])

    xstart = (vstart * voxel_size + img_origin).astype(xp.float32)
    xend = (vend * voxel_size + img_origin).astype(xp.float32)

    img_fwd = xp.zeros(xstart.shape[0], dtype=xp.float32)

    parallelproj.joseph3d_fwd(xstart, xend, img, img_origin, voxel_size,
                              img_fwd)

    # setup the expected values for the projection
    expected_projections = xp.zeros_like(img_fwd)
    expected_projections[0] = img[0, :, 0].sum() * voxel_size[1]
    expected_projections[1] = img[0, :, 0].sum() * voxel_size[1]
    expected_projections[2] = img[0, :, 1].sum() * voxel_size[1]
    expected_projections[3] = 0.5 * (expected_projections[0] +
                                     expected_projections[2])
    expected_projections[4] = img[0, 0, :].sum() * voxel_size[2]
    expected_projections[5] = img[:, 0, 0].sum() * voxel_size[0]
    expected_projections[6] = img[n0 - 1, :, 0].sum() * voxel_size[1]
    expected_projections[7] = img[n0 - 1, :, n2 - 1].sum() * voxel_size[1]
    expected_projections[8] = img[n0 - 1, 0, :].sum() * voxel_size[2]
    expected_projections[9] = img[n0 - 1, n1 - 1, :].sum() * voxel_size[2]

    if verbose:
        print(
            f'module = {xp.__name__}  -  cuda_enabled {parallelproj.cuda_enabled}'
        )
        print('calculated projection = ', img_fwd)
        print('expected   projection = ', expected_projections)
        print('abs diff              = ',
              xp.abs(img_fwd - expected_projections))
        print('rel diff              = ',
              xp.abs(img_fwd - expected_projections) / expected_projections)
        print('')

    return xp.all(xp.isclose(img_fwd, expected_projections))


#--------------------------------------------------------------------------


class TestNonTOFJoseph(unittest.TestCase):

    def test_fwd(self):
        self.assertTrue(fwd_test(np))

        if parallelproj.cupy_enabled:
            import cupy as cp
            self.assertTrue(fwd_test(cp))

        if parallelproj.cuda_enabled:
            parallelproj.cuda_enabled = False
            self.assertTrue(fwd_test(np))
            parallelproj.cuda_enabled = True


#--------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()