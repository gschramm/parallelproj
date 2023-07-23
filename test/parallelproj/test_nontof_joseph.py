import unittest
import parallelproj
import numpy as np
import numpy.array_api as nparr

from types import ModuleType


def fwd_test(xp: ModuleType, verbose=True) -> bool:
    n0, n1, n2 = (2, 3, 4)

    img_dim = (n0, n1, n2)
    voxel_size = xp.asarray([4., 3., 2.], dtype=xp.float32)
    img_origin = ((-xp.asarray(img_dim, dtype=xp.float32) / 2 + 0.5) *
                  voxel_size)
    img = xp.reshape(xp.arange(n0 * n1 * n2, dtype=xp.float32), (n0, n1, n2))

    # LOR start points in voxel coordinates
    vstart = xp.asarray([
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

    vend = xp.asarray([
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

    xstart = (vstart * voxel_size + img_origin)
    xend = (vend * voxel_size + img_origin)

    img_fwd = parallelproj.joseph3d_fwd(xstart, xend, img, img_origin,
                                        voxel_size)

    # setup the expected values for the projection
    expected_projections = xp.zeros_like(img_fwd)
    expected_projections[0] = xp.sum(img[0, :, 0]) * voxel_size[1]
    expected_projections[1] = xp.sum(img[0, :, 0]) * voxel_size[1]
    expected_projections[2] = xp.sum(img[0, :, 1]) * voxel_size[1]
    expected_projections[3] = 0.5 * (expected_projections[0] +
                                     expected_projections[2])
    expected_projections[4] = xp.sum(img[0, 0, :]) * voxel_size[2]
    expected_projections[5] = xp.sum(img[:, 0, 0]) * voxel_size[0]
    expected_projections[6] = xp.sum(img[n0 - 1, :, 0]) * voxel_size[1]
    expected_projections[7] = xp.sum(img[n0 - 1, :, n2 - 1]) * voxel_size[1]
    expected_projections[8] = xp.sum(img[n0 - 1, 0, :]) * voxel_size[2]
    expected_projections[9] = xp.sum(img[n0 - 1, n1 - 1, :]) * voxel_size[2]

    if verbose:
        print(
            f'module = {xp.__name__}  -  cuda_enabled {parallelproj.num_visible_cuda_devices > 0}'
        )
        print('calculated projection = ', img_fwd)
        print('expected   projection = ', expected_projections)
        print('abs diff              = ',
              xp.abs(img_fwd - expected_projections))
        print('rel diff              = ',
              xp.abs(img_fwd - expected_projections) / expected_projections)
        print('')

    return bool(np.allclose(img_fwd, expected_projections))


#--------------------------------------------------------------------------


def adjointness_test(xp: ModuleType,
                     nLORs=1000000,
                     seed=1,
                     verbose=True) -> bool:
    """test whether backprojection is the adjoint of forward projection
       indirect test whether back projection is correct (assuming fwd projection is correct)
    """

    np.random.seed(seed)
    n0, n1, n2 = (16, 15, 17)

    img_dim = (n0, n1, n2)
    voxel_size = xp.asarray([0.7, 0.8, 0.6], dtype=xp.float32)
    img_origin = ((-xp.asarray(img_dim, dtype=xp.float32) / 2 + 0.5) *
                  voxel_size)

    img = xp.asarray(np.random.rand(n0, n1, n2), dtype=xp.float32)

    # generate random LORs on a sphere around the image volume
    R = 0.8 * xp.max((xp.asarray(img_dim, dtype=xp.float32) * voxel_size))

    phis = xp.asarray(np.random.rand(nLORs) * 2 * np.pi)
    costheta = xp.asarray(np.random.rand(nLORs) * 2 - 1)
    sintheta = xp.sqrt(1 - costheta**2)

    xstart = xp.zeros((nLORs, 3), dtype=xp.float32)
    xstart[:, 0] = R * sintheta * xp.cos(phis)
    xstart[:, 1] = R * sintheta * xp.sin(phis)
    xstart[:, 2] = R * costheta

    phis = xp.asarray(np.random.rand(nLORs) * 2 * np.pi)
    costheta = xp.asarray(np.random.rand(nLORs) * 2 - 1)
    sintheta = xp.sqrt(1 - costheta**2)

    xend = xp.zeros((nLORs, 3), dtype=xp.float32)
    xend[:, 0] = R * sintheta * xp.cos(phis)
    xend[:, 1] = R * sintheta * xp.sin(phis)
    xend[:, 2] = R * costheta

    # forward project
    img_fwd = parallelproj.joseph3d_fwd(xstart, xend, img, img_origin,
                                        voxel_size)

    # backward project
    sino = xp.asarray(np.random.rand(*img_fwd.shape), dtype=xp.float32)
    back_img = parallelproj.joseph3d_back(xstart, xend, img.shape, img_origin,
                                          voxel_size, sino)

    ip_a = float(xp.sum((back_img * img)))
    ip_b = float(xp.sum((img_fwd * sino)))

    if verbose:
        print(
            f'module = {xp.__name__}  -  cuda_enabled {parallelproj.num_visible_cuda_devices > 0}'
        )
        print('ip_a = ', ip_a)
        print('ip_b = ', ip_b)
        print('')

    return bool(np.isclose(ip_a, ip_b))


#--------------------------------------------------------------------------


class TestNonTOFJoseph(unittest.TestCase):
    """test for non TOF joseph projections"""

    def test_fwd(self):
        """test non TOF joseph forward projection using different backends"""
        self.assertTrue(fwd_test(np))
        self.assertTrue(fwd_test(nparr))

        if parallelproj.cupy_enabled:
            import cupy as cp
            self.assertTrue(fwd_test(cp))

        if parallelproj.torch_enabled:
            import torch
            self.assertTrue(fwd_test(torch))

    def test_adjoint(self):
        """test non TOF joseph forward projection using different backends"""
        self.assertTrue(adjointness_test(np))
        self.assertTrue(adjointness_test(nparr))

        if parallelproj.cupy_enabled:
            import cupy as cp
            self.assertTrue(adjointness_test(cp))

        if parallelproj.torch_enabled:
            import torch
            self.assertTrue(adjointness_test(torch))


#--------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
