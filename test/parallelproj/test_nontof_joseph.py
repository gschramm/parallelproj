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
            f'module = {xp.__name__}  -  cuda_enabled {parallelproj.num_visible_cuda_devices > 0}'
        )
        print('calculated projection = ', img_fwd)
        print('expected   projection = ', expected_projections)
        print('abs diff              = ',
              xp.abs(img_fwd - expected_projections))
        print('rel diff              = ',
              xp.abs(img_fwd - expected_projections) / expected_projections)
        print('')

    return xp.allclose(img_fwd, expected_projections)


#--------------------------------------------------------------------------


def adjointness_test(xp: ModuleType,
                     nLORs=1000000,
                     seed=1,
                     verbose=True) -> bool:
    """test whether backprojection is the adjoint of forward projection
       indirect test whether back projection is correct (assuming fwd projection is correct)
    """
    xp.random.seed(seed)
    n0, n1, n2 = (16, 15, 17)

    img_dim = xp.array([n0, n1, n2])
    voxel_size = xp.array([0.7, 0.8, 0.6], dtype=xp.float32)
    img_origin = ((-img_dim / 2 + 0.5) * voxel_size).astype(xp.float32)
    img = xp.random.rand(n0, n1, n2).astype(xp.float32)

    # generate random LORs on a sphere around the image volume
    R = 0.8 * (img_dim * voxel_size).max()

    phis = xp.random.rand(nLORs) * 2 * xp.pi
    costheta = xp.random.rand(nLORs) * 2 - 1
    sintheta = xp.sqrt(1 - costheta**2)

    xstart = xp.zeros((nLORs, 3), dtype=xp.float32)
    xstart[:, 0] = R * sintheta * xp.cos(phis)
    xstart[:, 1] = R * sintheta * xp.sin(phis)
    xstart[:, 2] = R * costheta

    phis = xp.random.rand(nLORs) * 2 * xp.pi
    costheta = xp.random.rand(nLORs) * 2 - 1
    sintheta = xp.sqrt(1 - costheta**2)

    xend = xp.zeros((nLORs, 3), dtype=xp.float32)
    xend[:, 0] = R * sintheta * xp.cos(phis)
    xend[:, 1] = R * sintheta * xp.sin(phis)
    xend[:, 2] = R * costheta

    # forward project
    img_fwd = xp.zeros(xstart.shape[0], dtype=xp.float32)
    parallelproj.joseph3d_fwd(xstart, xend, img, img_origin, voxel_size,
                              img_fwd)

    # backward project
    back_img = xp.zeros_like(img)
    sino = xp.random.rand(nLORs).astype(xp.float32)
    parallelproj.joseph3d_back(xstart, xend, back_img, img_origin, voxel_size,
                               sino)

    ip_a = (back_img * img).sum()
    ip_b = (img_fwd * sino).sum()

    res = np.isclose(ip_a, ip_b)

    if verbose:
        print(
            f'module = {xp.__name__}  -  cuda_enabled {parallelproj.num_visible_cuda_devices > 0}'
        )
        print('ip_a = ', ip_a)
        print('ip_b = ', ip_b)
        print('')

    return res


#--------------------------------------------------------------------------


class TestNonTOFJoseph(unittest.TestCase):
    """test for non TOF joseph projections"""

    def test_fwd(self):
        """test non TOF joseph forward projection using different backends"""
        self.assertTrue(fwd_test(np))

        if parallelproj.cupy_enabled:
            import cupy as cp
            self.assertTrue(fwd_test(cp))

    def test_adjoint(self):
        """test non TOF joseph forward projection using different backends"""
        self.assertTrue(adjointness_test(np))

        if parallelproj.cupy_enabled:
            import cupy as cp
            self.assertTrue(adjointness_test(cp))


#--------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
