import unittest
import parallelproj
import numpy as np

from types import ModuleType


def adjointness_test(xp: ModuleType,
                     nLORs: int = 10000,
                     num_tof_bins: int = 21,
                     seed: int = 1,
                     verbose: bool = True) -> bool:
    """test whether backprojection is the adjoint of forward projection
       indirect test whether back projection is correct (assuming fwd projection is correct)
    """

    xp.random.seed(seed)

    n0, n1, n2 = (17, 17, 17)

    img_dim = xp.array([n0, n1, n2])
    voxel_size = xp.array([1., 1., 1.], dtype=xp.float32)
    img_origin = ((-img_dim / 2 + 0.5) * voxel_size).astype(xp.float32)
    img = xp.zeros((n0, n1, n2)).astype(xp.float32)
    img[n0 // 2, n1 // 2, n2 // 2] = 1

    # generate random LORs on a sphere around the image volume
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
    tofbin_width = 2.
    num_tof_bins = 11
    nsigmas = 3.
    sigma_tof = xp.array([5 / 2.35], dtype=xp.float32)
    tofcenter_offset = xp.array([0], dtype=xp.float32)

    img_fwd = xp.zeros((xstart.shape[0], num_tof_bins), dtype=xp.float32)

    parallelproj.joseph3d_fwd_tof_sino(xstart, xend, img, img_origin,
                                       voxel_size, img_fwd, tofbin_width,
                                       sigma_tof, tofcenter_offset, nsigmas,
                                       num_tof_bins)

    # backward project
    back_img = xp.zeros_like(img)
    sino = xp.random.rand(nLORs, num_tof_bins).astype(xp.float32)

    parallelproj.joseph3d_back_tof_sino(xstart, xend, back_img, img_origin,
                                        voxel_size, sino, tofbin_width,
                                        sigma_tof, tofcenter_offset, nsigmas,
                                        num_tof_bins)
    ip_a = (back_img * img).sum()
    ip_b = (img_fwd * sino).sum()

    res = np.isclose(ip_a, ip_b)

    if verbose:
        print(
            f'module = {xp.__name__}  -  cuda_enabled {parallelproj.cuda_enabled}'
        )
        print('ip_a = ', ip_a)
        print('ip_b = ', ip_b)
        print('')

    return res


#--------------------------------------------------------------------------

class TestTOFJoseph(unittest.TestCase):
    """test for TOF joseph projections"""

    def test_adjoint(self):
        """test TOF joseph forward projection using different backends"""
        self.assertTrue(adjointness_test(np))

        if parallelproj.cupy_enabled:
            import cupy as cp
            self.assertTrue(adjointness_test(cp))

        if parallelproj.cuda_enabled:
            parallelproj.cuda_enabled = False
            self.assertTrue(adjointness_test(np))
            parallelproj.cuda_enabled = True


#--------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
