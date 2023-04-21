import unittest
import parallelproj
import numpy as np

from types import ModuleType


def tof_sino_fwd_test(xp: ModuleType, verbose: bool = True, atol: float = 1e-6) -> None:
    """test fwd sinogram TOF projection of a point source"""
    nLORs: int = 1
    voxsize: float = 0.1
    seed: int = 1

    xp.random.seed(seed)

    n0, n1, n2 = (171, 171, 171)

    img_dim = xp.array([n0, n1, n2])
    voxel_size = xp.array([voxsize, voxsize, voxsize], dtype=xp.float32)
    img_origin = ((-img_dim / 2 + 0.5) * voxel_size).astype(xp.float32)
    img = xp.zeros((n0, n1, n2), dtype=xp.float32)
    img[n0 // 2, n1 // 2, n2 // 2] = 1

    # generate random LORs on a sphere around the image volume
    # generate random LORs on a sphere around the image volume
    xstart = xp.zeros((nLORs, 3), dtype=xp.float32)
    xstart[:, 0] = 0
    xstart[:, 0] = 0
    xstart[:, 0] = 100

    xend = xp.zeros((nLORs, 3), dtype=xp.float32)
    xend[:, 0] = 0
    xend[:, 0] = 0
    xend[:, 0] = -100

    # forward project
    tofbin_width = 0.05
    num_tof_bins = 501
    nsigmas = 9.
    fwhm_tof = 6.
    sigma_tof = xp.array([fwhm_tof / (2 * np.sqrt(2 * np.log(2)))],
                         dtype=xp.float32)
    tofcenter_offset = xp.array([0], dtype=xp.float32)

    img_fwd = xp.zeros((xstart.shape[0], num_tof_bins), dtype=xp.float32)

    parallelproj.joseph3d_fwd_tof_sino(xstart, xend, img, img_origin,
                                       voxel_size, img_fwd, tofbin_width,
                                       sigma_tof, tofcenter_offset, nsigmas,
                                       num_tof_bins)

    # check if sum of the projection is correct (should be equal to the voxel size)
    res1 = xp.isclose(img_fwd.sum(), voxsize)

    # check if the FWHM in the projected profile is correct
    # to do so, we check if the interpolated profile - 0.5max(profile) at +/- FWHM/2 is 0
    r = (xp.arange(num_tof_bins) - 0.5 * num_tof_bins + 0.5) * tofbin_width

    res2 = xp.isclose(
        float(
            xp.interp(xp.array([fwhm_tof / 2]), r,
                      img_fwd[0, :] - 0.5 * img_fwd[0, :].max())[0]), 0, atol = atol)
    res3 = xp.isclose(
        float(
            xp.interp(xp.array([-fwhm_tof / 2]), r,
                      img_fwd[0, :] - 0.5 * img_fwd[0, :].max())[0]), 0, atol = atol)

    if verbose:
        print(
            f'module = {xp.__name__}  -  cuda_enabled {parallelproj.num_visible_cuda_devices > 0}'
        )
        print(
            f'sum of TOF profile / expected:    {float(img_fwd.sum()):.4E} / {voxsize:.4E}'
        )
        print('')

    return res1 * res2 * res3


def adjointness_test(xp: ModuleType,
                     nLORs: int = 10000,
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
    img = xp.random.rand(n0, n1, n2).astype(xp.float32)

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
            f'module = {xp.__name__}  -  cuda_enabled {parallelproj.num_visible_cuda_devices > 0}'
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

    def test_forward(self):
        """test TOF joseph forward projection using different backends"""
        self.assertTrue(tof_sino_fwd_test(np))

        if parallelproj.cupy_enabled:
            import cupy as cp
            self.assertTrue(tof_sino_fwd_test(cp))


#--------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
