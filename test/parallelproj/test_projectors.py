from __future__ import annotations

import unittest
import parallelproj
import numpy as np
import numpy.array_api as nparr
import array_api_compat


def allclose(x, y, atol: float = 1e-8, rtol: float = 1e-5) -> bool:
    """check if two arrays are close to each other, given absolute and relative error
       inspired by numpy.allclose
    """
    xp = array_api_compat.array_namespace(x)
    return bool(xp.all(xp.less_equal(xp.abs(x - y), atol + rtol * xp.abs(y))))


def parallelviewprojector_test(xp, verbose=True):
    image_shape = (2, 2)
    voxel_size = (2., 2.)
    image_origin = (-1., -1.)

    radial_positions = xp.asarray([-1, 0, 1], dtype=xp.float32)
    view_angles = xp.asarray([0, xp.pi / 2], dtype=xp.float32)
    radius = 3.

    proj2d = parallelproj.ParallelViewProjector2D(image_shape,
                                                  radial_positions,
                                                  view_angles, radius,
                                                  image_origin, voxel_size)

    proj2d.adjointness_test(verbose=verbose)

    # test a simple 2D projection
    x2d = xp.reshape(xp.arange(4, dtype=xp.float32), (2, 2))
    x_fwd = proj2d(x2d)

    exp_result = xp.asarray([[2., 8.], [6., 6.], [10., 4.]], dtype=xp.float32)

    if verbose:
        print(
            f'module = {xp.__name__}  -  cuda_enabled {parallelproj.num_visible_cuda_devices > 0}'
        )
        print('calculated 2d projection = ', x_fwd)
        print('expected   2d projection = ', exp_result)
        print('abs diff                 = ', xp.abs(x_fwd - exp_result))
        print('rel diff                 = ',
              xp.abs(x_fwd - exp_result) / exp_result)
        print('')

    assert allclose(x_fwd, exp_result)

    # setup a simple 3D projector with 2 rings

    image_shape3d = (2, 2, 2)
    image_origin3d = (-1, -1., -1.)
    voxel_size3d = (2., 2., 2.)
    ring_positions = xp.asarray([-1, 0, 1.], dtype=xp.float32)

    proj3d = parallelproj.ParallelViewProjector3D(image_shape3d,
                                                  radial_positions,
                                                  view_angles,
                                                  radius,
                                                  image_origin3d,
                                                  voxel_size3d,
                                                  ring_positions,
                                                  max_ring_diff=1)
    proj3d.adjointness_test(verbose=verbose)

    # test a simple 3D projection
    x3d = xp.reshape(xp.arange(8, dtype=xp.float32), (2, 2, 2))
    x3d_fwd = proj3d(x3d)

    # check if we get the expected results for the 3 direct planes
    exp_result_dp0 = xp.asarray([[4., 16.], [12., 12.], [20., 8.]],
                                dtype=xp.float32)
    exp_result_dp1 = xp.asarray([[6., 18.], [14., 14.], [22., 10.]],
                                dtype=xp.float32)
    exp_result_dp2 = xp.asarray([[8., 20.], [16., 16.], [24., 12.]],
                                dtype=xp.float32)

    assert allclose(x3d_fwd[..., 0], exp_result_dp0)
    assert allclose(x3d_fwd[..., 1], exp_result_dp1)
    assert allclose(x3d_fwd[..., 2], exp_result_dp2)


#--------------------------------------------------------------------------


class TestParallelViewProjector(unittest.TestCase):

    def test(self):
        parallelviewprojector_test(np)

        parallelviewprojector_test(nparr)

    if parallelproj.cupy_enabled:

        def test_cp(self):
            import array_api_compat.cupy as cp
            parallelviewprojector_test(cp)
            import cupy.array_api as cparr
            parallelviewprojector_test(cparr)

    if parallelproj.torch_enabled:

        def test_torch(self):
            import torch
            parallelviewprojector_test(torch)


if __name__ == '__main__':
    unittest.main()
