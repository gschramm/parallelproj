import unittest
import parallelproj
import numpy as np
import numpy.array_api as nparr


def test_parallelviewprojector(xp, verbose=True):
    image_shape = (1, 2, 2)
    voxel_size = xp.asarray([2., 2., 2.], dtype=xp.float32)
    image_origin = xp.asarray([0, -1., -1.], dtype=xp.float32)

    radial_positions = xp.asarray([-1, 0, 1], dtype=xp.float32)
    view_angles = xp.asarray([0, xp.pi / 2], dtype=xp.float32)
    radius = 3.

    proj2d = parallelproj.ParallelViewProjector2D(image_shape,
                                                  radial_positions,
                                                  view_angles, radius,
                                                  image_origin, voxel_size)

    proj2d.adjointness_test(verbose=verbose)

    # test a simple 2D projection
    x2d = xp.reshape(xp.arange(4, dtype=xp.float32), (1, 2, 2))
    x_fwd = proj2d(x2d)

    exp_result = xp.asarray([[2., 6., 10.], [8., 6., 4.]], dtype=xp.float32)

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

    assert bool(np.allclose(x_fwd, exp_result))

    # setup a simple 3D projector with 2 rings

    image_shape3d = (2, 2, 2)
    image_origin3d = xp.asarray([-1, -1., -1.], dtype=xp.float32)
    ring_positions = xp.asarray([-1, 0, 1.], dtype=xp.float32)

    proj3d = parallelproj.ParallelViewProjector3D(image_shape3d,
                                                  image_origin3d, proj2d,
                                                  ring_positions)

    proj3d.adjointness_test(verbose=verbose)

    # test a simple 3D projection
    x3d = xp.reshape(xp.arange(8, dtype=xp.float32), (2, 2, 2))
    x3d_fwd = proj3d(x3d)

    # check if we get the expected results for the 3 direct planes
    exp_result_dp0 = xp.asarray([[2., 6., 10.], [8., 6., 4.]],
                                dtype=xp.float32)
    exp_result_dp1 = xp.asarray([[10., 14., 18.], [16., 14., 12.]],
                                dtype=xp.float32)
    exp_result_dp2 = xp.asarray([[18., 22., 26.], [24., 22., 20.]],
                                dtype=xp.float32)

    assert bool(np.allclose(x3d_fwd[0, ...], exp_result_dp0))
    assert bool(np.allclose(x3d_fwd[1, ...], exp_result_dp1))
    assert bool(np.allclose(x3d_fwd[2, ...], exp_result_dp2))


#--------------------------------------------------------------------------


class TestParallelViewProjector(unittest.TestCase):

    def test(self):
        test_parallelviewprojector(nparr)

    if parallelproj.cupy_enabled:

        def test_cp(self):
            import array_api_compat.cupy as cp
            test_parallelviewprojector(cp)

    if parallelproj.torch_enabled:

        def test_torch(self):
            import array_api_compat.torch as torch
            test_parallelviewprojector(torch)


if __name__ == '__main__':
    unittest.main()
