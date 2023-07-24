import unittest
import parallelproj
import numpy as np
import numpy.array_api as nparr


def test_parallelviewprojector2d(xp, verbose=True):
    image_shape = (1, 2, 2)
    voxel_size = xp.asarray([2., 2., 2.], dtype=xp.float32)
    image_origin = xp.asarray([0, -1., -1.], dtype=xp.float32)

    radial_positions = xp.asarray([-1, 0, 1], dtype=xp.float32)
    view_angles = xp.asarray([0, xp.pi / 2], dtype=xp.float32)
    radius = 3.

    proj = parallelproj.ParallelViewProjector2D(image_shape, radial_positions,
                                                view_angles, radius,
                                                image_origin, voxel_size)

    proj.adjointness_test()

    x = xp.reshape(xp.arange(4, dtype=xp.float32), (1, 2, 2))
    x_fwd = proj(x)

    exp_result = xp.asarray([[2., 6., 10.], [8., 6., 4.]], dtype=xp.float32)

    if verbose:
        print(
            f'module = {xp.__name__}  -  cuda_enabled {parallelproj.num_visible_cuda_devices > 0}'
        )
        print('calculated projection = ', x_fwd)
        print('expected   projection = ', exp_result)
        print('abs diff              = ', xp.abs(x_fwd - exp_result))
        print('rel diff              = ',
              xp.abs(x_fwd - exp_result) / exp_result)
        print('')

    assert bool(np.allclose(x_fwd, exp_result))


#--------------------------------------------------------------------------


class TestParallelViewProjector2D(unittest.TestCase):

    def test2d(self):
        test_parallelviewprojector2d(nparr)

    if parallelproj.cupy_enabled:

        def test2d_cp(self):
            import array_api_compat.cupy as cp
            test_parallelviewprojector2d(cp)

    if parallelproj.torch_enabled:

        def test2d_torch(self):
            import array_api_compat.torch as torch
            test_parallelviewprojector2d(torch)


if __name__ == '__main__':
    unittest.main()
