import unittest
import parallelproj
import numpy as np


class TestParallelViewProjector2D(unittest.TestCase):

    def test_parallelviewprojector2d(self):
        image_shape = np.array([1, 2, 2])
        voxel_size = np.array([2., 2., 2.], dtype=np.float32)
        image_origin = np.array([0, -1., -1.], dtype=np.float32)

        radial_positions = np.array([-1, 0, 1], dtype=np.float32)
        view_angles = np.array([0, np.pi / 2], dtype=np.float32)
        radius = 3.

        proj = parallelproj.ParallelViewProjector2D(image_shape,
                                                    radial_positions,
                                                    view_angles, radius,
                                                    image_origin, voxel_size,
                                                    np)

        proj.adjointness_test(np)

        x = np.arange(4).reshape(1, 2, 2).astype(np.float32)

        x_fwd = proj(x)

        exp_result = np.array([[2., 6., 10.], [8., 6., 4.]], dtype=np.float32)

        assert np.allclose(x_fwd, exp_result)


#--------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
