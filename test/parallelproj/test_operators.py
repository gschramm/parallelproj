import unittest
import parallelproj
import numpy as np

import numpy.array_api as xp
#import array_api_compat.cupy as xp
#import array_api_compat.torch as xp


class TestProjectors(unittest.TestCase):

    def test_matrix(self):
        np.random.seed(0)

        A = xp.asarray([[1., 2.], [-3., 2.], [-1., -1.]])
        x = xp.asarray([-2., 1.])

        op = parallelproj.MatrixOperator(A)

        op.adjointness_test(xp)
        assert np.allclose(A @ x, op(x))

    def test_elementwise(self):
        np.random.seed(0)

        v = xp.asarray([3., -1.])
        x = xp.asarray([-2., 1.])

        op = parallelproj.ElementwiseMultiplicationOperator(v)

        op.adjointness_test(xp)
        assert np.allclose(v * x, op(x))

    def test_gaussian(self):
        np.random.seed(0)
        in_shape = (32, 32)
        sigma = 2.3

        op = parallelproj.GaussianFilterOperator(in_shape, xp, sigma=sigma)
        op.adjointness_test(xp)

    def test_composite(self):
        A = xp.asarray([[1., 2.], [-3., 2.], [-1., -1.]])
        x = xp.asarray([-2., 1.])
        v = xp.asarray([3., -1., 2.])

        op1 = parallelproj.ElementwiseMultiplicationOperator(v)
        op2 = parallelproj.MatrixOperator(A)

        op = parallelproj.CompositeLinearOperator([op1, op2])

        op.adjointness_test(xp)
        assert np.allclose(v * (A @ x), op(x))

    def test_vstack(self):
        np.random.seed(0)
        in_shape = (16, 11)

        A1 = parallelproj.GaussianFilterOperator(in_shape, xp, sigma=1.)
        A2 = parallelproj.ElementwiseMultiplicationOperator(
            xp.asarray(np.random.rand(*in_shape)))
        A3 = parallelproj.GaussianFilterOperator(in_shape, xp, sigma=2.)

        A = parallelproj.VstackOperator((A1, A2, A3))

        A.adjointness_test()

        x = xp.asarray(np.random.rand(*in_shape))
        x_fwd = A(x)

        assert np.allclose(
            x_fwd,
            xp.concat((xp.reshape(A1(x), (-1, )), xp.reshape(A2(x), (-1, )),
                       xp.reshape(A3(x), (-1, )))))

    def test_subsets(self):
        np.random.seed(0)
        in_shape = (3, )

        A1 = parallelproj.MatrixOperator(xp.asarray(np.random.randn(4, 3)))
        A2 = parallelproj.MatrixOperator(xp.asarray(np.random.randn(5, 3)))
        A3 = parallelproj.MatrixOperator(xp.asarray(np.random.randn(2, 3)))

        A = parallelproj.SubsetOperator((A1, A2, A3))

        x = xp.asarray(np.random.rand(*in_shape))

        x_fwd = A(x)

        for i in range(A.num_subsets):
            assert np.allclose(x_fwd[i], A.apply_subset(x, i))

        y = A.adjoint(x_fwd)

        for i in range(A.num_subsets):
            assert np.allclose(y[i], A.adjoint_subset(x_fwd[i], i))


#--------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
