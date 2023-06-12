import unittest
import parallelproj
import numpy as np

import scipy.ndimage as ndi


class TestProjectors(unittest.TestCase):

    def test_matrix(self):
        A = np.array([[1., 2.], [-3., 2.], [-1., -1.]])
        x = np.array([-2., 1.])

        op = parallelproj.MatrixOperator(A, np)

        op.adjointness_test(np)
        assert np.allclose(A @ x, op(x))

    def test_elementwise(self):
        v = np.array([3., -1.])
        x = np.array([-2., 1.])

        op = parallelproj.ElementwiseMultiplicationOperator(v, np)

        op.adjointness_test(np)
        assert np.allclose(v * x, op(x))

    def test_gaussian(self):
        np.random.seed(0)
        in_shape = (32, 32)
        x = np.random.randn(*in_shape)
        sigma = 2.3

        op = parallelproj.GaussianFilterOperator(in_shape, ndi, np, sigma=sigma)

        op.adjointness_test(np)
        assert np.allclose(ndi.gaussian_filter(x, sigma=sigma), op(x))

    def test_composite(self):
        A = np.array([[1., 2.], [-3., 2.], [-1., -1.]])
        x = np.array([-2., 1.])
        v = np.array([3., -1., 2.])

        op1 = parallelproj.ElementwiseMultiplicationOperator(v, np)
        op2 = parallelproj.MatrixOperator(A, np)

        op = parallelproj.CompositeLinearOperator([op1, op2])

        op.adjointness_test(np)
        assert np.allclose(v * (A @ x), op(x))

    def test_vstack(self):
        np.random.seed(0)
        in_shape = (16,11)

        A1  = parallelproj.GaussianFilterOperator(in_shape, ndi, np, sigma = 1.) 
        A2  = parallelproj.ElementwiseMultiplicationOperator(np.random.rand(*in_shape), np)
        A3  = parallelproj.GaussianFilterOperator(in_shape, ndi, np, sigma = 2.) 

        A = parallelproj.VstackOperator((A1, A2, A3))

        A.adjointness_test()

        x = np.random.rand(*in_shape)
        x_fwd = A(x)

        assert np.allclose(x_fwd, np.concatenate((A1(x).ravel(), A2(x). ravel(), A3(x).ravel())))

    def test_subsets(self):
        np.random.seed(0)
        in_shape = (3,)

        A1 = parallelproj.MatrixOperator(np.random.randn(4, 3), np)
        A2 = parallelproj.MatrixOperator(np.random.randn(5, 3), np)
        A3 = parallelproj.MatrixOperator(np.random.randn(2, 3), np)

        A = parallelproj.SubsetOperator((A1, A2, A3))

        x = np.random.rand(*in_shape)

        x_fwd = A(x)

        for i in range(A.num_subsets):
            assert np.allclose(x_fwd[i], A.apply_subset(x, i))

        y = A.adjoint(x_fwd)

        for i in range(A.num_subsets):
            assert np.allclose(y[i], A.adjoint_subset(x_fwd[i], i))

#--------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
