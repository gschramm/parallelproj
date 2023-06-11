import unittest
import parallelproj
import numpy as np

import scipy.ndimage as ndi


class TestProjectors(unittest.TestCase):

    def test_matrix(self):
        A = np.array([[1., 2.], [-3., 2.], [-1., -1.]])
        x = np.array([-2., 1.])

        op = parallelproj.MatrixOperator(A)

        op.adjointness_test(np)
        assert (np.all(np.isclose(A @ x, op(x))))

    def test_elementwise(self):
        v = np.array([3., -1.])
        x = np.array([-2., 1.])

        op = parallelproj.ElementwiseMultiplicationOperator(v)

        op.adjointness_test(np)
        assert (np.all(np.isclose(v * x, op(x))))

    def test_gaussian(self):

        in_shape = (32, 32)
        x = np.random.randn(*in_shape)
        sigma = 2.3

        op = parallelproj.GaussianFilterOperator(in_shape, ndi, sigma=sigma)

        op.adjointness_test(np)
        assert (np.all(np.isclose(ndi.gaussian_filter(x, sigma=sigma), op(x))))

    def test_composite(self):

        A = np.array([[1., 2.], [-3., 2.], [-1., -1.]])
        x = np.array([-2., 1.])
        v = np.array([3., -1., 2.])

        op1 = parallelproj.ElementwiseMultiplicationOperator(v)
        op2 = parallelproj.MatrixOperator(A)

        op = parallelproj.CompositeLinearOperator([op1, op2])

        op.adjointness_test(np)
        assert (np.all(np.isclose(v * (A @ x), op(x))))


#--------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
