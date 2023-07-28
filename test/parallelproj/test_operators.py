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


def test_matrix(xp):
    np.random.seed(0)

    A = xp.asarray([[1., 2.], [-3., 2.], [-1., -1.]])
    x = xp.asarray([-2., 1.])

    op = parallelproj.MatrixOperator(A)

    op.adjointness_test(xp)
    assert allclose(A @ x, op(x))


def test_elementwise(xp):
    np.random.seed(0)

    v = xp.asarray([3., -1.])
    x = xp.asarray([-2., 1.])

    op = parallelproj.ElementwiseMultiplicationOperator(v)

    op.adjointness_test(xp)
    assert allclose(v * x, op(x))


def test_gaussian(xp):
    np.random.seed(0)
    in_shape = (32, 32)
    sigma = 2.3

    op = parallelproj.GaussianFilterOperator(in_shape, xp, sigma=sigma)
    op.adjointness_test(xp)


def test_composite(xp):
    np.random.seed(0)

    A = xp.asarray([[1., 2.], [-3., 2.], [-1., -1.]])
    x = xp.asarray([-2., 1.])
    v = xp.asarray([3., -1., 2.])

    op1 = parallelproj.ElementwiseMultiplicationOperator(v)
    op2 = parallelproj.MatrixOperator(A)

    op = parallelproj.CompositeLinearOperator([op1, op2])

    op.adjointness_test(xp)
    assert allclose(v * (A @ x), op(x))


def test_vstack(xp):
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

    assert allclose(
        x_fwd,
        xp.concat((xp.reshape(A1(x), (-1, )), xp.reshape(A2(x), (-1, )),
                   xp.reshape(A3(x), (-1, )))))


def test_subsets(xp):
    np.random.seed(0)
    in_shape = (3, )

    A1 = parallelproj.MatrixOperator(xp.asarray(np.random.randn(4, 3)))
    A2 = parallelproj.MatrixOperator(xp.asarray(np.random.randn(5, 3)))
    A3 = parallelproj.MatrixOperator(xp.asarray(np.random.randn(2, 3)))

    A = parallelproj.SubsetOperator((A1, A2, A3))

    x = xp.asarray(np.random.rand(*in_shape))

    x_fwd = A(x)

    for i in range(A.num_subsets):
        assert allclose(x_fwd[i], A.apply_subset(x, i))

    y = A.adjoint(x_fwd)

    for i in range(A.num_subsets):
        assert allclose(y[i], A.adjoint_subset(x_fwd[i], i))


#--------------------------------------------------------------------------
class TestOperators(unittest.TestCase):

    def testmatrix(self):
        test_matrix(np)
        test_matrix(nparr)

    if parallelproj.cupy_enabled:

        def testmatrix_cp(self):
            import array_api_compat.cupy as cp
            test_matrix(cp)

    if parallelproj.torch_enabled:

        def testmatrix_torch(self):
            import array_api_compat.torch as torch
            test_matrix(torch)

    #-----------------------------------------------
    def testelementwise(self):
        test_elementwise(np)
        test_elementwise(nparr)

    if parallelproj.cupy_enabled:

        def testelementwise_cp(self):
            import array_api_compat.cupy as cp
            test_elementwise(cp)

    if parallelproj.torch_enabled:

        def testelementwise_torch(self):
            import array_api_compat.torch as torch
            test_elementwise(torch)

    #-----------------------------------------------
    def testgaussian(self):
        test_gaussian(nparr)

    if parallelproj.cupy_enabled:

        def testgaussian_cp(self):
            import array_api_compat.cupy as cp
            test_gaussian(cp)

    if parallelproj.torch_enabled:

        def testgaussian_torch(self):
            import array_api_compat.torch as torch
            test_gaussian(torch)

    #-----------------------------------------------
    def testcomposite(self):
        test_composite(nparr)

    if parallelproj.cupy_enabled:

        def testcomposite_cp(self):
            import array_api_compat.cupy as cp
            test_composite(cp)

    if parallelproj.torch_enabled:

        def testcomposite_torch(self):
            import array_api_compat.torch as torch
            test_composite(torch)

    #-----------------------------------------------
    def testvstack(self):
        test_vstack(nparr)

    if parallelproj.cupy_enabled:

        def testvstack_cp(self):
            import array_api_compat.cupy as cp
            test_vstack(cp)

    if parallelproj.torch_enabled:

        def testvstack_torch(self):
            import array_api_compat.torch as torch
            test_vstack(torch)


#--------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
