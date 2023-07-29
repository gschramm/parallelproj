from __future__ import annotations

import unittest
import parallelproj
import numpy.array_api as nparr
import array_api_compat
import array_api_compat.numpy as np

from types import ModuleType


def allclose(x, y, atol: float = 1e-8, rtol: float = 1e-5) -> bool:
    """check if two arrays are close to each other, given absolute and relative error
       inspired by numpy.allclose
    """
    xp = array_api_compat.array_namespace(x)
    return bool(xp.all(xp.less_equal(xp.abs(x - y), atol + rtol * xp.abs(y))))


def matrix_test(xp: ModuleType):
    np.random.seed(0)

    A = xp.asarray([[1., 2.], [-3., 2.], [-1., -1.]])
    x = xp.asarray([-2., 1.])

    op = parallelproj.MatrixOperator(A)

    op.adjointness_test(xp)
    assert allclose(A @ x, op(x))


def elemenwise_test(xp: ModuleType):
    np.random.seed(0)

    v = xp.asarray([3., -1.])
    x = xp.asarray([-2., 1.])

    op = parallelproj.ElementwiseMultiplicationOperator(v)

    op.adjointness_test(xp)
    assert allclose(v * x, op(x))


def gaussian_test(xp: ModuleType):
    np.random.seed(0)
    in_shape = (32, 32)
    sigma = 2.3

    op = parallelproj.GaussianFilterOperator(in_shape, xp, sigma=sigma)
    op.adjointness_test(xp)


def composite_test(xp: ModuleType):
    np.random.seed(0)

    A = xp.asarray([[1., 2.], [-3., 2.], [-1., -1.]])
    x = xp.asarray([-2., 1.])
    v = xp.asarray([3., -1., 2.])

    op1 = parallelproj.ElementwiseMultiplicationOperator(v)
    op2 = parallelproj.MatrixOperator(A)

    op = parallelproj.CompositeLinearOperator([op1, op2])

    op.adjointness_test(xp)
    assert allclose(v * (A @ x), op(x))


def vstack_test(xp: ModuleType):
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


def subsets_test(xp: ModuleType):
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
        matrix_test(np)
        if np.__version__ >= '1.25':
            matrix_test(nparr)

    if parallelproj.cupy_enabled:

        def testmatrix_cp(self):
            import array_api_compat.cupy as cp
            matrix_test(cp)

    if parallelproj.torch_enabled:

        def testmatrix_torch(self):
            import array_api_compat.torch as torch
            matrix_test(torch)

    #-----------------------------------------------
    def testelementwise(self):
        elemenwise_test(np)
        if np.__version__ >= '1.25':
            elemenwise_test(nparr)

    if parallelproj.cupy_enabled:

        def testelementwise_cp(self):
            import array_api_compat.cupy as cp
            elemenwise_test(cp)

    if parallelproj.torch_enabled:

        def testelementwise_torch(self):
            import array_api_compat.torch as torch
            elemenwise_test(torch)

    #-----------------------------------------------
    def testgaussian(self):
        gaussian_test(np)
        if np.__version__ >= '1.25':
            gaussian_test(nparr)

    if parallelproj.cupy_enabled:

        def testgaussian_cp(self):
            import array_api_compat.cupy as cp
            gaussian_test(cp)

    if parallelproj.torch_enabled:

        def testgaussian_torch(self):
            import array_api_compat.torch as torch
            gaussian_test(torch)

    #-----------------------------------------------
    def testcomposite(self):
        composite_test(np)
        if np.__version__ >= '1.25':
            composite_test(nparr)

    if parallelproj.cupy_enabled:

        def testcomposite_cp(self):
            import array_api_compat.cupy as cp
            composite_test(cp)

    if parallelproj.torch_enabled:

        def testcomposite_torch(self):
            import array_api_compat.torch as torch
            composite_test(torch)

    #-----------------------------------------------
    def testvstack(self):
        vstack_test(np)
        if np.__version__ >= '1.25':
            vstack_test(nparr)

    if parallelproj.cupy_enabled:

        def testvstack_cp(self):
            import array_api_compat.cupy as cp
            vstack_test(cp)

    if parallelproj.torch_enabled:

        def testvstack_torch(self):
            import array_api_compat.torch as torch
            vstack_test(torch)

    #-----------------------------------------------
    def testsubsets(self):
        subsets_test(np)
        if np.__version__ >= '1.25':
            subsets_test(nparr)

    if parallelproj.cupy_enabled:

        def testsubsets(self):
            import array_api_compat.cupy as cp
            subsets_test(cp)

    if parallelproj.torch_enabled:

        def testsubsets_torch(self):
            import array_api_compat.torch as torch
            subsets_test(torch)


#--------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
