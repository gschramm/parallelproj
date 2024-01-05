from __future__ import annotations

import pytest
import parallelproj
import array_api_compat
import array_api_compat.numpy as np

from types import ModuleType
from math import prod

from config import pytestmark


def allclose(x, y, atol: float = 1e-8, rtol: float = 1e-5) -> bool:
    """check if two arrays are close to each other, given absolute and relative error
       inspired by numpy.allclose
    """
    xp = array_api_compat.array_namespace(x)
    return bool(xp.all(xp.less_equal(xp.abs(x - y), atol + rtol * xp.abs(y))))


#---------------------------------------------------------------------------------------


def test_matrix(xp: ModuleType, dev: str):
    np.random.seed(0)

    A = xp.asarray([[1., 2.], [-3., 2.], [-1., -1.]], device=dev)
    x = xp.asarray([-2., 1.], device=dev)

    op = parallelproj.MatrixOperator(A)

    # set scale that is not 1
    scale_fac = -2.5
    op.scale = scale_fac

    assert op.scale == scale_fac
    assert allclose(op.A, A)

    with pytest.raises(ValueError):
        op.scale = xp.ones(1, device=dev)

    # test call to norm
    op_norm = op.norm(xp, dev)

    op.adjointness_test(xp, dev)
    assert allclose(scale_fac * (A @ x), op(x))


def test_complex_matrix(xp: ModuleType, dev: str):
    np.random.seed(0)

    A = xp.asarray([[1., 2j], [-3., 2.], [-1., -1.]],
                   device=dev,
                   dtype=xp.complex128)
    x = xp.asarray([-2., 1.], device=dev, dtype=xp.complex128)

    op = parallelproj.MatrixOperator(A)
    op.adjointness_test(xp, dev, iscomplex=True)

    n = op.norm(xp, dev, iscomplex=True)

    assert allclose((A @ x), op(x))


def test_elementwise(xp: ModuleType, dev: str):
    np.random.seed(0)

    v = xp.asarray([3., -1.], device=dev)
    x = xp.asarray([-2., 1.], device=dev)

    op = parallelproj.ElementwiseMultiplicationOperator(v)
    # test call to norm
    op_norm = op.norm(xp, dev)

    assert xp.all(op.values == v)

    op.adjointness_test(xp, dev)
    assert allclose(v * x, op(x))

def test_tofnontofelemenwise(xp: ModuleType, dev: str):
    np.random.seed(0)

    x = xp.reshape(xp.arange(3*3*2, device = dev, dtype = xp.float32), (3,3,2))
    v = xp.reshape(xp.arange(3*3, device = dev, dtype = xp.float32), (3,3))

    op = parallelproj.TOFNonTOFElementwiseMultiplicationOperator(x.shape, v)
    # test call to norm

    assert xp.all(op.values == v)

    op.adjointness_test(xp, dev)
    assert allclose(v * x[...,0], op(x)[...,0])
    assert allclose(v * x[...,1], op(x)[...,1])

def test_elemenwise_complex(xp: ModuleType, dev: str):
    np.random.seed(0)

    v = xp.asarray([3j, -1.], device=dev, dtype=xp.complex128)
    x = xp.asarray([-2., 1j], device=dev, dtype=xp.complex128)

    op = parallelproj.ElementwiseMultiplicationOperator(v)
    # test call to norm
    op_norm = op.norm(xp, dev, iscomplex=True)

    op.adjointness_test(xp, dev, iscomplex=True)
    assert allclose(v * x, op(x))

def test_tofnontofelemenwise_complex(xp: ModuleType, dev: str):
    np.random.seed(0)

    x = xp.ones((3,3,2), device = dev, dtype = xp.complex128)
    x[0,0,1] = 2. + 1j
    x[1,1,0] = 1. - 2j
    v = xp.ones((3,3), device = dev, dtype = xp.complex128)
    v[2,2] = 3. + 2j
    v[1,2] = -4. + 1j

    op = parallelproj.TOFNonTOFElementwiseMultiplicationOperator(x.shape, v)
    # test call to norm

    op.adjointness_test(xp, dev, iscomplex=True)
    assert allclose(v * x[...,0], op(x)[...,0])
    assert allclose(v * x[...,1], op(x)[...,1])



def test_gaussian(xp: ModuleType, dev: str):
    np.random.seed(0)
    in_shape = (32, 32)
    sigma1 = 2.3

    op = parallelproj.GaussianFilterOperator(in_shape, sigma=sigma1)
    op.adjointness_test(xp, dev)

    sigma2 = xp.asarray([2.3, 1.2], device=dev)
    op = parallelproj.GaussianFilterOperator(in_shape, sigma=sigma2)
    op.adjointness_test(xp, dev)


def test_composite(xp: ModuleType, dev: str):
    np.random.seed(0)

    A = xp.asarray([[1., 2.], [-3., 2.], [-1., -1.]], device=dev)
    x = xp.asarray([-2., 1.], device=dev)
    v = xp.asarray([3., -1., 2.], device=dev)

    op1 = parallelproj.ElementwiseMultiplicationOperator(v)
    op2 = parallelproj.MatrixOperator(A)

    op = parallelproj.CompositeLinearOperator([op1, op2])

    assert op.operators == [op1, op2]

    # test call to norm
    op_norm = op.norm(xp, dev)

    op.adjointness_test(xp, dev)
    assert allclose(v * (A @ x), op(x))


def test_vstack(xp: ModuleType, dev: str):
    np.random.seed(0)
    in_shape = (16, 11)

    A1 = parallelproj.GaussianFilterOperator(in_shape, sigma=1.)
    A2 = parallelproj.ElementwiseMultiplicationOperator(
        xp.asarray(np.random.rand(*in_shape), device=dev))
    A3 = parallelproj.GaussianFilterOperator(in_shape, sigma=2.)

    A = parallelproj.VstackOperator((A1, A2, A3))
    # test call to norm
    A_norm = A.norm(xp, dev)

    A.adjointness_test(xp, dev)

    x = xp.asarray(np.random.rand(*in_shape), device=dev)
    x_fwd = A(x)

    assert allclose(
        x_fwd,
        xp.concat((xp.reshape(A1(x), (-1, )), xp.reshape(A2(x), (-1, )),
                   xp.reshape(A3(x), (-1, )))))


def test_subsets(xp: ModuleType, dev: str):
    np.random.seed(0)
    in_shape = (3, )

    A1 = parallelproj.MatrixOperator(
        xp.asarray(np.random.randn(4, 3), device=dev))
    A2 = parallelproj.MatrixOperator(
        xp.asarray(np.random.randn(5, 3), device=dev))
    A3 = parallelproj.MatrixOperator(
        xp.asarray(np.random.randn(2, 3), device=dev))

    A = parallelproj.SubsetOperator((A1, A2, A3))

    # test call to norms
    ns = A.norms(xp, dev)

    x = xp.asarray(np.random.rand(*in_shape), device=dev)
    x_fwd = A(x)

    for i in range(A.num_subsets):
        assert allclose(x_fwd[i], A.apply_subset(x, i))

    y = A.adjoint(x_fwd)
    tmp = sum([A.adjoint_subset(x_fwd[i], i) for i in range(A.num_subsets)])

    assert allclose(y, tmp)

    assert A.in_shape == A3.in_shape
    assert A.out_shapes == (A1.out_shape, A2.out_shape, A3.out_shape)
    assert A.operators == (A1, A2, A3)


def test_finite_difference(xp: ModuleType, dev: str):

    # 1D tests
    A = parallelproj.FiniteForwardDifference((3, ))
    x = xp.reshape(xp.arange(prod(A.in_shape), device=dev), A.in_shape)

    n = A.norm(xp, dev)
    # test adjointness
    A.adjointness_test(xp, dev)

    # test simple forward
    y = A(x)
    assert xp.all(y[0, :-1] == 1)

    # 2D tests
    A = parallelproj.FiniteForwardDifference((5, 3))
    x = xp.reshape(xp.arange(prod(A.in_shape), device=dev), A.in_shape)

    # test call to norm
    n = A.norm(xp, dev)
    # test adjointness
    A.adjointness_test(xp, dev)

    # test simple forward
    y = A(x)
    assert xp.all(y[0, :-1, :] == 3)
    assert xp.all(y[1, :, :-1] == 1)

    # 3D tests
    A = parallelproj.FiniteForwardDifference((5, 3, 4))
    x = xp.reshape(xp.arange(prod(A.in_shape), device=dev), A.in_shape)

    # test adjointness
    A.adjointness_test(xp, dev)

    # test simple forward
    y = A(x)
    assert xp.all(y[0, :-1, :, :] == 12)
    assert xp.all(y[1, :, :-1, :] == 4)
    assert xp.all(y[2, :, :, :-1] == 1)

    # 4D tests
    A = parallelproj.FiniteForwardDifference((5, 3, 4, 6))
    x = xp.reshape(xp.arange(prod(A.in_shape), device=dev), A.in_shape)

    # test adjointness
    A.adjointness_test(xp, dev)

    # test simple forward
    y = A(x)
    assert xp.all(y[0, :-1, :, :, :] == 72)
    assert xp.all(y[1, :, :-1, :, :] == 24)
    assert xp.all(y[2, :, :, :-1, :] == 6)
    assert xp.all(y[3, :, :, :, :-1] == 1)

    with pytest.raises(ValueError):
        A = parallelproj.FiniteForwardDifference((3, 3, 3, 3, 3))
