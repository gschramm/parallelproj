from __future__ import annotations
from types import ModuleType
import abc
import numpy as np
import array_api_compat
from array_api_compat import device

import parallelproj


class LinearOperator(abc.ABC):
    """abstract base class for linear operators"""

    def __init__(self):
        self._scale = 1

    @property
    @abc.abstractmethod
    def in_shape(self) -> tuple[int, ...]:
        """shape of the input array"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def out_shape(self) -> tuple[int, ...]:
        """shape of the output array"""
        raise NotImplementedError

    @property
    def scale(self) -> int | float | complex:
        """scalar factor applied to the linear operator"""
        return self._scale

    @scale.setter
    def scale(self, value: int | float | complex):
        if not np.isscalar(value):
            raise ValueError
        self._scale = value

    @abc.abstractmethod
    def _apply(self, x):
        """forward step y = Ax"""
        raise NotImplementedError

    @abc.abstractmethod
    def _adjoint(self, y):
        """adjoint step x = A^H y"""
        raise NotImplementedError

    def apply(self, x):
        """forward step y = scale * Ax

        Parameters
        ----------
        x : numpy or cupy array

        Returns
        -------
        numpy or cupy array
        """
        if self._scale == 1:
            return self._apply(x)
        else:
            return self._scale * self._apply(x)

    def __call__(self, x):
        return self.apply(x)

    def adjoint(self, y):
        """adjoint step x = conj(scale) * A^H y

        Parameters
        ----------
        y : numpy or cupy array

        Returns
        -------
        numpy or cupy array
        """
        if self._scale == 1:
            return self._adjoint(y)
        else:
            return np.conj(self._scale) * self._adjoint(y)

    def adjointness_test(self,
                         xp: ModuleType,
                         verbose: bool = False,
                         iscomplex: bool = False,
                         **kwargs):
        """test whether the adjoint is correctly implemented

        Parameters
        ----------
        xp : ModuleType
            array module to use
        verbose : bool, optional
            verbose output
        iscomplex : bool, optional
            use complex arrays
        **kwargs : dict
            passed to np.isclose
        """

        x = xp.asarray(np.random.rand(*self.in_shape))
        y = xp.asarray(np.random.rand(*self.out_shape))

        if iscomplex:
            x = x + 1j * xp.random.rand(*self.in_shape)

        if iscomplex:
            y = y + 1j * xp.random.rand(*self.out_shape)

        x_fwd = self.apply(x)
        y_adj = self.adjoint(y)

        if iscomplex:
            ip1 = complex(xp.sum(xp.conj(x_fwd) * y))
            ip2 = complex(xp.sum(xp.conj(x) * y_adj))
        else:
            ip1 = float(xp.sum(x_fwd * y))
            ip2 = float(xp.sum(x * y_adj))

        if verbose:
            print(ip1, ip2)

        assert (np.isclose(ip1, ip2, **kwargs))

    def norm(self,
             xp: ModuleType,
             num_iter: int = 30,
             iscomplex: bool = False,
             verbose: bool = False) -> float:
        """estimate norm of the linear operator using power iterations

        Parameters
        ----------
        xp : ModuleType
            array module to use
        num_iter : int, optional
            number of power iterations
        iscomplex : bool, optional
            use complex arrays
        verbose : bool, optional
            verbose output

        Returns
        -------
        float
            the norm of the linear operator
        """

        x = xp.asarray(np.random.rand(*self.in_shape))

        if iscomplex:
            x = x + 1j * xp.asarray(np.random.rand(*self.in_shape))

        for i in range(num_iter):
            x = self.adjoint(self.apply(x))
            norm_squared = xp.sqrt(xp.sum(xp.abs(x)**2))
            x /= norm_squared

            if verbose:
                print(f'{(i+1):03} {xp.sqrt(norm_squared):.2E}')

        return float(xp.sqrt(norm_squared))


class MatrixOperator(LinearOperator):
    """Linear Operator defined by dense matrix multiplication"""

    def __init__(self, A):
        """init method

        Parameters
        ----------
        A : 2D numpy or cupy real of complex array
        """
        super().__init__()
        self._A = A

    @property
    def in_shape(self):
        return (self._A.shape[1], )

    @property
    def out_shape(self):
        return (self._A.shape[0], )

    @property
    def A(self):
        return self._A

    @property
    def xp(self):
        return array_api_compat.get_namespace(self._A)

    def iscomplex(self):
        return self.xp.isdtype(self._A.dtype,
                               self.xp.complex64) or self.xp.isdtype(
                                   self._A.dtype, self.xp.complex128)

    def _apply(self, x):
        """forward step y = Ax"""
        return self.xp.matmul(self._A, x)

    def _adjoint(self, y):
        """adjoint step x = A^H y"""

        if self.iscomplex():
            return self.xp.matmul(self.xp.conj(self._A).T, y)
        else:
            return self.xp.matmul(self._A.T, y)


class CompositeLinearOperator(LinearOperator):
    """Composite Linear Operator defined by a sequence of Linear Operators
    
       Given a tuple of operators (A_0, ..., A_{n-1}) the composite operator is defined as
       A(x) = A0( A1( ... ( A_{n-1}(x) ) ) ) 
    """

    def __init__(self, operators: tuple[LinearOperator, ...]):
        """init method

        Parameters
        ----------
        operators : tuple[LinearOperator, ...]
            tuple of linear operators
        """
        super().__init__()
        self._operators = operators

    @property
    def in_shape(self):
        return self._operators[-1].in_shape

    @property
    def out_shape(self):
        return self._operators[0].out_shape

    @property
    def operators(self) -> tuple[LinearOperator, ...]:
        """tuple of linear operators"""
        return self._operators

    def _apply(self, x):
        """forward step y = Ax"""
        y = x
        for op in self._operators[::-1]:
            y = op(y)
        return y

    def _adjoint(self, y):
        """adjoint step x = A^H y"""
        x = y
        for op in self._operators:
            x = op.adjoint(x)
        return x


class ElementwiseMultiplicationOperator(LinearOperator):
    """Element-wise multiplication operator (multiplication with a diagonal matrix)"""

    def __init__(self, values):
        """init method

        Parameters
        ----------
        values : numpy or cupy array
            values of the diagonal matrix
        """
        super().__init__()
        self._values = values

    @property
    def in_shape(self) -> tuple[int, ...]:
        return self._values.shape

    @property
    def out_shape(self) -> tuple[int, ...]:
        return self._values.shape

    @property
    def xp(self) -> ModuleType:
        return array_api_compat.get_namespace(self._values)

    def _apply(self, x):
        """forward step y = Ax"""
        return self._values * x

    def _adjoint(self, y):
        """adjoint step x = A^H y"""

        if self.iscomplex():
            return self.xp.conj(self._values) * y
        else:
            return self._values * y

    def iscomplex(self):
        return self.xp.isdtype(self._values.dtype,
                               self.xp.complex64) or self.xp.isdtype(
                                   self._values.dtype, self.xp.complex128)


class GaussianFilterOperator(LinearOperator):
    """Gaussian filter operator"""

    def __init__(self, in_shape, **kwargs):
        """init method

        Parameters
        ----------
        in_shape : tuple[int, ...]
            shape of the input array
        **kwargs : sometype
            passed to the ndimage gaussian_filter function
        """
        super().__init__()
        self._in_shape = in_shape
        self._kwargs = kwargs

    @property
    def in_shape(self):
        return self._in_shape

    @property
    def out_shape(self):
        return self._in_shape

    def _apply(self, x):
        """forward step y = Ax"""
        xp = array_api_compat.get_namespace(x)

        if parallelproj.is_cuda_array(x):
            import cupy as cp
            import cupyx.scipy.ndimage as ndimagex
            return xp.asarray(
                ndimagex.gaussian_filter(cp.asarray(x), **self._kwargs))
        else:
            import scipy.ndimage as ndimage
            return xp.asarray(
                ndimage.gaussian_filter(np.asarray(x), **self._kwargs))

    def _adjoint(self, y):
        """adjoint step x = A^H y"""
        return self._apply(y)


class VstackOperator(LinearOperator):
    """Stacking operator for stacking multiple linear operators vertically"""

    def __init__(self, operators: tuple[LinearOperator, ...]) -> None:
        """init method

        Parameters
        ----------
        operators : tuple[LinearOperator, ...]
            tuple of linear operators
        """
        super().__init__()
        self._operators = operators
        self._in_shape = self._operators[0].in_shape
        self._out_shapes = tuple([x.out_shape for x in operators])
        self._raveled_out_shapes = tuple(
            [np.prod(x) for x in self._out_shapes])
        self._out_shape = (sum(self._raveled_out_shapes), )

        # setup the slices for slicing the raveled output array
        self._slices = []
        start = 0
        for length in self._raveled_out_shapes:
            end = start + length
            self._slices.append(slice(start, end, None))
            start = end
        self._slices = tuple(self._slices)

    @property
    def in_shape(self):
        return self._in_shape

    @property
    def out_shape(self):
        return self._out_shape

    def _apply(self, x):
        xp = array_api_compat.get_namespace(x)
        y = xp.zeros(self._out_shape, dtype=x.dtype)

        for i, op in enumerate(self._operators):
            y[self._slices[i]] = xp.reshape(op(x), (-1, ))

        return y

    def _adjoint(self, y):
        xp = array_api_compat.get_namespace(y)
        x = xp.zeros(self._in_shape, dtype=y.dtype)

        for i, op in enumerate(self._operators):
            x += op.adjoint(xp.reshape(y[self._slices[i]],
                                       self._out_shapes[i]))

        return x


class SubsetOperator:
    """Operator split into subsets"""

    def __init__(self, operators: tuple[LinearOperator, ...]) -> None:
        """init method

        Parameters
        ----------
        operators : tuple[LinearOperator, ...]
            tuple of linear operators
        """
        self._operators = operators
        self._in_shape = self._operators[0].in_shape
        self._out_shapes = tuple([x.out_shape for x in operators])
        self._num_subsets = len(operators)

    @property
    def in_shape(self):
        return self._in_shape

    @property
    def out_shapes(self):
        return self._out_shapes

    @property
    def operators(self) -> tuple[LinearOperator, ...]:
        return self._operators

    @property
    def num_subsets(self):
        return self._num_subsets

    def apply(self, x):
        """A_i x for all subsets i"""

        y = [op(x) for op in self._operators]

        return y

    def __call__(self, x):
        return self.apply(x)

    def adjoint(self, y):
        """A_i^H x for all subsets i"""

        x = []

        for i, op in enumerate(self._operators):
            x.append(op.adjoint(y[i]))

        return x

    def apply_subset(self, x, i):
        """A_i x for a given subset i"""
        return self._operators[i](x)

    def adjoint_subset(self, x, i):
        """A_i^H x for a given subset i"""
        return self._operators[i].adjoint(x)

    def norms(self, xp: ModuleType):
        """norm(A_i) for all subsets i"""
        return [op.norm(xp) for op in self._operators]


class FiniteForwardDifference(LinearOperator):
    """finite difference gradient operator"""

    def __init__(self, in_shape: tuple[int, ...]) -> None:

        if len(in_shape) > 4:
            raise ValueError('only up to 4 dimensions supported')

        self._ndim = len(in_shape)
        self._in_shape = in_shape
        self._out_shape = (self.ndim, ) + in_shape
        super().__init__()

    @property
    def in_shape(self) -> tuple[int, ...]:
        return self._in_shape

    @property
    def out_shape(self) -> tuple[int, ...]:
        return self._out_shape

    @property
    def ndim(self) -> int:
        return self._ndim

    def _apply(self, x):
        xp = array_api_compat.get_namespace(x)

        g = xp.zeros(self.out_shape, dtype=x.dtype, device=device(x))

        if self.ndim == 1:
            g[0, :-1] = x[1:] - x[:-1]
        elif self.ndim == 2:
            g[0, :-1, :] = x[1:, :] - x[:-1, :]
            g[1, :, :-1] = x[:, 1:] - x[:, :-1]
        elif self.ndim == 3:
            g[0, :-1, :, :] = x[1:, :, :] - x[:-1, :, :]
            g[1, :, :-1, :] = x[:, 1:, :] - x[:, :-1, :]
            g[2, :, :, :-1] = x[:, :, 1:] - x[:, :, :-1]
        elif self.ndim == 4:
            g[0, :-1, :, :, :] = x[1:, :, :, :] - x[:-1, :, :, :]
            g[1, :, :-1, :, :] = x[:, 1:, :, :] - x[:, :-1, :, :]
            g[2, :, :, :-1, :] = x[:, :, 1:, :] - x[:, :, :-1, :]
            g[3, :, :, :, :-1] = x[:, :, :, 1:] - x[:, :, :, :-1]

        return g

    def _adjoint(self, y):
        xp = array_api_compat.get_namespace(y)

        if self.ndim == 1:
            tmp0 = xp.asarray(y[0, ...], copy=True)
            tmp0[-1] = 0

            div0 = xp.zeros(self.in_shape, dtype=y.dtype, device=device(y))
            div0[1:] = -tmp0[1:] + tmp0[:-1]
            div0[0] = -tmp0[0]

            res = div0

        elif self.ndim == 2:
            tmp0 = xp.asarray(y[0, ...], copy=True)
            tmp1 = xp.asarray(y[1, ...], copy=True)
            tmp0[-1, :] = 0
            tmp1[:, -1] = 0

            div0 = xp.zeros(self.in_shape, dtype=y.dtype, device=device(y))
            div1 = xp.zeros(self.in_shape, dtype=y.dtype, device=device(y))

            div0[1:, :] = -tmp0[1:, :] + tmp0[:-1, :]
            div1[:, 1:] = -tmp1[:, 1:] + tmp1[:, :-1]

            div0[0, :] = -tmp0[0, :]
            div1[:, 0] = -tmp1[:, 0]

            res = div0 + div1

        elif self.ndim == 3:
            tmp0 = xp.asarray(y[0, ...], copy=True)
            tmp1 = xp.asarray(y[1, ...], copy=True)
            tmp2 = xp.asarray(y[2, ...], copy=True)
            tmp0[-1, :, :] = 0
            tmp1[:, -1, :] = 0
            tmp2[:, :, -1] = 0

            div0 = xp.zeros(self.in_shape, dtype=y.dtype, device=device(y))
            div1 = xp.zeros(self.in_shape, dtype=y.dtype, device=device(y))
            div2 = xp.zeros(self.in_shape, dtype=y.dtype, device=device(y))

            div0[1:, :, :] = -tmp0[1:, :, :] + tmp0[:-1, :, :]
            div1[:, 1:, :] = -tmp1[:, 1:, :] + tmp1[:, :-1, :]
            div2[:, :, 1:] = -tmp2[:, :, 1:] + tmp2[:, :, :-1]

            div0[0, :, :] = -tmp0[0, :, :]
            div1[:, 0, :] = -tmp1[:, 0, :]
            div2[:, :, 0] = -tmp2[:, :, 0]

            res = div0 + div1 + div2

        elif self.ndim == 4:
            tmp0 = xp.asarray(y[0, ...], copy=True)
            tmp1 = xp.asarray(y[1, ...], copy=True)
            tmp2 = xp.asarray(y[2, ...], copy=True)
            tmp3 = xp.asarray(y[3, ...], copy=True)
            tmp0[-1, :, :, :] = 0
            tmp1[:, -1, :, :] = 0
            tmp2[:, :, -1, :] = 0
            tmp3[:, :, :, -1] = 0

            div0 = xp.zeros(self.in_shape, dtype=y.dtype, device=device(y))
            div1 = xp.zeros(self.in_shape, dtype=y.dtype, device=device(y))
            div2 = xp.zeros(self.in_shape, dtype=y.dtype, device=device(y))
            div3 = xp.zeros(self.in_shape, dtype=y.dtype, device=device(y))

            div0[1:, :, :, :] = -tmp0[1:, :, :, :] + tmp0[:-1, :, :, :]
            div1[:, 1:, :, :] = -tmp1[:, 1:, :, :] + tmp1[:, :-1, :, :]
            div2[:, :, 1:, :] = -tmp2[:, :, 1:, :] + tmp2[:, :, :-1, :]
            div3[:, :, :, 1:] = -tmp3[:, :, :, 1:] + tmp3[:, :, :, :-1]

            div0[0, :, :, :] = -tmp0[0, :, :, :]
            div1[:, 0, :, :] = -tmp1[:, 0, :, :]
            div2[:, :, 0, :] = -tmp2[:, :, 0, :]
            div3[:, :, :, 0] = -tmp3[:, :, :, 0]

            res = div0 + div1 + div2 + div3

        return res
