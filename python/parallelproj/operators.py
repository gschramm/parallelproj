from __future__ import annotations
from types import ModuleType
import abc
import numpy as np

class LinearOperator(abc.ABC):
    """abstract base class for linear operators"""

    def __init__(self, xp: ModuleType):
        """init method

        Parameters
        ----------
        xp : ModuleType
            numpy or cupy module
        """        
        self._scale = 1
        self._xp = xp

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

    @property
    def xp(self) -> ModuleType:
        """module type (numpy or cupy) of the operator"""
        return self._xp

    @abc.abstractmethod
    def _call(self, x):
        """forward step y = Ax"""
        raise NotImplementedError

    @abc.abstractmethod
    def _adjoint(self, y):
        """adjoint step x = A^H y"""
        raise NotImplementedError

    def __call__(self, x):
        """forward step y = scale * Ax

        Parameters
        ----------
        x : numpy or cupy array

        Returns
        -------
        numpy or cupy array
        """
        if self._scale == 1:
            return self._call(x)
        else:
            return self._scale * self._call(x)

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

    def adjointness_test(self, verbose=False, iscomplex=False, **kwargs):
        """test whether the adjoint is correctly implemented

        Parameters
        ----------
        verbose : bool, optional
            verbose output
        iscomplex : bool, optional
            use complex arrays
        """
        x = self.xp.random.rand(*self.in_shape)
        y = self.xp.random.rand(*self.out_shape)

        if iscomplex:
            x = x + 1j * self.xp.random.rand(*self.in_shape)

        if iscomplex:
            y = y + 1j * self.xp.random.rand(*self.out_shape)

        x_fwd = self.__call__(x)
        y_adj = self.adjoint(y)

        ip1 = (x_fwd.conj() * y).sum()
        ip2 = (x.conj() * y_adj).sum()

        if verbose:
            print(ip1, ip2)

        assert (self.xp.isclose(ip1, ip2, **kwargs))

    def norm(self, num_iter=30, iscomplex=False, verbose=False) -> float:
        """estimate norm of the linear operator using power iterations

        Parameters
        ----------
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
        x = self.xp.random.rand(*self.in_shape)

        if iscomplex:
            x = x + 1j * self.xp.random.rand(*self.in_shape)

        for i in range(num_iter):
            x = self.adjoint(self.__call__(x))
            norm_squared = self.xp.linalg.norm(x)
            x /= norm_squared

            if verbose:
                print(f'{(i+1):03} {self.xp.sqrt(norm_squared):.2E}')

        return self.xp.sqrt(norm_squared)


class MatrixOperator(LinearOperator):
    """Linear Operator defined by dense matrix multiplication"""

    def __init__(self, A, xp):
        """init method

        Parameters
        ----------
        A : 2D numpy or cupy real of complex array
        xp: ModuleType
            numpy or cupy module
        """
        super().__init__(xp)
        self._A = A

        if self._A.dtype.kind == 'c':
            self._dtype_kind = 'complex'
        else:
            self._dtype_kind = 'float'

    @property
    def in_shape(self):
        return (self._A.shape[1], )

    @property
    def out_shape(self):
        return (self._A.shape[0], )

    @property
    def A(self):
        return self._A

    def _call(self, x):
        """forward step y = Ax"""
        return self._A @ x

    def _adjoint(self, y):
        """adjoint step x = A^H y"""
        return self._A.conj().T @ y


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
        super().__init__(operators[0].xp)
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

    def _call(self, x):
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

    def __init__(self, values, xp):
        """init method

        Parameters
        ----------
        values : numpy or cupy array
            values of the diagonal matrix
        xp: ModuleType
            numpy or cupy module
        """
        super().__init__(xp)
        self._values = values

    @property
    def in_shape(self):
        return self._values.shape

    @property
    def out_shape(self):
        return self._values.shape

    def _call(self, x):
        """forward step y = Ax"""
        return self._values * x

    def _adjoint(self, y):
        """adjoint step x = A^H y"""
        return self._values.conj() * y


class GaussianFilterOperator(LinearOperator):
    """Gaussian filter operator"""

    def __init__(self, in_shape, ndimage_module, xp, **kwargs):
        """init method

        Parameters
        ----------
        in_shape : tuple[int, ...]
            shape of the input array
        ndimage_module : scipy or cupyx.scipy module
        xp: ModuleType
            numpy or cupy module
        **kwargs : sometype
            passed to the ndimage gaussian_filter function
        """
        super().__init__(xp)
        self._in_shape = in_shape
        self._ndimage_module = ndimage_module
        self._kwargs = kwargs

    @property
    def in_shape(self):
        return self._in_shape

    @property
    def out_shape(self):
        return self._in_shape

    def _call(self, x):
        """forward step y = Ax"""
        return self._ndimage_module.gaussian_filter(x, **self._kwargs)

    def _adjoint(self, y):
        """adjoint step x = A^H y"""
        return self._call(y)


class VstackOperator(LinearOperator):

    def __init__(self, operators: tuple[LinearOperator, ...]) -> None:
        """init method

        Parameters
        ----------
        operators : tuple[LinearOperator, ...]
            tuple of linear operators
        """
        super().__init__(operators[0].xp)
        self._operators = operators
        self._in_shape = self._operators[0].in_shape
        self._out_shapes = tuple([x.out_shape for x in operators])
        self._raveled_out_shapes = tuple([np.prod(x) for x in self._out_shapes])
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

    def _call(self, x):
        y = self.xp.zeros(self._out_shape, dtype=x.dtype)

        for i, op in enumerate(self._operators):
            y[self._slices[i]] = op(x).ravel()

        return y

    def _adjoint(self, y):
        x = self.xp.zeros(self._in_shape, dtype=y.dtype)

        for i, op in enumerate(self._operators):
            x += op.adjoint(y[self._slices[i]].reshape(self._out_shapes[i]))

        return x