from __future__ import annotations
import abc
import numpy as np


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

    def adjointness_test(self, xp, verbose=False, iscomplex=False, **kwargs):
        """test whether the adjoint is correctly implemented

        Parameters
        ----------
        xp : numpy or cupy module
        verbose : bool, optional
            verbose output
        iscomplex : bool, optional
            use complex arrays
        """
        x = xp.random.rand(*self.in_shape)
        y = xp.random.rand(*self.out_shape)

        if iscomplex:
            x = x + 1j * xp.random.rand(*self.in_shape)

        if iscomplex:
            y = y + 1j * xp.random.rand(*self.out_shape)

        x_fwd = self.__call__(x)
        y_adj = self.adjoint(y)

        ip1 = (x_fwd.conj() * y).sum()
        ip2 = (x.conj() * y_adj).sum()

        if verbose:
            print(ip1, ip2)

        assert (xp.isclose(ip1, ip2, **kwargs))

    def norm(self, xp, num_iter=30, iscomplex=False, verbose=False) -> float:
        """estimate norm of the linear operator using power iterations

        Parameters
        ----------
        xp : numpy or cupy module
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
        x = xp.random.rand(*self.in_shape)

        if iscomplex:
            x = x + 1j * xp.random.rand(*self.in_shape)

        for i in range(num_iter):
            x = self.adjoint(self.__call__(x))
            norm_squared = xp.linalg.norm(x)
            x /= norm_squared

            if verbose:
                print(f'{(i+1):03} {xp.sqrt(norm_squared):.2E}')

        return xp.sqrt(norm_squared)


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

    def __init__(self, in_shape, ndimage_module, **kwargs):
        """init method

        Parameters
        ----------
        in_shape : tuple[int, ...]
            shape of the input array
        ndimage_module : scipy or cupyx.scipy module
        **kwargs : sometype
            passed to the ndimage gaussian_filter function
        """
        super().__init__()
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
