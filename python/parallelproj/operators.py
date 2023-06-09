from __future__ import annotations
import abc
import numpy as np


class LinearOperator(abc.ABC):

    def __init__(self):
        self._scale = 1

    @property
    @abc.abstractmethod
    def in_shape(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def out_shape(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def in_dtype_kind(self):
        """must return 'float' for real or 'complex' for complex """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def out_dtype_kind(self):
        """must return 'float' for real or 'complex' for complex """
        raise NotImplementedError

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        if not np.isscalar(value):
            raise ValueError
        self._scale = value

    @abc.abstractmethod
    def _call(self, x):
        """ forward step y = Ax"""
        raise NotImplementedError

    @abc.abstractmethod
    def _adjoint(self, y):
        """ adjoint step x = A^H y"""
        raise NotImplementedError

    def __call__(self, x):
        """ forward step y = scale * Ax"""
        if self._scale == 1:
            return self._call(x)
        else:
            return self._scale * self._call(x)

    def adjoint(self, y):
        """ adjoint step x = conj(scale) * A^H y"""
        if self._scale == 1:
            return self._adjoint(y)
        else:
            return np.conj(self._scale) * self._adjoint(y)

    def adjointness_test(self, xp, verbose=False, **kwargs):
        x = xp.random.rand(self.in_shape)
        y = xp.random.rand(self.out_shape)

        if self.in_dtype_kind == 'complex':
            x = x + 1j * xp.random.rand(self.in_shape)

        if self.out_dtype_kind == 'complex':
            y = y + 1j * xp.random.rand(self.out_shape)

        x_fwd = self.__call__(x)
        y_adj = self.adjoint(y)

        ip1 = (x_fwd.conj() * y).sum()
        ip2 = (x.conj() * y_adj).sum()

        if verbose:
            print(ip1, ip2)

        assert (xp.isclose(ip1, ip2, **kwargs))


class MatrixOperator(LinearOperator):

    def __init__(self, A):
        super().__init__()
        self._A = A

        if self._A.dtype.kind == 'c':
            self._dtype_kind = 'complex'
        else:
            self._dtype_kind = 'float'

    @property
    def in_shape(self):
        return self._A.shape[1]

    @property
    def out_shape(self):
        return self._A.shape[0]

    @property
    def in_dtype_kind(self):
        return self._dtype_kind

    @property
    def out_dtype_kind(self):
        return self._dtype_kind

    @property
    def A(self):
        return self._A

    def _call(self, x):
        """ forward step y = Ax"""
        return self._A @ x

    def _adjoint(self, y):
        """ adjoint step x = A^H y"""
        return self._A.conj().T @ y


class CompositeLinearOperator(LinearOperator):

    def __init__(self, operators: tuple[LinearOperator, ...]):
        super().__init__()
        self._operators = operators

    @property
    def in_shape(self):
        return self._operators[-1].in_shape

    @property
    def out_shape(self):
        return self._operators[0].out_shape

    @property
    def in_dtype_kind(self):
        return self._operators[-1].in_dtype_kind

    @property
    def out_dtype_kind(self):
        return self._operators[0].in_dtype_kind

    @property
    def operators(self) -> tuple[LinearOperator, ...]:
        return self._operators

    def _call(self, x):
        """ forward step y = Ax"""
        y = x
        for op in self._operators[::-1]:
            y = op(y)
        return y

    def _adjoint(self, y):
        """ adjoint step x = A^H y"""
        x = y
        for op in self._operators:
            x = op.adjoint(x)
        return x


#----------------------------------------------------

if __name__ == '__main__':
    np.random.seed(1)

    G0 = MatrixOperator(np.random.rand(3, 2) + np.random.rand(3, 2) * 1j)
    G1 = MatrixOperator(np.random.rand(5, 3) + np.random.rand(5, 3) * 1j)
    G2 = MatrixOperator(np.random.rand(4, 5) + np.random.rand(4, 5) * 1j)

    G0.scale = 2. - 3j
    G1.scale = 3. + 3j
    G2.scale = 1. + 2j

    G0.adjointness_test(np)
    G1.adjointness_test(np)
    G2.adjointness_test(np)

    G = CompositeLinearOperator((G2, G1, G0))
    G.scale = 2 - 1.4j

    G.adjointness_test(np)