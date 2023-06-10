from __future__ import annotations
import abc
import numpy as np

class LinearOperator(abc.ABC):

    def __init__(self):
        self._scale = 1

    @property
    @abc.abstractmethod
    def in_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def out_shape(self) -> tuple[int, ...]:
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

    def adjointness_test(self, xp, verbose=False, iscomplex = False, **kwargs):
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

    def norm(self, xp, num_iter=30, iscomplex = False, verbose=False):
        """estimate the norm of the operator using power iterations"""
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

    def __init__(self, A):
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


class ElementwiseMultiplicationOperator(LinearOperator):

    def __init__(self, values):
        super().__init__()
        self._values = values

    @property
    def in_shape(self):
        return self._values.shape

    @property
    def out_shape(self):
        return self._values.shape

    def _call(self, x):
        """ forward step y = Ax"""
        return self._values * x

    def _adjoint(self, y):
        """ adjoint step x = A^H y"""
        return self._values.conj() * y


class GaussianFilterOperator(LinearOperator):

    def __init__(self, in_shape, ndimage_module, **kwargs):
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
        """ forward step y = Ax"""
        return self._ndimage_module.gaussian_filter(x, **self._kwargs)

    def _adjoint(self, y):
        """ adjoint step x = A^H y"""
        return self._call(y)



#----------------------------------------------------

if __name__ == '__main__':
    #import numpy as xp
    #import scipy.ndimage as ndi

    import cupy as xp
    import cupyx.scipy.ndimage as ndi

    from parallelproj.projectors import ParallelViewProjector2D

    #----------------------------------------------------------------------
    projector = ParallelViewProjector2D((1, 128, 128),
                                xp.linspace(-200, 200, 131, dtype=xp.float32),
                                210, 300.,
                                xp.array([0, -63.5, -63.5], dtype=xp.float32),
                                xp.ones(3, dtype=xp.float32), xp)


    image_space_filter = GaussianFilterOperator(projector.in_shape,
                               ndi,
                               sigma=1.3)

    sensitivity = ElementwiseMultiplicationOperator(xp.full(projector.out_shape, 2.3, dtype=xp.float32))

    fwd_model = CompositeLinearOperator((sensitivity, projector, image_space_filter))

    fwd_model.adjointness_test(xp, verbose=True)
    fwd_model_norm = fwd_model.norm(xp)

    fig = projector.show_views()
    fig.show()