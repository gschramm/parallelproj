from __future__ import annotations
import abc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import parallelproj


def tonumpy(x, xp):
    if xp.__name__ == 'numpy':
        return x
    elif xp.__name__ == 'cupy':
        return xp.asnumpy(x)
    else:
        raise ValueError


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


class ParallelViewProjector2D(LinearOperator):

    def __init__(self, image_shape, radial_positions, num_views, radius,
                 image_origin, voxel_size, xp):

        super().__init__()

        self._image_shape = image_shape
        self._radial_positions = radial_positions

        self._num_rad = radial_positions.shape[0]
        self._num_views = num_views

        self._radius = radius
        self._image_origin = image_origin
        self._voxel_size = voxel_size

        self._xp = xp

        # array of projection angles
        self._view_angles = self._xp.linspace(0,
                                              xp.pi,
                                              self._num_views,
                                              endpoint=False)

        self._xstart = self._xp.zeros((self._num_views, self._num_rad, 3),
                                      dtype=xp.float32)
        self._xend = self._xp.zeros((self._num_views, self._num_rad, 3),
                                    dtype=xp.float32)

        for i, phi in enumerate(self._view_angles):
            # world coordinates of LOR start points
            self._xstart[
                i, :,
                1] = self._xp.cos(phi) * self._radial_positions + self._xp.sin(
                    phi) * self._radius
            self._xstart[i, :, 2] = -self._xp.sin(
                phi) * self._radial_positions + self._xp.cos(
                    phi) * self._radius
            # world coordinates of LOR endpoints
            self._xend[
                i, :,
                1] = self._xp.cos(phi) * self._radial_positions - self._xp.sin(
                    phi) * self._radius
            self._xend[i, :, 2] = -self._xp.sin(
                phi) * self._radial_positions - self._xp.cos(
                    phi) * self._radius

    @property
    def in_shape(self):
        return self._image_shape

    @property
    def out_shape(self):
        return (self._num_views, self._num_rad)

    def _call(self, x):
        y = self._xp.zeros(self.out_shape, dtype=self._xp.float32)
        parallelproj.joseph3d_fwd(self._xstart.reshape(-1, 3),
                                  self._xend.reshape(-1, 3),
                                  x.astype(self._xp.float32),
                                  self._image_origin, self._voxel_size, y)
        return y

    def _adjoint(self, y):
        x = self._xp.zeros(self.in_shape, dtype=self._xp.float32)
        parallelproj.joseph3d_back(self._xstart.reshape(-1, 3),
                                   self._xend.reshape(-1, 3), x,
                                   self._image_origin, self._voxel_size,
                                   y.astype(self._xp.float32))
        return x

    def show_views(self, views_to_show=None, image=None, **kwargs):
        if views_to_show is None:
            views_to_show = np.linspace(0, self._num_views - 1, 5).astype(int)

        num_cols = len(views_to_show)
        fig, ax = plt.subplots(1, num_cols, figsize=(num_cols * 3, 3))

        tmp1 = float(self._image_origin[1] - 0.5 * self._voxel_size[1])
        tmp2 = float(self._image_origin[2] - 0.5 * self._voxel_size[2])
        img_extent = [tmp1, -tmp1, tmp2, -tmp2]

        for i, ip in enumerate(views_to_show):
            ax[i].plot(tonumpy(self._xstart[ip, :, 1], self._xp),
                       tonumpy(self._xstart[ip, :, 2], self._xp),
                       '.',
                       ms=0.5)
            ax[i].plot(tonumpy(self._xend[ip, :, 1], self._xp),
                       tonumpy(self._xend[ip, :, 2], self._xp),
                       '.',
                       ms=0.5)
            for k in np.linspace(0, self._num_rad - 1, 7).astype(int):
                ax[i].plot([
                    float(self._xstart[ip, k, 1]),
                    float(self._xend[ip, k, 1])
                ], [
                    float(self._xstart[ip, k, 2]),
                    float(self._xend[ip, k, 2])
                ],
                           'k-',
                           lw=0.5)
                ax[i].annotate(f'{k}', (float(
                    self._xstart[ip, k, 1]), float(self._xstart[ip, k, 2])),
                               fontsize='xx-small')
            ax[i].set_xlim(-500, 500)
            ax[i].set_ylim(-500, 500)
            ax[i].grid(ls=':')
            ax[i].set_aspect('equal')

            if image is not None:
                ax[i].add_patch(
                    Rectangle((tmp1, tmp2),
                              float(self.in_shape[1] * self._voxel_size[1]),
                              float(self.in_shape[2] * self._voxel_size[2]),
                              edgecolor='r',
                              facecolor='none',
                              linestyle=':'))
                ax[i].imshow(tonumpy(image[0, ...], self._xp).T,
                             origin='lower',
                             extent=img_extent,
                             **kwargs)
            ax[i].set_title(
                f'view {ip:03} - phi {(180/np.pi)*self._view_angles[ip]:.1f} deg',
                fontsize='small')

        fig.tight_layout()

        return fig


#----------------------------------------------------

if __name__ == '__main__':
    import scipy.ndimage as ndi

    #----------------------------------------------------------------------
    projector = ParallelViewProjector2D((1, 128, 128),
                                np.linspace(-200, 200, 131, dtype=np.float32),
                                210, 300.,
                                np.array([0, -63.5, -63.5], dtype=np.float32),
                                np.ones(3, dtype=np.float32), np)


    image_space_filter = GaussianFilterOperator(projector.in_shape,
                               ndi,
                               sigma=1.3)

    sensitivity = ElementwiseMultiplicationOperator(np.full(projector.out_shape, 2.3, dtype=np.float32))

    fwd_model = CompositeLinearOperator((sensitivity, projector, image_space_filter))

    fwd_model.adjointness_test(np, verbose=True)
    fwd_model_norm = fwd_model.norm(np)

    fig = projector.show_views()
    fig.show()