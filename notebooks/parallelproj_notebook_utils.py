#TODO get xp definition

import abc
import parallelproj
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from notebook_utils_parallelproj import tonumpy


class LinearOperator(abc.ABC):

    @property
    @abc.abstractmethod
    def ishape(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def oshape(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, x):
        """ forward step y = Ax"""
        raise NotImplementedError

    @abc.abstractmethod
    def adjoint(self, y):
        """ adjoint step x = A^H y"""
        raise NotImplementedError


class ParallelViewProjector2D(LinearOperator):

    def __init__(self, image_shape, radial_positions, num_views, radius,
                 image_origin, voxel_size):
        self._image_shape = image_shape
        self._radial_positions = radial_positions

        self._num_rad = radial_positions.shape[0]
        self._num_views = num_views

        self._radius = radius
        self._image_origin = image_origin
        self._voxel_size = voxel_size

        # array of projection angles
        self._view_angles = xp.linspace(0,
                                        xp.pi,
                                        self._num_views,
                                        endpoint=False)

        self._xstart = xp.zeros((self._num_views, self._num_rad, 3),
                                dtype=xp.float32)
        self._xend = xp.zeros((self._num_views, self._num_rad, 3),
                              dtype=xp.float32)

        for i, phi in enumerate(self._view_angles):
            # world coordinates of LOR start points
            self._xstart[i, :,
                         1] = xp.cos(phi) * r + xp.sin(phi) * self._radius
            self._xstart[i, :,
                         2] = -xp.sin(phi) * r + xp.cos(phi) * self._radius
            # world coordinates of LOR endpoints
            self._xend[i, :, 1] = xp.cos(phi) * r - xp.sin(phi) * self._radius
            self._xend[i, :, 2] = -xp.sin(phi) * r - xp.cos(phi) * self._radius

    @property
    def ishape(self):
        return self._image_shape

    @property
    def oshape(self):
        return (self._num_views, self._num_rad)

    def __call__(self, x):
        y = xp.zeros(self.oshape, dtype=xp.float32)
        parallelproj.joseph3d_fwd(self._xstart.reshape(-1, 3),
                                  self._xend.reshape(-1, 3), x,
                                  self._image_origin, self._voxel_size, y)
        return y

    def adjoint(self, y):
        x = xp.zeros(self.ishape, dtype=xp.float32)
        parallelproj.joseph3d_back(self._xstart.reshape(-1, 3),
                                   self._xend.reshape(-1, 3), x,
                                   self._image_origin, self._voxel_size, y)
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
            ax[i].plot(tonumpy(self._xstart[ip, :, 1]),
                       tonumpy(self._xstart[ip, :, 2]),
                       '.',
                       ms=0.5)
            ax[i].plot(tonumpy(self._xend[ip, :, 1]),
                       tonumpy(self._xend[ip, :, 2]),
                       '.',
                       ms=0.5)
            for k in np.linspace(0, num_rad - 1, 7).astype(int):
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

            if img is not None:
                ax[i].add_patch(
                    Rectangle((tmp1, tmp2),
                              float(self.ishape[1] * self._voxel_size[1]),
                              float(self.ishape[2] * self._voxel_size[2]),
                              edgecolor='r',
                              facecolor='none',
                              linestyle=':'))
                ax[i].imshow(tonumpy(image[0, ...]).T,
                             origin='lower',
                             extent=img_extent,
                             **kwargs)
            ax[i].set_title(
                f'view {ip:03} - phi {(180/np.pi)*self._view_angles[ip]:.1f} deg',
                fontsize='small')

        fig.tight_layout()

        return fig