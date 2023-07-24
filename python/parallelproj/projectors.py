from __future__ import annotations

import numpy.typing as npt
import array_api_compat
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import parallelproj


class ParallelViewProjector2D(parallelproj.LinearOperator):
    """2D non-TOF parallel view projector"""

    def __init__(self, image_shape: tuple[int, int, int],
                 radial_positions: npt.ArrayLike, view_angles: npt.ArrayLike,
                 radius: float, image_origin: npt.ArrayLike,
                 voxel_size: npt.ArrayLike):
        """init method

        Parameters
        ----------
        image_shape : tuple[int, int, int]
            shape of the input image (1, n1, n2)
        radial_positions : npt.ArrayLike (numpy, cupy or torch array)
            radial positions of the projection views in world coordinates
        view angles : np.ArrayLike (numpy, cupy or torch array)
            angles of the projection views in radians
        radius : float
            radius of the scanner
        image_origin : 3 element npt.ArrayLike (numpy, cupy or torch array)
            world coordinates of the [0,0,0] voxel
        voxel_size : 3 element npt.ArrayLike (numpy, cupy or torch array)
            the voxel size
        """
        super().__init__(array_api_compat.get_namespace(radial_positions))

        self._image_shape = image_shape
        self._view_angles = view_angles
        self._num_views = self._view_angles.shape[0]
        self._radial_positions = radial_positions

        self._num_rad = radial_positions.shape[0]

        self._radius = radius
        self._image_origin = image_origin
        self._voxel_size = voxel_size

        self._xstart = self.xp.zeros((self._num_views, self._num_rad, 3),
                                     dtype=self.xp.float32)
        self._xend = self.xp.zeros((self._num_views, self._num_rad, 3),
                                   dtype=self.xp.float32)

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

    def _apply(self, x):
        y = parallelproj.joseph3d_fwd(self._xstart, self._xend, x,
                                      self._image_origin, self._voxel_size)
        return y

    def _adjoint(self, y):
        x = parallelproj.joseph3d_back(self._xstart, self._xend,
                                       self._image_shape, self._image_origin,
                                       self._voxel_size, y)
        return x


#    def show_views(self, views_to_show=None, image=None, **kwargs):
#        """visualize the geometry of certrain projection views
#
#        Parameters
#        ----------
#        views_to_show : numpy array of integers
#            view numbers to show
#        image : numpy array or cupy array, optional
#            show an image inside the projector geometry
#        **kwargs : some type
#            passed to matplotlib.pyplot.imshow
#
#        """
#        if views_to_show is None:
#            views_to_show = np.linspace(0, self._num_views - 1, 5).astype(int)
#
#        num_cols = len(views_to_show)
#        fig, ax = plt.subplots(1, num_cols, figsize=(num_cols * 3, 3))
#
#        tmp1 = float(self._image_origin[1] - 0.5 * self._voxel_size[1])
#        tmp2 = float(self._image_origin[2] - 0.5 * self._voxel_size[2])
#        img_extent = [tmp1, -tmp1, tmp2, -tmp2]
#
#        for i, ip in enumerate(views_to_show):
#            ax[i].plot(parallelproj.tonumpy(self._xstart[ip, :, 1], self._xp),
#                       parallelproj.tonumpy(self._xstart[ip, :, 2], self._xp),
#                       '.',
#                       ms=0.5)
#            ax[i].plot(parallelproj.tonumpy(self._xend[ip, :, 1], self._xp),
#                       parallelproj.tonumpy(self._xend[ip, :, 2], self._xp),
#                       '.',
#                       ms=0.5)
#            for k in np.linspace(0, self._num_rad - 1, 7).astype(int):
#                ax[i].plot([
#                    float(self._xstart[ip, k, 1]),
#                    float(self._xend[ip, k, 1])
#                ], [
#                    float(self._xstart[ip, k, 2]),
#                    float(self._xend[ip, k, 2])
#                ],
#                           'k-',
#                           lw=0.5)
#                ax[i].annotate(f'{k}', (float(
#                    self._xstart[ip, k, 1]), float(self._xstart[ip, k, 2])),
#                               fontsize='xx-small')
#            ax[i].set_xlim(-500, 500)
#            ax[i].set_ylim(-500, 500)
#            ax[i].grid(ls=':')
#            ax[i].set_aspect('equal')
#
#            if image is not None:
#                ax[i].add_patch(
#                    Rectangle((tmp1, tmp2),
#                              float(self.in_shape[1] * self._voxel_size[1]),
#                              float(self.in_shape[2] * self._voxel_size[2]),
#                              edgecolor='r',
#                              facecolor='none',
#                              linestyle=':'))
#                ax[i].imshow(parallelproj.tonumpy(image[0, ...], self._xp).T,
#                             origin='lower',
#                             extent=img_extent,
#                             **kwargs)
#            ax[i].set_title(
#                f'view {ip:03} - phi {(180/np.pi)*self._view_angles[ip]:.1f} deg',
#                fontsize='small')
#
#        fig.tight_layout()
#
#        return fig
#