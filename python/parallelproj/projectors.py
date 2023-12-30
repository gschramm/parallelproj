from __future__ import annotations

import array_api_compat.numpy as np
from numpy.array_api._array_object import Array
import array_api_compat
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from types import ModuleType
import parallelproj

from .operators import LinearOperator
from .pet_lors import RegularPolygonPETLORDescriptor
from .tof import TOFParameters


class ParallelViewProjector2D(LinearOperator):
    """2D non-TOF parallel view projector"""

    def __init__(self, image_shape: tuple[int, int],
                 radial_positions: Array, view_angles: Array,
                 radius: float, image_origin: tuple[float, float],
                 voxel_size: tuple[float, float]) -> None:
        """init method

        Parameters
        ----------
        image_shape : tuple[int, int]
            shape of the input image (n1, n2)
        radial_positions : Array
            radial positions of the projection views in world coordinates
        view_angles : Array
            angles of the projection views in radians
        radius : float
            radius of the scanner
        image_origin : tuple[float, float]
            world coordinates of the [0,0] voxel
        voxel_size : tuple[float, float]
            the voxel size in both directions
        """
        super().__init__()
        self._xp = array_api_compat.get_namespace(radial_positions)

        self._radial_positions = radial_positions
        self._device = array_api_compat.device(radial_positions)

        self._image_shape = image_shape
        self._image_origin = array_api_compat.to_device(
            self.xp.asarray((0, ) + image_origin, dtype=self.xp.float32),
            self._device)
        self._voxel_size = array_api_compat.to_device(
            self.xp.asarray((1, ) + voxel_size, dtype=self.xp.float32),
            self._device)

        self._view_angles = view_angles
        self._num_views = self._view_angles.shape[0]

        self._num_rad = radial_positions.shape[0]

        self._radius = radius

        self._xstart = array_api_compat.to_device(
            self.xp.zeros((self._num_rad, self._num_views, 3),
                          dtype=self.xp.float32), self._device)
        self._xend = array_api_compat.to_device(
            self.xp.zeros((self._num_rad, self._num_views, 3),
                          dtype=self.xp.float32), self._device)

        for i, phi in enumerate(self._view_angles):
            # world coordinates of LOR start points
            self._xstart[:, i, 1] = self._xp.cos(
                phi) * self._radial_positions + self._xp.sin(
                    phi) * self._radius
            self._xstart[:, i, 2] = -self._xp.sin(
                phi) * self._radial_positions + self._xp.cos(
                    phi) * self._radius
            # world coordinates of LOR endpoints
            self._xend[:, i, 1] = self._xp.cos(
                phi) * self._radial_positions - self._xp.sin(
                    phi) * self._radius
            self._xend[:, i, 2] = -self._xp.sin(
                phi) * self._radial_positions - self._xp.cos(
                    phi) * self._radius

    @property
    def xp(self) -> ModuleType:
        return self._xp

    @property
    def in_shape(self) -> tuple[int, int]:
        return self._image_shape

    @property
    def out_shape(self) -> tuple[int, int]:
        return (self._num_rad, self._num_views)

    @property
    def num_views(self) -> int:
        return self._num_views

    @property
    def num_rad(self) -> int:
        return self._num_rad

    @property
    def xstart(self) -> Array:
        return self._xstart

    @property
    def xend(self) -> Array:
        return self._xend

    @property
    def image_origin(self) -> Array:
        return self._image_origin

    @property
    def image_shape(self) -> tuple[int, int]:
        return self._image_shape

    @property
    def voxel_size(self) -> Array:
        return self._voxel_size

    @property
    def device(self) -> str:
        return self._device

    def _apply(self, x: Array) -> Array:
        y = parallelproj.joseph3d_fwd(self._xstart, self._xend,
                                      self.xp.expand_dims(x, axis=0),
                                      self._image_origin, self._voxel_size)
        return y

    def _adjoint(self, y: Array) -> Array:
        x = parallelproj.joseph3d_back(self._xstart, self._xend,
                                       (1, ) + self._image_shape,
                                       self._image_origin, self._voxel_size, y)
        return self.xp.squeeze(x, axis=0)

    def show_views(self, views_to_show : None | Array = None, image: None | Array = None, **kwargs) -> plt.Figure:
        """visualize the geometry of certrain projection views

        Parameters
        ----------
        views_to_show : None | Array
            view numbers to show
        image : None | Array
            show an image inside the projector geometry
        **kwargs : some type
            passed to matplotlib.pyplot.imshow

        """
        if views_to_show is None:
            views_to_show = np.linspace(0, self._num_views - 1, 5).astype(int)

        num_cols = len(views_to_show)
        fig, ax = plt.subplots(1, num_cols, figsize=(num_cols * 3, 3))

        tmp1 = float(self._image_origin[1] - 0.5 * self._voxel_size[1])
        tmp2 = float(self._image_origin[2] - 0.5 * self._voxel_size[2])
        img_extent = [tmp1, -tmp1, tmp2, -tmp2]

        for i, ip in enumerate(views_to_show):
            ax[i].plot(np.asarray(array_api_compat.to_device(self._xstart[:, ip, 1],
                                                  'cpu')),
                       np.asarray(array_api_compat.to_device(self._xstart[:, ip, 2],
                                                  'cpu')),
                       '.',
                       ms=0.5)
            ax[i].plot(np.asarray(array_api_compat.to_device(self._xend[:, ip, 1], 'cpu')),
                       np.asarray(array_api_compat.to_device(self._xend[:, ip, 2], 'cpu')),
                       '.',
                       ms=0.5)

            for k in np.linspace(0, self._num_rad - 1, 7).astype(int):
                ax[i].plot([
                    float(self._xstart[k, ip, 1]),
                    float(self._xend[k, ip, 1])
                ], [
                    float(self._xstart[k, ip, 2]),
                    float(self._xend[k, ip, 2])
                ],
                           'k-',
                           lw=0.5)
                ax[i].annotate(f'{k}', (float(
                    self._xstart[k, ip, 1]), float(self._xstart[k, ip, 2])),
                               fontsize='xx-small')

            pmax = 1.5 * float(self.xp.max(self._xstart[..., 1]))
            ax[i].set_xlim(-pmax, pmax)
            ax[i].set_ylim(-pmax, pmax)
            ax[i].grid(ls=':')
            ax[i].set_aspect('equal')

            if image is not None:
                ax[i].add_patch(
                    Rectangle((tmp1, tmp2),
                              float(self.in_shape[0] * self._voxel_size[1]),
                              float(self.in_shape[1] * self._voxel_size[2]),
                              edgecolor='r',
                              facecolor='none',
                              linestyle=':'))

                ax[i].imshow(array_api_compat.to_device(image, 'cpu').T,
                             origin='lower',
                             extent=img_extent,
                             **kwargs)

            ax[i].set_title(
                f'view {ip:03} - phi {(180/np.pi)*self._view_angles[ip]} deg',
                fontsize='small')

        fig.tight_layout()

        return fig


#-------------------------------------------------------------------------------


class ParallelViewProjector3D(LinearOperator):
    """3D non-TOF parallel view projector"""

    def __init__(self,
                 image_shape: tuple[int, int, int],
                 radial_positions: Array,
                 view_angles: Array,
                 radius: float,
                 image_origin: tuple[float, float, float],
                 voxel_size: tuple[float, float],
                 ring_positions: Array,
                 span: int = 1,
                 max_ring_diff: int | None = None) -> None:
        """init method

        Parameters
        ----------
        image_shape : tuple[int, int, int]
            shape of the input image (n0, n1, n2) (last direction is axial)
        radial_positions : Array
            radial positions of the projection views in world coordinates
        view_angles : Array
            angles of the projection views in radians
        radius : float
            radius of the scanner
        image_origin : tuple[float, float, float]
            world coordinates of the [0,0,0] voxel
        voxel_size : tuple[float, float, float]
            the voxel size in all directions (last direction is axial)
        ring_positions : Array
            position of the rings in world coordinates
        span : int
            span of the sinogram - default is 1
        max_ring_diff : int | None
            maximum ring difference - default is None (no limit)
        """

        super().__init__()

        self._xp = array_api_compat.get_namespace(radial_positions)

        self._radial_positions = radial_positions
        self._device = array_api_compat.device(radial_positions)

        self._image_shape = image_shape
        self._image_origin = array_api_compat.to_device(
            self.xp.asarray(image_origin, dtype=self.xp.float32), self._device)
        self._voxel_size = array_api_compat.to_device(
            self.xp.asarray(voxel_size, dtype=self.xp.float32), self._device)

        self._view_angles = view_angles
        self._num_views = self._view_angles.shape[0]

        self._num_rad = radial_positions.shape[0]

        self._radius = radius

        xstart2d = array_api_compat.to_device(
            self.xp.zeros((self._num_rad, self._num_views, 2),
                          dtype=self.xp.float32), self._device)
        xend2d = array_api_compat.to_device(
            self.xp.zeros((self._num_rad, self._num_views, 2),
                          dtype=self.xp.float32), self._device)

        for i, phi in enumerate(self._view_angles):
            # world coordinates of LOR start points
            xstart2d[:, i, 0] = self._xp.cos(
                phi) * self._radial_positions + self._xp.sin(
                    phi) * self._radius
            xstart2d[:, i, 1] = -self._xp.sin(
                phi) * self._radial_positions + self._xp.cos(
                    phi) * self._radius
            # world coordinates of LOR endpoints
            xend2d[:, i, 0] = self._xp.cos(
                phi) * self._radial_positions - self._xp.sin(
                    phi) * self._radius
            xend2d[:, i, 1] = -self._xp.sin(
                phi) * self._radial_positions - self._xp.cos(
                    phi) * self._radius

        self._ring_positions = ring_positions
        self._num_rings = ring_positions.shape[0]
        self._span = span

        if max_ring_diff is None:
            self._max_ring_diff = self._num_rings - 1
        else:
            self._max_ring_diff = max_ring_diff

        if self._span == 1:
            self._num_segments = 2 * self._max_ring_diff + 1
            self._segment_numbers = np.zeros(self._num_segments,
                                             dtype=np.int32)
            self._segment_numbers[0::2] = np.arange(self._max_ring_diff + 1)
            self._segment_numbers[1::2] = -np.arange(1,
                                                     self._max_ring_diff + 1)

            self._num_planes_per_segment = self._num_rings - np.abs(
                self._segment_numbers)

            self._start_plane_number = []
            self._end_plane_number = []

            for i, seg_number in enumerate(self._segment_numbers):
                tmp = np.arange(self._num_planes_per_segment[i])

                if seg_number < 0:
                    tmp -= seg_number

                self._start_plane_number.append(tmp)
                self._end_plane_number.append(tmp + seg_number)

            self._start_plane_number = np.concatenate(self._start_plane_number)
            self._end_plane_number = np.concatenate(self._end_plane_number)
            self._num_planes = self._start_plane_number.shape[0]
        else:
            raise ValueError('span > 1 not implemented yet')

        self._xstart = array_api_compat.to_device(
            self._xp.zeros(
                (self._num_rad, self._num_views, self._num_planes, 3),
                dtype=self._xp.float32), self._device)
        self._xend = array_api_compat.to_device(
            self._xp.zeros(
                (self._num_rad, self._num_views, self._num_planes, 3),
                dtype=self._xp.float32), self._device)

        for i in range(self._num_planes):
            self._xstart[:, :, i, :2] = xstart2d
            self._xend[:, :, i, :2] = xend2d

            self._xstart[:, :, i,
                         2] = self._ring_positions[self._start_plane_number[i]]
            self._xend[:, :, i,
                       2] = self._ring_positions[self._end_plane_number[i]]

    @property
    def max_ring_diff(self) -> int:
        return self._max_ring_diff

    @property
    def xp(self) -> ModuleType:
        return self._xp

    @property
    def in_shape(self) -> tuple[int, int, int]:
        return self._image_shape

    @property
    def out_shape(self) -> tuple[int, int, int]:
        return (self._num_rad, self._num_views, self._num_planes)

    @property
    def voxel_size(self) -> Array:
        return self._voxel_size

    @property
    def image_origin(self) -> Array:
        return self._image_origin

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return self._image_shape

    @property
    def xstart(self) -> Array:
        return self._xstart

    @property
    def xend(self) -> Array:
        return self._xend

    def _apply(self, x: Array) -> Array:
        y = parallelproj.joseph3d_fwd(self._xstart, self._xend, x,
                                      self.image_origin, self.voxel_size)
        return y

    def _adjoint(self, y: Array) -> Array:
        x = parallelproj.joseph3d_back(self._xstart, self._xend,
                                       self.image_shape, self.image_origin,
                                       self.voxel_size, y)
        return x


class RegularPolygonPETProjector(LinearOperator):

    def __init__(self,
                 lor_descriptor: RegularPolygonPETLORDescriptor,
                 img_shape: tuple[int, int, int],
                 voxel_size: tuple[float, float, float],
                 img_origin: None | Array = None,
                 views: None | Array = None) -> None:
        """Regular polygon PET projector

        Parameters
        ----------
        lor_descriptor : RegularPolygonPETLORDescriptor
            descriptor of the LOR start / end points
        img_shape : tuple[int, int, int]
            shape of the image to be projected
        voxel_size : tuple[float, float, float]
            the voxel size of the image to be projected
        img_origin : None | Array, optional
            the origin of the image to be projected, by default None
            means that image is "centered" in the scanner
        views : None | Array, optional
            sinogram views to be projected, by default None
            means that all views are being projected
        """

        super().__init__()
        self._dev = lor_descriptor.dev

        self._lor_descriptor = lor_descriptor
        self._img_shape = img_shape
        self._voxel_size = self.xp.asarray(voxel_size,
                                           dtype=self.xp.float32,
                                           device=self._dev)

        if img_origin is None:
            self._img_origin = (-(self.xp.asarray(
                self._img_shape, dtype=self.xp.float32, device=self._dev) / 2)
                                + 0.5) * self._voxel_size
        else:
            self._img_origin = self.xp.asarray(img_origin,
                                               dtype=self.xp.float32,
                                               device=self._dev)

        if views is None:
            self._views = self.xp.arange(self._lor_descriptor.num_views,
                                         device=self._dev)
        else:
            self._views = views

        self._xstart, self._xend = lor_descriptor.get_lor_coordinates(views=self._views)

        self._tof_parameters = None
        self._tof = False

    @property
    def in_shape(self) -> tuple[int, int, int]:
        return self._img_shape

    @property
    def out_shape(self) -> tuple[int, int, int]:
        if self.tof:
            out_shape = (self._lor_descriptor.num_rad, self._views.shape[0],
                         self._lor_descriptor.num_planes,
                         self.tof_parameters.num_tofbins)
        else:
            out_shape = (self._lor_descriptor.num_rad, self._views.shape[0],
                         self._lor_descriptor.num_planes)

        return out_shape

    @property
    def xp(self) -> ModuleType:
        return self._lor_descriptor.xp

    @property
    def tof(self) -> bool:
        return self._tof

    @tof.setter
    def tof(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError('tof must be a boolean')
        self._tof = value

    @property
    def tof_parameters(self) -> TOFParameters | None:
        return self._tof_parameters

    @tof_parameters.setter
    def tof_parameters(self, value: TOFParameters | None) -> None:
        if not (isinstance(value, TOFParameters) or value is None):
            raise ValueError('tof_parameters must be a TOFParameters object or None')
        self._tof_parameters = value

        if value is None:
            self._tof = False
        else:
            self._tof = True

    @property
    def img_origin(self) -> Array:
        return self._img_origin

    def _apply(self, x):
        """nonTOF forward projection of input image x including image based resolution model"""

        dev = array_api_compat.device(x)

        if not self.tof:
            x_fwd = parallelproj.joseph3d_fwd(self._xstart, self._xend, x,
                                              self._img_origin,
                                              self._voxel_size)
        else:
            x_fwd = parallelproj.joseph3d_fwd_tof_sino(
                self._xstart, self._xend, x, self._img_origin,
                self._voxel_size, self._tof_parameters.tofbin_width,
                self.xp.asarray([self._tof_parameters.sigma_tof],
                                dtype=self.xp.float32,
                                device=dev),
                self.xp.asarray([self._tof_parameters.tofcenter_offset],
                                dtype=self.xp.float32,
                                device=dev), self.tof_parameters.num_sigmas,
                self.tof_parameters.num_tofbins)

        return x_fwd

    def _adjoint(self, y):
        """nonTOF back projection of sinogram y"""
        dev = array_api_compat.device(y)

        if not self.tof:
            y_back = parallelproj.joseph3d_back(self._xstart, self._xend,
                                                self._img_shape,
                                                self._img_origin,
                                                self._voxel_size, y)
        else:
            y_back = parallelproj.joseph3d_back_tof_sino(
                self._xstart, self._xend, self._img_shape, self._img_origin,
                self._voxel_size, y, self._tof_parameters.tofbin_width,
                self.xp.asarray([self._tof_parameters.sigma_tof],
                                dtype=self.xp.float32,
                                device=dev),
                self.xp.asarray([self._tof_parameters.tofcenter_offset],
                                dtype=self.xp.float32,
                                device=dev), self.tof_parameters.num_sigmas,
                self.tof_parameters.num_tofbins)

        return y_back
