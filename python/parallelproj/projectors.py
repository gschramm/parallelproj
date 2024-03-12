"""high-level geometrical forward and back projectors"""

from __future__ import annotations

import array_api_compat.numpy as np
from numpy.array_api._array_object import Array
import array_api_compat
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from types import ModuleType
from array_api_compat import device, to_device, get_namespace, size
import parallelproj

from .operators import LinearOperator
from .pet_lors import RegularPolygonPETLORDescriptor
from .tof import TOFParameters
from .backend import is_cuda_array, empty_cuda_cache, to_numpy_array


class ParallelViewProjector2D(LinearOperator):
    """2D non-TOF parallel view projector"""

    def __init__(
        self,
        image_shape: tuple[int, int],
        radial_positions: Array,
        view_angles: Array,
        radius: float,
        image_origin: tuple[float, float],
        voxel_size: tuple[float, float],
    ) -> None:
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
            self.xp.asarray((0,) + image_origin, dtype=self.xp.float32), self._device
        )
        self._voxel_size = array_api_compat.to_device(
            self.xp.asarray((1,) + voxel_size, dtype=self.xp.float32), self._device
        )

        self._view_angles = view_angles
        self._num_views = self._view_angles.shape[0]

        self._num_rad = radial_positions.shape[0]

        self._radius = radius

        self._xstart = array_api_compat.to_device(
            self.xp.zeros((self._num_rad, self._num_views, 3), dtype=self.xp.float32),
            self._device,
        )
        self._xend = array_api_compat.to_device(
            self.xp.zeros((self._num_rad, self._num_views, 3), dtype=self.xp.float32),
            self._device,
        )

        for i, phi in enumerate(self._view_angles):
            # world coordinates of LOR start points
            self._xstart[:, i, 1] = (
                self._xp.cos(phi) * self._radial_positions
                + self._xp.sin(phi) * self._radius
            )
            self._xstart[:, i, 2] = (
                -self._xp.sin(phi) * self._radial_positions
                + self._xp.cos(phi) * self._radius
            )
            # world coordinates of LOR endpoints
            self._xend[:, i, 1] = (
                self._xp.cos(phi) * self._radial_positions
                - self._xp.sin(phi) * self._radius
            )
            self._xend[:, i, 2] = (
                -self._xp.sin(phi) * self._radial_positions
                - self._xp.cos(phi) * self._radius
            )

    @property
    def xp(self) -> ModuleType:
        """array module"""
        return self._xp

    @property
    def in_shape(self) -> tuple[int, int]:
        return self._image_shape

    @property
    def out_shape(self) -> tuple[int, int]:
        return (self._num_rad, self._num_views)

    @property
    def num_views(self) -> int:
        """number of views"""
        return self._num_views

    @property
    def num_rad(self) -> int:
        """number of radial elements"""
        return self._num_rad

    @property
    def xstart(self) -> Array:
        """coordinates of LOR start points"""
        return self._xstart

    @property
    def xend(self) -> Array:
        """coordinates of LOR end points"""
        return self._xend

    @property
    def image_origin(self) -> Array:
        """image origin - world coordinates of the [0,0] voxel"""
        return self._image_origin

    @property
    def image_shape(self) -> tuple[int, int]:
        """image shape"""
        return self._image_shape

    @property
    def voxel_size(self) -> Array:
        """voxel size"""
        return self._voxel_size

    @property
    def dev(self) -> str:
        """device used for storage of LOR endpoints"""
        return self._device

    def _apply(self, x: Array) -> Array:
        y = parallelproj.joseph3d_fwd(
            self._xstart,
            self._xend,
            self.xp.expand_dims(x, axis=0),
            self._image_origin,
            self._voxel_size,
        )
        return y

    def _adjoint(self, y: Array) -> Array:
        x = parallelproj.joseph3d_back(
            self._xstart,
            self._xend,
            (1,) + self._image_shape,
            self._image_origin,
            self._voxel_size,
            y,
        )
        return self.xp.squeeze(x, axis=0)

    def show_views(
        self, views_to_show: None | Array = None, image: None | Array = None, **kwargs
    ) -> plt.Figure:
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
            # ax[i].plot(
            #    np.array(array_api_compat.to_device(self._xstart[:, ip, 1], "cpu")),
            #    np.array(array_api_compat.to_device(self._xstart[:, ip, 2], "cpu")),
            #    ".",
            #    ms=0.5,
            # )
            # ax[i].plot(
            #    np.array(array_api_compat.to_device(self._xend[:, ip, 1], "cpu")),
            #    np.array(array_api_compat.to_device(self._xend[:, ip, 2], "cpu")),
            #    ".",
            #    ms=0.5,
            # )

            for k in np.linspace(0, self._num_rad - 1, 7).astype(int):
                ax[i].plot(
                    [float(self._xstart[k, ip, 1]), float(self._xend[k, ip, 1])],
                    [float(self._xstart[k, ip, 2]), float(self._xend[k, ip, 2])],
                    "k-",
                    lw=0.5,
                )
                ax[i].annotate(
                    f"{k}",
                    (float(self._xstart[k, ip, 1]), float(self._xstart[k, ip, 2])),
                    fontsize="xx-small",
                )

            pmax = 1.5 * float(self.xp.max(self._xstart[..., 1]))
            ax[i].set_xlim(-pmax, pmax)
            ax[i].set_ylim(-pmax, pmax)
            ax[i].grid(ls=":")
            ax[i].set_aspect("equal")

            if image is not None:
                ax[i].add_patch(
                    Rectangle(
                        (tmp1, tmp2),
                        float(self.in_shape[0] * self._voxel_size[1]),
                        float(self.in_shape[1] * self._voxel_size[2]),
                        edgecolor="r",
                        facecolor="none",
                        linestyle=":",
                    )
                )

                ax[i].imshow(
                    np.array(array_api_compat.to_device(image, "cpu")).T,
                    origin="lower",
                    extent=img_extent,
                    **kwargs,
                )

            ax[i].set_title(
                f"view {ip:03} - phi {(180/np.pi)*self._view_angles[ip]} deg",
                fontsize="small",
            )

        fig.tight_layout()

        return fig


class ParallelViewProjector3D(LinearOperator):
    """3D non-TOF parallel view projector"""

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        radial_positions: Array,
        view_angles: Array,
        radius: float,
        image_origin: tuple[float, float, float],
        voxel_size: tuple[float, float],
        ring_positions: Array,
        span: int = 1,
        max_ring_diff: int | None = None,
    ) -> None:
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
            self.xp.asarray(image_origin, dtype=self.xp.float32), self._device
        )
        self._voxel_size = array_api_compat.to_device(
            self.xp.asarray(voxel_size, dtype=self.xp.float32), self._device
        )

        self._view_angles = view_angles
        self._num_views = self._view_angles.shape[0]

        self._num_rad = radial_positions.shape[0]

        self._radius = radius

        xstart2d = array_api_compat.to_device(
            self.xp.zeros((self._num_rad, self._num_views, 2), dtype=self.xp.float32),
            self._device,
        )
        xend2d = array_api_compat.to_device(
            self.xp.zeros((self._num_rad, self._num_views, 2), dtype=self.xp.float32),
            self._device,
        )

        for i, phi in enumerate(self._view_angles):
            # world coordinates of LOR start points
            xstart2d[:, i, 0] = (
                self._xp.cos(phi) * self._radial_positions
                + self._xp.sin(phi) * self._radius
            )
            xstart2d[:, i, 1] = (
                -self._xp.sin(phi) * self._radial_positions
                + self._xp.cos(phi) * self._radius
            )
            # world coordinates of LOR endpoints
            xend2d[:, i, 0] = (
                self._xp.cos(phi) * self._radial_positions
                - self._xp.sin(phi) * self._radius
            )
            xend2d[:, i, 1] = (
                -self._xp.sin(phi) * self._radial_positions
                - self._xp.cos(phi) * self._radius
            )

        self._ring_positions = ring_positions
        self._num_rings = ring_positions.shape[0]
        self._span = span

        if max_ring_diff is None:
            self._max_ring_diff = self._num_rings - 1
        else:
            self._max_ring_diff = max_ring_diff

        if self._span == 1:
            self._num_segments = 2 * self._max_ring_diff + 1
            self._segment_numbers = np.zeros(self._num_segments, dtype=np.int32)
            self._segment_numbers[0::2] = np.arange(self._max_ring_diff + 1)
            self._segment_numbers[1::2] = -np.arange(1, self._max_ring_diff + 1)

            self._num_planes_per_segment = self._num_rings - np.abs(
                self._segment_numbers
            )

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
            raise ValueError("span > 1 not implemented yet")

        self._xstart = array_api_compat.to_device(
            self._xp.zeros(
                (self._num_rad, self._num_views, self._num_planes, 3),
                dtype=self._xp.float32,
            ),
            self._device,
        )
        self._xend = array_api_compat.to_device(
            self._xp.zeros(
                (self._num_rad, self._num_views, self._num_planes, 3),
                dtype=self._xp.float32,
            ),
            self._device,
        )

        for i in range(self._num_planes):
            self._xstart[:, :, i, :2] = xstart2d
            self._xend[:, :, i, :2] = xend2d

            self._xstart[:, :, i, 2] = self._ring_positions[self._start_plane_number[i]]
            self._xend[:, :, i, 2] = self._ring_positions[self._end_plane_number[i]]

    @property
    def max_ring_diff(self) -> int:
        """maximum ring difference"""
        return self._max_ring_diff

    @property
    def xp(self) -> ModuleType:
        """array module"""
        return self._xp

    @property
    def in_shape(self) -> tuple[int, int, int]:
        return self._image_shape

    @property
    def out_shape(self) -> tuple[int, int, int]:
        return (self._num_rad, self._num_views, self._num_planes)

    @property
    def voxel_size(self) -> Array:
        """the voxel size in all directions"""
        return self._voxel_size

    @property
    def image_origin(self) -> Array:
        """image origin - world coordinates of the [0,0,0] voxel"""
        return self._image_origin

    @property
    def image_shape(self) -> tuple[int, int, int]:
        """image shape"""
        return self._image_shape

    @property
    def xstart(self) -> Array:
        """coordinates of LOR start points"""
        return self._xstart

    @property
    def xend(self) -> Array:
        """coordinates of LOR end points"""
        return self._xend

    def _apply(self, x: Array) -> Array:
        y = parallelproj.joseph3d_fwd(
            self._xstart, self._xend, x, self.image_origin, self.voxel_size
        )
        return y

    def _adjoint(self, y: Array) -> Array:
        x = parallelproj.joseph3d_back(
            self._xstart,
            self._xend,
            self.image_shape,
            self.image_origin,
            self.voxel_size,
            y,
        )
        return x


class RegularPolygonPETProjector(LinearOperator):
    """geometric non-TOF and TOF sinogram projector for regular polygon PET scanners

    Examples
    --------
    .. minigallery:: parallelproj.RegularPolygonPETProjector
    """

    def __init__(
        self,
        lor_descriptor: RegularPolygonPETLORDescriptor,
        img_shape: tuple[int, int, int],
        voxel_size: tuple[float, float, float],
        img_origin: None | Array = None,
        views: None | Array = None,
        cache_lor_endpoints: bool = True,
    ) -> None:
        """
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
            means that the center of the image is at world coordinate (0,0,0)
        views : None | Array, optional
            sinogram views to be projected, by default None
            means that all views are being projected
        cache_lor_endpoints : bool, optional
            whether to cache the LOR endpoints, by default True
            setting it to False will save memory but will slow down computations
        """

        super().__init__()
        self._dev = lor_descriptor.dev

        self._lor_descriptor = lor_descriptor
        self._img_shape = img_shape
        self._voxel_size = self.xp.asarray(
            voxel_size, dtype=self.xp.float32, device=self._dev
        )

        if img_origin is None:
            self._img_origin = (
                -(
                    self.xp.asarray(
                        self._img_shape, dtype=self.xp.float32, device=self._dev
                    )
                    / 2
                )
                + 0.5
            ) * self._voxel_size
        else:
            self._img_origin = self.xp.asarray(
                img_origin, dtype=self.xp.float32, device=self._dev
            )

        if views is None:
            self._views = self.xp.arange(
                self._lor_descriptor.num_views, device=self._dev
            )
        else:
            self._views = views

        self._tof_parameters = None
        self._tof = False

        self._cache_lor_endpoints = cache_lor_endpoints

        self._xstart = None
        self._xend = None

    @property
    def in_shape(self) -> tuple[int, int, int]:
        return self._img_shape

    @property
    def out_shape(self) -> tuple[int, int, int]:
        out_shape = list(self._lor_descriptor.spatial_sinogram_shape)
        out_shape[self._lor_descriptor.view_axis_num] = self._views.shape[0]

        if self.tof:
            out_shape += [self.tof_parameters.num_tofbins]

        return tuple(out_shape)

    @property
    def xp(self) -> ModuleType:
        """array module"""
        return self._lor_descriptor.xp

    @property
    def tof(self) -> bool:
        """bool indicating whether to use TOF or not"""
        return self._tof

    @tof.setter
    def tof(self, value: bool) -> None:
        self._tof = value

        if self.tof_parameters is None:
            raise ValueError("tof_parameters must not be None")

    @property
    def tof_parameters(self) -> TOFParameters | None:
        """TOF parameters"""
        return self._tof_parameters

    @tof_parameters.setter
    def tof_parameters(self, value: TOFParameters | None) -> None:
        if not (isinstance(value, TOFParameters) or value is None):
            raise ValueError("tof_parameters must be a TOFParameters object or None")
        self._tof_parameters = value

        if value is None:
            self._tof = False
        else:
            self._tof = True

    @property
    def lor_descriptor(self) -> RegularPolygonPETLORDescriptor:
        """LOR descriptor"""
        return self._lor_descriptor

    @property
    def img_origin(self) -> Array:
        """image origin - world coordinates of the [0,0,0] voxel"""
        return self._img_origin

    @property
    def views(self) -> Array:
        """view numbers to be projected"""
        return self._views

    @views.setter
    def views(self, value: Array) -> None:
        self._views = value
        # we need to reset the LOR start and end points in case
        # they were cached
        self.clear_cached_lor_endpoints()

    @property
    def xstart(self) -> Array | None:
        """cached coordinates of LOR start points"""
        return self._xstart

    @property
    def xend(self) -> Array | None:
        """cached coordinates of LOR end points"""
        return self._xend

    @property
    def voxel_size(self) -> Array:
        """voxel size"""
        return self._voxel_size

    def clear_cached_lor_endpoints(self) -> None:
        """clear cached LOR endpoints"""
        was_cuda_start = False
        was_cuda_end = False
        if self._xstart is not None:
            was_cuda_start = is_cuda_array(self._xstart)
        if self._xend is not None:
            was_cuda_end = is_cuda_array(self._xend)

        self._xstart = None
        self._xend = None

        if was_cuda_start or was_cuda_end:
            empty_cuda_cache(self.xp)

    def __str__(self) -> str:
        """string representation"""

        st = (
            self.__class__.__name__
            + " with sinogram shape ("
            + ", ".join(
                [
                    f"{self.lor_descriptor.spatial_sinogram_shape[i]} {self.lor_descriptor.sinogram_order.name[i]}"
                    for i in range(3)
                ]
            )
        )

        if self.tof:
            st += f", {self.tof_parameters.num_tofbins} TOF bins"

        st += ")"

        return st

    def _apply(self, x):
        """nonTOF forward projection of input image x including image based resolution model"""

        dev = array_api_compat.device(x)

        # calculate LOR endpoints if not done yet
        if (self.xstart is None) or (self.xend is None):
            xstart, xend = self._lor_descriptor.get_lor_coordinates(views=self._views)
            if is_cuda_array(xstart):
                empty_cuda_cache(self.xp)
        else:
            xstart = self.xstart
            xend = self.xend

        # cache LOR endpoints if requested
        if self._cache_lor_endpoints and ((self.xstart is None) or (self.xend is None)):
            self._xstart = xstart
            self._xend = xend

        if not self.tof:
            x_fwd = parallelproj.joseph3d_fwd(
                xstart, xend, x, self._img_origin, self._voxel_size
            )
        else:
            x_fwd = parallelproj.joseph3d_fwd_tof_sino(
                xstart,
                xend,
                x,
                self._img_origin,
                self._voxel_size,
                self._tof_parameters.tofbin_width,
                self.xp.asarray(
                    [self._tof_parameters.sigma_tof], dtype=self.xp.float32, device=dev
                ),
                self.xp.asarray(
                    [self._tof_parameters.tofcenter_offset],
                    dtype=self.xp.float32,
                    device=dev,
                ),
                self.tof_parameters.num_sigmas,
                self.tof_parameters.num_tofbins,
            )

        return x_fwd

    def _adjoint(self, y):
        """nonTOF back projection of sinogram y"""
        dev = array_api_compat.device(y)

        # calculate LOR endpoints if not done yet
        if (self.xstart is None) or (self.xend is None):
            xstart, xend = self._lor_descriptor.get_lor_coordinates(views=self._views)
            if is_cuda_array(xstart):
                empty_cuda_cache(self.xp)
        else:
            xstart = self.xstart
            xend = self.xend

        # cache LOR endpoints if requested
        if self._cache_lor_endpoints and ((self.xstart is None) or (self.xend is None)):
            self._xstart = xstart
            self._xend = xend

        if not self.tof:
            y_back = parallelproj.joseph3d_back(
                xstart,
                xend,
                self._img_shape,
                self._img_origin,
                self._voxel_size,
                y,
            )
        else:
            y_back = parallelproj.joseph3d_back_tof_sino(
                xstart,
                xend,
                self._img_shape,
                self._img_origin,
                self._voxel_size,
                y,
                self._tof_parameters.tofbin_width,
                self.xp.asarray(
                    [self._tof_parameters.sigma_tof], dtype=self.xp.float32, device=dev
                ),
                self.xp.asarray(
                    [self._tof_parameters.tofcenter_offset],
                    dtype=self.xp.float32,
                    device=dev,
                ),
                self.tof_parameters.num_sigmas,
                self.tof_parameters.num_tofbins,
            )

        return y_back

    def show_geometry(
        self,
        ax: plt.Axes,
        color: tuple[float, float, float] = (1.0, 0.0, 0.0),
        edgecolor: str = "grey",
        alpha: float = 0.1,
    ) -> None:
        """show the geometry of the scanner and the FOV of the image

        Parameters
        ----------
        ax : plt.Axes
            matplotlib axes object with projection = '3d'
        color : tuple[float, float, float], optional
            color to use for the FOV cube, by default (1.,0.,0.)
        edgecolor : str, optional
            edgecolor to use for the FOV cube, by default 'grey'
        alpha : float, optional
            alpha value of the FOV cube, by default 0.1
        """

        # dimensions of the "voxel" array for the FOV cube
        # (1,1,1) means that FOV cube is represented by a single voxel
        sh = (1, 1, 1)

        x, y, z = np.indices((sh[0] + 1, sh[1] + 1, sh[2] + 1)).astype(float)

        x /= sh[0]
        y /= sh[1]
        z /= sh[2]

        x *= int(self.in_shape[0]) * float(self.voxel_size[0])
        y *= int(self.in_shape[1]) * float(self.voxel_size[1])
        z *= int(self.in_shape[2]) * float(self.voxel_size[2])

        x += float(self.img_origin[0]) - 0.5 * float(self.voxel_size[0])
        y += float(self.img_origin[1]) - 0.5 * float(self.voxel_size[1])
        z += float(self.img_origin[2]) - 0.5 * float(self.voxel_size[2])

        colors = np.empty(sh + (4,), dtype=np.float32)
        colors[..., 0] = color[0]
        colors[..., 1] = color[1]
        colors[..., 2] = color[2]
        colors[..., 3] = alpha

        ax.voxels(
            x,
            y,
            z,
            filled=np.ones(sh, dtype=np.bool),
            facecolors=colors,
            edgecolors=edgecolor,
        )

        self.lor_descriptor.scanner.show_lor_endpoints(ax)

    def convert_sinogram_to_listmode(
        self, sinogram: Array
    ) -> tuple[Array, Array, Array | None]:
        """convert a non-TOF or TOF emission sinogram to listmode events

        Parameters
        ----------
        sinogram : Array
            an integer (TOF or non-TOF) emission sinogram

        Returns
        -------
        tuple[Array, Array, Array | None]
            event_start_coordinates, event_end_coordinates, event_tofbin
            in case of non-TOF, event_tofbin is None
        """

        num_events = int(self.xp.sum(sinogram))

        event_start_coords = self.xp.empty(
            (num_events, 3), device=self._dev, dtype=self.xp.float32
        )
        event_end_coords = self.xp.empty(
            (num_events, 3), device=self._dev, dtype=self.xp.float32
        )

        if self.tof:
            num_tofbins = self.tof_parameters.num_tofbins
            event_tofbins = self.xp.empty(
                (num_events,), device=self._dev, dtype=self.xp.int16
            )
        else:
            num_tofbins = 1
            event_tofbins = None

        event_offset = 0

        # we convert view by view and tofbin by tofbin to save memory
        for view in range(self.lor_descriptor.num_views):
            xstart, xend = self.lor_descriptor.get_lor_coordinates(
                views=self.xp.asarray([view], device=self._dev)
            )
            xstart = self.xp.reshape(xstart, (-1, 3))
            xend = self.xp.reshape(xend, (-1, 3))

            sino_view = self.xp.take(
                sinogram,
                self.xp.asarray([view], device=self._dev),
                axis=self.lor_descriptor.view_axis_num,
            )
            sino_view = self.xp.squeeze(
                sino_view, axis=self.lor_descriptor.view_axis_num
            )

            for it in range(num_tofbins):
                if self.tof:
                    ss = sino_view[..., it]
                else:
                    ss = sino_view

                ss = self.xp.reshape(ss, (size(ss),))

                # currently there is no "repeat" function in array-api, so
                # we convert back and forth to numpy cpu array
                event_sino_inds = self.xp.asarray(
                    np.repeat(np.arange(ss.shape[0]), to_numpy_array(ss)),
                    device=self._dev,
                )

                num_events_ss = int(self.xp.sum(ss))

                event_start_coords[event_offset : (event_offset + num_events_ss), :] = (
                    self.xp.take(xstart, event_sino_inds, axis=0)
                )
                event_end_coords[event_offset : (event_offset + num_events_ss), :] = (
                    self.xp.take(xend, event_sino_inds, axis=0)
                )

                if self.tof:
                    event_tofbins[event_offset : (event_offset + num_events_ss)] = (
                        self.xp.full(
                            num_events_ss,
                            it - num_tofbins // 2,
                            device=self._dev,
                            dtype=self.xp.int16,
                        )
                    )

                event_offset += num_events_ss

        return event_start_coords, event_end_coords, event_tofbins


class ListmodePETProjector(LinearOperator):
    """non-TOF and TOF listmode projector for regular polygon PET scanners

    Examples
    --------
    .. minigallery:: parallelproj.ListmodePETProjector
    """

    def __init__(
        self,
        event_start_coordinates: Array,
        event_end_coordinates: Array,
        img_shape: tuple[int, int, int],
        voxel_size: tuple[float, float, float],
        img_origin: None | Array = None,
    ) -> None:
        """
        Parameters
        ----------
        event_start_coordinates : Array
            float world coordinates of event LOR start points, shape (num_events, 3)
        event_end_coordinates : Array
            float world coordinates of event LOR end points, shape (num_events, 3)
        img_shape : tuple[int, int, int]
            shape of the image to be projected
        voxel_size : tuple[float, float, float]
            the voxel size of the image to be projected
        img_origin : None | Array, optional
            the origin of the image to be projected, by default None
            means that the center of the image is at world coordinate (0,0,0)
        """

        super().__init__()

        self._xstart = event_start_coordinates
        self._xend = event_end_coordinates

        self._xp = get_namespace(self._xstart)

        self._dev = device(event_start_coordinates)

        self._img_shape = img_shape
        self._voxel_size = self.xp.asarray(
            voxel_size, dtype=self.xp.float32, device=self._dev
        )

        if img_origin is None:
            self._img_origin = (
                -(
                    self.xp.asarray(
                        self._img_shape, dtype=self.xp.float32, device=self._dev
                    )
                    / 2
                )
                + 0.5
            ) * self._voxel_size
        else:
            self._img_origin = self.xp.asarray(
                img_origin, dtype=self.xp.float32, device=self._dev
            )

        self._tof_parameters = None
        self._tof = False
        self._tofbin = None

    @property
    def in_shape(self) -> tuple[int, int, int]:
        return self._img_shape

    @property
    def out_shape(self) -> tuple[int]:
        return (self._xstart.shape[0],)

    @property
    def num_events(self) -> int:
        """number of events"""
        return self._xstart.shape[0]

    @property
    def xp(self) -> ModuleType:
        """array module"""
        return self._xp

    @property
    def tof(self) -> bool:
        """bool indicating whether to use TOF projections or not"""
        return self._tof

    @tof.setter
    def tof(self, value: bool) -> None:
        if (value) and (self.tof_parameters is None):
            raise ValueError("must set tof_parameters first")
        if (value) and (self.event_tofbins is None):
            raise ValueError("must set event_tofbins first")

        self._tof = value

    @property
    def tof_parameters(self) -> TOFParameters | None:
        """TOF parameters"""
        return self._tof_parameters

    @tof_parameters.setter
    def tof_parameters(self, value: TOFParameters | None) -> None:
        if not (isinstance(value, TOFParameters) or value is None):
            raise ValueError("tof_parameters must be a TOFParameters object or None")
        self._tof_parameters = value

        if value is None:
            self._tof = False

    @property
    def event_tofbins(self) -> None | Array:
        """TOF bin of each event"""
        return self._tofbin

    @event_tofbins.setter
    def event_tofbins(self, value: None | Array) -> None:
        if value is None:
            self._tofbin = None
            self._tof = False
        else:
            if value.shape[0] != self.num_events:
                raise ValueError(
                    "tofbin must have the same number of elements as events"
                )
            self._tofbin = value

    @property
    def event_start_coordinates(self) -> Array:
        """coordinates of LOR start points"""
        return self._xstart

    @property
    def event_end_coordinates(self) -> Array:
        """coordinates of LOR end points"""
        return self._xend

    @property
    def voxel_size(self) -> Array:
        """voxel size"""
        return self._voxel_size

    def _apply(self, x: Array) -> Array:
        dev = array_api_compat.device(x)

        if not self.tof:
            x_fwd = parallelproj.joseph3d_fwd(
                self._xstart, self._xend, x, self._img_origin, self._voxel_size
            )
        else:
            x_fwd = parallelproj.joseph3d_fwd_tof_lm(
                self._xstart,
                self._xend,
                x,
                self._img_origin,
                self._voxel_size,
                self._tof_parameters.tofbin_width,
                self.xp.asarray(
                    [self._tof_parameters.sigma_tof], dtype=self.xp.float32, device=dev
                ),
                self.xp.asarray(
                    [self._tof_parameters.tofcenter_offset],
                    dtype=self.xp.float32,
                    device=dev,
                ),
                self.tof_parameters.num_sigmas,
                self._tofbin,
            )

        return x_fwd

    def _adjoint(self, y: Array) -> Array:
        dev = array_api_compat.device(y)

        if not self.tof:
            y_back = parallelproj.joseph3d_back(
                self._xstart,
                self._xend,
                self._img_shape,
                self._img_origin,
                self._voxel_size,
                y,
            )
        else:
            y_back = parallelproj.joseph3d_back_tof_lm(
                self._xstart,
                self._xend,
                self._img_shape,
                self._img_origin,
                self._voxel_size,
                y,
                self._tof_parameters.tofbin_width,
                self.xp.asarray(
                    [self._tof_parameters.sigma_tof], dtype=self.xp.float32, device=dev
                ),
                self.xp.asarray(
                    [self._tof_parameters.tofcenter_offset],
                    dtype=self.xp.float32,
                    device=dev,
                ),
                self.tof_parameters.num_sigmas,
                self._tofbin,
            )

        return y_back
