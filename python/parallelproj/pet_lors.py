"""description of PET LORs (and sinograms bins) consisting of two detector endpoints"""
from __future__ import annotations

import abc
import enum
import array_api_compat.numpy as np
from numpy.array_api._array_object import Array
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from types import ModuleType
from array_api_compat import to_device

from .pet_scanners import ModularizedPETScannerGeometry, RegularPolygonPETScannerGeometry


class SinogramSpatialAxisOrder(enum.Enum):
    """order of spatial axis in a sinogram R (radial), V (view), P (plane)"""

    RVP = enum.auto()
    """[radial,view,plane]"""
    RPV = enum.auto()
    """[radial,plane,view]"""
    VRP = enum.auto()
    """[view,radial,plane]"""
    VPR = enum.auto()
    """[view,plane,radial]"""
    PRV = enum.auto()
    """[plane,radial,view]"""
    PVR = enum.auto()
    """[plane,view,radial]"""


class PETLORDescriptor(abc.ABC):
    """abstract base class to describe which modules / indices in modules of a
       modularized PET scanner are in coincidence; defining geometrical LORs"""

    def __init__(self, scanner: ModularizedPETScannerGeometry) -> None:
        """
        Parameters
        ----------
        scanner : ModularizedPETScannerGeometry
            a modularized PET scanner
        """
        self._scanner = scanner

    @abc.abstractmethod
    def get_lor_indices(self) -> tuple[Array, Array, Array, Array]:
        """return the start and end indices of all LORs / or a subset of LORs"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_lor_coordinates(self) -> tuple[Array, Array]:
        """return the start and end coordinates of all (or a subset of) LORs"""
        raise NotImplementedError

    @property
    def scanner(self) -> ModularizedPETScannerGeometry:
        """the scanner for which coincidences are described"""
        return self._scanner

    @property
    def xp(self) -> ModuleType:
        """array module to use for storing the LOR endpoints"""
        return self.scanner.xp

    @property
    def dev(self) -> str:
        """device to use for storing the LOR endpoints"""
        return self.scanner.dev


class RegularPolygonPETLORDescriptor(PETLORDescriptor):
    """Coincidence descriptor for a regular polygon PET scanner where
       we have coincidences within and between "rings (polygons of modules)"
       The geometrical LORs can be sorted into a sinogram having a
       "plane", "view" and "radial" axis."""

    def __init__(
        self,
        scanner: RegularPolygonPETScannerGeometry,
        radial_trim: int = 3,
        max_ring_difference: int | None = None,
        sinogram_order: SinogramSpatialAxisOrder = SinogramSpatialAxisOrder.RVP
    ) -> None:
        """

        Parameters
        ----------
        scanner : RegularPolygonPETScannerGeometry
            a regular polygon PET scanner
        radial_trim : int, optional
            number of geometrial LORs to disregard in the radial direction, by default 3
        max_ring_difference : int | None, optional
            maximim ring difference to consider for coincidences, by default None means
            all ring differences are included
        sinogram_order : SinogramSpatialAxisOrder, optional
            the order of the sinogram axes, by default SinogramSpatialAxisOrder.RVP
        """

        super().__init__(scanner)

        self._scanner = scanner
        self._radial_trim = radial_trim

        if max_ring_difference is None:
            self._max_ring_difference = scanner.num_rings - 1
        else:
            self._max_ring_difference = max_ring_difference

        self._num_rad = (scanner.num_lor_endpoints_per_ring +
                         1) - 2 * self._radial_trim
        self._num_views = scanner.num_lor_endpoints_per_ring // 2

        self._sinogram_order = sinogram_order

        self._setup_plane_indices()
        self._setup_view_indices()

    @property
    def radial_trim(self) -> int:
        """number of geometrial LORs to disregard in the radial direction"""
        return self._radial_trim

    @property
    def max_ring_difference(self) -> int:
        """the maximum ring difference"""
        return self._max_ring_difference

    @property
    def num_planes(self) -> int:
        """number of planes in the sinogram"""
        return self._num_planes

    @property
    def num_rad(self) -> int:
        """number of radial elements in the sinogram"""
        return self._num_rad

    @property
    def num_views(self) -> int:
        """number of views in the sinogram"""
        return self._num_views

    @property
    def start_plane_index(self) -> Array:
        """start plane for all planes"""
        return self._start_plane_index

    @property
    def end_plane_index(self) -> Array:
        """end plane for all planes"""
        return self._end_plane_index

    @property
    def start_in_ring_index(self) -> Array:
        """start index within ring for all views - shape (num_view, num_rad)"""
        return self._start_in_ring_index

    @property
    def end_in_ring_index(self) -> Array:
        """end index within ring for all views - shape (num_view, num_rad)"""
        return self._end_in_ring_index

    @property
    def sinogram_order(self) -> SinogramSpatialAxisOrder:
        """the order of the sinogram axes"""
        return self._sinogram_order

    @property
    def plane_axis_num(self) -> int:
        """the axis number of the plane axis"""
        return self.sinogram_order.name.find('P')

    @property
    def radial_axis_num(self) -> int:
        """the axis number of the radial axis"""
        return self.sinogram_order.name.find('R')

    @property
    def view_axis_num(self) -> int:
        """the axis number of the view axis"""
        return self.sinogram_order.name.find('V')

    def _setup_plane_indices(self) -> None:
        """setup the start / end plane indices (similar to a Michelogram)
        """
        self._start_plane_index = self.xp.arange(self.scanner.num_rings,
                                                 dtype=self.xp.int32,
                                                 device=self.dev)
        self._end_plane_index = self.xp.arange(self.scanner.num_rings,
                                               dtype=self.xp.int32,
                                               device=self.dev)

        for i in range(1, self._max_ring_difference + 1):
            tmp1 = self.xp.arange(self.scanner.num_rings - i,
                                  dtype=self.xp.int16,
                                  device=self.dev)
            tmp2 = self.xp.arange(self.scanner.num_rings - i,
                                  dtype=self.xp.int16,
                                  device=self.dev) + i

            self._start_plane_index = self.xp.concat(
                (self._start_plane_index, tmp1, tmp2))
            self._end_plane_index = self.xp.concat(
                (self._end_plane_index, tmp2, tmp1))

        self._num_planes = self._start_plane_index.shape[0]

    def _setup_view_indices(self) -> None:
        """setup the start / end view indices
        """
        n = self.scanner.num_lor_endpoints_per_ring

        m = 2 * (n // 2)

        self._start_in_ring_index = self.xp.zeros(
            (self._num_views, self._num_rad),
            dtype=self.xp.int32,
            device=self.dev)
        self._end_in_ring_index = self.xp.zeros(
            (self._num_views, self._num_rad),
            dtype=self.xp.int32,
            device=self.dev)

        for view in np.arange(self._num_views):
            self._start_in_ring_index[view, :] = (
                self.xp.concat(
                    (self.xp.arange(m) // 2, self.xp.asarray([n // 2]))) -
                view)[self._radial_trim:-self._radial_trim]
            self._end_in_ring_index[view, :] = (
                self.xp.concat(
                    (self.xp.asarray([-1]), -((self.xp.arange(m) + 4) // 2))) -
                view)[self._radial_trim:-self._radial_trim]

        # shift the negative indices
        self._start_in_ring_index = self.xp.where(
            self._start_in_ring_index >= 0, self._start_in_ring_index,
            self._start_in_ring_index + n)
        self._end_in_ring_index = self.xp.where(self._end_in_ring_index >= 0,
                                                self._end_in_ring_index,
                                                self._end_in_ring_index + n)

    def get_lor_indices(
            self,
            views: None | Array = None) -> tuple[Array, Array, Array, Array]:
        """return the start and end indices of all LORs / or a subset of views

        Parameters
        ----------
        views : None | Array, optional
            the views to consider, by default None means all views
        Returns
        -------
        start_mods, end_mods, start_inds, end_inds
        """

        if views is None:
            views = self.xp.arange(self.num_views, device=self.dev)

        # setup the module and in_module (in_ring) indices for all LORs in PVR order
        start_inring_inds = self.xp.reshape(
            self.xp.take(self.start_in_ring_index, views, axis=0), (-1, ))
        end_inring_inds = self.xp.reshape(
            self.xp.take(self.end_in_ring_index, views, axis=0), (-1, ))

        start_mods, start_inds = self.xp.meshgrid(self.start_plane_index,
                                                  start_inring_inds,
                                                  indexing='ij')
        end_mods, end_inds = self.xp.meshgrid(self.end_plane_index,
                                              end_inring_inds,
                                              indexing='ij')

        # reshape to PVR dimensions (radial moving fastest, planes moving slowest)
        sinogram_spatial_shape = (self.num_planes, views.shape[0],
                                  self.num_rad)
        start_mods = self.xp.reshape(start_mods, sinogram_spatial_shape)
        end_mods = self.xp.reshape(end_mods, sinogram_spatial_shape)
        start_inds = self.xp.reshape(start_inds, sinogram_spatial_shape)
        end_inds = self.xp.reshape(end_inds, sinogram_spatial_shape)

        new_order = (0, 1, 2)  # new order for PVR sinogram shape

        if self.sinogram_order is not SinogramSpatialAxisOrder.PVR:
            if self.sinogram_order is SinogramSpatialAxisOrder.RVP:
                new_order = (2, 1, 0)
            elif self.sinogram_order is SinogramSpatialAxisOrder.RPV:
                new_order = (2, 0, 1)
            elif self.sinogram_order is SinogramSpatialAxisOrder.VRP:
                new_order = (1, 2, 0)
            elif self.sinogram_order is SinogramSpatialAxisOrder.VPR:
                new_order = (1, 0, 2)
            elif self.sinogram_order is SinogramSpatialAxisOrder.PRV:
                new_order = (0, 2, 1)

            start_mods = self.xp.permute_dims(start_mods, new_order)
            end_mods = self.xp.permute_dims(end_mods, new_order)

            start_inds = self.xp.permute_dims(start_inds, new_order)
            end_inds = self.xp.permute_dims(end_inds, new_order)

        return start_mods, end_mods, start_inds, end_inds

    def get_lor_coordinates(
        self,
        views: None | Array = None,
    ) -> tuple[Array, Array]:
        """return the start and end coordinates of all LORs / or a subset of views

        Parameters
        ----------
        views : None | Array, optional
            the views to consider, by default None means all views
        sinogram_order : SinogramSpatialAxisOrder, optional
            the order of the sinogram axes, by default SinogramSpatialAxisOrder.RVP

        Returns
        -------
        xstart, xend : Array
           2 dimensional floating point arrays containing the start and end coordinates of all LORs
        """

        start_mods, end_mods, start_inds, end_inds = self.get_lor_indices(
            views)
        sinogram_spatial_shape = start_mods.shape

        start_mods = self.xp.reshape(start_mods, (-1, ))
        start_inds = self.xp.reshape(start_inds, (-1, ))

        end_mods = self.xp.reshape(end_mods, (-1, ))
        end_inds = self.xp.reshape(end_inds, (-1, ))

        x_start = self.xp.reshape(
            self.scanner.get_lor_endpoints(start_mods, start_inds),
            sinogram_spatial_shape + (3, ))
        x_end = self.xp.reshape(
            self.scanner.get_lor_endpoints(end_mods, end_inds),
            sinogram_spatial_shape + (3, ))

        return x_start, x_end

    def show_views(self,
                   ax: plt.Axes,
                   views: Array,
                   planes: Array,
                   lw: float = 0.2,
                   **kwargs) -> None:
        """show all LORs of a single view in a given plane

        Parameters
        ----------
        ax : plt.Axes
            a 3D matplotlib axes
        view : int
            the view number
        plane : int
            the plane number
        lw : float, optional
            the line width, by default 0.2
        """

        xs, xe = self.get_lor_coordinates(views=views)
        print(xs.shape)

        xs = self.xp.reshape(
            self.xp.take(xs, planes, axis=self.plane_axis_num), (-1, 3))
        xe = self.xp.reshape(
            self.xp.take(xe, planes, axis=self.plane_axis_num), (-1, 3))

        p1s = np.asarray(to_device(xs, 'cpu'))
        p2s = np.asarray(to_device(xe, 'cpu'))

        ls = np.hstack([p1s, p2s]).copy()
        ls = ls.reshape((-1, 2, 3))
        lc = Line3DCollection(ls, linewidths=lw, **kwargs)
        ax.add_collection(lc)
