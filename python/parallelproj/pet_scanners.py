"""description of PET scanner geometries (detector coordinates)"""
from __future__ import annotations

import abc
import array_api_compat.numpy as np
from numpy.array_api._array_object import Array
import matplotlib.pyplot as plt

from types import ModuleType
from array_api_compat import to_device, size


class PETScannerModule(abc.ABC):
    """abstract base class for PET scanner module"""

    def __init__(self,
                 xp: ModuleType,
                 dev: str,
                 num_lor_endpoints: int,
                 affine_transformation_matrix: Array | None = None) -> None:
        """

        Parameters
        ----------
        xp: ModuleType
            array module to use for storing the LOR endpoints
        dev: str
            device to use for storing the LOR endpoints
        num_lor_endpoints : int
            number of LOR endpoints in the module
        affine_transformation_matrix : Array | None, optional
            4x4 affine transformation matrix applied to the LOR endpoint coordinates, default None
            if None, the 4x4 identity matrix is used
        """

        self._xp = xp
        self._dev = dev
        self._num_lor_endpoints = num_lor_endpoints
        self._lor_endpoint_numbers = self.xp.arange(num_lor_endpoints,
                                                    device=self.dev)

        if affine_transformation_matrix is None:
            aff_mat = self.xp.eye(4, device=self.dev)
            aff_mat[-1,-1] = 0
            self._affine_transformation_matrix = aff_mat
        else:
            self._affine_transformation_matrix = affine_transformation_matrix

    @property
    def xp(self) -> ModuleType:
        """array module to use for storing the LOR endpoints"""
        return self._xp

    @property
    def dev(self) -> str:
        """device to use for storing the LOR endpoints"""
        return self._dev

    @property
    def num_lor_endpoints(self) -> int:
        """total number of LOR endpoints in the module

        Returns
        -------
        int
        """
        return self._num_lor_endpoints

    @property
    def lor_endpoint_numbers(self) -> Array:
        """array enumerating all the LOR endpoints in the module

        Returns
        -------
        Array
        """
        return self._lor_endpoint_numbers

    @property
    def affine_transformation_matrix(self) -> Array:
        """4x4 affine transformation matrix

        Returns
        -------
        Array
        """
        return self._affine_transformation_matrix

    @abc.abstractmethod
    def get_raw_lor_endpoints(self, inds: Array | None = None) -> Array:
        """mapping from LOR endpoint indices within module to an array of "raw" world coordinates

        Parameters
        ----------
        inds : Array | None, optional
            an non-negative integer array of indices, default None
            if None means all possible indices [0, ... , num_lor_endpoints - 1]

        Returns
        -------
        Array
            a 3 x len(inds) float array with the world coordinates of the LOR endpoints
        """
        if inds is None:
            inds = self.lor_endpoint_numbers
        raise NotImplementedError

    def get_lor_endpoints(self, inds: Array | None = None) -> Array:
        """mapping from LOR endpoint indices within module to an array of "transformed" world coordinates

        Parameters
        ----------
        inds : Array | None, optional
            an non-negative integer array of indices, default None
            if None means all possible indices [0, ... , num_lor_endpoints - 1]

        Returns
        -------
        Array
            a 3 x len(inds) float array with the world coordinates of the LOR endpoints including an affine transformation
        """

        raw_lor_endpoints = self.get_raw_lor_endpoints(inds)

        tmp = self.xp.ones((raw_lor_endpoints.shape[0], 4), device=self.dev)
        tmp[:, :-1] = raw_lor_endpoints

        return (tmp @ self.affine_transformation_matrix.T)[:, :3]

    def show_lor_endpoints(self,
                           ax: plt.Axes,
                           annotation_fontsize: float = 0,
                           annotation_prefix: str = '',
                           annotation_offset: int = 0,
                           transformed: bool = True,
                           **kwargs) -> None:
        """show the LOR coordinates in a 3D scatter plot

        Parameters
        ----------
        ax : plt.Axes
            3D matplotlib axes
        annotation_fontsize : float, optional
            fontsize of LOR endpoint number annotation, by default 0
        annotation_prefix : str, optional
            prefix for annotation, by default ''
        annotation_offset : int, optional
            number to add to crystal number, by default 0
        transformed : bool, optional
            use transformed instead of raw coordinates, by default True
        """

        if transformed:
            all_lor_endpoints = self.get_lor_endpoints()
        else:
            all_lor_endpoints = self.get_raw_lor_endpoints()

        # convert to numpy array
        all_lor_endpoints = np.asarray(to_device(all_lor_endpoints, 'cpu'))

        ax.scatter(all_lor_endpoints[:, 0], all_lor_endpoints[:, 1],
                   all_lor_endpoints[:, 2], **kwargs)

        ax.set_box_aspect([
            ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')
        ])

        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_zlabel('x2')

        if annotation_fontsize > 0:
            for i in self.lor_endpoint_numbers:
                ax.text(all_lor_endpoints[i, 0],
                        all_lor_endpoints[i, 1],
                        all_lor_endpoints[i, 2],
                        f'{annotation_prefix}{i+annotation_offset}',
                        fontsize=annotation_fontsize)


class RegularPolygonPETScannerModule(PETScannerModule):
    """Regular polygon PET scanner module (detectors on a regular polygon)"""

    def __init__(self,
                 xp: ModuleType,
                 dev: str,
                 radius: float,
                 num_sides: int,
                 num_lor_endpoints_per_side: int,
                 lor_spacing: float,
                 ax0: int = 2,
                 ax1: int = 1,
                 affine_transformation_matrix: Array | None = None) -> None:
        """

        Parameters
        ----------
        xp: ModuleType
            array module to use for storing the LOR endpoints
        device: str
            device to use for storing the LOR endpoints
        radius : float
            inner radius of the regular polygon
        num_sides: int
            number of sides of the regular polygon
        num_lor_endpoints_per_sides: int
            number of LOR endpoints per side
        lor_spacing : float
            spacing between the LOR endpoints in the polygon direction
        ax0 : int, optional
            axis number for the first direction, by default 2
        ax1 : int, optional
            axis number for the second direction, by default 1
        affine_transformation_matrix : Array | None, optional
            4x4 affine transformation matrix applied to the LOR endpoint coordinates, default None
            if None, the 4x4 identity matrix is used
        """

        self._radius = radius
        self._num_sides = num_sides
        self._num_lor_endpoints_per_side = num_lor_endpoints_per_side
        self._ax0 = ax0
        self._ax1 = ax1
        self._lor_spacing = lor_spacing
        super().__init__(xp, dev, num_sides * num_lor_endpoints_per_side,
                         affine_transformation_matrix)

    @property
    def radius(self) -> float:
        """inner radius of the regular polygon

        Returns
        -------
        float
        """
        return self._radius

    @property
    def num_sides(self) -> int:
        """number of sides of the regular polygon

        Returns
        -------
        int
        """
        return self._num_sides

    @property
    def num_lor_endpoints_per_side(self) -> int:
        """number of LOR endpoints per side

        Returns
        -------
        int
        """
        return self._num_lor_endpoints_per_side

    @property
    def ax0(self) -> int:
        """axis number for the first module direction

        Returns
        -------
        int
        """
        return self._ax0

    @property
    def ax1(self) -> int:
        """axis number for the second module direction

        Returns
        -------
        int
        """
        return self._ax1

    @property
    def lor_spacing(self) -> float:
        """spacing between the LOR endpoints in a module along the polygon

        Returns
        -------
        float
        """
        return self._lor_spacing

    # abstract method from base class to be implemented
    def get_raw_lor_endpoints(self, inds: Array | None = None) -> Array:
        if inds is None:
            inds = self.lor_endpoint_numbers

        side = inds // self.num_lor_endpoints_per_side
        tmp = inds - side * self.num_lor_endpoints_per_side
        tmp = self.xp.astype(
            tmp, float) - (self.num_lor_endpoints_per_side / 2 - 0.5)

        phi = 2 * self.xp.pi * self.xp.astype(side, float) / self.num_sides

        lor_endpoints = self.xp.zeros((self.num_lor_endpoints, 3),
                                      device=self.dev)
        lor_endpoints[:, self.ax0] = self.xp.cos(
            phi) * self.radius - self.xp.sin(phi) * self.lor_spacing * tmp
        lor_endpoints[:, self.ax1] = self.xp.sin(
            phi) * self.radius + self.xp.cos(phi) * self.lor_spacing * tmp

        return lor_endpoints


class ModularizedPETScannerGeometry:
    """description of a PET scanner geometry consisting of LOR endpoint modules"""

    def __init__(self, modules: tuple[PETScannerModule]):
        """
        Parameters
        ----------
        modules : tuple[PETScannerModule]
            a tuple of scanner modules
        """

        # member variable that determines whether we want to use
        # a numpy or cupy array to store the array of all lor endpoints
        self._modules = modules
        self._num_modules = len(self._modules)
        self._num_lor_endpoints_per_module = self.xp.asarray(
            [x.num_lor_endpoints for x in self._modules], device=self.dev)
        self._num_lor_endpoints = int(
            self.xp.sum(self._num_lor_endpoints_per_module))

        self.setup_all_lor_endpoints()

    def setup_all_lor_endpoints(self) -> None:
        """calculate the position of all lor endpoints by iterating over
           the modules and calculating the transformed coordinates of all
           module endpoints
        """

        self._all_lor_endpoints_index_offset = self.xp.asarray([
            int(sum(self._num_lor_endpoints_per_module[:i]))
            for i in range(size(self._num_lor_endpoints_per_module))
        ],
                                                               device=self.dev)

        self._all_lor_endpoints = self.xp.zeros((self._num_lor_endpoints, 3),
                                                device=self.dev,
                                                dtype=self.xp.float32)

        for i, module in enumerate(self._modules):
            self._all_lor_endpoints[
                int(self._all_lor_endpoints_index_offset[i]):int(
                    self._all_lor_endpoints_index_offset[i] +
                    module.num_lor_endpoints), :] = module.get_lor_endpoints()

        self._all_lor_endpoints_module_number = [
            int(self._num_lor_endpoints_per_module[i]) * [i]
            for i in range(self._num_modules)
        ]

        self._all_lor_endpoints_module_number = self.xp.asarray(
            [i for r in self._all_lor_endpoints_module_number for i in r],
            device=self.dev)

    @property
    def modules(self) -> tuple[PETScannerModule]:
        """tuple of modules defining the scanner"""
        return self._modules

    @property
    def num_modules(self) -> int:
        """the number of modules defining the scanner"""
        return self._num_modules

    @property
    def num_lor_endpoints_per_module(self) -> Array:
        """numpy array showing how many LOR endpoints are in every module"""
        return self._num_lor_endpoints_per_module

    @property
    def num_lor_endpoints(self) -> int:
        """the total number of LOR endpoints in the scanner"""
        return self._num_lor_endpoints

    @property
    def all_lor_endpoints_index_offset(self) -> Array:
        """the offset in the linear (flattend) index for all LOR endpoints"""
        return self._all_lor_endpoints_index_offset

    @property
    def all_lor_endpoints_module_number(self) -> Array:
        """the module number of all LOR endpoints"""
        return self._all_lor_endpoints_module_number

    @property
    def all_lor_endpoints(self) -> Array:
        """the world coordinates of all LOR endpoints"""
        return self._all_lor_endpoints

    @property
    def xp(self) -> ModuleType:
        """module indicating whether the LOR endpoints are stored as numpy or cupy array"""
        return self._modules[0].xp

    @property
    def dev(self) -> str:
        return self._modules[0].dev

    def linear_lor_endpoint_index(
        self,
        module: Array,
        index_in_module: Array,
    ) -> Array:
        """transform the module + index_in_modules indices into a flattened / linear LOR endpoint index

        Parameters
        ----------
        module : Array
            containing module numbers
        index_in_module : Array
            containing index in modules

        Returns
        -------
        Array
            the flattened LOR endpoint index
        """
        #    index_in_module = self._xp.asarray(index_in_module)

        return self.xp.take(self.all_lor_endpoints_index_offset,
                            module, axis = 0) + index_in_module

    def get_lor_endpoints(self, module: Array,
                          index_in_module: Array) -> Array:
        """get the coordinates for LOR endpoints defined by module and index in module

        Parameters
        ----------
        module : Array
            the module number of the LOR endpoints
        index_in_module : Array
            the index in module number of the LOR endpoints

        Returns
        -------
        Array
            the 3 world coordinates of the LOR endpoints
        """
        return self.xp.take(self.all_lor_endpoints,
                            self.linear_lor_endpoint_index(
                                module, index_in_module),
                            axis=0)

    def show_lor_endpoints(self,
                           ax: plt.Axes,
                           show_linear_index: bool = True,
                           **kwargs) -> None:
        """show all LOR endpoints in a 3D plot

        Parameters
        ----------
        ax : plt.Axes
            a 3D matplotlib axes
        show_linear_index : bool, optional
            annotate the LOR endpoints with the linear LOR endpoint index
        **kwargs : keyword arguments
            passed to show_lor_endpoints() of the scanner module
        """
        for i, module in enumerate(self.modules):
            if show_linear_index:
                offset = np.asarray(
                    to_device(self.all_lor_endpoints_index_offset[i], 'cpu'))
                prefix = f''
            else:
                offset = 0
                prefix = f'{i},'

            module.show_lor_endpoints(ax,
                                      annotation_offset=offset,
                                      annotation_prefix=prefix,
                                      **kwargs)


class RegularPolygonPETScannerGeometry(ModularizedPETScannerGeometry):
    """description of a PET scanner geometry consisting stacked regular polygons"""

    def __init__(self, xp: ModuleType, dev: str, radius: float, num_sides: int,
                 num_lor_endpoints_per_side: int, lor_spacing: float,
                 num_rings: int, ring_positions: Array,
                 symmetry_axis: int) -> None:
        """
        Parameters
        ----------
        xp: ModuleType
            array module to use for storing the LOR endpoints
        dev: str
            device to use for storing the LOR endpoints
        radius : float
            radius of the scanner
        num_sides : int
            number of sides (faces) of each regular polygon
        num_lor_endpoints_per_side : int
            number of LOR endpoints in each side (face) of each polygon
        lor_spacing : float
            spacing between the LOR endpoints in each side
        num_rings : int
            the number of rings (regular polygons)
        ring_positions : Array
            1D array with the coordinate of the rings along the ring axis
        symmetry_axis : int
            the ring axis (0,1,2)
        """

        self._radius = radius
        self._num_sides = num_sides
        self._num_lor_endpoints_per_side = num_lor_endpoints_per_side
        self._num_rings = num_rings
        self._lor_spacing = lor_spacing
        self._symmetry_axis = symmetry_axis
        self._ring_positions = ring_positions

        if symmetry_axis == 0:
            self._ax0 = 2
            self._ax1 = 1
        elif symmetry_axis == 1:
            self._ax0 = 0
            self._ax1 = 2
        elif symmetry_axis == 2:
            self._ax0 = 1
            self._ax1 = 0

        modules = []

        for ring in range(num_rings):
            aff_mat = xp.eye(4, device=dev)
            aff_mat[symmetry_axis, -1] = ring_positions[ring]

            modules.append(
                RegularPolygonPETScannerModule(
                    xp,
                    dev,
                    radius,
                    num_sides,
                    num_lor_endpoints_per_side=num_lor_endpoints_per_side,
                    lor_spacing=lor_spacing,
                    affine_transformation_matrix=aff_mat,
                    ax0=self._ax0,
                    ax1=self._ax1))

        super().__init__(tuple(modules))

        self._all_lor_endpoints_index_in_ring = self.xp.arange(
            self.num_lor_endpoints, device=dev
        ) - self.all_lor_endpoints_ring_number * self.num_lor_endpoints_per_module[
            0]

    @property
    def radius(self) -> float:
        """radius of the scanner"""
        return self._radius

    @property
    def num_sides(self) -> int:
        """number of sides (faces) of each polygon"""
        return self._num_sides

    @property
    def num_lor_endpoints_per_side(self) -> int:
        """number of LOR endpoints per side (face) in each polygon"""
        return self._num_lor_endpoints_per_side

    @property
    def num_rings(self) -> int:
        """number of rings (regular polygons)"""
        return self._num_rings

    @property
    def lor_spacing(self) -> float:
        """the spacing between the LOR endpoints in every side (face) of each polygon"""
        return self._lor_spacing

    @property
    def symmetry_axis(self) -> int:
        """The symmetry axis. Also called axial (or ring) direction."""
        return self._symmetry_axis

    @property
    def all_lor_endpoints_ring_number(self) -> Array:
        """the ring (regular polygon) number of all LOR endpoints"""
        return self._all_lor_endpoints_module_number

    @property
    def all_lor_endpoints_index_in_ring(self) -> Array:
        """the index within the ring (regular polygon) of all LOR endpoints"""
        return self._all_lor_endpoints_index_in_ring

    @property
    def num_lor_endpoints_per_ring(self) -> int:
        """the number of LOR endpoints per ring (regular polygon)"""
        return int(self._num_lor_endpoints_per_module[0])

    @property
    def ring_positions(self) -> Array:
        """the ring (regular polygon) positions"""
        return self._ring_positions


class DemoPETScannerGeometry(RegularPolygonPETScannerGeometry):
    """Demo PET scanner geometry consisting of a 34-ogon with 16 LOR endpoints per side and 36 rings"""

    def __init__(self,
                 xp: ModuleType,
                 dev: str,
                 radius: float = 0.5 * (744.1 + 2 * 8.51),
                 num_sides: int = 34,
                 num_lor_endpoints_per_side: int = 16,
                 lor_spacing: float = 4.03125,
                 num_rings: int = 36,
                 ring_positions: Array | None = None,
                 symmetry_axis: int = 2) -> None:
        """
        Parameters
        ----------
        xp : ModuleType
            array module
        dev : str
            the device to use
        radius : float, optional
            radius of the regular polygon, by default 0.5*(744.1 + 2 * 8.51)
        num_sides : int, optional
            number of sides of the polygon, by default 34
        num_lor_endpoints_per_side : int, optional
            number of LOR endpoints per side, by default 16
        lor_spacing : float, optional
            spacing between the LOR endpoints, by default 4.03125
        num_rings : int, optional
            number of rings, by default 36
        ring_positions : Array | None, optional
            position of the rings, by default None means equally spaced rings 5.32mm apart
        symmetry_axis : int, optional
            symmetry (axial) axis of the scanner, by default 2
        """

        if ring_positions is None:
            ring_positions = 5.32 * xp.arange(
                num_rings, device=dev, dtype=xp.float32) + (xp.astype(
                    xp.arange(num_rings, device=dev) // 9, xp.float32)) * 2.8
            ring_positions -= 0.5 * xp.max(ring_positions)

        super().__init__(xp,
                         dev,
                         radius=radius,
                         num_sides=num_sides,
                         num_lor_endpoints_per_side=num_lor_endpoints_per_side,
                         lor_spacing=lor_spacing,
                         num_rings=num_rings,
                         ring_positions=ring_positions,
                         symmetry_axis=symmetry_axis)
