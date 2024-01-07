from __future__ import annotations

import parallelproj
import matplotlib.pyplot as plt

from types import ModuleType

from config import pytestmark


def test_pet_lors(xp: ModuleType, dev: str) -> None:
    num_rings = 3
    symmetry_axis = 2
    scanner = parallelproj.DemoPETScannerGeometry(
        xp, dev, num_rings, symmetry_axis=symmetry_axis
    )

    radial_trim = 65
    max_ring_difference = 2

    for sinogram_order in parallelproj.SinogramSpatialAxisOrder:
        lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
            scanner,
            radial_trim=radial_trim,
            max_ring_difference=max_ring_difference,
            sinogram_order=sinogram_order,
        )

        assert lor_desc.scanner == scanner
        assert lor_desc.max_ring_difference == max_ring_difference
        assert lor_desc.radial_trim == radial_trim
        assert lor_desc.num_views == scanner.num_lor_endpoints_per_ring // 2

        assert lor_desc.plane_axis_num == sinogram_order.name.find("P")
        assert lor_desc.radial_axis_num == sinogram_order.name.find("R")
        assert lor_desc.view_axis_num == sinogram_order.name.find("V")

        lor_coords = lor_desc.get_lor_coordinates()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        scanner.show_lor_endpoints(ax, show_linear_index=False)
        lor_desc.show_views(
            ax,
            views=xp.asarray([0], device=dev),
            planes=xp.asarray([0], device=dev),
            lw=0.1,
        )
        fig.show()
        plt.close(fig)

    # test lor descriptor without max_ring_difference and radial_trim
    lor_desc2 = parallelproj.RegularPolygonPETLORDescriptor(
        scanner, radial_trim=radial_trim, max_ring_difference=None
    )
