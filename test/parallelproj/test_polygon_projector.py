from __future__ import annotations

import parallelproj
import pytest
from types import ModuleType

from config import pytestmark


def test_polygon_projector(xp: ModuleType, dev: str) -> None:
    num_rings = 3
    symmetry_axis = 2
    num_sides = 17
    radius = 180
    radial_trim = 51
    max_ring_difference = 1

    voxel_size = (4., 4., 2.66)
    img_shape = (53, 53, 5)
    sinogram_order = parallelproj.SinogramSpatialAxisOrder.RVP

    # setup a test image with 3 hot rods
    x = xp.zeros(img_shape, dtype=xp.float32, device=dev)
    x[img_shape[0] // 2, img_shape[1] // 2, 1:] = 1.0
    x[-3, img_shape[1] // 2, :-1] = 1.0
    x[img_shape[0] // 2, -3, 1:] = 1.0

    # define the scanner geometry, lor descriptor and projector
    scanner = parallelproj.DemoPETScannerGeometry(xp,
                                                  dev,
                                                  num_rings=num_rings,
                                                  num_sides=num_sides,
                                                  radius=radius,
                                                  symmetry_axis=symmetry_axis)

    lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
        scanner,
        radial_trim=radial_trim,
        max_ring_difference=max_ring_difference,
        sinogram_order=sinogram_order)

    proj = parallelproj.RegularPolygonPETProjector(lor_desc, img_shape,
                                                   voxel_size)
    assert proj.out_shape == (lor_desc.num_rad, lor_desc.num_views,
                              lor_desc.num_planes)

    # non-TOF projections
    x_fwd = proj(x)
    y = xp.ones(x_fwd.shape, dtype=xp.float32, device=dev)
    y_back = proj.adjoint(y)

    # TOF projections
    tof_params = parallelproj.TOFParameters(num_tofbins=7, tofbin_width=30.6)
    proj.tof_parameters = tof_params
    assert proj.out_shape == (lor_desc.num_rad, lor_desc.num_views,
                              lor_desc.num_planes, tof_params.num_tofbins)

    x_fwd_tof = proj(x)
    y_tof = xp.ones(x_fwd_tof.shape, dtype=xp.float32, device=dev)
    y_back_tof = proj.adjoint(y_tof)

    # setup a projector with non default image origin and views
    views = xp.asarray([0, 1], device=dev)
    img_origin = xp.asarray([-100, -100, -5], device=dev, dtype=xp.float32)
    proj2 = parallelproj.RegularPolygonPETProjector(lor_desc,
                                                    img_shape,
                                                    voxel_size,
                                                    views=views,
                                                    img_origin=img_origin)

    assert xp.all(proj2.views == views)
    assert xp.all(proj2.img_origin == img_origin)

    assert proj2.in_shape == img_shape
    assert proj2.out_shape == (lor_desc.num_rad, views.shape[0],
                               lor_desc.num_planes)

    with pytest.raises(ValueError):
        # should raise an error since we have not set the TOF parameters
        proj2.tof = True

    proj2.tof_parameters = tof_params
    proj2.tof = True

    assert proj2.tof_parameters == tof_params
    assert proj2.tof == True

    proj2.tof = False
    assert proj2.tof == False

    proj2.tof_parameters = tof_params
    assert proj2.tof == True

    # setting tof_parameters to None should set tof to False
    proj2.tof_parameters = None
    assert proj2.tof == False

    with pytest.raises(ValueError):
        # should raise an error if we don't pass None | TOFParameters
        proj2.tof_parameters = 3.5
