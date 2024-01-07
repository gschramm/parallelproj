from __future__ import annotations

import parallelproj
import array_api_compat.numpy as np
from array_api_compat import to_device
import pytest
import matplotlib.pyplot as plt
from types import ModuleType

from config import pytestmark


def test_polygon_projector(xp: ModuleType, dev: str) -> None:
    num_rings = 3
    symmetry_axis = 2
    num_sides = 17
    radius = 180
    radial_trim = 51
    max_ring_difference = 1

    voxel_size = (4.0, 4.0, 2.66)
    img_shape = (53, 53, 5)
    sinogram_order = parallelproj.SinogramSpatialAxisOrder.RVP

    # setup a test image with 3 hot rods
    x = xp.zeros(img_shape, dtype=xp.float32, device=dev)
    x[img_shape[0] // 2, img_shape[1] // 2, 1:] = 1.0
    x[-3, img_shape[1] // 2, :-1] = 1.0
    x[img_shape[0] // 2, -3, 1:] = 1.0

    # define the scanner geometry, lor descriptor and projector
    scanner = parallelproj.DemoPETScannerGeometry(
        xp,
        dev,
        num_rings=num_rings,
        num_sides=num_sides,
        radius=radius,
        symmetry_axis=symmetry_axis,
    )

    lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
        scanner,
        radial_trim=radial_trim,
        max_ring_difference=max_ring_difference,
        sinogram_order=sinogram_order,
    )

    proj = parallelproj.RegularPolygonPETProjector(lor_desc, img_shape, voxel_size)
    assert proj.out_shape == (lor_desc.num_rad, lor_desc.num_views, lor_desc.num_planes)

    # non-TOF projections
    x_fwd = proj(x)
    y = xp.ones(x_fwd.shape, dtype=xp.float32, device=dev)
    y_back = proj.adjoint(y)

    # TOF projections
    tof_params = parallelproj.TOFParameters(num_tofbins=7, tofbin_width=30.6)
    proj.tof_parameters = tof_params
    assert proj.out_shape == (
        lor_desc.num_rad,
        lor_desc.num_views,
        lor_desc.num_planes,
        tof_params.num_tofbins,
    )

    x_fwd_tof = proj(x)
    y_tof = xp.ones(x_fwd_tof.shape, dtype=xp.float32, device=dev)
    y_back_tof = proj.adjoint(y_tof)

    # setup a projector with non default image origin and views
    views = xp.asarray([0, 1], device=dev)
    img_origin = xp.asarray([-100, -100, -5], device=dev, dtype=xp.float32)
    proj2 = parallelproj.RegularPolygonPETProjector(
        lor_desc, img_shape, voxel_size, views=views, img_origin=img_origin
    )

    assert xp.all(proj2.views == views)
    assert xp.all(proj2.img_origin == img_origin)

    assert proj2.in_shape == img_shape
    assert proj2.out_shape == (lor_desc.num_rad, views.shape[0], lor_desc.num_planes)

    with pytest.raises(ValueError):
        # should raise an error since we have not set the TOF parameters
        proj2.tof = True

    proj2.tof_parameters = tof_params
    proj2.tof = True

    assert proj2.tof_parameters == tof_params
    assert proj2.tof

    proj2.tof = False
    assert proj2.tof == False

    proj2.tof_parameters = tof_params
    assert proj2.tof

    # setting tof_parameters to None should set tof to False
    proj2.tof_parameters = None
    assert proj2.tof == False

    with pytest.raises(ValueError):
        # should raise an error if we don't pass None | TOFParameters
        proj2.tof_parameters = 3.5

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(121, projection="3d")
    proj.show_geometry(ax)
    plt.close(fig)


def _get_slice(r, v, p, lor_descriptor):
    sl = [slice(0), slice(0), slice(0)]

    sl[lor_descriptor.radial_axis_num] = slice(r, r + 1, None)
    sl[lor_descriptor.view_axis_num] = slice(v, v + 1, None)
    sl[lor_descriptor.plane_axis_num] = slice(p, p + 1, None)

    return tuple(sl)


def test_minimal_reg_polygon_projector(xp, dev) -> None:
    """test forward joseph non-tof forward projection with a minimal scanner geometry
    using a 3x3x3 image and a single voxel !=0 at the center of the image
    """

    radius = 12.0
    z = radius / float(np.sqrt(2.0))
    vox_size = 1.8
    vox_value = 2.7
    img_size = 3

    img_shape = 3 * (img_size,)

    # setup a test image with a single voxel != 0 at the center of the image
    x = xp.zeros(img_shape, device=dev)
    x[img_shape[0] // 2, img_shape[1] // 2, img_shape[2] // 2] = vox_value
    # setup a test image where all voxels have the same value
    x2 = vox_value * xp.ones(img_shape, device=dev)

    for symmetry_axis in [0, 1, 2]:
        scanner = parallelproj.RegularPolygonPETScannerGeometry(
            xp,
            dev,
            radius=radius,
            num_sides=8,
            num_lor_endpoints_per_side=1,
            lor_spacing=1.0,
            ring_positions=xp.asarray([-z, 0, z], device=dev),
            symmetry_axis=symmetry_axis,
        )

        for sinogram_order in parallelproj.SinogramSpatialAxisOrder:
            lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
                scanner, radial_trim=1, sinogram_order=sinogram_order
            )

            proj = parallelproj.RegularPolygonPETProjector(
                lor_desc, img_shape=img_shape, voxel_size=3 * (vox_size,)
            )

            x_fwd = proj(x)

            # check "corner to corner" projection which should be vox size *
            # vox value * sqrt(3)
            assert np.isclose(
                float(
                    x_fwd[_get_slice(2, 0, lor_desc.num_planes - 1, lor_desc)][0, 0, 0]
                ),
                vox_value * vox_size * np.sqrt(3),
            )
            assert np.isclose(
                float(
                    x_fwd[_get_slice(2, 2, lor_desc.num_planes - 1, lor_desc)][0, 0, 0]
                ),
                vox_value * vox_size * np.sqrt(3),
            )
            assert np.isclose(
                float(
                    x_fwd[_get_slice(2, 0, lor_desc.num_planes - 2, lor_desc)][0, 0, 0]
                ),
                vox_value * vox_size * np.sqrt(3),
            )
            assert np.isclose(
                float(
                    x_fwd[_get_slice(2, 2, lor_desc.num_planes - 2, lor_desc)][0, 0, 0]
                ),
                vox_value * vox_size * np.sqrt(3),
            )

            # check "central" (straight through) projection which should be vox
            # size * vox value
            assert np.isclose(
                float(x_fwd[_get_slice(2, 1, 1, lor_desc)][0, 0, 0]),
                vox_value * vox_size,
            )
            assert np.isclose(
                float(x_fwd[_get_slice(2, 3, 1, lor_desc)][0, 0, 0]),
                vox_value * vox_size,
            )

            # check "corner to corner" projection which should be vox size *
            # vox value * sqrt(2)
            assert np.isclose(
                float(x_fwd[_get_slice(2, 0, 1, lor_desc)][0, 0, 0]),
                vox_value * vox_size * np.sqrt(2),
            )
            assert np.isclose(
                float(x_fwd[_get_slice(2, 2, 1, lor_desc)][0, 0, 0]),
                vox_value * vox_size * np.sqrt(2),
            )

            x_fwd2 = proj(x2)

            # check "corner to corner" projection which should be vox size *
            # vox value * sqrt(3)
            assert np.isclose(
                float(
                    x_fwd2[_get_slice(2, 0, lor_desc.num_planes - 1, lor_desc)][0, 0, 0]
                ),
                img_size * vox_value * vox_size * np.sqrt(3),
            )
            assert np.isclose(
                float(
                    x_fwd2[_get_slice(2, 2, lor_desc.num_planes - 1, lor_desc)][0, 0, 0]
                ),
                img_size * vox_value * vox_size * np.sqrt(3),
            )
            assert np.isclose(
                float(
                    x_fwd2[_get_slice(2, 0, lor_desc.num_planes - 2, lor_desc)][0, 0, 0]
                ),
                img_size * vox_value * vox_size * np.sqrt(3),
            )
            assert np.isclose(
                float(
                    x_fwd2[_get_slice(2, 2, lor_desc.num_planes - 2, lor_desc)][0, 0, 0]
                ),
                img_size * vox_value * vox_size * np.sqrt(3),
            )

            # check "central" (straight through) projection which should be vox
            # size * vox value
            assert np.isclose(
                float(x_fwd2[_get_slice(2, 1, 1, lor_desc)][0, 0, 0]),
                img_size * vox_value * vox_size,
            )
            assert np.isclose(
                float(x_fwd2[_get_slice(2, 3, 1, lor_desc)][0, 0, 0]),
                img_size * vox_value * vox_size,
            )

            # check "corner to corner" projection which should be vox size *
            # vox value * sqrt(2)
            assert np.isclose(
                float(x_fwd2[_get_slice(2, 0, 1, lor_desc)][0, 0, 0]),
                img_size * vox_value * vox_size * np.sqrt(2),
            )
            assert np.isclose(
                float(x_fwd2[_get_slice(2, 2, 1, lor_desc)][0, 0, 0]),
                img_size * vox_value * vox_size * np.sqrt(2),
            )

            # setup the same projector without caching of the LOR endpoints
            projb = parallelproj.RegularPolygonPETProjector(
                lor_desc,
                img_shape=img_shape,
                voxel_size=3 * (vox_size,),
                cache_lor_endpoints=False,
            )
            x_fwd2b = projb(x2)

            assert projb.adjointness_test(xp, dev)

            assert projb.xstart is None
            assert projb.xend is None
            assert projb.lor_descriptor == lor_desc

            # check whether the projections with and without caching the LOR
            # endpoints are the same
            assert np.allclose(
                np.asarray(to_device(x_fwd2b, "cpu")),
                np.asarray(to_device(x_fwd2, "cpu")),
            )
