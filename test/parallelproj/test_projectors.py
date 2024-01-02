from __future__ import annotations

import pytest
import parallelproj
import array_api_compat
import array_api_compat.numpy as np

from config import pytestmark


def allclose(x, y, atol: float = 1e-8, rtol: float = 1e-5) -> bool:
    """check if two arrays are close to each other, given absolute and relative error
       inspired by numpy.allclose
    """
    xp = array_api_compat.array_namespace(x)
    return bool(xp.all(xp.less_equal(xp.abs(x - y), atol + rtol * xp.abs(y))))


def test_parallelviewprojector(xp, dev, verbose=True):
    image_shape = (2, 2)
    voxel_size = (2., 2.)
    image_origin = (-1., -1.)

    radial_positions = xp.asarray([-1, 0, 1], dtype=xp.float32, device=dev)
    view_angles = xp.asarray([0, xp.pi / 2], dtype=xp.float32, device=dev)
    radius = 3.

    proj2d = parallelproj.ParallelViewProjector2D(image_shape,
                                                  radial_positions,
                                                  view_angles, radius,
                                                  image_origin, voxel_size)

    assert proj2d.num_views == array_api_compat.size(view_angles)
    assert proj2d.num_rad == array_api_compat.size(radial_positions)

    xstart = proj2d.xstart
    xend = proj2d.xend

    assert allclose(proj2d.image_origin[1:],
                    xp.asarray(image_origin, device=dev))
    assert proj2d.image_shape == image_shape
    assert allclose(proj2d.voxel_size[1:], xp.asarray(voxel_size, device=dev))
    assert proj2d.dev == array_api_compat.device(xstart)

    proj2d.adjointness_test(xp, dev, verbose=verbose)

    # test a simple 2D projection
    x2d = xp.reshape(xp.arange(4, dtype=xp.float32, device=dev), (2, 2))
    x_fwd = proj2d(x2d)

    exp_result = xp.asarray([[2., 8.], [6., 6.], [10., 4.]],
                            dtype=xp.float32,
                            device=dev)

    if verbose:
        print(
            f'module = {xp.__name__}  -  cuda_enabled {parallelproj.num_visible_cuda_devices > 0}'
        )
        print('calculated 2d projection = ', x_fwd)
        print('expected   2d projection = ', exp_result)
        print('abs diff                 = ', xp.abs(x_fwd - exp_result))
        print('rel diff                 = ',
              xp.abs(x_fwd - exp_result) / exp_result)
        print('')

    assert allclose(x_fwd, exp_result)

    fig = proj2d.show_views(
        image=xp.ones(image_shape, dtype=xp.float32, device=dev))

    # setup a simple 3D projector with 2 rings

    image_shape3d = (2, 2, 2)
    image_origin3d = (-1, -1., -1.)
    voxel_size3d = (2., 2., 2.)
    ring_positions = xp.asarray([-1, 0, 1.], dtype=xp.float32, device=dev)

    proj3d = parallelproj.ParallelViewProjector3D(image_shape3d,
                                                  radial_positions,
                                                  view_angles,
                                                  radius,
                                                  image_origin3d,
                                                  voxel_size3d,
                                                  ring_positions,
                                                  max_ring_diff=1)

    xstart = proj3d.xstart
    xend = proj3d.xend

    proj3d.adjointness_test(xp, dev, verbose=verbose)

    # test a simple 3D projection
    x3d = xp.reshape(xp.arange(8, dtype=xp.float32, device=dev), (2, 2, 2))
    x3d_fwd = proj3d(x3d)

    # check if we get the expected results for the 3 direct planes
    exp_result_dp0 = xp.asarray([[4., 16.], [12., 12.], [20., 8.]],
                                dtype=xp.float32,
                                device=dev)
    exp_result_dp1 = xp.asarray([[6., 18.], [14., 14.], [22., 10.]],
                                dtype=xp.float32,
                                device=dev)
    exp_result_dp2 = xp.asarray([[8., 20.], [16., 16.], [24., 12.]],
                                dtype=xp.float32,
                                device=dev)

    assert allclose(x3d_fwd[..., 0], exp_result_dp0)
    assert allclose(x3d_fwd[..., 1], exp_result_dp1)
    assert allclose(x3d_fwd[..., 2], exp_result_dp2)

    # test is max_ring_diff = None works
    proj3d_2 = parallelproj.ParallelViewProjector3D(image_shape3d,
                                                    radial_positions,
                                                    view_angles,
                                                    radius,
                                                    image_origin3d,
                                                    voxel_size3d,
                                                    ring_positions,
                                                    max_ring_diff=None)

    assert proj3d_2.max_ring_diff == array_api_compat.size(ring_positions) - 1

    # test whether span > 1 raises execption
    with pytest.raises(Exception) as e_info:
        proj3d_2 = parallelproj.ParallelViewProjector3D(image_shape3d,
                                                        radial_positions,
                                                        view_angles,
                                                        radius,
                                                        image_origin3d,
                                                        voxel_size3d,
                                                        ring_positions,
                                                        max_ring_diff=None,
                                                        span=3)

def test_minimal_reg_polygon_projector(xp, dev) -> None:
    """test forward joseph non-tof forward projection with a minimal scanner geometry
       using a 3x3x3 image and a single voxel !=0 at the center of the image
    """

    radius = 12.
    z = radius / float(np.sqrt(2.))
    vox_size = 1.8
    vox_value = 2.7
    img_size = 3

    img_shape = 3*(img_size,)
    
    scanner = parallelproj.RegularPolygonPETScannerGeometry(
        xp, dev, radius = radius, num_sides = 8,
        num_lor_endpoints_per_side = 1, lor_spacing = 1.,
        num_rings = 3, ring_positions = xp.asarray([-z, 0, z], device = dev),
        symmetry_axis = 2)
    
    lor_desc = parallelproj.RegularPolygonPETLORDescriptor(scanner, radial_trim = 1)
    
    proj = parallelproj.RegularPolygonPETProjector(lor_desc, img_shape = img_shape, voxel_size = 3*(vox_size,))
    
    # setup a test image with a single voxel != 0 at the center of the image
    x = xp.zeros(img_shape, device = dev)
    x[img_shape[0]//2, img_shape[1]//2, img_shape[2]//2] = vox_value
    x_fwd = proj(x)
    
    # check "corner to corner" projection which should be vox size * vox value * sqrt(3)
    assert np.isclose(float(x_fwd[2,0,-1]), vox_value * vox_size * np.sqrt(3))
    assert np.isclose(float(x_fwd[2,2,-1]), vox_value * vox_size * np.sqrt(3))
    assert np.isclose(float(x_fwd[2,0,-2]), vox_value * vox_size * np.sqrt(3))
    assert np.isclose(float(x_fwd[2,2,-2]), vox_value * vox_size * np.sqrt(3))
    
    # check "central" (straight through) projection which should be vox size * vox value
    assert np.isclose(float(x_fwd[2,1,1]), vox_value * vox_size)
    assert np.isclose(float(x_fwd[2,3,1]), vox_value * vox_size)
    
    # check "corner to corner" projection which should be vox size * vox value * sqrt(2)
    assert np.isclose(float(x_fwd[2,0,1]), vox_value * vox_size * np.sqrt(2))
    assert np.isclose(float(x_fwd[2,2,1]), vox_value * vox_size * np.sqrt(2))

    # setup a test image where all voxels have the same value
    x2 = vox_value*xp.ones(img_shape, device = dev)
    x_fwd2 = proj(x2)

    # check "corner to corner" projection which should be vox size * vox value * sqrt(3)
    assert np.isclose(float(x_fwd2[2,0,-1]), img_size * vox_value * vox_size * np.sqrt(3))
    assert np.isclose(float(x_fwd2[2,2,-1]), img_size * vox_value * vox_size * np.sqrt(3))
    assert np.isclose(float(x_fwd2[2,0,-2]), img_size * vox_value * vox_size * np.sqrt(3))
    assert np.isclose(float(x_fwd2[2,2,-2]), img_size * vox_value * vox_size * np.sqrt(3))
    
    # check "central" (straight through) projection which should be vox size * vox value
    assert np.isclose(float(x_fwd2[2,1,1]), img_size * vox_value * vox_size)
    assert np.isclose(float(x_fwd2[2,3,1]), img_size * vox_value * vox_size)
    
    # check "corner to corner" projection which should be vox size * vox value * sqrt(2)
    assert np.isclose(float(x_fwd2[2,0,1]), img_size * vox_value * vox_size * np.sqrt(2))
    assert np.isclose(float(x_fwd2[2,2,1]), img_size * vox_value * vox_size * np.sqrt(2))

 