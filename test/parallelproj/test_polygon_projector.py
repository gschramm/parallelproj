from __future__ import annotations

import parallelproj
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
    x = xp.zeros(img_shape, dtype = xp.float32, device = dev)
    x[img_shape[0] // 2, img_shape[1] // 2, 1:] = 1.0
    x[-3, img_shape[1] // 2, :-1] = 1.0
    x[img_shape[0] // 2, -3, 1:] = 1.0
    
    # define the scanner geometry, lor descriptor and projector
    scanner = parallelproj.DemoPETScannerGeometry(xp,
                             dev,
                             num_rings = num_rings,
                             num_sides = num_sides,
                             radius = radius,
                             symmetry_axis=symmetry_axis)
    
    
    lor_desc = parallelproj.RegularPolygonPETLORDescriptor(scanner,
                     radial_trim=radial_trim,
                     max_ring_difference=max_ring_difference,
                     sinogram_order = sinogram_order)
    
    proj = parallelproj.RegularPolygonPETProjector(lor_desc, img_shape, voxel_size)
    
    # non-TOF projections
    x_fwd = proj(x)
    y = xp.ones(x_fwd.shape, dtype = xp.float32, device = dev)
    y_back = proj.adjoint(y)
    
    # TOF projections
    tof_params = parallelproj.TOFParameters(num_tofbins=7, tofbin_width=30.6)
    proj.tof_parameters = tof_params
    
    x_fwd_tof = proj(x)
    y_tof = xp.ones(x_fwd_tof.shape, dtype = xp.float32, device = dev)
    y_back_tof = proj.adjoint(y_tof)
    