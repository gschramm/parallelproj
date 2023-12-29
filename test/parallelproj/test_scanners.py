from __future__ import annotations

import pytest
import parallelproj
import numpy.array_api as nparr
import array_api_compat.numpy as np
import matplotlib.pyplot as plt

from types import ModuleType

# generate list of array_api modules / device combinations to test
xp_dev_list = [(np, 'cpu')]

if np.__version__ >= '1.25':
    xp_dev_list.append((nparr, 'cpu'))

if parallelproj.cupy_enabled:
    import array_api_compat.cupy as cp
    xp_dev_list.append((cp, 'cuda'))

if parallelproj.torch_enabled:
    import array_api_compat.torch as torch
    xp_dev_list.append((torch, 'cpu'))

    if parallelproj.cuda_present:
        xp_dev_list.append((torch, 'cuda'))

pytestmark = pytest.mark.parametrize("xp,dev", xp_dev_list)

#---------------------------------------------------------------------------------------

def test_regular_polygon_module(xp: ModuleType,
             dev: str) -> None:
    
    radius = 700.
    num_sides = 28
    num_lor_endpoints_per_side = 16
    lor_spacing = 4.

    aff_mat = xp.zeros((4,4), dtype = xp.float32)
    aff_mat[0,0] = 1.
    aff_mat[1,1] = 1.
    aff_mat[2,2] = 1.
    aff_mat[0,-1] = 0.
    aff_mat[1,-1] = 0.
    aff_mat[2,-1] = 10.

    ax0 = 1
    ax1 = 0

    mod = parallelproj.RegularPolygonPETScannerModule(xp, dev, radius = radius,
                 num_sides = num_sides,
                 num_lor_endpoints_per_side = num_lor_endpoints_per_side,
                 lor_spacing = lor_spacing, affine_transformation_matrix = aff_mat, ax0 = ax0, ax1 = ax1)

    assert mod.radius == radius
    assert mod.num_sides == num_sides
    assert mod.num_lor_endpoints_per_side == num_lor_endpoints_per_side
    assert mod.lor_spacing == lor_spacing
    assert mod.xp == xp
    assert mod.dev == dev

    assert mod.num_lor_endpoints == num_sides * num_lor_endpoints_per_side
    assert bool(xp.all(mod.lor_endpoint_numbers == xp.arange(mod.num_lor_endpoints)))
    assert xp.all(mod.affine_transformation_matrix == aff_mat)

    assert ax0 == mod.ax0
    assert ax1 == mod.ax1
    assert lor_spacing == mod.lor_spacing

    raw_points = mod.get_raw_lor_endpoints()
    transformed_points = mod.get_lor_endpoints()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mod.show_lor_endpoints(ax)
    fig.show()