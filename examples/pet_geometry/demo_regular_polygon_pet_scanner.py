"""
Regular polygon PET scanner
---------------------------

This example shows how to create and visualize PET scanner where the LOR
endpoints can be modeled as a stack of regular polygons.
"""

# %%
# parallelproj supports the numpy, cupy and pytorch array API and different devices
# choose your preferred array API uncommenting the corresponding line

import array_api_compat.numpy as xp
#import array_api_compat.cupy as xp
#import array_api_compat.torch as xp

# %%
import parallelproj
from array_api_compat import to_device, device
import matplotlib.pyplot as plt

# choose a device (CPU or CUDA GPU)
if 'numpy' in xp.__name__:
    # using numpy, device must be cpu
    dev = 'cpu'
elif 'cupy' in xp.__name__:
    # using cupy, only cuda devices are possible
    dev = xp.cuda.Device(0)
elif 'torch' in xp.__name__:
    # using torch valid choices are 'cpu' or 'cuda'
    dev = 'cuda'


# %%
scanner1 = parallelproj.RegularPolygonPETScannerGeometry(xp,
                                              dev,
                                              radius = 65.,
                                              num_sides = 12,
                                              num_lor_endpoints_per_side = 8,
                                              lor_spacing = 4.,
                                              ring_positions = xp.linspace(-4,4,3),
                                              symmetry_axis=2)

scanner2 = parallelproj.RegularPolygonPETScannerGeometry(xp,
                                              dev,
                                              radius = 65.,
                                              num_sides = 12,
                                              num_lor_endpoints_per_side = 8,
                                              lor_spacing = 4.2,
                                              ring_positions = xp.linspace(-4,4,3),
                                              symmetry_axis=1)

scanner3 = parallelproj.RegularPolygonPETScannerGeometry(xp,
                                              dev,
                                              radius = 400.,
                                              num_sides = 32,
                                              num_lor_endpoints_per_side = 16,
                                              lor_spacing = 4.3,
                                              ring_positions = xp.linspace(-70,70,36),
                                              symmetry_axis=2)

scanner4 = parallelproj.RegularPolygonPETScannerGeometry(xp,
                                              dev,
                                              radius = 400.,
                                              num_sides = 32,
                                              num_lor_endpoints_per_side = 16,
                                              lor_spacing = 4.3,
                                              ring_positions = xp.linspace(-70,70,36),
                                              symmetry_axis=0)


# %%
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222, projection='3d')
ax3 = fig.add_subplot(223, projection='3d')
ax4 = fig.add_subplot(224, projection='3d')
scanner1.show_lor_endpoints(ax1)
scanner2.show_lor_endpoints(ax2)
scanner3.show_lor_endpoints(ax3)
scanner4.show_lor_endpoints(ax4)
fig.tight_layout()
fig.show()
