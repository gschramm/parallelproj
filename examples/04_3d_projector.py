from __future__ import annotations

from array_api_compat import to_device
import parallelproj

# parallelproj supports the numpy, cupy and pytorch array API
# choose your preferred array API uncommenting the corresponding line
import array_api_compat.numpy as xp
#import array_api_compat.cupy as xp
#import array_api_compat.torch as xp

# choose our device depending on the array API and the availability of CUDA
if 'numpy' in xp.__name__:
    device = 'cpu'
elif 'cupy' in xp.__name__:
    device = xp.cuda.Device(0)
elif 'torch' in xp.__name__:
    # using torch valid choises are cpu or cuda
    device = 'cpu'

print(f'running on {device} device using {xp.__name__}')

image_shape = (128, 128, 8)
voxel_size = (2., 2., 4.)
image_origin = (-127., -127., -14.)

radial_positions = to_device(xp.linspace(-128, 128, 200), device)
view_angles = to_device(xp.linspace(0, xp.pi, 180, endpoint=False), device)
radius = 200.
ring_positions = to_device(xp.linspace(-14, 14, 8), device)

proj3d = parallelproj.ParallelViewProjector3D(image_shape,
                                              radial_positions,
                                              view_angles,
                                              radius,
                                              image_origin,
                                              voxel_size,
                                              ring_positions,
                                              max_ring_diff=5)

img = to_device(xp.zeros(proj3d.in_shape, dtype=xp.float32), device)
img[32:96, 32:64, 1:-1] = 1.
img[48:64, 48:54, 3:-3] = 2.

img_fwd = proj3d(img)
img_fwd_back = proj3d.adjoint(img_fwd)