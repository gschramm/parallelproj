from __future__ import annotations

# parallelproj supports the numpy, cupy and pytorch array API
# choose your preferred array API uncommenting the corresponding line
import array_api_compat.numpy as xp
#import array_api_compat.cupy as xp
#import array_api_compat.torch as xp

import parallelproj
from array_api_compat import to_device

# choose our device depending on the array API and the availability of CUDA
if 'numpy' in xp.__name__:
    dev = 'cpu'
elif 'cupy' in xp.__name__:
    dev = xp.cuda.Device(0)
elif 'torch' in xp.__name__:
    # using torch valid choises are cpu or cuda
    dev = 'cuda'

print(f'running on {dev} device using {xp.__name__}')

image_shape = (128, 128)
voxel_size = (2., 2.)
image_origin = (-127., -127.)

radial_positions = to_device(xp.linspace(-128, 128, 200), dev)
view_angles = to_device(xp.linspace(0, xp.pi, 180, endpoint=False), dev)
radius = 200.

proj2d = parallelproj.ParallelViewProjector2D(image_shape, radial_positions,
                                              view_angles, radius,
                                              image_origin, voxel_size)

img = to_device(xp.zeros(proj2d.in_shape, dtype=xp.float32), dev)
img[32:96, 32:64] = 1.

img_fwd = proj2d(img)
img_fwd_back = proj2d.adjoint(img_fwd)

fig = proj2d.show_views(image=img)
fig.show()