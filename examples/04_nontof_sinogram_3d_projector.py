from __future__ import annotations

# parallelproj supports the numpy, cupy and pytorch array API
# choose your preferred array API uncommenting the corresponding line
import array_api_compat.numpy as xp
#import array_api_compat.cupy as xp
#import array_api_compat.torch as xp

import parallelproj
import matplotlib.pyplot as plt
from array_api_compat import to_device

# choose our device depending on the array API and the availability of CUDA
if 'numpy' in xp.__name__:
    dev = 'cpu'
elif 'cupy' in xp.__name__:
    dev = xp.cuda.Device(0)
elif 'torch' in xp.__name__:
    # using torch valid choises are cpu or cuda
    dev = 'cpu'

print(f'running on {dev} device using {xp.__name__}')

# shape of the 2D image
image_shape = (128, 128, 8)
# voxel size
voxel_size = (2., 2., 4.)
# world coordinates of the [0, 0] pixel
image_origin = (-127., -127., -14.)

# radial positions of the projection lines
radial_positions = to_device(xp.linspace(-128, 128, 200), dev)
# projection angles
view_angles = to_device(xp.linspace(0, xp.pi, 180, endpoint=False), dev)
# distance between the image center and the start / end of the center line
radius = 200.
# axial coordinates of the projection "rings"
ring_positions = to_device(xp.linspace(-14, 14, 8), dev)

proj3d = parallelproj.ParallelViewProjector3D(image_shape,
                                              radial_positions,
                                              view_angles,
                                              radius,
                                              image_origin,
                                              voxel_size,
                                              ring_positions,
                                              max_ring_diff=5)

img = to_device(xp.zeros(proj3d.in_shape, dtype=xp.float32), dev)
img[32:96, 32:64, 1:-1] = 1.
img[48:64, 48:54, 3:-3] = 2.

img_fwd = proj3d(img)
img_fwd_back = proj3d.adjoint(img_fwd)

#----------------------------------------------------------------------------
# show the geometry of the projector and the projections

fig, ax = plt.subplots(3, 8, figsize=(16, 6))
for i in range(8):
    ax[0, i].imshow(to_device(img[..., i], 'cpu'), vmin=0, vmax=xp.max(img))
    ax[1, i].imshow(to_device(img_fwd[..., i], 'cpu'),
                    vmin=0,
                    vmax=xp.max(img_fwd))
    ax[2, i].imshow(to_device(img_fwd_back[..., i], 'cpu'),
                    vmin=0,
                    vmax=xp.max(img_fwd_back))
    ax[0, i].set_title(f'x slice {i}')
    ax[1, i].set_title(f'(A x) plane {i}')
    ax[2, i].set_title(f'(A^H A x) slice {i}')
for axx in ax.ravel():
    axx.set_axis_off()
fig.tight_layout()
fig.show()