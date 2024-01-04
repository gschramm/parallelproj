"""
PET non-TOF sinogram projector
==============================

In this example we will show how to setup and use PET sinogram projector
consisting of the geometrical forward projection, resolution model and 
correction for attenuation.
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
# setup a small regular polygon PET scanner with 5 rings (polygons)

num_rings = 5
scanner = parallelproj.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=65.,
    num_sides=12,
    num_lor_endpoints_per_side=15,
    lor_spacing=2.3,
    ring_positions=xp.linspace(-10, 10, num_rings),
    symmetry_axis=1)

# %%
# setup the LOR descriptor that defines the sinogram

lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=10,
    max_ring_difference=2,
    sinogram_order=parallelproj.SinogramSpatialAxisOrder.RVP)

# %%
# Defining a non-TOF projector
# ----------------------------
#
# :class:`.RegularPolygonPETProjector` can be used to define a non-TOF projector
# that combines the scanner, LOR and image geometry. The letter is defined by
# the image shape and the voxel size. 

# define a first projector using an image with 40x8x40 voxels of size 2x2x2 mm
# where the image center is at world coordinate (0, 0, 0)
proj = parallelproj.RegularPolygonPETProjector(lor_desc,
                                               img_shape=(40, 8, 40),
                                               voxel_size=(2., 2., 2.))

# define a second projector using an image with 20x8x30 voxels of size 3x2x2 mm
# that is off-center
proj2 = parallelproj.RegularPolygonPETProjector(lor_desc,
                                                img_shape=(20, 8, 30),
                                                voxel_size=(3., 2., 2.),
                                                img_origin=(-19, -7, -19))

# %%
# Visualize the scanner and image geometry
# ----------------------------------------
#
# :meth:`.RegularPolygonPETProjector.show_geometry` can be used
# to visualize the scanner and image geometry

fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
proj.show_geometry(ax1)
proj2.show_geometry(ax2, color=(0, 0, 1))
fig.tight_layout()
fig.show()

# %% 
# Simple geometrical forward projections
# --------------------------------------
#
# :meth:`.RegularPolygonPETProjector.__call__` allows us to calculate
# the geometrical forward projection (line integrals by Joseph's method)
# though a voxelize image.

# setup a simple test image containing a few "point rods"

x = xp.zeros(proj.in_shape, device = dev, dtype = xp.float32)
x[proj.in_shape[0]//2, :, proj.in_shape[2]//2] = 2.
x[0, :, proj.in_shape[2]//2] = 1.5
x[proj.in_shape[0]//2, :, 0] = 1.

x_fwd = proj(x)

fig, ax = plt.subplots(4,5, figsize = (2*5, 2*4))
vmax = float(xp.max(x_fwd))
for i in range(20):
    axx = ax.ravel()[i]
    if i < proj.lor_descriptor.num_planes:
        axx.imshow(x_fwd[:,:,i].T, cmap='Greys', vmin = 0, vmax = vmax)
        axx.set_title(f'sino plane {i}', fontsize = 'medium')
    else:
     axx.set_axis_off()
fig.tight_layout()
fig.show()


# %%
# Adding an image-based resolution model
# --------------------------------------

# setup a simple image-based resolution model with an Gaussian FWHM of 4.5mm
res_model = parallelproj.GaussianFilterOperator(proj.in_shape, sigma = 4.5 / (2.35*proj.voxel_size))

proj_with_res_model = parallelproj.CompositeLinearOperator((proj, res_model))

x_fwd2 = proj_with_res_model(x)

fig, ax = plt.subplots(4,5, figsize = (2*5, 2*4))
vmax = float(xp.max(x_fwd2))
for i in range(20):
    axx = ax.ravel()[i]
    if i < proj.lor_descriptor.num_planes:
        axx.imshow(x_fwd2[:,:,i].T, cmap='Greys', vmin = 0, vmax = vmax)
        axx.set_title(f'sino plane {i}', fontsize = 'medium')
    else:
     axx.set_axis_off()
fig.tight_layout()
fig.show()


# %%
# Adding the effect of attenuation
# --------------------------------

# setup an attenuation image containing the attenuation coeff. of water (in 1/mm)
x_att = xp.full(proj.in_shape, 0.01, device = dev, dtype = xp.float32)

# forward project the attenuation image
x_att_fwd = proj(x_att)

# calculate the attenuation sinogram
att_sino = xp.exp(-x_att_fwd)
att_op = parallelproj.ElementwiseMultiplicationOperator(att_sino)

# setup a forward projector containing the attenuation and resolution
proj_with_att_and_res_model = parallelproj.CompositeLinearOperator((att_op, proj, res_model))

x_fwd3 = proj_with_att_and_res_model(x)

fig, ax = plt.subplots(4,5, figsize = (2*5, 2*4))
vmax = float(xp.max(x_fwd3))
for i in range(20):
    axx = ax.ravel()[i]
    if i < proj.lor_descriptor.num_planes:
        axx.imshow(x_fwd3[:,:,i].T, cmap='Greys', vmin = 0, vmax = vmax)
        axx.set_title(f'sino plane {i}', fontsize = 'medium')
    else:
     axx.set_axis_off()
fig.tight_layout()
fig.show()


