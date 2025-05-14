"""
Neg Poisson logL gradient layer (in listmode)
=============================================
"""

# %%
from __future__ import annotations
from parallelproj import Array

import array_api_compat.numpy as np
import array_api_compat.torch as torch

import parallelproj
from array_api_compat import to_device, size
import array_api_compat.numpy as np
import matplotlib.pyplot as plt
import json
from copy import copy
from pathlib import Path
from dataclasses import asdict

# using torch valid choices are 'cpu' or 'cuda'
if parallelproj.cuda_present:
    dev = "cuda"
else:
    dev = "cpu"

seed = 1
fwhm_data_mm = 4.5

# %%
# Simulation of PET data in sinogram space
# ----------------------------------------
#
# In this example, we use simulated listmode data for which we first
# need to setup a sinogram forward model to create a noise-free and noisy
# emission sinogram that can be converted to listmode data.

# %%
# Sinogram forward model setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We setup a linear forward operator :math:`A` consisting of an
# image-based resolution model, a non-TOF PET projector and an attenuation model
#

num_rings = 5
scanner = parallelproj.RegularPolygonPETScannerGeometry(
    torch,
    dev,
    radius=65.0,
    num_sides=12,
    num_lor_endpoints_per_side=15,
    lor_spacing=2.3,
    ring_positions=torch.linspace(-10, 10, num_rings),
    symmetry_axis=2,
)

# setup the LOR descriptor that defines the sinogram

img_shape = (40, 40, 8)
voxel_size = (2.0, 2.0, 2.0)

lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=10,
    max_ring_difference=2,
    sinogram_order=parallelproj.SinogramSpatialAxisOrder.RVP,
)

proj = parallelproj.RegularPolygonPETProjector(
    lor_desc, img_shape=img_shape, voxel_size=voxel_size
)

# setup a simple test image containing a few "hot rods"
x_true = torch.ones(proj.in_shape, device=dev, dtype=torch.float32)
c0 = proj.in_shape[0] // 2
c1 = proj.in_shape[1] // 2
x_true[(c0 - 2) : (c0 + 2), (c1 - 2) : (c1 + 2), :] = 5.0
x_true[4, c1, 2:] = 5.0
x_true[c0, 4, :-2] = 5.0

x_true[:2, :, :] = 0
x_true[-2:, :, :] = 0
x_true[:, :2, :] = 0
x_true[:, -2:, :] = 0


# %%
# Attenuation image and sinogram setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# setup an attenuation image
x_att = 0.01 * torch.astype(x_true > 0, torch.float32)
# calculate the attenuation sinogram
att_sino = torch.exp(-proj(x_att))

# %%
# Complete PET forward model setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We combine an image-based resolution model,
# a non-TOF or TOF PET projector and an attenuation model
# into a single linear operator.

# enable TOF - comment if you want to run non-TOF
proj.tof_parameters = parallelproj.TOFParameters(
    num_tofbins=13, tofbin_width=12.0, sigma_tof=12.0
)

# setup the attenuation multiplication operator which is different
# for TOF and non-TOF since the attenuation sinogram is always non-TOF
att_op = parallelproj.TOFNonTOFElementwiseMultiplicationOperator(
    proj.out_shape, att_sino
)

res_model = parallelproj.GaussianFilterOperator(
    proj.in_shape, sigma=fwhm_data_mm / (2.35 * proj.voxel_size)
)

# compose all 3 operators into a single linear operator
pet_lin_op = parallelproj.CompositeLinearOperator((att_op, proj, res_model))

# %%
# calculate the adjoint of the forward model applied to a ones sinogram
# needed to calculate the gradient of the negative Poisson log-likelihood (in listmode)

adjoint_ones = pet_lin_op.adjoint(torch.ones(pet_lin_op.out_shape, device=dev, dtype=torch.float32))

# %%
# Simulation of sinogram projection data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We setup an arbitrary ground truth :math:`x_{true}` and simulate
# noise-free and noisy data :math:`y` by adding Poisson noise.

# simulated noise-free data
noise_free_data = pet_lin_op(x_true)

# generate a contant contamination sinogram
contamination = torch.full(
    noise_free_data.shape,
    0.5 * float(torch.mean(noise_free_data)),
    device=dev,
    dtype=torch.float32,
)

noise_free_data += contamination

# add Poisson noise
np.random.seed(seed)
y = torch.asarray(
    np.random.poisson(parallelproj.to_numpy_array(noise_free_data)),
    device=dev,
    dtype=torch.int16,
)

# %%
# Conversion of the emission sinogram to listmode
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Using :meth:`.RegularPolygonPETProjector.convert_sinogram_to_listmode` we can convert an
# integer non-TOF or TOF sinogram to an event list for listmode processing.
#
# **Note:** The create event list is sorted and should be shuffled running LM-MLEM.

event_start_coords, event_end_coords, event_tofbins = proj.convert_sinogram_to_listmode(
    y
)

# shuffle the event list using torch
perm = torch.randperm(event_start_coords.shape[0])
event_start_coords = event_start_coords[perm]
event_end_coords = event_end_coords[perm]
event_tofbins = event_tofbins[perm]

# %%
# Setup of the LM projector and LM forward model
# ----------------------------------------------

lm_proj = parallelproj.ListmodePETProjector(
    event_start_coords,
    event_end_coords,
    proj.in_shape,
    proj.voxel_size,
    proj.img_origin,
)

# recalculate the attenuation factor for all LM events
# this needs to be a non-TOF projection
att_list = torch.exp(-lm_proj(x_att))
lm_att_op = parallelproj.ElementwiseMultiplicationOperator(att_list)

# enable TOF in the LM projector
lm_proj.tof_parameters = proj.tof_parameters
lm_proj.event_tofbins = event_tofbins
lm_proj.tof = proj.tof

# create the contamination list
contamination_list = torch.full(
    event_start_coords.shape[0],
    float(torch.reshape(contamination, (size(contamination),))[0]),
    device=dev,
    dtype=torch.float32,
)

lm_pet_lin_op = parallelproj.CompositeLinearOperator((lm_att_op, lm_proj, res_model))

# %%
# save all data to disk such that we can re-use it later (e.g. using a torch data loader)

odir = Path(f"lm_dataset_{seed:03}")
odir.mkdir(exist_ok=True)

torch.save({"event_start_coords": event_start_coords,
           "event_end_coords": event_end_coords,
           "event_tofbins": event_tofbins,
           "att_list": att_list,
           "contamination_list": contamination_list,
           "adjoint_ones": adjoint_ones,
}, odir / "input_tensors.pt")

with open(odir / "projector_parameters.json", "w") as f:
    json.dump({"tof_parameters":asdict(lm_proj.tof_parameters), "in_shape": proj.in_shape, 
    "voxel_size": voxel_size, "img_origin": tuple(proj.img_origin.tolist()), "fwhm_data_mm": fwhm_data_mm,
    "seed": seed}, f, indent=4)


## %%
## calculate what is needed for a pytorch negative Poisson logL gradient descent layer 
## note that pet_lin_op in the current form, because of attenuation, is object dependent
## the same holds for adjoint_ones (should be precomputed and saved to disk)
#
## calculate the gradient of the negative Poisson log-likelihood for a random image x
#
#batch_size = 1
#
#x = torch.rand(
#    (batch_size, 1) + lm_pet_lin_op.in_shape,
#    device=dev,
#    dtype=torch.float32,
#    requires_grad=True,
#)
#
## affine forward model evaluated at x
#z_sino = pet_lin_op(x[0,...].squeeze().detach()) + contamination
#
#sino_grad = adjoint_ones - pet_lin_op.adjoint(y/z_sino)
#
## %% 
## calculate the gradient of the negative Poisson log-likelihood using LM data and projector
#
#z_lm = lm_pet_lin_op(x[0,...].squeeze().detach()) + contamination_list
#lm_grad = adjoint_ones - lm_pet_lin_op.adjoint(1/z_lm)
#
## now sino_grad and lm_grad should be numerically very close demonstrating that
## both approaches are equivalent
## but for 3D real world low count PET data, the 2nd approach is much faster and
## more memory efficient
#
#assert torch.allclose(lm_grad,sino_grad, atol = 1e-2) # lower limit to the abs tolerance is needed
#
## to minimize computation time, we should keep z_sino / z_lm in memory (using the ctx object) after the fwd pass
## however, if memory is limited, we could also recompute it in the backward pass
#
#
## %%
## calculate the Hessian(x) applied to another random image w
## if the forwad pass computes the gradient of the negative Poisson log-likelihood
## the backward pass needs to compute the Hessian(x) applied to an image w
#
## grad_output next to ctx is usually the input passed to the backward pass
#grad_output = torch.rand(
#    (batch_size,) + lm_pet_lin_op.in_shape,
#    device=dev,
#    dtype=torch.float32,
#    requires_grad=False,
#)
#
#
#hess_app_grad_output =  pet_lin_op.adjoint(y * pet_lin_op(grad_output[0,...].squeeze()) / (z_sino**2))
#hess_app_grad_output_lm = lm_pet_lin_op.adjoint(lm_pet_lin_op(grad_output[0,...].squeeze()) / z_lm**2)
#
## again both ways of computing the Hessian to grad_output should be numerically very close
## the 2nd way should be faster and more memory efficient for real world low count PET data
#
#assert torch.allclose(hess_app_grad_output, hess_app_grad_output_lm, atol = 1e-2) # lower limit to the abs tolerance is needed
#
## the only thing that is now left is to properly wrap everything in a pytorch autograd layer
## as done in ../examples/07_torch/01_run_projection_layer.py
#
## %%
## calculate the gradient of the negative Poisson log-likelihood in listmode using a dedicated pytorch autograd layer
#
#from utils import LMPoissonLogLDescent
#lm_grad_layer = LMPoissonLogLDescent.apply
#
#lm_grad2 = lm_grad_layer(x, lm_pet_lin_op, contamination_list.unsqueeze(0), adjoint_ones.unsqueeze(0))
#
#assert torch.allclose(lm_grad, lm_grad2, atol = 1e-3) # lower limit to the abs tolerance is needed
#
#loss = 0.5*(lm_grad2*lm_grad2).sum()
#loss.backward()
#