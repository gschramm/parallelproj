"""
TOF listmode SPDHG with projection data
=======================================

This example demonstrates the use of the listmode SPDHG algorithm to minimize the negative Poisson log-likelihood function.

.. math::
    f(x) = \sum_{i=1}^m \\bar{y}_i - \\bar{y}_i (x) \log(y_i)

subject to

.. math::
    x \geq 0

using the listmode linear forward model

.. math::
    \\bar{y}_{LM}(x) = A_{LM} x + s

and data stored in listmode format (event by event).

.. tip::
    parallelproj is python array API compatible meaning it supports different 
    array backends (e.g. numpy, cupy, torch, ...) and devices (CPU or GPU).
    Choose your preferred array API ``xp`` and device ``dev`` below.

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/gschramm/parallelproj/master?labpath=examples
"""

# %%
from __future__ import annotations
from array_api_strict._array_object import Array

# Running this example using GPU arrays is highly recommended
# due to "long" execution times with CPU arrays

# import array_api_compat.numpy as xp

import array_api_compat.cupy as xp

# import array_api_compat.torch as xp

import parallelproj
from array_api_compat import to_device, size
import array_api_compat.numpy as np
import matplotlib.pyplot as plt

# choose a device (CPU or CUDA GPU)
if "numpy" in xp.__name__:
    # using numpy, device must be cpu
    dev = "cpu"
elif "cupy" in xp.__name__:
    # using cupy, only cuda devices are possible
    dev = xp.cuda.Device(0)
elif "torch" in xp.__name__:
    # using torch valid choices are 'cpu' or 'cuda'
    if parallelproj.cuda_present:
        dev = "cuda"
    else:
        dev = "cpu"

num_subsets = 100
num_iter_mlem = 2000
num_iter_lmspdhg = 25

# %%
# Simulation of PET data in sinogram space
# ----------------------------------------
#
# In this example, we use simulated listmode data for which we first
# need to setup a sinogram forward model to create a noise-free and noisy
# emission sinogram that can be converted to listmode data.

# %%
# Setup of the sinogram forward model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We setup a linear forward operator :math:`A` consisting of an
# image-based resolution model, a non-TOF PET projector and an attenuation model
#

num_rings = 5
scanner = parallelproj.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=65.0,
    num_sides=12,
    num_lor_endpoints_per_side=15,
    lor_spacing=2.3,
    ring_positions=xp.linspace(-10, 10, num_rings),
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
x_true = xp.ones(proj.in_shape, device=dev, dtype=xp.float32)
c0 = proj.in_shape[0] // 2
c1 = proj.in_shape[1] // 2
x_true[(c0 - 2) : (c0 + 2), (c1 - 2) : (c1 + 2), :] = 5.0
x_true[4, c1, 2:] = 5.0
x_true[c0, 4, :-2] = 5.0

tmp_n = proj.in_shape[0] // 4
x_true[:tmp_n, :, :] = 0
x_true[-tmp_n:, :, :] = 0
x_true[:, :2, :] = 0
x_true[:, -2:, :] = 0

# scale image to get more counts
x_true *= 1.0

# %%
# Attenuation image and sinogram setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# setup an attenuation image
x_att = 0.01 * xp.astype(x_true > 0, xp.float32)
# calculate the attenuation sinogram
att_sino = xp.exp(-proj(x_att))

# %%
# Complete sinogram PET forward model setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
if proj.tof:
    att_op = parallelproj.TOFNonTOFElementwiseMultiplicationOperator(
        proj.out_shape, att_sino
    )
else:
    att_op = parallelproj.ElementwiseMultiplicationOperator(att_sino)

res_model = parallelproj.GaussianFilterOperator(
    proj.in_shape, sigma=4.5 / (2.35 * proj.voxel_size)
)

# compose all 3 operators into a single linear operator
pet_lin_op = parallelproj.CompositeLinearOperator((att_op, proj, res_model))

# %%
# Simulation of sinogram projection data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We setup an arbitrary ground truth :math:`x_{true}` and simulate
# noise-free and noisy data :math:`y` by adding Poisson noise.

# simulated noise-free data
noise_free_data = pet_lin_op(x_true)

# generate a contant contamination sinogram
contamination = xp.full(
    noise_free_data.shape,
    0.5 * float(xp.mean(noise_free_data)),
    device=dev,
    dtype=xp.float32,
)

noise_free_data += contamination

# add Poisson noise
np.random.seed(1)
y = xp.asarray(
    np.random.poisson(np.asarray(to_device(noise_free_data, "cpu"))),
    device=dev,
    dtype=xp.int16,
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

# %%
# shuffle the events
random_inds = np.random.permutation(event_start_coords.shape[0])
event_start_coords = event_start_coords[random_inds, :]
event_end_coords = event_end_coords[random_inds, :]
event_tofbins = event_tofbins[random_inds]

# %%
# Setup of the LM subset projectors and LM subset forward models
# --------------------------------------------------------------

subset_slices_lm = [slice(i, None, num_subsets) for i in range(num_subsets)]

lm_pet_subset_linop_seq = []

for i, sl in enumerate(subset_slices_lm):
    subset_lm_proj = parallelproj.ListmodePETProjector(
        event_start_coords[sl, :],
        event_end_coords[sl, :],
        proj.in_shape,
        proj.voxel_size,
        proj.img_origin,
    )

    # recalculate the attenuation factor for all LM events
    # this needs to be a non-TOF projection
    subset_att_list = xp.exp(-subset_lm_proj(x_att))

    # enable TOF in the LM projector
    subset_lm_proj.tof_parameters = proj.tof_parameters
    if proj.tof:
        # we need to make a copy of the 1D subset event_tofbins array
        # stupid way of doing this, but torch asarray copy doesn't seem to work
        subset_lm_proj.event_tofbins = 1 * event_tofbins[sl]
        subset_lm_proj.tof = proj.tof

    subset_lm_att_op = parallelproj.ElementwiseMultiplicationOperator(subset_att_list)

    lm_pet_subset_linop_seq.append(
        parallelproj.CompositeLinearOperator(
            (subset_lm_att_op, subset_lm_proj, res_model)
        )
    )

lm_pet_subset_linop_seq = parallelproj.LinearOperatorSequence(lm_pet_subset_linop_seq)

# create the contamination list
contamination_list = xp.full(
    event_start_coords.shape[0],
    float(xp.reshape(contamination, (size(contamination),))[0]),
    device=dev,
    dtype=xp.float32,
)

# %%
# RUN MLEM as reference
# ---------------------


def em_update(
    x_cur: Array,
    data: Array,
    op: parallelproj.LinearOperator,
    s: Array,
    adjoint_ones: Array,
) -> Array:
    """EM update

    Parameters
    ----------
    x_cur : Array
        current solution
    data : Array
        data
    op : parallelproj.LinearOperator
        linear forward operator
    s : Array
        contamination
    adjoint_ones : Array
        adjoint of ones

    Returns
    -------
    Array
        _description_
    """
    ybar = op(x_cur) + s
    return x_cur * op.adjoint(data / ybar) / adjoint_ones


# %%
# Run the MLEM iterations
# -----------------------


# initialize x
x = xp.ones(pet_lin_op.in_shape, dtype=xp.float32, device=dev)
# calculate A^H 1
adjoint_ones = pet_lin_op.adjoint(
    xp.ones(pet_lin_op.out_shape, dtype=xp.float32, device=dev)
)

for i in range(num_iter_mlem):
    print(f"MLEM iteration {(i + 1):03} / {num_iter_mlem:03}", end="\r")
    x = em_update(x, y, pet_lin_op, contamination, adjoint_ones)

x_mlem = 1.0 * x

# %%
# LM OSEM reconstruction
# ----------------------
#
# The EM update that can be used in LM-OSEM is given by
#
# .. math::
#     x^+ = \frac{x}{(A^k)^H 1} (A_{LM}^k)^H \frac{1}{A_{LM}^k x + s_{LM}^k}
#
# to calculate the minimizer of :math:`f(x)` iteratively.


def lm_em_update(
    x_cur: Array,
    op: parallelproj.LinearOperator,
    s: Array,
    adjoint_ones: Array,
) -> Array:
    """LM EM update

    Parameters
    ----------
    x_cur : Array
        current solution
    op : parallelproj.LinearOperator
        subset listmode linear forward operator
    s : Array
        subset contamination list
    adjoint_ones : Array
        adjoint of ones of the non-LM (the complete) operator
        divided by the number of subsets

    Returns
    -------
    Array
        _description_
    """
    ybar = op(x_cur) + s
    return x_cur * op.adjoint(1 / ybar) / adjoint_ones


# %%
# Run a quick LM OSEM to initialize LM-SPDHG
# ------------------------------------------

# initialize x
x_lmosem = xp.ones(pet_lin_op.in_shape, dtype=xp.float32, device=dev)
# calculate A^H 1
adjoint_ones = pet_lin_op.adjoint(
    xp.ones(pet_lin_op.out_shape, dtype=xp.float32, device=dev)
)

for i in range(1):
    for k, sl in enumerate(subset_slices_lm):
        print(f"LM OSEM iteration {(k+1):03} / {(i + 1):03} / 001", end="\r")
        x_lmosem = lm_em_update(
            x_lmosem,
            lm_pet_subset_linop_seq[k],
            contamination_list[sl],
            adjoint_ones / num_subsets,
        )

# %%
# Listmode SPDHG
# --------------
#
# .. admonition:: Listmode SPDHG algorithm to minimize negative Poisson log-likelihood
#
#   | **Input** event list :math:`N`, contamination list :math:`s_N`
#   | **Calculate** event counts :math:`\mu_e` for each :math:`e` in :math:`N`
#   | **Initialize** :math:`x,(S_i)_i,T,(p_i)_i`
#   | **Initialize list** :math:`y_{N} = 1 - (\mu_N /(A^{LM}_{N} x + s_{N}))`
#   | **Preprocessing** :math:`\overline{z} = z = {A^T} 1 - {A^{LM}}^T (y_N-1)/\mu_N`
#   | **Split lists** :math:`N`, :math:`s_N` and :math:`y_N` into :math:`n` sublists :math:`N_i`, :math:`y_{N_i}` and :math:`s_{N_i}`
#   | **Repeat**, until stopping criterion fulfilled
#   |     **Update** :math:`x \gets \text{proj}_{\geq 0} \left( x - T \overline{z} \right)`
#   |     **Select** :math:`i \in \{ 1,\ldots,n+1\}` randomly according to :math:`(p_i)_i`
#   |     **Update** :math:`y_{N_i}^+ \gets \text{prox}_{D^*}^{S_i} \left( y_{N_i} + S_i \left(A^{LM}_{N_i} x + s^{LM}_{N_i} \right) \right)`
#   |     **Update** :math:`\delta z \gets {A^{LM}_{N_i}}^T \left(\frac{y_{N_i}^+ - y_{N_i}}{\mu_{N_i}}\right)`
#   |     **Update** :math:`y_{N_i} \gets y_{N_i}^+`
#   |     **Update** :math:`z \gets z + \delta z`
#   |     **Update** :math:`\bar{z} \gets z + (\delta z/p_i)`
#   | **Return** :math:`x`
#
# .. admonition:: Proximal operator of the convex dual of the negative Poisson log-likelihood
#
#  :math:`(\text{prox}_{D_i^*}^{S_i}(y))_i = \text{prox}_{D_i^*}^{S_i}(y_i) = \frac{1}{2} \left(y_i + 1 - \sqrt{ (y_i-1)^2 + 4 S_i d_i} \right)`
#

# %%
# Calculate event multiplicity :math:`\mu` for each event in the list
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
events = xp.concat(
    [event_start_coords, event_end_coords, xp.expand_dims(event_tofbins, -1)], axis=1
)
mu = parallelproj.count_event_multiplicity(events)

# %%
# Initialize variables
# ^^^^^^^^^^^^^^^^^^^^

# Intialize image x with solution from quick LM OSEM
x = 1.0 * x_lmosem

# setup dual variable for data subsets
ys = []
for k, sl in enumerate(subset_slices_lm):
    ys.append(1 - (mu[sl] / (lm_pet_subset_linop_seq[k](x) + contamination_list[sl])))

z = 1.0 * adjoint_ones
for k, sl in enumerate(subset_slices_lm):
    z += lm_pet_subset_linop_seq[k].adjoint((ys[k] - 1) / mu[sl])
zbar = 1.0 * z

# %%
# Calculate the step sizes
# ^^^^^^^^^^^^^^^^^^^^^^^^

gamma = 10.0 / xp.max(x_true)
rho = 0.999

p_g = 0.0
p_p = (1 - p_g) / num_subsets

S = []
ones_img = xp.ones(img_shape, dtype=xp.float32, device=dev)

for lm_op in lm_pet_subset_linop_seq:
    tmp = lm_op(ones_img)
    tmp = xp.where(tmp == 0, xp.min(tmp[tmp > 0]), tmp)
    S.append(gamma * rho / tmp)

T = (rho * p_p / gamma) / (adjoint_ones / num_subsets)

# %%
# LM SPDHG iterations
# ^^^^^^^^^^^^^^^^^^^

for it in range(num_iter_lmspdhg):
    subset_sequence = np.random.permutation(num_subsets)

    for k in subset_sequence:
        sl = subset_slices_lm[k]
        x -= T * zbar
        x = xp.where(x < 0, xp.zeros_like(x), x)

        y_plus = ys[k] + S[k] * (lm_pet_subset_linop_seq[k](x) + contamination_list[sl])
        y_plus = 0.5 * (y_plus + 1 - xp.sqrt((y_plus - 1) ** 2 + 4 * S[k] * mu[sl]))

        dz = lm_pet_subset_linop_seq[k].adjoint((y_plus - ys[k]) / mu[sl])

        ys[k] = y_plus
        z = z + dz
        zbar = z + dz / p_p

        print(
            f"LM SPDHG iteration {(k+1):03} / {(it + 1):03} / {num_iter_lmspdhg:03}",
            end="\r",
        )

# %%
# Calculate the final cost
# ^^^^^^^^^^^^^^^^^^^^^^^^

exp = pet_lin_op(x) + contamination
cost = float(xp.sum(exp - y * xp.log(exp)))
print("")
print(f"\nLM SPDHG cost {cost:.8E} after {num_iter_lmspdhg:03} iterations")

exp_mlem = pet_lin_op(x_mlem) + contamination
cost_mlem = float(xp.sum(exp_mlem - y * xp.log(exp_mlem)))
print("")
print(f"\nMLEM cost {cost_mlem:.8E} after {num_iter_mlem:03} iterations")


# %%
# Show the results
# ^^^^^^^^^^^^^^^^

x_true_np = np.asarray(to_device(x_true, "cpu"))
x_np = np.asarray(to_device(x, "cpu"))
x_mlem_np = np.asarray(to_device(x_mlem, "cpu"))

pl = x_np.shape[2] // 2

fig, ax = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)
vmax = 1.2 * x_true_np.max()
ax[0].imshow(x_true_np[:, :, pl], cmap="Greys", vmin=0, vmax=vmax)
ax[1].imshow(x_mlem_np[:, :, pl], cmap="Greys", vmin=0, vmax=vmax)
ax[2].imshow(x_np[:, :, pl], cmap="Greys", vmin=0, vmax=vmax)
fig.show()
