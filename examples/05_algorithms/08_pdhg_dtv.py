"""
PDHG to optimize the Poisson logL and directional TV (structural prior)
=======================================================================

This example demonstrates the use of the primal dual hybrid gradient (PDHG) algorithm, 
to minimize the negative  Poisson log-likelihood function combined with a 
directional total variation regularizer (a structural prior):

.. math::
    f(x) = \\sum_{i=1}^m \\bar{d}_i (x) - d_i \\log(\\bar{d}_i (x)) + \\beta \\|P_\\{\\xi} \\nabla x \\|_{1,2}

subject to

.. math::
    x \\geq 0
    
using the linear forward model

.. math::
    \\bar{d}(x) = A x + s

See Ehrhardt and Betcke:
"Multi-Contrast MRI Reconstruction with Structure-Guided Total Variation"
SIAM, https://doi.org/10.1137/15M1047325
    
.. tip::
    parallelproj is python array API compatible meaning it supports different 
    array backends (e.g. numpy, cupy, torch, ...) and devices (CPU or GPU).
    Choose your preferred array API ``xp`` and device ``dev`` below.

.. warning::
    Running this example using GPU arrays (e.g. using cupy as array backend) 
    is highly recommended due to "longer" execution times with CPU arrays

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/gschramm/parallelproj/master?labpath=examples
"""

# %%
from __future__ import annotations

import array_api_compat.numpy as xp

# import array_api_compat.cupy as xp
# import array_api_compat.torch as xp

import parallelproj
from array_api_compat import to_device
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

# %%
# **Input Parameters**

# image scale (can be used to simulated more or less counts)
img_scale = 0.1
# number of MLEM iterations to init. PDHG and LM-SPDHG
num_iter_mlem = 10
# number of PDHG iterations
num_iter_pdhg = 1000
# prior weight
beta = 6.0
# step size ratio for PDHG
gamma = 1.0 / img_scale
# rho value for PDHG
rho = 0.9999
# contaminaton in every sinogram bin relative to mean of trues sinogram
contam = 1.0


track_cost = True

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

num_rings = 2
scanner = parallelproj.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=350.0,
    num_sides=28,
    num_lor_endpoints_per_side=16,
    lor_spacing=4.0,
    ring_positions=xp.linspace(-2.5, 2.5, num_rings),
    symmetry_axis=2,
)

# setup the LOR descriptor that defines the sinogram

img_shape = (40, 40, 4)
voxel_size = (4.0, 4.0, 2.5)

lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=170,
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

# setup a structural prior image
x_struct = -1.0 * xp.sqrt(x_true)
x_struct[(c0) : (c0 + 2), (c1) : (c1 + 2), :] = -1.0

# scale image to get more counts
x_true *= img_scale


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

# enable TOF - uncomment if you want to run TOF recons
# proj.tof_parameters = parallelproj.TOFParameters(
#    num_tofbins=17, tofbin_width=12.0, sigma_tof=12.0
# )

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
    contam * float(xp.mean(noise_free_data)),
    device=dev,
    dtype=xp.float32,
)

noise_free_data += contamination

# add Poisson noise
np.random.seed(1)
d = xp.asarray(
    np.random.poisson(np.asarray(to_device(noise_free_data, "cpu"))),
    device=dev,
    dtype=xp.int16,
)

# %%
# Run quick MLEM as initialization
# --------------------------------

x_mlem = xp.ones(pet_lin_op.in_shape, dtype=xp.float32, device=dev)
# calculate A^H 1
adjoint_ones = pet_lin_op.adjoint(
    xp.ones(pet_lin_op.out_shape, dtype=xp.float32, device=dev)
)

for i in range(num_iter_mlem):
    print(f"MLEM iteration {(i + 1):03} / {num_iter_mlem:03}", end="\r")
    dbar = pet_lin_op(x_mlem) + contamination
    x_mlem *= pet_lin_op.adjoint(d / dbar) / adjoint_ones

# %%
# Setup the cost function
# ^^^^^^^^^^^^^^^^^^^^^^^


def cost_function(img):
    exp = pet_lin_op(img) + contamination
    res = float(xp.sum(exp - d * xp.log(exp)))
    res += beta * float(xp.sum(xp.linalg.vector_norm(op_G(img), axis=0)))
    return res


# %%
# PDHG
# ----
#
# .. admonition:: PDHG algorithm to minimize negative Poisson log-likelihood + regularization
#
#   | **Input** Poisson data :math:`d`
#   | **Initialize** :math:`x,y,w,S_A,S_G,T`
#   | **Preprocessing** :math:`\overline{z} = z = A^T y + \nabla^T w`
#   | **Repeat**, until stopping criterion fulfilled
#   |     **Update** :math:`x \gets \text{proj}_{\geq 0} \left( x - T \overline{z} \right)`
#   |     **Update** :math:`y^+ \gets \text{prox}_{D^*}^{S_A} ( y + S_A  ( A x + s))`
#   |     **Update** :math:`w^+ \gets \beta \, \text{prox}_{R^*}^{S_G/\beta} ((w + S_G  \nabla x)/\beta)`
#   |     **Update** :math:`\Delta z \gets A^T (y^+ - y) + \nabla^T (w^+ - w)`
#   |     **Update** :math:`z \gets z + \Delta z`
#   |     **Update** :math:`\bar{z} \gets z + \Delta z`
#   |     **Update** :math:`y \gets y^+`
#   |     **Update** :math:`w \gets w^+`
#   | **Return** :math:`x`
#
# See :cite:p:`Ehrhardt2019` :cite:p:`Schramm2022` for more details.
#
# .. admonition:: Proximal operator of the convex dual of the negative Poisson log-likelihood
#
#  :math:`(\text{prox}_{D^*}^{S}(y))_i = \text{prox}_{D^*}^{S}(y_i) = \frac{1}{2} \left(y_i + 1 - \sqrt{ (y_i-1)^2 + 4 S d_i} \right)`
#
# .. admonition:: Step sizes
#
#  :math:`S_A = \gamma \, \text{diag}(\frac{\rho}{A 1})`
#
#  :math:`S_G = \gamma \, \text{diag}(\frac{\rho}{|\nabla|})`
#
#  :math:`T_A = \gamma^{-1} \text{diag}(\frac{\rho}{A^T 1})`
#
#  :math:`T_G = \gamma^{-1} \text{diag}(\frac{\rho}{|\nabla|})`
#
#  :math:`T = \min T_A, T_G` pointwise
#

# setup the "normal" gradient operator
G = parallelproj.FiniteForwardDifference(pet_lin_op.in_shape)
# calculate the joint vector field based on the structural prior image
joint_vector_field = G(x_struct)
# setup the projected gradient operator
P = parallelproj.GradientFieldProjectionOperator(joint_vector_field, eta=1e-4)
op_G = parallelproj.CompositeLinearOperator((P, G))

# initialize primal and dual variables
x_pdhg = 1.0 * x_mlem
y = 1 - d / (pet_lin_op(x_pdhg) + contamination)

# initialize dual variable for the gradient
w = xp.zeros(op_G.out_shape, dtype=xp.float32, device=dev)

z = pet_lin_op.adjoint(y) + op_G.adjoint(w)
zbar = 1.0 * z

# %%

# calculate PHDG step sizes
tmp = pet_lin_op(xp.ones(pet_lin_op.in_shape, dtype=xp.float32, device=dev))
tmp = xp.where(tmp == 0, xp.min(tmp[tmp > 0]), tmp)
S_A = gamma * rho / tmp

T_A = (
    (1 / gamma)
    * rho
    / pet_lin_op.adjoint(xp.ones(pet_lin_op.out_shape, dtype=xp.float64, device=dev))
)

op_G_norm = op_G.norm(xp, dev, num_iter=100)
S_G = gamma * rho / op_G_norm
T_G = (1 / gamma) * rho / op_G_norm

T = xp.where(T_A < T_G, T_A, xp.full(pet_lin_op.in_shape, T_G))


# %%
# Run PDHG
# ^^^^^^^^

print("")
cost_pdhg = np.zeros(num_iter_pdhg, dtype=xp.float32)

for i in range(num_iter_pdhg):
    x_pdhg -= T * zbar
    x_pdhg = xp.where(x_pdhg < 0, xp.zeros_like(x_pdhg), x_pdhg)

    if track_cost:
        cost_pdhg[i] = cost_function(x_pdhg)

    y_plus = y + S_A * (pet_lin_op(x_pdhg) + contamination)
    # prox of convex conjugate of negative Poisson logL
    y_plus = 0.5 * (y_plus + 1 - xp.sqrt((y_plus - 1) ** 2 + 4 * S_A * d))

    w_plus = (w + S_G * op_G(x_pdhg)) / beta
    # prox of convex conjugate of TV
    denom = xp.linalg.vector_norm(w_plus, axis=0)
    w_plus /= xp.where(denom < 1, xp.ones_like(denom), denom)
    w_plus *= beta

    delta_z = pet_lin_op.adjoint(y_plus - y) + op_G.adjoint(w_plus - w)
    y = 1.0 * y_plus
    w = 1.0 * w_plus

    z = z + delta_z
    zbar = z + delta_z

    print(f"PDHG iter {(i+1):04} / {num_iter_pdhg}, cost {cost_pdhg[i]:.7e}", end="\r")

# %%
# Vizualizations
# --------------

x_true_np = parallelproj.to_numpy_array(x_true)
x_struct_np = parallelproj.to_numpy_array(x_struct)
x_pdhg_np = parallelproj.to_numpy_array(x_pdhg)

pl2 = x_true_np.shape[2] // 2
pl1 = x_true_np.shape[1] // 2
pl0 = x_true_np.shape[0] // 2

fig, ax = plt.subplots(2, 3, figsize=(9, 5), tight_layout=True)
vmax = 1.2 * x_true_np.max()
ax[0, 0].imshow(x_true_np[:, :, pl2], cmap="Greys", vmin=0, vmax=vmax)
ax[0, 1].imshow(x_pdhg_np[:, :, pl2], cmap="Greys", vmin=0, vmax=vmax)
ax[0, 2].imshow(x_struct_np[:, :, pl2], cmap="Greys")

ax[1, 0].imshow(x_true_np[pl0, :, :].T, cmap="Greys", vmin=0, vmax=vmax)
ax[1, 1].imshow(x_pdhg_np[pl0, :, :].T, cmap="Greys", vmin=0, vmax=vmax)
ax[1, 2].imshow(x_struct_np[pl0, :, :].T, cmap="Greys")

ax[0, 0].set_title("true img", fontsize="medium")
ax[0, 1].set_title(f"DTV PDHG {num_iter_pdhg} it.", fontsize="medium")
ax[0, 2].set_title("structural img", fontsize="medium")
fig.show()

# %%

if track_cost:
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4), tight_layout=True)
    ax2.plot(cost_pdhg, ".-", label="PDHG")
    ax2.grid(ls=":")
    ax2.legend()
    ax2.set_xlabel("iteration")
    ax2.set_title("cost", fontsize="medium")
    fig2.show()
