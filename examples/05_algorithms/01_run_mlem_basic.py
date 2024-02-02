"""
Basic MLEM
==========

This example demonstrates the use of the MLEM algorithm to minimize the negative Poisson log-likelihood function.

.. math::
    f(x) = \sum_{i=1}^m \\bar{y}_i - \\bar{y}_i (x) \log(y_i)

subject to

.. math::
    x \geq 0
    
using the linear forward model

.. math::
    \\bar{y}(x) = A x + s

.. tip::
    parallelproj is python array API compatible meaning it supports different 
    array backends (e.g. numpy, cupy, torch, ...) and devices (CPU or GPU).
    Choose your preferred array API ``xp`` and device ``dev`` below.

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/gschramm/parallelproj/master?labpath=examples
"""

# %%
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
    dev = "cuda"


# %%
# Setup of the forward model :math:`\bar{y}(x) = A x + s`
# --------------------------------------------------------
#
# We setup a minimal linear forward operator :math:`A` respresented by a 4x4 matrix
# and an arbritrary contamination vector :math:`s` of length 4.
#
# .. note::
#     The OSEM implementation below works with all linear operators that
#     subclass :class:`.LinearOperator` (e.g. the high-level projectors).

# setup an arbitrary 4x4 matrix
mat = xp.asarray(
    [
        [2.5, 1.2, 0.3, 0.1],
        [0.4, 3.1, 0.7, 0.2],
        [0.1, 0.3, 4.1, 2.5],
        [0.2, 0.5, 0.2, 0.9],
    ],
    dtype=xp.float64,
    device=dev,
)

op_A = parallelproj.MatrixOperator(mat)
# setup an arbitrary contamination vector that has shape op_A.out_shape
contamination = xp.asarray([0.3, 0.2, 0.1, 0.4], dtype=xp.float64, device=dev)

# %%
# Setup of ground truth and data simulation
# -----------------------------------------
#
# We setup an arbitrary ground truth :math:`x_{true}` and simulate
# noise-free and noisy data :math:`y` by adding Poisson noise.

# ground truth
x_true = xp.asarray([5.5, 10.7, 8.2, 7.9], dtype=xp.float64, device=dev)

# simulated noise-free data
noise_free_data = op_A(x_true) + contamination

# add Poisson noise
np.random.seed(1)
y = xp.asarray(
    np.random.poisson(np.asarray(to_device(noise_free_data, "cpu"))),
    device=dev,
    dtype=xp.float64,
)

# %%
# Analytic calculation of the optimal point (as reference)
# --------------------------------------------------------
#
# Since our linear forward operator :math:`A` is small and invertible
# (*which is usually not the case in practice*),
# we can calculate the optimal point :math:`x^* = A^{-1} (y - s)`
# and the corresponding optimal value of :math:`f(x^*)`.

# calculate the reference solution by inverting A
mat_inv = xp.linalg.inv(mat)
x_ref = mat_inv @ (y - contamination)

# also calculate the cost of the reference solution
exp_ref = op_A(x_ref) + contamination
cost_ref = float(xp.sum(exp_ref - y * xp.log(exp_ref)))

# %%
# MLEM iterations to minimize :math:`f(x)`
# ----------------------------------------
#
# We apply multiple MLEM updates
#
# .. math::
#     x^+ = \frac{x}{A^H 1} A^H \frac{y}{A x + s}
#
# to calculate the minimizer of :math:`f(x)` iteratively.
#
# To monitor the convergence we calculate the relative cost
#
# .. math::
#    \frac{f(x) - f(x^*)}{|f(x^*)|}
#
# and the distance to the optimal point
#
# .. math::
#    \frac{\|x - x^*\|}{\|x^*\|}.

# number MLEM iterations
num_iter = 500

# initialize x
x = xp.ones(op_A.in_shape, dtype=xp.float64, device=dev)
# calculate A^H 1
adjoint_ones = op_A.adjoint(xp.ones(op_A.out_shape, dtype=xp.float64, device=dev))

# allocate arrays for the relative cost and the relative distance to the
# optimal point
rel_cost = xp.zeros(num_iter, dtype=xp.float64, device=dev)
rel_dist = xp.zeros(num_iter, dtype=xp.float64, device=dev)

for i in range(num_iter):
    # evaluate the forward model
    exp = op_A(x) + contamination
    # calculate the relative cost and distance to the optimal point
    rel_cost[i] = (xp.sum(exp - y * xp.log(exp)) - cost_ref) / abs(cost_ref)
    rel_dist[i] = xp.linalg.vector_norm(x - x_ref) / xp.linalg.vector_norm(x_ref)
    # MLEM update
    ratio = y / exp
    x *= op_A.adjoint(ratio) / adjoint_ones


# %%
# Convergences plots
# ------------------

fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True)
ax[0].semilogx(np.asarray(to_device(rel_cost, "cpu")))
ax[1].loglog(np.asarray(to_device(rel_dist, "cpu")))
ax[0].set_ylim(-rel_cost[2], rel_cost[2])
ax[0].set_ylabel(r"( f($x$) - f($x^*$) )   /   | f($x^*$) |")
ax[1].set_ylabel(r"rel. distance to optimum $\|x - x^*\| / \|x^*\|$")
ax[0].set_xlabel("iteration")
ax[1].set_xlabel("iteration")
ax[0].grid(ls=":")
ax[1].grid(ls=":")
fig.tight_layout()
fig.show()
