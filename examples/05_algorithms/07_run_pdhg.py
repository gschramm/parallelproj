"""
PDHG to optimize the Poisson logL with total variation
======================================================

This example demonstrates the use of the primal dual hybrid gradient (PDHG) algorithm to minimize the negative 
Poisson log-likelihood function combined with a total variation regularizer:

.. math::
    f(x) = \\underbrace{\sum_{i=1}^m \\bar{y}_i - \\bar{y}_i (x) \log(y_i)}_{D(x)} + \\beta \\underbrace{\\|\\nabla x \\|_{1,2}}_{R(x)}

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
        [1.0, 1.2, 6.3, 0.1],
        [0.1, 0.2, 0.1, 0.1],
        [1.1, 0.3, 4.1, 0.1],
        [1.2, 3.5, 7.2, 0.1],
        [1.3, 8.1, 4.7, 0.1],
    ],
    dtype=xp.float64,
    device=dev,
)

op_A = parallelproj.MatrixOperator(mat)
# normalize the forward operator
op_A.scale = 1 / op_A.norm(xp, dev)
# setup an arbitrary contamination vector that has shape op_A.out_shape
contamination = xp.asarray([1.3, 0.2, 0.1, 0.4, 0.1], dtype=xp.float64, device=dev)

# %%
# Setup of ground truth and data simulation
# -----------------------------------------
#
# We setup an arbitrary ground truth :math:`x_{true}` and simulate
# noise-free and noisy data :math:`y` by adding Poisson noise.

# ground truth
x_true = 3 * xp.asarray([5.5, 10.7, 8.2, 7.9], dtype=xp.float64, device=dev)

# simulated noise-free data
noise_free_data = op_A(x_true) + contamination

# add Poisson noise
np.random.seed(1)
d = xp.asarray(
    np.random.poisson(np.asarray(to_device(noise_free_data, "cpu"))),
    device=dev,
    dtype=xp.float64,
)

# %%
beta = 1e-2
# setup of gradient operator
op_G = parallelproj.FiniteForwardDifference(op_A.in_shape)


# %%
def cost_function(x):
    exp = op_A(x) + contamination
    if (xp.min(exp) < 0) or (xp.min(x) < 0):
        res = xp.finfo(xp.float64).max
    else:
        res = float(
            xp.sum(exp - d * xp.log(exp))
            + beta * xp.sum(xp.linalg.vector_norm(op_G(x), axis=0))
        )
    return res


# %%
# PDHG update to minimize :math:`f(x)`
# ------------------------------------
#
# We apply multiple pre-conditioned PDHG updates
# to calculate the minimizer of :math:`f(x)` iteratively.
#
# .. math::
#   \DeclareMathOperator{\proj}{proj}
#   \DeclareMathOperator{\prox}{prox}
#   \DeclareMathOperator*{\argmin}{argmin}
# 	x = \proj_{\geq 0} (x - T (z + \Delta z))
#
# .. math::
# 	y^+ = \prox_{D^*}^{S_A} ( y + S_A  ( A x + s))
#
# .. math::
#   w^+ = \beta \prox_{R^*}^{S_G/\beta} ((w + S_G  \nabla x)/\beta)
#
# .. math::
#   \Delta z = A^T (y^+ - y) + \nabla^T (w^+ - w)
#
# .. math::
#    z = z + \Delta z
#
# .. math::
#    y = y^+
#
# .. math::
#    w = w^+
#
# See :cite:p:`Ehrhardt2019` :cite:p:`Schramm2022` for more details.

num_iter = 500
gamma = 1e-2  # should be roughly 1 / max(x_true) for fast convergence
rho = 0.9999

# initialize primal and dual variables
x = op_A.adjoint(d)
y = 1 - d / (op_A(x) + contamination)
w = beta * xp.sign(op_G(x))

z = op_A.adjoint(y) + op_G.adjoint(w)

# calculate PHDG step sizes
S_A = gamma * rho / op_A(xp.ones(op_A.in_shape, dtype=xp.float64, device=dev))
T_A = (
    (1 / gamma)
    * rho
    / op_A.adjoint(xp.ones(op_A.out_shape, dtype=xp.float64, device=dev))
)

op_G_norm = op_G.norm(xp, dev, num_iter=100)
S_G = gamma * rho / op_G_norm
T_G = (1 / gamma) * rho / op_G_norm

T = xp.where(T_A < T_G, T_A, xp.full(op_A.in_shape, T_G))

# run PHDG iterations
cost = xp.zeros(num_iter, dtype=xp.float64, device=dev)

delta_z = 0.0

for i in range(num_iter):
    x -= T * (z + delta_z)
    x = xp.where(x < 0, xp.zeros_like(x), x)

    cost[i] = cost_function(x)

    y_plus = y + S_A * (op_A(x) + contamination)
    # prox of convex conjugate of negative Poisson logL
    y_plus = 0.5 * (y_plus + 1 - xp.sqrt((y_plus - 1) ** 2 + 4 * S_A * d))

    w_plus = (w + S_G * op_G(x)) / beta
    # prox of convex conjugate of TV
    denom = xp.linalg.vector_norm(w_plus, axis=0)
    w_plus /= xp.where(denom < 1, xp.ones_like(denom), denom)
    w_plus *= beta

    delta_z = op_A.adjoint(y_plus - y) + op_G.adjoint(w_plus - w)
    y = 1.0 * y_plus
    w = 1.0 * w_plus

    z = z + delta_z


# %%
# calculate reference solution using Powell optimizer
if xp.__name__.endswith("numpy"):
    from scipy.optimize import fmin_powell

    x_ref = fmin_powell(cost_function, x, xtol=1e-6, ftol=1e-6)
    rel_dist = float(xp.sum((x - x_ref) ** 2)) / float(xp.sum(x_ref**2))


# %%
print(f"\nSolution after {num_iter} PDHG iterations:")
print(x)
print(f"cost: {cost[-1]:.6e}")

if xp.__name__.endswith("numpy"):
    print(f"\nReference solution using Powell optimizer:")
    print(x_ref)
    print(f"rel. distance to DePierro solution: {rel_dist:.2e}")
    print(f"cost: {cost_function(x_ref):.6e}")

# %%
fig, ax = plt.subplots(1, 1, tight_layout=True)
if xp.__name__.endswith("numpy"):
    ax.axhline(cost_function(x_ref), color="k", linestyle="--")
ax.plot(np.asarray(to_device(cost, "cpu")))
ax.set_xlabel("iteration")
ax.set_ylabel("cost function")
ax.grid(ls=":")
fig.show()
