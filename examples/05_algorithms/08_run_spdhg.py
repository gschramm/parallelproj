"""
SPDHG to optimize the Poisson logL with total variation
=======================================================

This example demonstrates the use of the stochastic primal dual hybrid gradient (SPDHG) algorithm to minimize the negative 
Poisson log-likelihood function combined with a total variation regularizer:

.. math::
    f(x) = \sum_{i=1}^m \\bar{y}_i - \\bar{y}_i (x) \log(y_i) + \\beta \\|\\nabla x \\|_{1,2}

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
scale = 1 / op_A.norm(xp, dev)
op_A.scale = scale

# setup the forward operator split into three subsets
op_A1 = parallelproj.MatrixOperator(mat[0:2, :])
op_A1.scale = scale
op_A2 = parallelproj.MatrixOperator(mat[2:4, :])
op_A2.scale = scale
op_A3 = parallelproj.MatrixOperator(mat[4:5, :])
op_A3.scale = scale

subset_op_A = parallelproj.LinearOperatorSequence([op_A1, op_A2, op_A3])

# normalize the forward operator
# setup an arbitrary contamination vector that has shape op_A.out_shape
subset_contamination = [
    xp.asarray([1.3, 0.2], dtype=xp.float64, device=dev),
    xp.asarray([0.1, 0.4], dtype=xp.float64, device=dev),
    xp.asarray([0.1], dtype=xp.float64, device=dev),
]

# %%
# Setup of ground truth and data simulation
# -----------------------------------------
#
# We setup an arbitrary ground truth :math:`x_{true}` and simulate
# noise-free and noisy data :math:`y` by adding Poisson noise.

# ground truth
x_true = 3 * xp.asarray([5.5, 10.7, 8.2, 7.9], dtype=xp.float64, device=dev)

# simulated noise-free data
noise_free_data = [
    op(x_true) + subset_contamination[i] for i, op in enumerate(subset_op_A)
]

# add Poisson noise
np.random.seed(1)
subset_d = [
    xp.asarray(
        np.random.poisson(np.asarray(to_device(data, "cpu"))),
        device=dev,
        dtype=xp.float64,
    )
    for data in noise_free_data
]

# %%
beta = 1e-2
# setup of gradient operator
op_G = parallelproj.FiniteForwardDifference(op_A.in_shape)


# %%
def cost_function(x):
    if xp.min(x) < 0:
        return xp.finfo(xp.float64).max

    res = 0

    for i, op in enumerate(subset_op_A):
        subset_exp = op(x) + subset_contamination[i]
        if xp.min(subset_exp) < 0:
            return xp.finfo(xp.float64).max
        else:
            res += float(xp.sum(subset_exp - subset_d[i] * xp.log(subset_exp)))

    res += beta * float(xp.sum(xp.linalg.vector_norm(op_G(x), axis=0)))

    return res


# %%
# SPDHG updates to minimize :math:`f(x)`
# --------------------------------------
#
# We apply multiple pre-conditioned SPDHG updates
# to calculate the minimizer of :math:`f(x)` iteratively.
#
# .. math::
#   \DeclareMathOperator{\proj}{proj}
#   \DeclareMathOperator{\prox}{prox}
#   \DeclareMathOperator*{\argmin}{argmin}
#
# .. math::
#
#   &\text{select a random data subset number} \ i \ \text{or do prior update} \\\\
# 	x &= \proj_{\geq 0} (x - T \bar{z}) \\\\
# 	y_i^+ &= \prox_{D^*}^{S_{A_i}} ( y_i + S_{A_i}  ( A_i x + s)) \\\\
#   \Delta z &= A_i^T (y_i^+ - y_i) \\\\
#   \text{or} \\\\
#   w^+& = \beta \prox_{R^*}^{S_G/\beta} ((w + S_G  \nabla x)/\beta) \\\\
#   \Delta z &= \nabla^T (w^+ - w) \\\\
#   z &= z + \Delta z \\\\
#   \bar{z} &= z + \frac{1}{p_i}\Delta z \\\\
#   y &= y^+ \\\\
#   w &= w^+
#
# See :cite:p:`Ehrhardt2019` :cite:p:`Schramm2022` for more details.


num_iter = 500
gamma = 1e-2  # should be roughly 1 / max(x_true) for fast convergence
rho = 0.9999

# initialize primal and dual variables
x = subset_op_A.adjoint(subset_d)
subset_y = [
    1 - subset_d[i] / (op(x) + subset_contamination[i])
    for i, op in enumerate(subset_op_A)
]
w = beta * xp.sign(op_G(x))

z = subset_op_A.adjoint(subset_y) + op_G.adjoint(w)
zbar = 1.0 * z


# calculate PHDG step sizes
subset_S_A = [
    gamma * rho / op(xp.ones(op.in_shape, dtype=xp.float64, device=dev))
    for op in subset_op_A
]
subset_T_A = [
    (
        (1 / gamma)
        * rho
        / op.adjoint(xp.ones(op.out_shape, dtype=xp.float64, device=dev))
    )
    for op in subset_op_A
]

# calculate the element wise min over all subsets
T_A = xp.min(xp.asarray(subset_T_A), axis=0)

op_G_norm = op_G.norm(xp, dev, num_iter=100)
S_G = gamma * rho / op_G_norm
T_G = (1 / gamma) * rho / op_G_norm

T = xp.where(T_A < T_G, T_A, xp.full(4, T_G))

# run SPHDG iterations
cost = xp.zeros(num_iter, dtype=xp.float64, device=dev)

for i in range(num_iter):
    x -= T * zbar
    x = xp.where(x < 0, xp.zeros_like(x), x)

    cost[i] = cost_function(x)

    # select a random subset
    # in 50% of the cases we select a subset of the forward operator
    # in the other 50% select the gradient operator
    i_ss = np.random.randint(0, 2 * len(subset_op_A))

    if i_ss < len(subset_op_A):
        y_plus = subset_y[i_ss] + subset_S_A[i_ss] * (
            subset_op_A[i_ss](x) + subset_contamination[i_ss]
        )
        # prox of convex conjugate of negative Poisson logL
        y_plus = 0.5 * (
            y_plus
            + 1
            - xp.sqrt((y_plus - 1) ** 2 + 4 * subset_S_A[i_ss] * subset_d[i_ss])
        )

        delta_z = subset_op_A[i_ss].adjoint(y_plus - subset_y[i_ss])
        subset_y[i_ss] = y_plus
        p = 0.5 / len(subset_op_A)
    else:
        w_plus = (w + S_G * op_G(x)) / beta
        # prox of convex conjugate of TV
        denom = xp.linalg.vector_norm(w_plus, axis=0)
        w_plus /= xp.where(denom < 1, xp.ones_like(denom), denom)
        w_plus *= beta

        delta_z = op_G.adjoint(w_plus - w)
        w = 1.0 * w_plus
        p = 0.5

    z = z + delta_z
    zbar = z + delta_z / p
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
