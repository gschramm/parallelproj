"""
SPDHG to optimize the Poisson logL and total variation
======================================================

This example demonstrates the use of the stochastic primal dual hybrid gradient (SPDHG) algorithm to minimize the negative 
Poisson log-likelihood function combined with a total variation regularizer:

.. math::
    f(x) = \sum_{i=1}^m \\bar{d}_i (x) - d_i \log(\\bar{d}_i (x)) + \\beta \\|\\nabla x \\|_{1,2}

subject to

.. math::
    x \geq 0
    
using the linear forward model

.. math::
    \\bar{d}(x) = A x + s

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
        np.random.poisson(parallelproj.to_numpy_array(data)),
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
#
# .. admonition:: SPDHG algorithm to minimize negative Poisson log-likelihood + regularization
#
#   | **Input** Poisson data :math:`d`
#   | **Initialize** :math:`x,y_i,w,S_{A_i},S_G,T,p_i`
#   | **Preprocessing** :math:`\overline{z} = z = \sum_i A_i^T y + \nabla^T w`
#   | **Repeat**, until stopping criterion fulfilled
#   |     **Update** :math:`x \gets \text{proj}_{\geq 0} \left( x - T \overline{z} \right)`
#   |     **select a random data subset number i or do prior update according to** :math:`p_i`
#   |       **Update** :math:`y_i^+ \gets \text{prox}_{D^*}^{S_{A_i}} ( y_i + S_{A_i}  ( {A_i} x + s))`
#   |       **Update** :math:`\Delta z \gets A_i^T (y_i^+ - y_i)`
#   |       **Update** :math:`y_i \gets y_i^+`
#   |     **or**
#   |       **Update** :math:`w^+ \gets \beta \, \text{prox}_{R^*}^{S_G/\beta} ((w + S_G  \nabla x)/\beta)`
#   |       **Update** :math:`\Delta z \gets \nabla^T (w^+ - w)`
#   |       **Update** :math:`w \gets w^+`
#   |     **Update** :math:`z \gets z + \Delta z`
#   |     **Update** :math:`\bar{z} \gets z + (\Delta z \ p_i)`
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
#  :math:`S_{A_i} = \gamma \, \text{diag}(\frac{\rho}{A_i 1})`
#
#  :math:`S_G = \gamma \, \text{diag}(\frac{\rho}{|\nabla|})`
#
#  :math:`T_{A_i} = \gamma^{-1} \text{diag}(\frac{\rho}{A_i^T 1})`
#
#  :math:`T_G = \gamma^{-1} \text{diag}(\frac{\rho}{|\nabla|})`
#
#  :math:`T = \min T_{A_i}, T_G` pointwise
#

num_iter = 500
gamma = 1e-2  # should be roughly 1 / max(x_true) for fast convergence
rho = 0.9999

p_g = 0.5
p_a = (1 - p_g) / len(subset_op_A)

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
        (1 / gamma) * rho * p_a / op.adjoint(xp.ones(op.out_shape, dtype=xp.float64, device=dev))
    )
    for op in subset_op_A
]

# calculate the element wise min over all subsets
T_A = xp.min(xp.asarray(subset_T_A), axis=0)

op_G_norm = op_G.norm(xp, dev, num_iter=100)
S_G = gamma * rho / op_G_norm
T_G = (1 / gamma) * rho * p_g / op_G_norm

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
        p = p_a
    else:
        w_plus = (w + S_G * op_G(x)) / beta
        # prox of convex conjugate of TV
        denom = xp.linalg.vector_norm(w_plus, axis=0)
        w_plus /= xp.where(denom < 1, xp.ones_like(denom), denom)
        w_plus *= beta

        delta_z = op_G.adjoint(w_plus - w)
        w = 1.0 * w_plus
        p = p_g

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
    print(f"rel. distance to Powell solution: {rel_dist:.2e}")
    print(f"cost: {cost_function(x_ref):.6e}")

# %%
fig, ax = plt.subplots(1, 1, tight_layout=True)
if xp.__name__.endswith("numpy"):
    ax.axhline(cost_function(x_ref), color="k", linestyle="--")
ax.plot(parallelproj.to_numpy_array(cost))
ax.set_xlabel("iteration")
ax.set_ylabel("cost function")
ax.grid(ls=":")
fig.show()
