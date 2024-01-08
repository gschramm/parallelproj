"""
Basic MLEM
==========

This example demonstrates the use of the MLEM algorithm to minimize the negative Poisson log-likelihood function.

.. math::
    f(x) = \sum_{i=1}^m \\bar{y}_i - \\bar{y}_i (x) \log(y_i)

using the linear forward model

.. math::
    \\bar{y}(x) = A x + s
"""
# %%
# parallelproj supports the numpy, cupy and pytorch array API and different devices
# choose your preferred array API uncommenting the corresponding line

import numpy.array_api as xp

# import array_api_compat.numpy as xp

# import array_api_compat.cupy as xp
# import array_api_compat.torch as xp

# %%
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

A = xp.asarray(
    [[2.5, 1.2, 0, 0], [0, 3.1, 0.7, 0], [0, 0, 4.1, 2.5], [0.2, 0, 0, 0.9]],
    dtype=xp.float64,
    device=dev,
)

op = parallelproj.MatrixOperator(A)

x_true = xp.asarray([5.5, 10.7, 8.2, 7.9], dtype=xp.float64, device=dev)

noise_free_data = op(x_true)

np.random.seed(1)
noisy_data = xp.asarray(
    np.random.poisson(np.asarray(to_device(noise_free_data, "cpu"))),
    device=dev,
    dtype=xp.float64,
)

# noisy_data = xp.floor(noise_free_data)

contamination = xp.asarray([0.3, 0.2, 0.1, 0.4], dtype=xp.float64, device=dev)

# %%

# calculate the reference solution by inverting A
A_inv = xp.linalg.inv(A)
x_ref = A_inv @ (noisy_data - contamination)

exp_ref = op(x_ref) + contamination
cost_ref = float(xp.sum(exp_ref - noisy_data * xp.log(exp_ref)))
# %%
num_iter = 500

x = xp.ones(op.in_shape, dtype=xp.float64, device=dev)
ones_back = op.adjoint(xp.ones(op.out_shape, dtype=xp.float64, device=dev))

rel_cost = xp.zeros(num_iter, dtype=xp.float64, device=dev)
rel_dist = xp.zeros(num_iter, dtype=xp.float64, device=dev)

for i in range(num_iter):
    exp = op(x) + contamination
    rel_cost[i] = (xp.sum(exp - noisy_data * xp.log(exp)) - cost_ref) / abs(cost_ref)
    rel_dist[i] = xp.linalg.vector_norm(x - x_ref) / xp.linalg.vector_norm(x_ref)
    ratio = noisy_data / exp
    x *= op.adjoint(ratio) / ones_back


# %%

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
