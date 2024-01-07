"""
MLEM
====

foo bar
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

A = xp.asarray([[1, 4], [3, 2]], dtype=xp.float64, device=dev)

op = parallelproj.MatrixOperator(A)

x_true = xp.asarray([5.5, 10.5], dtype=xp.float64, device=dev)

noise_free_data = op(x_true)

np.random.seed(1)
noisy_data = xp.asarray(
    np.random.poisson(np.asarray(to_device(noise_free_data, "cpu"))),
    device=dev,
    dtype=xp.float64,
)

# noisy_data = xp.floor(noise_free_data)

contamination = xp.asarray([0.1, 0.1], dtype=xp.float64, device=dev)

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

cost = xp.zeros(num_iter, dtype=xp.float64, device=dev)
dist = xp.zeros(num_iter, dtype=xp.float64, device=dev)

for i in range(num_iter):
    exp = op(x) + contamination
    cost[i] = xp.sum(exp - noisy_data * xp.log(exp))
    dist[i] = xp.linalg.vector_norm(x - x_ref) / xp.linalg.vector_norm(x_ref)
    ratio = noisy_data / exp
    x *= op.adjoint(ratio) / ones_back


# %%

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].plot(np.asarray(to_device(cost, "cpu")))
ax[1].loglog(np.asarray(to_device(dist, "cpu")))
ax[0].set_ylim(cost_ref - 0.1 * (float(cost[1]) - cost_ref), float(cost[2]))
fig.tight_layout()
fig.show()
