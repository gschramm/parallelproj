import pyparallelproj as ppp
import numpy as np
import matplotlib.pyplot as plt
from warnings import warn

try:
    import cupy as cp
except:
    warn('cupy is not available')

#--------------------------------------------------------------------------------------------------------

on_gpu = False
norm_name = 'l2_l1'
n = 200

#--------------------------------------------------------------------------------------------------------

if on_gpu:
    xp = cp
else:
    xp = np

xp.random.seed(1)

img = ppp.ellipse2d_phantom(n=n, c=3)
if on_gpu:
    img = xp.asarray(img)

img += 2 * xp.random.rand(*img.shape)

#--------------------------------------------------------------------------------------------------------
# PDHG L2 denoise

# pixel dependent weight for data fidelity term
weights = xp.full(img.shape, 3e0, dtype=xp.float32)
weights[:n // 2, :] /= 3
prior = ppp.GradientBasedPrior(ppp.GradientOperator(xp),
                               ppp.GradientNorm(xp, name=norm_name))

pdhg_denoise = ppp.PDHG_L2_Denoise(prior, xp, verbose=True)
pdhg_denoise.init(img, weights)
pdhg_denoise.run(100, calculate_cost=True)

#--------------------------------------------------------------------------------------------------------
# visualizations

fig, ax = plt.subplots(1, 3, figsize=(9, 3))
if xp.__name__ == 'numpy':
    ax[0].imshow(img, vmax=4)
    ax[1].imshow(pdhg_denoise.x, vmax=4)
else:
    ax[0].imshow(xp.asnumpy(img))
    ax[1].imshow(xp.asnumpy(pdhg_denoise.x))
ax[2].semilogy(np.arange(1, pdhg_denoise.cost.shape[0] + 1), pdhg_denoise.cost)
fig.tight_layout()
fig.show()
