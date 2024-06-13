import numpy as np
import matplotlib.pyplot as plt
import parallelproj
from scipy.special import erf


def tw(x, sig, bw):
    return 0.5 * (
        erf((x + 0.5 * bw) / (np.sqrt(2) * sig))
        - erf((x - 0.5 * bw) / (np.sqrt(2) * sig))
    )


def tw_int(x, sig, bw):
    # return erf((x + 0.5 * bw) / (np.sqrt(2) * sig))
    return erf((x) / (np.sqrt(2) * sig))


# %%
sig_t = 10.0
delta = 50.0
ns = 3.0

sig_eff = np.sqrt(sig_t**2 + (delta**2) / 12)

x, dx = np.linspace(-ns * sig_eff, ns * sig_eff, 100000, retstep=True)
kernel = tw(x, sig_t, delta)

# %%

n = 101
vsize = 1.0

voxsize = np.array([vsize, vsize, vsize], dtype=np.float32)
tmp = (-0.5 * n + 0.5) * voxsize[0]
img_origin = np.array([tmp, tmp, tmp], dtype=np.float32)

num_off = 15
tof_sums = np.zeros(num_off)

for offset in range(num_off):
    img = np.zeros((n, n, n), dtype=np.float32)

    # img[n // 2 + offset, n // 2, n // 2] = 1.0
    # xstart = np.array([[2 * img_origin[0], 0, 0]])
    # xend = np.array([[-2 * img_origin[0], 0, 0]])

    # img[n // 2, n // 2 + offset, n // 2] = 1.0
    # xstart = np.array([[0, 2 * img_origin[0], 0]])
    # xend = np.array([[0, -2 * img_origin[0], 0]])

    img[n // 2, n // 2, n // 2 + offset] = 1.0
    xstart = np.array([[0, 0, 2 * img_origin[0]]])
    xend = np.array([[0, 0, -2 * img_origin[0]]])

    p_nontof = parallelproj.joseph3d_fwd(
        xstart,
        xend,
        img,
        img_origin,
        voxsize,
    )

    p_tof = parallelproj.joseph3d_fwd_tof_sino(
        xstart,
        xend,
        img,
        img_origin,
        voxsize,
        tofbin_width=delta,
        sigma_tof=np.array([sig_t], dtype=np.float32),
        tofcenter_offset=np.array([0], dtype=np.float32),
        nsigmas=ns,
        ntofbins=max(4 * int(n * vsize / delta / 2) + 1, 11),
    )
    tof_sums[offset] = p_tof.sum()

print(tof_sums)
print(tof_sums.min())
