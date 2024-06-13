# import array_api_compat.cupy as xp
import array_api_strict as xp
import parallelproj

# %%
sig_t = 10.0
ns = 3.0

n = 101
vsize = 1.0

voxsize = xp.asarray([vsize, vsize, vsize], dtype=xp.float32)
tmp = (-0.5 * n + 0.5) * voxsize[0]
img_origin = xp.asarray([tmp, tmp, tmp], dtype=xp.float32)

num_off = 15
tof_sums = xp.zeros(num_off)

for delta in [sig_t / 10.0, sig_t, 3 * sig_t, 10 * sig_t]:
    for offset in range(num_off):
        img = xp.zeros((n, n, n), dtype=xp.float32)

        img[n // 2 + offset, n // 2, n // 2] = 1.0
        xstart = xp.asarray([[2 * int(img_origin[0]), 0, 0]])
        xend = xp.asarray([[-2 * int(img_origin[0]), 0, 0]])

        # img[n // 2, n // 2 + offset, n // 2] = 1.0
        # xstart = xp.asarray([[0, 2 * int(img_origin[0]), 0]])
        # xend = xp.asarray([[0, -2 * int(img_origin[0]), 0]])

        # img[n // 2, n // 2, n // 2 + offset] = 1.0
        # xstart = xp.asarray([[0, 0, 2 * int(img_origin[0])]])
        # xend = xp.asarray([[0, 0, -2 * int(img_origin[0])]])

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
            sigma_tof=xp.asarray([sig_t], dtype=xp.float32),
            tofcenter_offset=xp.asarray([0], dtype=xp.float32),
            nsigmas=ns,
            ntofbins=max(4 * int(n * vsize / delta / 2) + 1, 11),
        )
        tof_sums[offset] = float(xp.sum(p_tof))

    print(f"{(delta/sig_t):.2E}, {1 - float(xp.min(tof_sums))}")
