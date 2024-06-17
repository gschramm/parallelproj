import parallelproj
import math
import matplotlib.pyplot as plt

# import array_api_strict as xp
# dev = None

import array_api_compat.cupy as xp

dev = xp.cuda.Device(0)

# %%
block_shape = (3, 2, 2)
block_spacing = (1.5, 1.2, 1.7)
scanner_radius = 10

# %%
mods = []

delta_phi = 2 * xp.pi / 12

aff_mat_trans = xp.eye(4, device=dev)
aff_mat_trans[1, -1] = scanner_radius

for phi in [
    -delta_phi,
    0,
    delta_phi,
    5 * delta_phi,
    6 * delta_phi,
    7 * delta_phi,
    8 * delta_phi,
]:
    aff_mat_rot = xp.asarray(
        [
            [math.cos(phi), -math.sin(phi), 0, 0],
            [math.sin(phi), math.cos(phi), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    mods.append(
        parallelproj.BlockPETScannerModule(
            xp,
            dev,
            block_shape,
            block_spacing,
            affine_transformation_matrix=(aff_mat_rot @ aff_mat_trans),
        )
    )

scanner = parallelproj.ModularizedPETScannerGeometry(mods)

# %%
fig = plt.figure(tight_layout=True)
ax = fig.add_subplot(111, projection="3d")
scanner.show_lor_endpoints(ax, annotation_fontsize=4, show_linear_index=False)
fig.show()

# %%
lor_desc = parallelproj.BlockPETLORDescriptor(
    scanner,
    xp.asarray(
        [
            [0, 3],
            [0, 4],
            [0, 5],
            [0, 6],
            [1, 3],
            [1, 4],
            [1, 5],
            [1, 6],
            [2, 3],
            [2, 4],
            [2, 5],
            [2, 6],
        ]
    ),
)

# %%
fig2 = plt.figure(tight_layout=True)
ax2 = fig2.add_subplot(111, projection="3d")
scanner.show_lor_endpoints(ax2, annotation_fontsize=4, show_linear_index=False)
lor_desc.show_block_pair_lors(
    ax2, block_pair_nums=xp.asarray([0], device=dev), color=plt.cm.tab10(0)
)
lor_desc.show_block_pair_lors(
    ax2, block_pair_nums=xp.asarray([5], device=dev), color=plt.cm.tab10(1)
)
lor_desc.show_block_pair_lors(
    ax2, block_pair_nums=xp.asarray([11], device=dev), color=plt.cm.tab10(2)
)
fig2.show()

# %%
fig3 = plt.figure(tight_layout=True)
ax3 = fig3.add_subplot(111, projection="3d")
scanner.show_lor_endpoints(ax3, annotation_fontsize=4, show_linear_index=False)
lor_desc.show_block_pair_lors(ax3, block_pair_nums=None, color=plt.cm.tab10(0))
fig3.show()

# %%
# get the start and end coordinates of all LORs in block pair 0
xstart0, xend0 = lor_desc.get_lor_coordinates(xp.asarray([0], device=dev))
# get the start and end coordinates of all LORs of all block pairs
xstart, xend = lor_desc.get_lor_coordinates()
