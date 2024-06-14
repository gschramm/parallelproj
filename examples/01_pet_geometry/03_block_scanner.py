import parallelproj
import math
import matplotlib.pyplot as plt

import array_api_strict as xp
dev = None
import array_api_compat.cupy as xp
dev = xp.cuda.Device(0)

shape = (3, 2, 2)
spacing = (1.5, 1.2, 1.7)

R = 10

# %%


aff_mat_trans = xp.eye(4, device=dev)
aff_mat_trans[1, -1] = R

mods = []

delta = 2 * xp.pi / 12

for phi in [-delta, 0, delta, 5 * delta, 6 * delta, 7 * delta, 8 * delta]:
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
            shape,
            spacing,
            affine_transformation_matrix=(aff_mat_rot @ aff_mat_trans),
        )
    )

scanner = parallelproj.ModularizedPETScannerGeometry(mods)

lor_desc = parallelproj.BlockPETLORDescriptor(
    scanner,
    xp.asarray(
        [[0, 3], [0, 4], [0, 5], [0,6], 
         [1, 3], [1, 4], [1, 5], [1,6],
         [2, 3], [2, 4], [2, 5], [2,6]]
    ),
)

xstart0, xend0 = lor_desc.get_lor_coordinates(xp.asarray([0], device=dev))
xstart5, xend5 = lor_desc.get_lor_coordinates(xp.asarray([5], device=dev))
xstart11, xend11 = lor_desc.get_lor_coordinates(xp.asarray([11], device=dev))


# %%
# conver the arrays to numpy for plotting

xstart0 = parallelproj.to_numpy_array(xstart0)
xend0 = parallelproj.to_numpy_array(xend0)

xstart5 = parallelproj.to_numpy_array(xstart5)
xend5 = parallelproj.to_numpy_array(xend5)

xstart11 = parallelproj.to_numpy_array(xstart11)
xend11 = parallelproj.to_numpy_array(xend11)

fig = plt.figure(tight_layout=True)
ax = fig.add_subplot(111, projection="3d")
scanner.show_lor_endpoints(ax, annotation_fontsize=4, show_linear_index=False)
lor_desc.show_block_pair_lors(ax, block_pair_nums=xp.asarray([0], device=dev), color=plt.cm.tab10(0))
lor_desc.show_block_pair_lors(ax, block_pair_nums=xp.asarray([5], device=dev), color=plt.cm.tab10(1))
lor_desc.show_block_pair_lors(ax, block_pair_nums=xp.asarray([11], device=dev), color=plt.cm.tab10(2))
fig.show()
