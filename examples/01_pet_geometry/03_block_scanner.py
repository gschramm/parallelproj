import numpy.array_api as xp
import parallelproj
import math
import matplotlib.pyplot as plt

dev = "cpu"

shape = (3, 2, 2)
spacing = (1.5, 1.2, 1.7)

R = 10

# %%


aff_mat_trans = xp.eye(4, device=dev)
aff_mat_trans[1, -1] = R

mods = []

delta = 2 * xp.pi / 12

for phi in [-delta, 0, delta, 5 * delta, 6 * delta, 7 * delta]:
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
        [[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]]
    ),
)

xstart, xend = lor_desc.get_lor_coordinates()

fig = plt.figure(tight_layout=True)
ax = fig.add_subplot(111, projection="3d")
scanner.show_lor_endpoints(ax, annotation_fontsize=4, show_linear_index=False)

for i in range(xstart.shape[0]):
    ax.plot(
        [xstart[i, 0], xend[i, 0]],
        [xstart[i, 1], xend[i, 1]],
        [xstart[i, 2], xend[i, 2]],
        "r-",
        lw=0.2,
    )
fig.show()
