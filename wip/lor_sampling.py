import os
import pyparallelproj as ppp
import numpy as np

import matplotlib.pyplot as plt
plt.ion()

def plot_view_lors(proj, view, save = False):
    """ function to plot LORs of a 1 ring cylindrical scanner """
    # get start and end coordinates of a specific view
    xstart, xend = proj.get_subset_sino_coordinates(view)
    xstart = xstart.squeeze()[:-1]
    xend = xend.squeeze()[:-1]

    fig, ax = proj.scanner.show_crystal_config()
    for i in range(xstart.shape[0]):
        ax[0].plot([xstart[i,0], xend[i,0]], [xstart[i,1], xend[i,1]], 'g')
    ax[0].plot(xstart[:,0], xstart[:,1], 'g^')
    ax[0].set_aspect('equal')

    fig.tight_layout()
    fig.show()

    return fig, ax

#--------------------------------------------------------------------------

def main():
    # setup a scanner
    scanner = ppp.RegularPolygonPETScanner(ncrystals_per_module=np.array([15, 1]),
                                           nmodules=np.array([12, 1]),
                                           R=80.,
                                           crystal_size=np.array([2.2,2.2]))

    # setup a test image
    voxsize = np.array([2.2, 2.2, 1.1])
    n0 = int((1.8*scanner.R) / voxsize[0] )
    n1 = n0
    n2 = 1

    # setup a random image
    img = np.zeros((n0,n1,n2))
    vox = (10, n1//2)
    img[vox[0],vox[1],0] = 1
    img_origin = (-(np.array(img.shape) / 2) + 0.5) * voxsize

    # setup the projector
    sino_params = ppp.PETSinogramParameters(scanner, rtrim = 18)
    proj = ppp.SinogramProjector(scanner,
                                 sino_params,
                                 img.shape,
                                 nsubsets=1,
                                 voxsize=voxsize,
                                 img_origin=img_origin,
                                 tof=False)

    # use number of subsets equal to number of projection angles
    proj.init_subsets(proj.sino_params.nontof_shape[1])

    for view in [72,73,74]:
        fig, ax = plot_view_lors(proj, view, save = False)
        ax[0].plot([vox[0]*voxsize[0] + img_origin[0]],
                   [vox[1]*voxsize[1] + img_origin[1]],
                   'b+')

    ######### nontof projections
    # setup a random sinogram
    img_fwd = proj.fwd_project(img)

    fig2, ax2 = plt.subplots()
    ax2.imshow(img_fwd.squeeze().T)
    fig2.tight_layout()
    fig2.show()

#--------------------------------------------------------------------------

if __name__ == '__main__':
    main()
