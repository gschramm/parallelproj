import os
import pyparallelproj as ppp
import numpy as np

import matplotlib.pyplot as plt
plt.ion()

def plot_view_lors(proj, view, ax):
    """ function to plot LORs of a 1 ring cylindrical scanner """
    # get start and end coordinates of a specific view
    xstart, xend = proj.get_subset_sino_coordinates(view)
    xstart = xstart.squeeze()[:-1]
    xend = xend.squeeze()[:-1]

    for i in range(xstart.shape[0]):
        ax.plot([xstart[i,0], xend[i,0]], [xstart[i,1], xend[i,1]], 'g')
    ax.set_aspect('equal')

#--------------------------------------------------------------------------

def main(views = [80,81,82]):
    plt.rcParams['image.cmap'] = 'Greys'

    # setup a scanner
    scanner = ppp.RegularPolygonPETScanner(ncrystals_per_module=np.array([15, 1]),
                                           nmodules=np.array([12, 1]),
                                           R=80.,
                                           crystal_size=np.array([2.2,2.2]))

    # setup a test image
    voxsize = np.array([2.2, 2.2, 1.1])
    n0 = int((2.1*scanner.R) / voxsize[0] )
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

    # forward project image
    img_fwd = proj.fwd_project(img)
    # backporject sinogram of ones
    back = proj.back_project(0*img_fwd + 1)

    fig2, ax2 = plt.subplots(1,2, figsize=(12,6))
    ax2[0].imshow(img_fwd.squeeze().T, interpolation='none')
    ax2[0].set_title(f'fwd proj point source')
    ax2[1].imshow(back, interpolation='none')
    ax2[1].set_title(f'back proj ones')
    ax2[0].set_aspect('equal')
    ax2[1].set_aspect('equal')
    fig2.tight_layout()
    fig2.show()


    for view in views:
        fig, ax = plt.subplots(1,2, figsize=(12,6))
        plot_view_lors(proj, view, ax[0])
        ax[0].plot([vox[0]*voxsize[0] + img_origin[0]],
                   [vox[1]*voxsize[1] + img_origin[1]],
                   'r+', ms = 10)
        ax[0].set_title(f'view {view} LORs')
        ax[0].set_xlim(-90,90)

        sino = np.zeros(img_fwd.shape)
        sino[:,view,...] = 1
        ax[1].imshow(proj.back_project(sino).squeeze().T,
                     interpolation='none', origin='lower')
        ax[1].set_title(f'back proj view {view}')

        ax[0].set_aspect('equal')
        ax[1].set_aspect('equal')
        fig.tight_layout()
        fig.show()

#--------------------------------------------------------------------------

if __name__ == '__main__':
    main()
