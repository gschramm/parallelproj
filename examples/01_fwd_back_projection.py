"""minimal script to show how to do a (geometrical) forward and backprojection"""
import os
import pyparallelproj as ppp
import numpy as np
import argparse

nsubsets = 1
np.random.seed(1)

# setup a scanner
scanner = ppp.RegularPolygonPETScanner(ncrystals_per_module=np.array([16, 1]),
                                       nmodules=np.array([28, 1]))

# setup a test image
voxsize = np.array([2., 2., 2.])
n0 = 350
n1 = 350
n2 = max(1, int((scanner.xc2.max() - scanner.xc2.min()) / voxsize[2]))

# setup a random image
img = np.random.rand(n0, n1, n2)
img_origin = (-(np.array(img.shape) / 2) + 0.5) * voxsize

# setup the projector

sino_params = ppp.PETSinogramParameters(scanner, ntofbins=27, tofbin_width=28.)
proj = ppp.SinogramProjector(scanner,
                             sino_params,
                             img.shape,
                             nsubsets=nsubsets,
                             voxsize=voxsize,
                             img_origin=img_origin,
                             tof=True,
                             sigma_tof=60. / 2.35,
                             n_sigmas=3)

######## tof projections

# setup a random sinogram
tsino = np.random.rand(*proj.sino_params.shape)

img_fwd_tof = proj.fwd_project(img)
back_tof = proj.back_project(tsino)

# check if fwd and back projection are adjoint
print((img * back_tof).sum())
print((img_fwd_tof * tsino).sum())

######## nontof projections

proj.set_tof(False)

# setup a random sinogram
rsino = np.random.rand(*proj.sino_params.nontof_shape)

img_fwd_nontof = proj.fwd_project(img)
back_nontof = proj.back_project(rsino)

# check if fwd and back projection are adjoint
print((img * back_nontof).sum())
print((img_fwd_nontof * rsino).sum())
