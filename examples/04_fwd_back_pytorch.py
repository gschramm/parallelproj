"""minimal script to show how to do a (geometrical) forward and backprojection of 
   pytorch (cuda device) tensors"""
import os
import pyparallelproj as ppp
import numpy as np
import argparse

import torch
import cupy as cp

device = 'cuda'
xp = cp

nsubsets = 1
np.random.seed(1)

# setup a scanner
scanner = ppp.RegularPolygonPETScanner(ncrystals_per_module=np.array([16, 9]),
                                       nmodules=np.array([28, 2]), on_gpu = True)

# setup a test image
voxsize = np.array([2., 2., 2.])
n0 = 350
n1 = 350
n2 = max(1, int((scanner.xc2.max() - scanner.xc2.min()) / voxsize[2]))

# create a box shaped 3d image as torch tensor
torch_tensor_3d = torch.zeros((n0, n1, n2), device = device)
torch_tensor_3d[(n0//4):(3*n0//4),(n1//4):(3*n1//4),(n2//4):(3*n2//4)] = 1

# coordinates of the [0,0,0] voxel
img_origin = (-(np.array(torch_tensor_3d.shape) / 2) + 0.5) * voxsize


# setup the projector
sino_params = ppp.PETSinogramParameters(scanner, ntofbins=27, tofbin_width=28.)
proj = ppp.SinogramProjector(scanner,
                             sino_params,
                             torch_tensor_3d.shape,
                             nsubsets=nsubsets,
                             voxsize=voxsize,
                             img_origin=img_origin,
                             tof=True,
                             sigma_tof=60. / 2.35,
                             n_sigmas=3)

#----------------------------------------------------------------------------- 
# fwd and back projections of torch tensors are possible
# they just have to be converted into cupy arrays using cp.asarray()
# https://docs.cupy.dev/en/stable/user_guide/interoperability.html#pytorch
# cp.asarray is a zero-copy data exchange between cupy and pytorch and comes at
# no cost

fwd_projection = proj.fwd_project(cp.asarray(torch_tensor_3d))

# after the forward projection, the result is a cupy array
# if needed it can be converted back into a torch tensor
print(type(fwd_projection))
fwd_projection = torch.as_tensor(fwd_projection, device = device)
print(type(fwd_projection))

# same for the back projection
back_projection = proj.back_project(cp.asarray(fwd_projection))

print(type(back_projection))
back_projection = torch.as_tensor(back_projection, device = device)
print(type(back_projection))
