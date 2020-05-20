import numpy as np

from pet_scanners import RegularPolygonPETScanner
from sinogram import PETSinogram
from projectors import SinogramProjector

# setup a scanner
scanner = RegularPolygonPETScanner(ncrystals_per_module = np.array([16,1]),
                                   nmodules             = np.array([28,1]))

# setup a test image
voxsize = np.array([2.,2.,2.])
n0      = 120
n1      = 120
n2      = 1

img = np.random.rand(n0,n1,n2)
img_origin = (-(np.array(img.shape) / 2) +  0.5) * voxsize

######## nontof forward projection
sino = PETSinogram(scanner)
proj = SinogramProjector(scanner, sino, img.shape, nsubsets = 1, 
                         voxsize = voxsize, img_origin = img_origin)

img_fwd = proj.fwd_project(img, subset = 0)

rsino = np.random.rand(*img_fwd.shape)
back = proj.back_project(rsino, subset = 0)

######## tof forward projection
tofsino = PETSinogram(scanner, ntofbins = 27, tofbin_width = 28.)
tofproj = SinogramProjector(scanner, tofsino, img.shape, nsubsets = 1, 
                            voxsize = voxsize, img_origin = img_origin,
                            tof = True, sigma_tof = 60., n_sigmas = 3)

img_fwd_tof = tofproj.fwd_project(img, subset = 0)

tsino = np.random.rand(*img_fwd_tof.shape)
back_tof = proj.back_project(tsino, subset = 0)
