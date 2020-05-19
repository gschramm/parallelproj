import numpy as np
import matplotlib.pyplot as py
import ctypes

from pet_scanners import RegularPolygonPETScanner
from sinogram import PETSinogram
from projectors import SinogramProjector

# setup a scanner
scanner = RegularPolygonPETScanner(ncrystals_per_module = np.array([16,3]),
                                   nmodules             = np.array([28,1]))

# setup a test image
n = 250
img = np.zeros((n,n,1), dtype = ctypes.c_float)  
img[(n//4):(3*n//4),(n//4):(3*n//4),0] = 1
voxsize = np.array([2.,2.,2.], dtype = ctypes.c_float)

img_origin = (-(np.array(img.shape) / 2) +  0.5) * voxsize

######## nontof forward projection
sino = PETSinogram(scanner)
proj = SinogramProjector(scanner, sino, img.shape, nsubsets = 4, 
                         voxsize = voxsize, img_origin = img_origin)

img_fwd = proj.fwd_project(img, subset = 0)


######## tof forward projection


# setup TOF params for sino
tofsino = PETSinogram(scanner, ntofbins = 27, tofbin_width = 28.)
tofproj = SinogramProjector(scanner, tofsino, img.shape, nsubsets = 4, 
                            voxsize = voxsize, img_origin = img_origin,
                            tof = True, sigma_tof = 60., n_sigmas = 3)

img_fwd_tof = tofproj.fwd_project(img, subset = 0)
