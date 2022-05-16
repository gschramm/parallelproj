# minimal example showing how to use raw (external) CUDA kernels with cupy
#
# Aim: unerstand how to load and execute a raw kernel based on addition of two arrays

import pyparallelproj as ppp
from pyparallelproj.phantoms import ellipse2d_phantom
import numpy as np
import math
import argparse
import time

#-------------------------------------------------------------

import cupy as cp

# load a kernel defined in a external file
with open('../cuda/src/projector_kernels.cu','r') as f:
  lines = f.read()
  joseph3d_fwd_cuda_kernel  = cp.RawKernel(lines, 'joseph3d_fwd_cuda_kernel')
  joseph3d_back_cuda_kernel = cp.RawKernel(lines, 'joseph3d_back_cuda_kernel')

#-------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--img_mem_order', help = 'memory layout for image', default = 'C',
                                       choices = ['C','F'])
parser.add_argument('--sino_dim_order', help = 'axis order in sinogram', default = ['0','1','2'],
                     nargs = '+')
parser.add_argument('--voxsize', help = '3 voxel sizes (mm)', default = ['2','2','2'], nargs = '+')
parser.add_argument('--threads_per_block', help = 'threads per block',     default = 64,  type = int)

args = parser.parse_args()

#---------------------------------------------------------------------------------

nsubsets          = 1
threads_per_block = args.threads_per_block
img_mem_order     = args.img_mem_order
subset            = 0
tof               = False
ntofbins          = 1
voxsize           = np.array(args.voxsize, dtype = np.float32)
fov_mm            = 600

spatial_dim_order = np.array(args.sino_dim_order, dtype = int)


#---------------------------------------------------------------------------------

# setup a scanner
scanner = ppp.RegularPolygonPETScanner(ncrystals_per_module = np.array([16,9]),
                                       nmodules             = np.array([28,5]))

# setup a test image
n0      = int(fov_mm / voxsize[0])
n1      = int(fov_mm / voxsize[1])
n2      = max(1,int((scanner.xc2.max() - scanner.xc2.min()) / voxsize[2]))


# setup an ellipse image
tmp =  ellipse2d_phantom(n = n0).T
img = np.zeros((n0,n1,n2), dtype = np.float32, order = img_mem_order)

for i in range(n2):
  img[:,:,i] = tmp
img_origin = (-(np.array(img.shape) / 2) +  0.5) * voxsize

# generate sinogram parameters and the projector
sino_params = ppp.PETSinogramParameters(scanner, ntofbins = ntofbins, tofbin_width = 23.,
                                        spatial_dim_order = spatial_dim_order)
proj        = ppp.SinogramProjector(scanner, sino_params, img.shape, nsubsets = nsubsets, 
                                    voxsize = voxsize, img_origin = img_origin,
                                    tof = False, sigma_tof = 60./2.35, n_sigmas = 3.,
                                    threadsperblock = threads_per_block)

#---------------------------------------------------------------------------------

xstart, xend =  proj.get_subset_sino_coordinates(subset)
xstart = xstart.ravel()
xend   = xend.ravel()

img_ravel = img.ravel(order = img_mem_order)
subset_nLORs = proj.nLORs[subset]

#-----------------------------------------------------------------------------------

# send arrays to device

xstart_d      = cp.asarray(xstart)
xend_d        = cp.asarray(xend)
img_ravel_d   = cp.asarray(img_ravel)
bimg_ravel_d  = cp.zeros(img_ravel_d.shape[0], dtype = cp.float32)
img_fwd_d     = cp.zeros(int(subset_nLORs), dtype = cp.float32)
img_origin_d  = cp.asarray(proj.img_origin)
voxsize_d     = cp.asarray(proj.voxsize)
img_dim_d     = cp.array(proj.img_dim)

blocks_per_grid   = math.ceil(subset_nLORs/threads_per_block) 

start_gpu = cp.cuda.Event()
end_gpu   = cp.cuda.Event()
start_gpu.record()
start_cpu = time.perf_counter()

joseph3d_fwd_cuda_kernel((blocks_per_grid,), (threads_per_block,), (xstart_d, xend_d, img_ravel_d, img_origin_d, voxsize_d, img_fwd_d, subset_nLORs, img_dim_d))

joseph3d_back_cuda_kernel((blocks_per_grid,), (threads_per_block,), (xstart_d, xend_d, bimg_ravel_d, img_origin_d, voxsize_d, img_fwd_d, subset_nLORs, img_dim_d))

end_cpu = time.perf_counter()
end_gpu.record()
end_gpu.synchronize()
t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
t_cpu = end_cpu - start_cpu

#-----------------------------------------------------------------------------------

img_fwd = cp.asnumpy(img_fwd_d).reshape(proj.subset_sino_shapes[subset])
bimg    = cp.asnumpy(bimg_ravel_d).reshape((n0,n1,n2))
