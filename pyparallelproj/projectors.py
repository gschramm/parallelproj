import numpy as np
import numpy.ctypeslib as npct
import ctypes
import os

from .wrapper import joseph3d_fwd, joseph3d_fwd_tof, joseph3d_back, joseph3d_back_tof

class LMProjector:
  """ TOF and non TOF 3D listmode Joseph forward and back projector

  Parameters
  ----------
  scanner : RegularPolygonPETScanner
    an object containing the parameter of the cylindrical PET scanner

  img_dim : array like (3 integer elements)
    containing the 3 dimensions of the image to be projected

  tof : bool
    whether to do TOF or non-TOF projections
    Default: False

  img_origin : 3 element numpy float32 array
    containing the image origin (the world coordinates of voxel 0,0,0)
    Default: None which means (-(img_dim/2) + 0.5)*voxsize

   voxsize: 3 element numpy float32 array
     containing the voxel size (same units as scanner geometry description)
     Default: np.ones(3)

   tofbin_width: float
     width of a LM tof bin in spatial units (same as scanner geometry description)
     Default: None

   sigma_tof : float
     standard deviation of the Gaussian TOF kernel in spatial units
     Default: 60/2.35 (FWHM of 6cm, ca 400ps coincidence timing resolution)

   n_sigmas : int
     number of standard deviations used to trunacte the TOF kernel.
     Default: 3
   
   threadsperblock: int
     threads per block to use on a CUDA GPU
     Default: 64

   ngpus: int 
     number of GPUs to use
     0 means use CPU and openmp. 1 means 1 GPU, 2 means 2 interconnected GPUS ...
     -1 means use CUDA to detect all available GPUs.
     Default: 0
  """
  def __init__(self, scanner, img_dim, tof = False,
                     img_origin = None, voxsize = np.ones(3, dtype = np.float32),
                     tofbin_width = None, sigma_tof = 60./2.35,
                     n_sigmas = 3., threadsperblock = 64, ngpus = 0):

    self.scanner = scanner
    
    self.tof      = tof

    self.img_dim = img_dim
    if not isinstance(self.img_dim, np.ndarray):
      self.img_dim = np.array(img_dim)

    self.nvox    = np.prod(self.img_dim)
    self.voxsize = voxsize

    if img_origin is None:
      self.img_origin = (-(self.img_dim / 2) +  0.5) * self.voxsize
    else:
      self.img_origin = img_origin

    # tof parameters
    self.sigma_tof        = sigma_tof
    self.tofbin_width     = tofbin_width
    self.nsigmas          = float(n_sigmas)

    # gpu parameters (not relevant when not run on gpu)
    self.threadsperblock = threadsperblock
    self.ngpus           = ngpus

    self.voxsize    = self.voxsize.astype(ctypes.c_float)
    self.img_origin = self.img_origin.astype(ctypes.c_float)
    self.img_dim    = self.img_dim.astype(ctypes.c_int)

  #--------------------------------------------------------------------
  def fwd_project(self, img, events, tofcenter_offset = None, sigma_tof_per_lor = None):

    if not isinstance(img, ctypes.c_float):
      img = img.astype(ctypes.c_float)

    nevents = events.shape[0]

    img_fwd = np.zeros(nevents, dtype = ctypes.c_float)  

    xstart = self.scanner.get_crystal_coordinates(events[:,0:2])
    xend   = self.scanner.get_crystal_coordinates(events[:,2:4])

    if self.tof == False:
      ####### NONTOF fwd projection 
      ok = joseph3d_fwd(xstart.ravel(), xend.ravel(), 
                        img.ravel(), self.img_origin, self.voxsize, 
                        img_fwd, nevents, self.img_dim,
                        threadsperblock = self.threadsperblock, ngpus = self.ngpus) 
    else:
      ####### TOF fwd projection 
      if sigma_tof_per_lor is None:
        sigma_tof = np.full(nevents, self.sigma_tof, dtype = ctypes.c_float)
      else:
        sigma_tof = sigma_tof_per_lor.astype(ctypes.c_float)

      if not isinstance(tofcenter_offset, np.ndarray):
        tofcenter_offset = np.zeros(nevents, dtype = ctypes.c_float)

      tofbin = events[:,4].astype(ctypes.c_short)

      ok = joseph3d_fwd_tof(xstart.ravel(), xend.ravel(), 
                            img.ravel(), self.img_origin, self.voxsize, 
                            img_fwd, nevents, self.img_dim,
                            self.tofbin_width, sigma_tof.ravel(), tofcenter_offset, self.nsigmas,
                            tofbin, threadsperblock = self.threadsperblock, 
                            ngpus = self.ngpus, lm = True) 

    return img_fwd  

  #--------------------------------------------------------------------
  def back_project(self, values, events, tofcenter_offset = None, sigma_tof_per_lor = None):

    if not isinstance(values, ctypes.c_float):
      values = values.astype(ctypes.c_float)

    nevents = events.shape[0]

    back_img = np.zeros(self.nvox, dtype = ctypes.c_float)  

    xstart = self.scanner.get_crystal_coordinates(events[:,0:2])
    xend   = self.scanner.get_crystal_coordinates(events[:,2:4])

    if self.tof == False:
      ####### NONTOF back projection 
      ok = joseph3d_back(xstart.ravel(), xend.ravel(), 
                         back_img, self.img_origin, self.voxsize, 
                         values.ravel(), nevents, self.img_dim,
                         threadsperblock = self.threadsperblock, ngpus = self.ngpus) 
    else:
      ####### TOF back projection 
      if sigma_tof_per_lor is None:
        sigma_tof = np.full(nevents, self.sigma_tof, dtype = ctypes.c_float)
      else:
        sigma_tof = sigma_tof_per_lor.astype(ctypes.c_float)

      if not isinstance(tofcenter_offset, np.ndarray):
        tofcenter_offset = np.zeros(nevents, dtype = ctypes.c_float)

      tofbin = events[:,4].astype(ctypes.c_short)

      ok = joseph3d_back_tof(xstart.ravel(), xend.ravel(), 
                             back_img, self.img_origin, self.voxsize, 
                             values.ravel(), nevents, self.img_dim,
                             self.tofbin_width, sigma_tof.ravel(), tofcenter_offset, self.nsigmas, 
                             tofbin, threadsperblock = self.threadsperblock, 
                             ngpus = self.ngpus, lm = True) 


    return back_img.reshape(self.img_dim)


#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------


class SinogramProjector(LMProjector):
  """ TOF and non TOF 3D sinogram Joseph forward and back projector

  Parameters
  ----------
  scanner : RegularPolygonPETScanner
    an object containing the parameter of the cylindrical PET scanner

  sino_params: PETSinogramParameters
    object containing the description of the sinogram parameters

  img_dim : array like (3 integer elements)
    containing the 3 dimensions of the image to be projected

  nsubsets: int
    Number of subsets to be used.
    Default: 1

  tof : bool
    whether to do TOF or non-TOF projections
    Default: False

  img_origin : 3 element numpy float32 array
    containing the image origin (the world coordinates of voxel 0,0,0)
    Default: None which means (-(img_dim/2) + 0.5)*voxsize

   voxsize: 3 element numpy float32 array
     containing the voxel size (same units as scanner geometry description)
     Default: np.ones(3)

   random_subset_angles: bool
     whether to use random or "regular" angular sampling for angular subsets

   subset_dir : int
     dimension along which to sample subsets
     default: angular dimension, deduced from sino_params

   sigma_tof : float
     standard deviation of the Gaussian TOF kernel in spatial units
     Default: 60/2.35 (FWHM of 6cm, ca 400ps coincidence timing resolution)

   n_sigmas : int
     number of standard deviations used to trunacte the TOF kernel.
     Default: 3
   
   threadsperblock: int
     threads per block to use on a CUDA GPU
     Default: 64

   ngpus: int 
     number of GPUs to use
     0 means use CPU and openmp. 1 means 1 GPU, 2 means 2 interconnected GPUS ...
     -1 means use CUDA to detect all available GPUs.
     Default: 0
  """

  def __init__(self, scanner, sino_params, img_dim, nsubsets = 1, tof = False,
                     img_origin = None, voxsize = np.ones(3), random_subset_angles = False,
                     subset_dir = None, sigma_tof = 60./2.35, n_sigmas = 3,
                     threadsperblock = 64, ngpus = 0):
    
    LMProjector.__init__(self, scanner, img_dim, tof = tof,
                         img_origin = img_origin, voxsize = voxsize,
                         tofbin_width = sino_params.tofbin_width,
                         sigma_tof = sigma_tof, 
                         n_sigmas = n_sigmas, threadsperblock = threadsperblock, ngpus = ngpus)

    self.sino_params = sino_params
    self.ntofbins    = self.sino_params.ntofbins

    self.all_views = np.arange(self.sino_params.nviews)

    # get the crystals IDs for all views
    self.istart, self.iend = self.sino_params.get_view_crystal_indices(self.all_views)

    # get the world coordiates for all views
    self.xstart = self.scanner.get_crystal_coordinates(
                    self.istart.reshape(-1,2)).reshape(self.sino_params.spatial_shape + (3,))
    self.xend = self.scanner.get_crystal_coordinates(
                  self.iend.reshape(-1,2)).reshape(self.sino_params.spatial_shape + (3,))

    self.random_subset_angles = random_subset_angles

    if subset_dir is None:
      self.subset_dir = np.where(sino_params.spatial_dim_order == 1)[0][0]
      if sino_params.tof_dim == 0:
        self.subset_dir += 1
    else:
      self.subset_dir = subset_dir

    self.init_subsets(nsubsets)

  #-----------------------------------------------------------------------------------------------
  def init_subsets(self, nsubsets):

    self.nsubsets   = nsubsets

    self.subset_slices      = []
    self.subset_sino_shapes = []
    self.nLORs              = []

    if self.random_subset_angles:
      subset_table = self.all_views.copy()
      np.random.shuffle(subset_table)
      subset_table = subset_table.reshape(nsubsets,-1)

    for i in range(self.nsubsets):
      subset_slice = 4*[slice(None,None,None)]
      if self.random_subset_angles:
        subset_slice[self.subset_dir] = subset_table[i,:]
      else:
        subset_slice[self.subset_dir] = slice(i,None,nsubsets)
      self.subset_slices.append(tuple(subset_slice))
      self.nLORs.append(np.prod(self.xstart[self.subset_slices[i]].shape[:-1]).astype(np.int64))

      subset_shape = np.array(self.sino_params.shape)

      if i == (self.nsubsets - 1):
        subset_shape[self.subset_dir] -= (self.nsubsets - 1)*int(np.ceil(subset_shape[self.subset_dir]/self.nsubsets))
      else:
        subset_shape[self.subset_dir] = int(np.ceil(subset_shape[self.subset_dir]/self.nsubsets))

      self.subset_sino_shapes.append(subset_shape)

  #-----------------------------------------------------------------------------------------------
  def fwd_project(self, img, subset = 0, tofcenter_offset = None, sigma_tof_per_lor = None):

    if not isinstance(img, ctypes.c_float):
      img = img.astype(ctypes.c_float)

    subset_slice = self.subset_slices[subset]

    img_fwd = np.zeros(self.nLORs[subset]*self.ntofbins, dtype = ctypes.c_float)  

    if self.tof == False:
      ####### NONTOF fwd projection 
      ok = joseph3d_fwd(self.xstart[subset_slice].ravel(), 
                        self.xend[subset_slice].ravel(), 
                        img.ravel(), self.img_origin, self.voxsize, 
                        img_fwd, self.nLORs[subset], self.img_dim,
                        threadsperblock = self.threadsperblock, ngpus = self.ngpus) 
    else:
      ####### TOF fwd projection 
      if sigma_tof_per_lor is None:
        sigma_tof = np.full(self.nLORs[subset], self.sigma_tof, dtype = ctypes.c_float)
      else:
        sigma_tof = sigma_tof_per_lor.astype(ctypes.c_float)

      if not isinstance(tofcenter_offset, np.ndarray):
        tofcenter_offset = np.zeros(self.nLORs[subset], dtype = ctypes.c_float)

      ok = joseph3d_fwd_tof(self.xstart[subset_slice].ravel(), 
                            self.xend[subset_slice].ravel(), 
                            img.ravel(), self.img_origin, self.voxsize, 
                            img_fwd, self.nLORs[subset], self.img_dim,
                            self.tofbin_width, sigma_tof.ravel(), tofcenter_offset.ravel(), 
                            self.nsigmas, self.ntofbins, 
                            threadsperblock = self.threadsperblock, ngpus = self.ngpus, lm = False) 

    return img_fwd.reshape(self.subset_sino_shapes[subset])

  #-----------------------------------------------------------------------------------------------
  def back_project(self, sino, subset = 0, tofcenter_offset = None, sigma_tof_per_lor = None):

    if not isinstance(sino, ctypes.c_float):
      sino = sino.astype(ctypes.c_float)

    subset_slice = self.subset_slices[subset]

    back_img = np.zeros(self.nvox, dtype = ctypes.c_float)  

    if self.tof == False:
      ####### NONTOF back projection 
      ok = joseph3d_back(self.xstart[subset_slice].ravel(), 
                         self.xend[subset_slice].ravel(), 
                         back_img, self.img_origin, self.voxsize, 
                         sino.ravel(), self.nLORs[subset], self.img_dim,
                         threadsperblock = self.threadsperblock, ngpus = self.ngpus) 

    else:
      ####### TOF back projection 
      if sigma_tof_per_lor is None:
        sigma_tof = np.full(self.nLORs[subset], self.sigma_tof, dtype = ctypes.c_float)
      else:
        sigma_tof = sigma_tof_per_lor.astype(ctypes.c_float)

      if not isinstance(tofcenter_offset, np.ndarray):
        tofcenter_offset = np.zeros(self.nLORs[subset], dtype = ctypes.c_float)

      ok = joseph3d_back_tof(self.xstart[subset_slice].ravel(), 
                             self.xend[subset_slice].ravel(), 
                             back_img, self.img_origin, self.voxsize, 
                             sino.ravel(), self.nLORs[subset], self.img_dim,
                             self.tofbin_width, sigma_tof.ravel(), tofcenter_offset.ravel(), 
                             self.nsigmas, self.ntofbins, 
                             threadsperblock = self.threadsperblock, ngpus = self.ngpus, lm = False) 

    return back_img.reshape(self.img_dim)
