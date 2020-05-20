import numpy as np
import numpy.ctypeslib as npct
import ctypes
import os

from .wrapper import joseph3d_fwd, joseph3d_fwd_tof_sino, joseph3d_back, joseph3d_back_tof_sino

class LMProjector:
  def __init__(self, scanner, img_dim, tof = False,
                     img_origin = None, voxsize = np.ones(3),
                     sigma_tof = 60., tofcenter_offset = 0, n_sigmas = 3,
                     threadsperblock = 64, ngpus = 0):

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
    self.sigma_tof = sigma_tof
    self.tofcenter_offset = tofcenter_offset
    self.nsigmas = n_sigmas

    # gpu parameters (not relevant when not run on gpu)
    self.threadsperblock = threadsperblock
    self.ngpus           = ngpus

    # check and cast data type to match libparallel proj
    if not isinstance(self.voxsize, ctypes.c_float):
      self.voxsize = self.voxsize.astype(ctypes.c_float)

    if not isinstance(self.img_origin, ctypes.c_float):
      self.img_origin = self.img_origin.astype(ctypes.c_float)

    if not isinstance(self.img_dim, ctypes.c_int):
      self.img_dim = self.img_dim.astype(ctypes.c_int)

  #--------------------------------------------------------------------
  def fwd_project(self, img, events):

    if not isinstance(img, ctypes.c_float):
      img = img.astype(ctypes.c_float)

    nevents = events.shape[0]

    img_fwd = np.zeros(nevents, dtype = ctypes.c_float)  

    xstart = self.scanner.get_crystal_coordinates(events[:,0:2])
    xend   = self.scanner.get_crystal_coordinates(events[:,2:4])

    if self.tof == False:
      ####### NONTOF fwd projection 
      ok = joseph3d_fwd(xstart.flatten(), xend.flatten(), 
                        img.flatten(), self.img_origin, self.voxsize, 
                        img_fwd, nevents, self.img_dim,
                        threadsperblock = self.threadsperblock, ngpus = self.ngpus) 

    return img_fwd  

  #--------------------------------------------------------------------
  def back_project(self, values, events):

    if not isinstance(values, ctypes.c_float):
      values = values.astype(ctypes.c_float)

    nevents = events.shape[0]

    back_img = np.zeros(self.nvox, dtype = ctypes.c_float)  

    xstart = self.scanner.get_crystal_coordinates(events[:,0:2])
    xend   = self.scanner.get_crystal_coordinates(events[:,2:4])

    if self.tof == False:
      ####### NONTOF back projection 
      ok = joseph3d_back(xstart.flatten(), xend.flatten(), 
                         back_img, self.img_origin, self.voxsize, 
                         values.flatten(), nevents, self.img_dim,
                         threadsperblock = self.threadsperblock, ngpus = self.ngpus) 

    return back_img.reshape(self.img_dim)

#-----------------------------------------------------------------------------------------

class SinogramProjector(LMProjector):

  def __init__(self, scanner, sino, img_dim, nsubsets = 1, tof = False,
                     img_origin = None, voxsize = np.ones(3),
                     sigma_tof = 60., tofcenter_offset = 0, n_sigmas = 3,
                     threadsperblock = 64, ngpus = 0):
    
    LMProjector.__init__(self, scanner, img_dim, tof = tof,
                         img_origin = img_origin, voxsize = voxsize,
                         sigma_tof = sigma_tof, tofcenter_offset = tofcenter_offset, 
                         n_sigmas = n_sigmas, threadsperblock = threadsperblock, ngpus = ngpus)

    self.sino    = sino

    self.all_views = np.arange(self.sino.nviews)

    # get the crystals IDs for all views
    self.istart, self.iend = self.sino.get_view_crystal_indices(self.all_views)

    # get the world coordiates for all view
    self.xstart = self.scanner.get_crystal_coordinates(
                    self.istart.reshape(-1,2)).reshape((self.sino.nrad, self.sino.nviews, 
                                                        self.sino.nplanes,3))
    self.xend = self.scanner.get_crystal_coordinates(
                  self.iend.reshape(-1,2)).reshape((self.sino.nrad, self.sino.nviews, 
                                                    self.sino.nplanes,3))

    self.init_subsets(nsubsets)

  #-----------------------------------------------------------------------------------------------
  def init_subsets(self, nsubsets, subset_dir = 1):

    self.nsubsets   = nsubsets
    self.subset_dir = subset_dir

    self.subset_slices      = []
    self.subset_sino_shapes = []
    self.nLORs              = []

    for i in range(self.nsubsets):
      subset_slice = 4*[slice(None,None,None)]
      subset_slice[self.subset_dir] = slice(i,None,nsubsets)
      self.subset_slices.append(tuple(subset_slice))
      self.nLORs.append(np.prod(self.xstart[self.subset_slices[i]].shape[:-1]).astype(np.int32))

      subset_shape = np.array(self.sino.shape)

      if i == (self.nsubsets - 1):
        subset_shape[self.subset_dir] -= (self.nsubsets - 1)*int(np.ceil(subset_shape[self.subset_dir]/self.nsubsets))
      else:
        subset_shape[self.subset_dir] = int(np.ceil(subset_shape[self.subset_dir]/self.nsubsets))

      self.subset_sino_shapes.append(subset_shape)

  #-----------------------------------------------------------------------------------------------
  def fwd_project(self, img, subset = 0):

    if not isinstance(img, ctypes.c_float):
      img = img.astype(ctypes.c_float)

    subset_slice = self.subset_slices[subset]

    img_fwd = np.zeros(self.nLORs[subset]*self.sino.ntofbins, dtype = ctypes.c_float)  

    if self.tof == False:
      ####### NONTOF fwd projection 
      ok = joseph3d_fwd(self.xstart[subset_slice].flatten(), 
                        self.xend[subset_slice].flatten(), 
                        img.flatten(), self.img_origin, self.voxsize, 
                        img_fwd, self.nLORs[subset], self.img_dim,
                        threadsperblock = self.threadsperblock, ngpus = self.ngpus) 
    else:
      ####### TOF fwd projection 
      if not isinstance(self.sigma_tof, np.ndarray):
        sigma_tof = np.full(self.nLORs[subset], self.sigma_tof, dtype = ctypes.c_float)
      else:
        sigma_tof = self.sigma_tof[subset_slice[:-1]].flatten().astype(ctypes.c_float)

      if not isinstance(self.tofcenter_offset, np.ndarray):
        tofcenter_offset = np.full(self.nLORs[subset], self.tofcenter_offset, dtype = ctypes.c_float)
      else:
        tofcenter_offset = self.tofcenter_offset[subset_slice[:-1]].flatten().astype(ctypes.c_float)

      ok = joseph3d_fwd_tof_sino(self.xstart[subset_slice].flatten(), 
                                 self.xend[subset_slice].flatten(), 
                                 img.flatten(), self.img_origin, self.voxsize, 
                                 img_fwd, self.nLORs[subset], self.img_dim,
                                 self.sino.ntofbins, self.sino.tofbin_width, 
                                 sigma_tof, tofcenter_offset, self.nsigmas,
                                 threadsperblock = self.threadsperblock, ngpus = self.ngpus) 

    return img_fwd.reshape(self.subset_sino_shapes[subset])

  #-----------------------------------------------------------------------------------------------
  def back_project(self, sino, subset = 0):

    if not isinstance(sino, ctypes.c_float):
      sino = sino.astype(ctypes.c_float)

    subset_slice = self.subset_slices[subset]

    back_img = np.zeros(self.nvox, dtype = ctypes.c_float)  

    if self.tof == False:
      ####### NONTOF back projection 
      ok = joseph3d_back(self.xstart[subset_slice].flatten(), 
                         self.xend[subset_slice].flatten(), 
                         back_img, self.img_origin, self.voxsize, 
                         sino.flatten(), self.nLORs[subset], self.img_dim,
                         threadsperblock = self.threadsperblock, ngpus = self.ngpus) 

    else:
      ####### TOF back projection 
      if not isinstance(self.sigma_tof, np.ndarray):
        sigma_tof = np.full(self.nLORs[subset], self.sigma_tof, dtype = ctypes.c_float)
      else:
        sigma_tof = self.sigma_tof[subset_slice[:-1]].flatten().astype(ctypes.c_float)

      if not isinstance(self.tofcenter_offset, np.ndarray):
        tofcenter_offset = np.full(self.nLORs[subset], self.tofcenter_offset, dtype = ctypes.c_float)
      else:
        tofcenter_offset = self.tofcenter_offset[subset_slice[:-1]].flatten().astype(ctypes.c_float)

      ok = joseph3d_back_tof_sino(self.xstart[subset_slice].flatten(), 
                                  self.xend[subset_slice].flatten(), 
                                  back_img, self.img_origin, self.voxsize, 
                                  sino.flatten(), self.nLORs[subset], self.img_dim,
                                  self.sino.ntofbins, self.sino.tofbin_width, 
                                  sigma_tof, tofcenter_offset, self.nsigmas,
                                  threadsperblock = self.threadsperblock, ngpus = self.ngpus) 

    return back_img.reshape(self.img_dim)
