import numpy as np
import numpy.ctypeslib as npct
import math
import ctypes
import os

from .wrapper import joseph3d_fwd, joseph3d_fwd_tof_lm, joseph3d_fwd_tof_sino
from .wrapper import joseph3d_back, joseph3d_back_tof_lm, joseph3d_back_tof_sino

try:
    import cupy as cp
except:
    import numpy as np


class SinogramProjector:
    """ TOF and non TOF 3D sinogram Joseph forward and back projector

        Parameters
        ----------
        scanner : RegularPolygonPETScanner
          an object containing the parameter of the cylindrical PET scanner

        sino_params: PETSinogramParameters
          object containing the description of the sinogram parameters

        img_dim : array like (3 integer elements)
          containing the 3 dimensions of the image to be projected

        nontof_shapeubsets: int
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
    """

    def __init__(self,
                 scanner,
                 sino_params,
                 img_dim,
                 nsubsets=1,
                 tof=False,
                 img_origin=None,
                 voxsize=np.ones(3),
                 random_subset_angles=False,
                 subset_dir=None,
                 sigma_tof=60. / 2.35,
                 n_sigmas=3,
                 threadsperblock=64):

        self.scanner = scanner
        self.sino_params = sino_params
        self.all_views = np.arange(self.sino_params.nviews)
        self.__tof = tof

        if self.scanner._on_gpu:
            self._xp = cp
        else:
            self._xp = np

        # in case the projector is created as a TOF projector, the user can self.set_tof(False)
        # to also to non-TOF projections
        # to prohibit that a user calls self.set_tof(True) for a projector created in non-TOF mode
        # we have to store the original tof value
        self.was_created_as_tof_projector = tof

        self.img_dim = img_dim
        if not isinstance(self.img_dim, np.ndarray):
            self.img_dim = np.array(img_dim)
        self.img_dim = self.img_dim.astype(ctypes.c_int)

        self.nvox = np.prod(self.img_dim)
        self.voxsize = voxsize.astype(np.float32)

        if img_origin is None:
            self.img_origin = (-(self.img_dim / 2) + 0.5) * self.voxsize
        else:
            self.img_origin = img_origin
        self.img_origin = self.img_origin.astype(np.float32)

        # tof parameters
        self.sigma_tof = sigma_tof
        self.nsigmas = float(n_sigmas)
        self.tofbin_width = self.sino_params.tofbin_width
        self.ntofbins = self.sino_params.ntofbins

        # gpu parameters (not relevant when not run on gpu)
        self.threadsperblock = threadsperblock

        # sinogram related things
        self.random_subset_angles = random_subset_angles

        if subset_dir is None:
            self.subset_dir = np.where(
                sino_params.spatial_dim_order == 1)[0][0]
            if sino_params.tof_dim == 0:
                self.subset_dir += 1
        else:
            self.subset_dir = subset_dir

        self.init_subsets(nsubsets)

    # -----------------------------------------------------------------------------------------------
    def init_subsets(self, nsubsets):

        self.nsubsets = nsubsets

        self.subset_slices = []
        self.subset_sino_shapes = []
        self.nLORs = []

        if self.random_subset_angles:
            subset_table = self.all_views.copy()
            np.random.shuffle(subset_table)
            subset_table = subset_table.reshape(nsubsets, -1)

        for i in range(self.nsubsets):
            subset_slice = 4 * [slice(None, None, None)]
            if self.random_subset_angles:
                subset_slice[self.subset_dir] = subset_table[i, :]
            else:
                subset_slice[self.subset_dir] = slice(i, None, nsubsets)
            self.subset_slices.append(tuple(subset_slice))

            # if the projector was created as TOF projector, but is being used as non-TOF
            # projector by setting self.__tof = False, sino_params.shape is still the TOF sino shape
            if self.__tof:
                subset_shape = np.array(self.sino_params.shape)
            else:
                subset_shape = np.array(self.sino_params.nontof_shape)

            if i == (self.nsubsets - 1):
                subset_shape[self.subset_dir] -= (self.nsubsets - 1) * int(
                    np.ceil(subset_shape[self.subset_dir] / self.nsubsets))
            else:
                subset_shape[self.subset_dir] = int(
                    np.ceil(subset_shape[self.subset_dir] / self.nsubsets))

            self.nLORs.append(np.prod(subset_shape[:-1]).astype(np.int64))

            self.subset_sino_shapes.append(subset_shape)

    # -----------------------------------------------------------------------------------------------
    def get_subset_sino_coordinates(self, subset):
        # get the world coordiates for start and end point of all LORs in a subset

        subset_views = self.all_views[self.subset_slices[subset][
            self.subset_dir]]
        istart, iend = self.sino_params.get_view_crystal_indices(subset_views)

        ssh = tuple(self.subset_sino_shapes[subset][:-1]) + (3, )

        xstart = self.scanner.get_crystal_coordinates(istart.reshape(
            -1, 2)).reshape(ssh)
        xend = self.scanner.get_crystal_coordinates(iend.reshape(
            -1, 2)).reshape(ssh)

        return xstart, xend

    # -----------------------------------------------------------------------------------------------
    def set_tof(self, tof):
        if self.was_created_as_tof_projector:
            self.__tof = tof
            self.init_subsets(self.nsubsets)
        else:
            raise NotImplementedError(
                'set_tof() can be only called for projectors created in TOF mode'
            )

    def get_tof(self):
        return self.__tof

    # -----------------------------------------------------------------------------------------------
    def fwd_project_subset(self,
                           img,
                           subset,
                           tofcenter_offset=None,
                           sigma_tof_per_lor=None):

        if not img.dtype is self._xp.dtype('float32'):
            img = img.astype(self._xp.float32)

        xstart, xend = self.get_subset_sino_coordinates(subset)

        if self.__tof == False:
            # NONTOF fwd projection
            img_fwd = self._xp.zeros(int(self.nLORs[subset]),
                                     dtype=self._xp.float32)

            ok = joseph3d_fwd(xstart.ravel(),
                              xend.ravel(),
                              img.ravel(),
                              self.img_origin,
                              self.voxsize,
                              img_fwd,
                              self.nLORs[subset],
                              self.img_dim,
                              threadsperblock=self.threadsperblock)
        else:
            # TOF fwd projection
            if sigma_tof_per_lor is None:
                sigma_tof = self._xp.array([self.sigma_tof],
                                           dtype=self._xp.float32)
            else:
                sigma_tof = sigma_tof_per_lor.astype(self._xp.float32)

            if not isinstance(tofcenter_offset, self._xp.ndarray):
                tofcenter_offset = self._xp.zeros(1, dtype=self._xp.float32)

            img_fwd = self._xp.zeros(int(self.nLORs[subset]) * self.ntofbins,
                                     dtype=self._xp.float32)

            ok = joseph3d_fwd_tof_sino(xstart.ravel(),
                                       xend.ravel(),
                                       img.ravel(),
                                       self.img_origin,
                                       self.voxsize,
                                       img_fwd,
                                       self.nLORs[subset],
                                       self.img_dim,
                                       self.tofbin_width,
                                       sigma_tof.ravel(),
                                       tofcenter_offset.ravel(),
                                       self.nsigmas,
                                       self.ntofbins,
                                       threadsperblock=self.threadsperblock)

        img_fwd = img_fwd.reshape(self.subset_sino_shapes[subset])

        return img_fwd

    # -----------------------------------------------------------------------------------------------
    def fwd_project(self, img, **kwargs):

        if self.__tof:
            img_fwd = self._xp.zeros(self.sino_params.shape,
                                     dtype=self._xp.float32)
        else:
            img_fwd = self._xp.zeros(self.sino_params.nontof_shape,
                                     dtype=self._xp.float32)

        for i in range(self.nsubsets):
            img_fwd[self.subset_slices[i]] = self.fwd_project_subset(
                img, i, **kwargs)

        return img_fwd

    # -----------------------------------------------------------------------------------------------
    def back_project_subset(self,
                            sino,
                            subset,
                            tofcenter_offset=None,
                            sigma_tof_per_lor=None):

        if not sino.dtype is self._xp.dtype('float32'):
            sino = sino.astype(self._xp.float32)

        xstart, xend = self.get_subset_sino_coordinates(subset)

        back_img = self._xp.zeros(int(self.nvox), dtype=self._xp.float32)

        if self.__tof == False:
            # NONTOF back projection
            ok = joseph3d_back(xstart.ravel(),
                               xend.ravel(),
                               back_img,
                               self.img_origin,
                               self.voxsize,
                               sino.ravel(),
                               self.nLORs[subset],
                               self.img_dim,
                               threadsperblock=self.threadsperblock)
        else:
            # TOF back projection
            if sigma_tof_per_lor is None:
                sigma_tof = self._xp.array([self.sigma_tof],
                                           dtype=self._xp.float32)
            else:
                sigma_tof = sigma_tof_per_lor.astype(self._xp.float32)

            if not isinstance(tofcenter_offset, self._xp.ndarray):
                tofcenter_offset = self._xp.zeros(1, dtype=self._xp.float32)

            ok = joseph3d_back_tof_sino(xstart.ravel(),
                                        xend.ravel(),
                                        back_img,
                                        self.img_origin,
                                        self.voxsize,
                                        sino.ravel(),
                                        self.nLORs[subset],
                                        self.img_dim,
                                        self.tofbin_width,
                                        sigma_tof.ravel(),
                                        tofcenter_offset.ravel(),
                                        self.nsigmas,
                                        self.ntofbins,
                                        threadsperblock=self.threadsperblock)

        return back_img.reshape(self.img_dim)

    # -----------------------------------------------------------------------------------------------
    def back_project(self, sino, **kwargs):
        back_img = self._xp.zeros(self.img_dim, dtype=self._xp.float32)

        for i in range(self.nsubsets):
            back_img += self.back_project_subset(sino[self.subset_slices[i]],
                                                 i, **kwargs)

        return back_img

    # --------------------------------------------------------------------
    def fwd_project_lm(self,
                       img,
                       events,
                       tofcenter_offset=None,
                       sigma_tof_per_lor=None):

        if not img.dtype is self._xp.dtype('float32'):
            img = img.astype(self._xp.float32)

        nevents = events.shape[0]

        img_fwd = self._xp.zeros(nevents, dtype=self._xp.float32)

        xstart = self.scanner.get_crystal_coordinates(events[:, 0:2])
        xend = self.scanner.get_crystal_coordinates(events[:, 2:4])

        if self.__tof == False:
            # NONTOF fwd projection
            ok = joseph3d_fwd(xstart.ravel(),
                              xend.ravel(),
                              img.ravel(),
                              self.img_origin,
                              self.voxsize,
                              img_fwd,
                              nevents,
                              self.img_dim,
                              threadsperblock=self.threadsperblock)
        else:
            # TOF fwd projection
            if sigma_tof_per_lor is None:
                sigma_tof = self._xp.array([self.sigma_tof],
                                           dtype=self._xp.float32)
            else:
                sigma_tof = sigma_tof_per_lor.astype(self._xp.float32)

            if not isinstance(tofcenter_offset, self._xp.ndarray):
                tofcenter_offset = self._xp.zeros(1, dtype=self._xp.float32)

            tofbin = events[:, 4].astype(self._xp.int16)

            ok = joseph3d_fwd_tof_lm(xstart.ravel(),
                                     xend.ravel(),
                                     img.ravel(),
                                     self.img_origin,
                                     self.voxsize,
                                     img_fwd,
                                     nevents,
                                     self.img_dim,
                                     self.tofbin_width,
                                     sigma_tof.ravel(),
                                     tofcenter_offset,
                                     self.nsigmas,
                                     tofbin,
                                     threadsperblock=self.threadsperblock)

        return img_fwd

    # --------------------------------------------------------------------
    def back_project_lm(self,
                        values,
                        events,
                        tofcenter_offset=None,
                        sigma_tof_per_lor=None):

        if not values.dtype is self._xp.dtype('float32'):
            values = values.astype(self._xp.float32)

        nevents = events.shape[0]

        back_img = self._xp.zeros(int(self.nvox), dtype=self._xp.float32)

        xstart = self.scanner.get_crystal_coordinates(events[:, 0:2])
        xend = self.scanner.get_crystal_coordinates(events[:, 2:4])

        if self.__tof == False:
            # NONTOF back projection
            ok = joseph3d_back(xstart.ravel(),
                               xend.ravel(),
                               back_img,
                               self.img_origin,
                               self.voxsize,
                               values.ravel(),
                               nevents,
                               self.img_dim,
                               threadsperblock=self.threadsperblock)
        else:
            # TOF back projection
            if sigma_tof_per_lor is None:
                sigma_tof = self._xp.array([self.sigma_tof],
                                           dtype=self._xp.float32)
            else:
                sigma_tof = sigma_tof_per_lor.astype(self._xp.float32)

            if not isinstance(tofcenter_offset, self._xp.ndarray):
                tofcenter_offset = self._xp.zeros(1, dtype=self._xp.float32)

            tofbin = events[:, 4].astype(self._xp.int16)

            ok = joseph3d_back_tof_lm(xstart.ravel(),
                                      xend.ravel(),
                                      back_img,
                                      self.img_origin,
                                      self.voxsize,
                                      values.ravel(),
                                      nevents,
                                      self.img_dim,
                                      self.tofbin_width,
                                      sigma_tof.ravel(),
                                      tofcenter_offset,
                                      self.nsigmas,
                                      tofbin,
                                      threadsperblock=self.threadsperblock)

        return back_img.reshape(self.img_dim)
