import numpy as np
import scipy.ndimage as ndi


# ------------------------------------------------------------------------------------------------------
class PETAcqModel:

    def __init__(self, proj, attn_sino, sens_sino, image_based_res_model=None):
        self.proj = proj  # parllelproj PET projector
        self.attn_sino = attn_sino  # numpy / cupy array with attenuation sinogram
        self.sens_sino = sens_sino  # numpy / cupy array with sensitivity sinogram

        self.image_based_res_model = image_based_res_model  # image-based resolution model

    def forward(self, img, isub=None):
        if self.image_based_res_model is not None:
            img = self.image_based_res_model.forward(img)

        if isub is None:
            img_fwd = self.sens_sino * self.attn_sino * self.proj.fwd_project(
                img)
        else:
            ss = self.proj.subset_slices[isub]
            img_fwd = self.sens_sino[ss] * self.attn_sino[
                ss] * self.proj.fwd_project_subset(img, isub)

        return img_fwd

    def adjoint(self, sino, isub=None):
        if isub is None:
            back_img = self.proj.back_project(self.sens_sino * self.attn_sino *
                                              sino)
        else:
            ss = self.proj.subset_slices[isub]
            back_img = self.proj.back_project_subset(
                self.sens_sino[ss] * self.attn_sino[ss] * sino, isub)

        if self.image_based_res_model is not None:
            back_img = self.image_based_res_model.adjoint(back_img)

        return back_img


# ------------------------------------------------------------------------------------------------------
class LMPETAcqModel:

    def __init__(self,
                 proj,
                 events,
                 attn_list,
                 sens_list,
                 image_based_res_model=None):
        self.proj = proj  # parllelproj PET projector
        self.events = events  # numpy / cupy event 2D event array
        self.attn_list = attn_list  # numpy / cupy 1D array with attenuation values
        self.sens_list = sens_list  # numpy / cupy 1D array with sensitivity values

        self.image_based_res_model = image_based_res_model  # image-based resolution model

    def forward(self, img, isub=0, nsubsets=1):
        if self.image_based_res_model is not None:
            img = self.image_based_res_model.forward(img)

        ss = slice(isub, None, nsubsets)
        img_fwd = self.sens_list[ss] * self.attn_list[
            ss] * self.proj.fwd_project_lm(img, self.events[ss])

        return img_fwd

    def adjoint(self, values, isub=0, nsubsets=1):
        ss = slice(isub, None, nsubsets)
        back_img = self.proj.back_project_lm(
            self.sens_list[ss] * self.attn_list[ss] * values, self.events[ss])

        if self.image_based_res_model is not None:
            back_img = self.image_based_res_model.adjoint(back_img)

        return back_img


# ------------------------------------------------------------------------------------------------------
class ImageBasedResolutionModel:

    def __init__(self, fwhm, ndimage_module=None):
        # numpy array with Gauss FWHM (in voxels) for resolution model
        self.fwhm = fwhm

        if ndimage_module is None:
            self._ndi = ndi
        else:
            self._ndi = ndimage_module

    def forward(self, img):
        return self._ndi.gaussian_filter(img, self.fwhm / 2.35)

    def adjoint(self, img):
        return self._ndi.gaussian_filter(img, self.fwhm / 2.35)


# ------------------------------------------------------------------------------------------------------
class GradientNorm:
    """ 
  norm of a gradient field

  Parameters
  ----------

  name : str
    name of the norm
    'l2_l1' ... mixed L2/L1 (sum of pointwise Euclidean norms in every voxel)
    'l2_sq' ... squared l2 norm (sum of pointwise squared Euclidean norms in every voxel)

  beta : float
    factor multiplied to the norm (default 1)
  """

    def __init__(self, xp, name='l2_l1'):
        self.name = name
        self._xp = xp

        if not self.name in ['l2_l1', 'l2_sq']:
            raise NotImplementedError

    def __call__(self, x):
        if self.name == 'l2_l1':
            n = self._xp.linalg.norm(x, axis=0).sum()
        elif self.name == 'l2_sq':
            n = (x**2).sum()

        return n

    def prox_convex_dual(self, x, sigma=None):
        """ proximal operator of the convex dual of the norm
    """
        if self.name == 'l2_l1':
            gnorm = self._xp.linalg.norm(x, axis=0)
            r = x / self._xp.clip(gnorm, 1, None)
        elif self.name == 'l2_sq':
            r = x / (1 + sigma)

        return r


# ------------------------------------------------------------------------------------------------------
class GradientOperator:
    """
  (directional) gradient operator and its adjoint in 2,3 or 4 dimensions
  using finite forward / backward differences

  Parameters
  ----------

  joint_gradient_field : numpy array
    if given, only the gradient component perpenticular to the directions 
    given in the joint gradient field are specified (default None)
  """

    def __init__(self, xp, joint_grad_field=None):
        self._xp = xp

        # e is the normalized joint gradient field that
        # we are only interested in the gradient component
        # perpendicular to it
        self.e = None

        if joint_grad_field is not None:
            norm = self._xp.linalg.norm(joint_grad_field, axis=0)
            inds = self._xp.where(norm > 0)
            self.e = joint_grad_field.copy()

            for i in range(self.e.shape[0]):
                self.e[i,
                       ...][inds] = joint_grad_field[i, ...][inds] / norm[inds]

    def forward(self, x):
        g = []
        for i in range(x.ndim):
            g.append(self._xp.diff(x, axis=i, append=self._xp.take(x, [-1],
                                                                   i)))
        g = self._xp.array(g)

        if self.e is not None:
            g = g - (g * self.e).sum(0) * self.e

        return g

    def adjoint(self, y):
        d = self._xp.zeros(y[0, ...].shape, dtype=y.dtype)

        if self.e is not None:
            y2 = y - (y * self.e).sum(0) * self.e
        else:
            y2 = y

        for i in range(y.shape[0]):
            d -= self._xp.diff(y2[i, ...],
                               axis=i,
                               prepend=self._xp.take(y2[i, ...], [0], i))

        return d


# ------------------------------------------------------------------------------------------------------
class GradientBasedPrior:

    def __init__(self, gradient_operator, gradient_norm, beta=1.):
        self.gradient_operator = gradient_operator
        self.gradient_norm = gradient_norm
        self.beta = beta

    def __call__(self, x):
        return float(self.beta *
                     self.gradient_norm(self.gradient_operator.forward(x)))
