import numpy as np
import scipy.ndimage as ndi

#---------------------------------------------------------------------------------
def pet_fwd_model(img, proj, attn_sino, sens_sino, isub = None, fwhm = 0):
  """PET forward model

  Parameters
  ----------

  img : 3D numpy array
    containing the activity image

  proj : projector
    geometrical TOF or non-TOF projector

  attn_sino : numpy array of shape [nrad, nanlges_per_subset,nplanes]
    containing the non-TOF subset attenuation sinogram

  sens_sino : numpy array of shape [nrad, nanlges_per_subset,nplanes]
    containing the non-TOF subset sensivity sinogram

  isub : int
    the subset number, if None the complete sinogram (no subsets) is projected

  fwhm : float, optional
    FWHM (voxels) of Gaussian filter applied to image before projection

  Returns
  -------

  numpy array of shape [nrad, nanlges_per_subset,nplanes,ntofbins] containing
  the forward projected image
  """

  if np.any(fwhm > 0):
    img = ndi.gaussian_filter(img, fwhm/2.35)

  if isub is None:
    sino = sens_sino*attn_sino*proj.fwd_project(img)
  else:
    sino = sens_sino*attn_sino*proj.fwd_project_subset(img, isub)

  return sino

#---------------------------------------------------------------------------------
def pet_fwd_model_lm(img, proj, subset_events, attn_list, sens_list, fwhm = 0):
  """PET listmode forward model

  Parameters
  ----------

  img : 3D numpy array
    containing the activity image

  proj : projector
    geometrical TOF or non-TOF projector

  subset_events : 2D numpy array of shape [nevents per subset, 5]
    detector and TOF coordinates of the events in the subset

  attn_list : 1D numpy array
    containing the attenuation factors for the events in the subset

  sens_list : 1D numpy array
    containing the sensitivity factors for the events in the subset

  fwhm : float, optional
    FWHM (voxels) of Gaussian filter applied to image before projection

  Returns
  -------

  numpy array of shape [nevents per subset] containing the forward projected image
  """

  if np.any(fwhm > 0):
    img = ndi.gaussian_filter(img, fwhm/2.35)

  fwd_list = sens_list*attn_list*proj.fwd_project_lm(img, subset_events)

  return fwd_list


#---------------------------------------------------------------------------------
def pet_back_model(subset_sino, proj, attn_sino, sens_sino, isub = None, fwhm = 0):
  """Adjoint of PET forward model (backward model)

  Parameters
  ----------

  subset_sino : numpy array of shape [nrad, nanlges_per_subset,nplanes,ntofbins]
    containing the subset sinogram to be backprojected

  proj : projector
    geometrical TOF orf non-TOF projector

  attn_sino : numpy array of shape [nrad, nanlges_per_subset,nplanes]
    containing the non-TOF attenuation subset sinogram

  sens_sino : numpy array of shape [nrad, nanlges_per_subset,nplanes]
    containing the non-TOF sensivity subset sinogram

  isub : int
    the subset number, if None the complete sinogram (no subsets) is projected

  fwhm : float, optional
    FWHM (voxels) of Gaussian filter applied to image after back projection

  Returns
  -------

  3D numpy array containing the back projected image
  """

  if isub is None:
    back_img = proj.back_project(sens_sino*attn_sino*subset_sino)
  else:
    back_img = proj.back_project_subset(sens_sino*attn_sino*subset_sino, subset = isub)

  if np.any(fwhm > 0):
    back_img = ndi.gaussian_filter(back_img, fwhm/2.35)

  return back_img

#---------------------------------------------------------------------------------
def pet_back_model_lm(lst, proj, subset_events, attn_list, sens_list, fwhm = 0):
  """Adjoint of listmode PET forward model (backward model)

  Parameters
  ----------

  lst : 1D numpy array
    containing the values for each event LOR to be backprojected

  proj : projector
    geometrical TOF or non-TOF projector

  subset_events : 2D numpy array of shape [nevents per subset, 5]
    detector and TOF coordinates of the events in the subset

  attn_list : 1D numpy array
    containing the attenuation factors for the events in the subset

  sens_list : 1D numpy array
    containing the sensitivity factors for the events in the subset

  fwhm : float, optional
    FWHM (voxels) of Gaussian filter applied to image before projection

  Returns
  -------

  3D numpy array containing the back projected image
  """

  back_img = proj.back_project_lm(sens_list*attn_list*lst, subset_events)

  if np.any(fwhm > 0):
    back_img = ndi.gaussian_filter(back_img, fwhm/2.35)

  return back_img

#------------------------------------------------------------------------------------------------------
class LMPETAcqModel:
  def __init__(self, proj, events, attn_list, sens_list, image_based_res_model = None):
    self.proj        = proj         # parllelproj PET projector
    self.events      = events       # numpy / cupy event 2D event array
    self.attn_list   = attn_list    # numpy / cupy 1D array with attenuation values 
    self.sens_list   = sens_list    # numpy / cupy 1D array with sensitivity values

    self.image_based_res_model = image_based_res_model # image-based resolution model

  def forward(self, img, isub = 0, nsubsets = 1):
    if self.image_based_res_model is not None:
      img = self.image_based_res_model.forward(img)

    ss = slice(isub, None, nsubsets)
    img_fwd = self.sens_list[ss]*self.attn_list[ss]*self.proj.fwd_project_lm(img, self.events[ss])

    return img_fwd

  def adjoint(self, values, isub = 0, nsubsets = 1):
    ss = slice(isub, None, nsubsets)
    back_img = self.proj.back_project_lm(self.sens_list[ss]*self.attn_list[ss]*values, self.events[ss])

    if self.image_based_res_model is not None:
      back_img = self.image_based_res_model.adjoint(back_img)

    return back_img

#------------------------------------------------------------------------------------------------------
class ImageBasedResolutionModel:
  def __init__(self, fwhm, ndimage_module = None):
    self.fwhm  = fwhm         # numpy array with Gauss FWHM (in voxels) for resolution model

    if ndimage_module is None:
      self._ndi = ndi
    else:
      self._ndi = ndimage_module

  def forward(self, img):
    return self._ndi.gaussian_filter(img, self.fwhm / 2.35)

  def adjoint(self, img):
    return self._ndi.gaussian_filter(img, self.fwhm / 2.35)
