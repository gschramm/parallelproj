import numpy as np
from scipy.ndimage import gaussian_filter

#---------------------------------------------------------------------------------
def pet_fwd_model(img, proj, attn_sino, sens_sino, isub, fwhm = 0):
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
    the subset number

  fwhm : float, optional
    FWHM (voxels) of Gaussian filter applied to image before projection

  Returns
  -------

  numpy array of shape [nrad, nanlges_per_subset,nplanes,ntofbins] containing
  the forward projected image
  """

  if np.any(fwhm > 0):
    img = gaussian_filter(img, fwhm/2.35)

  sino = sens_sino*attn_sino*proj.fwd_project(img, subset = isub)

  return sino

#---------------------------------------------------------------------------------
def pet_fwd_model_lm(img, lmproj, subset_events, attn_list, sens_list, fwhm = 0):
  """PET listmode forward model

  Parameters
  ----------

  img : 3D numpy array
    containing the activity image

  lmproj : projector
    geometrical listmode TOF or non-TOF projector

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
    img = gaussian_filter(img, fwhm/2.35)

  fwd_list = sens_list*attn_list*lmproj.fwd_project(img, subset_events)

  return fwd_list


#---------------------------------------------------------------------------------
def pet_back_model(subset_sino, proj, attn_sino, sens_sino, isub, fwhm = 0):
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
    the subset number

  fwhm : float, optional
    FWHM (voxels) of Gaussian filter applied to image after back projection

  Returns
  -------

  3D numpy array containing the back projected image
  """

  back_img = proj.back_project(sens_sino*attn_sino*subset_sino, subset = isub)

  if np.any(fwhm > 0):
    back_img = gaussian_filter(back_img, fwhm/2.35)

  return back_img

#---------------------------------------------------------------------------------
def pet_back_model_lm(lst, lmproj, subset_events, attn_list, sens_list, fwhm = 0):
  """Adjoint of listmode PET forward model (backward model)

  Parameters
  ----------

  lst : 1D numpy array
    containing the values for each event LOR to be backprojected

  lmproj : projector
    geometrical listmode TOF or non-TOF projector

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

  back_img = lmproj.back_project(sens_list*attn_list*lst, subset_events)

  if np.any(fwhm > 0):
    back_img = gaussian_filter(back_img, fwhm/2.35)

  return back_img


