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
    geometrical TOF orf non-TOF projector

  attn_sino : numpy array of shape [nsubsets, nrad, nanlges_per_subset,nplanes]
    containing the non-TOF attenuation sinogram

  sens_sino : numpy array of shape [nsubsets, nrad, nanlges_per_subset,nplanes]
    containing the non-TOF sensivity sinogram

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

  sino = sens_sino[isub, ...]*attn_sino[isub, ...]*proj.fwd_project(img, subset = isub)

  return sino

#---------------------------------------------------------------------------------
def pet_back_model(subset_sino, proj, attn_sino, sens_sino, isub, fwhm = 0):
  """Adjoint of PET forward model (backward model)

  Parameters
  ----------

  subset_sino : numpy array of shape [nrad, nanlges_per_subset,nplanes,ntofbins]
    containing the subset sinogram to be backprojected

  proj : projector
    geometrical TOF orf non-TOF projector

  attn_sino : numpy array of shape [nsubsets, nrad, nanlges_per_subset,nplanes]
    containing the non-TOF attenuation sinogram

  sens_sino : numpy array of shape [nsubsets, nrad, nanlges_per_subset,nplanes]
    containing the non-TOF sensivity sinogram

  isub : int
    the subset number

  fwhm : float, optional
    FWHM (voxels) of Gaussian filter applied to image after back projection

  Returns
  -------

  3D numpy array containing the back projected image
  """

  back_img = proj.back_project(sens_sino[isub, ...]*attn_sino[isub, ...]*subset_sino, subset = isub)

  if np.any(fwhm > 0):
    back_img = gaussian_filter(back_img, fwhm/2.35)

  return back_img
