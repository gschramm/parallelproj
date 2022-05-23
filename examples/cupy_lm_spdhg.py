import pyparallelproj as ppp
import h5py
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

import cupyx.scipy.ndimage as ndi_cupy
import scipy.ndimage as ndi

from pathlib import Path
from time import time

def cupy_unique_axis0(ar, return_index = False, return_inverse = False, return_counts=False):
  """ analogon of numpy's unique() for 2D arrays for axis = 0
  """

  if len(ar.shape) != 2:
    raise ValueError("Input array must be 2D.")

  perm     = cp.lexsort(ar.T[::-2])
  aux      = ar[perm]
  mask     = cp.empty(ar.shape[0], dtype=cp.bool_)
  mask[0]  = True
  mask[1:] = cp.any(aux[1:] != aux[:-1], axis=1)

  ret = aux[mask]
  if not return_index and not return_inverse and not return_counts:
    return ret

  ret = ret,

  if return_index:
    ret += perm[mask],
  if return_inverse:
    imask = cp.cumsum(mask) - 1
    inv_idx = cp.empty(mask.shape, dtype = cp.intp)
    inv_idx[perm] = imask
    ret += inv_idx,
  if return_counts:
    nonzero = cp.nonzero(mask)[0]  # may synchronize
    idx = cp.empty((nonzero.size + 1,), nonzero.dtype)
    idx[:-1] = nonzero
    idx[-1] = mask.size
    ret += idx[1:] - idx[:-1],

  return ret

def count_event_multiplicity(events, xp):
  """ Count the multiplicity of events in an LM file

  Parameters
  ----------

  events : 2D numpy/cupy array
    of LM events of shape (n_events, 5) where the second axis encodes the event 
    (e.g. detectors numbers and TOF bins)

  xp : numpy or cupy module to use
  """

  if xp.__name__ == 'cupy':
    if not isinstance(events, xp.ndarray):
      events_d = xp.array(events)
    else:
      events_d = events

    tmp_d    = cupy_unique_axis0(events_d, return_counts = True, return_inverse = True)
    mu_d     = tmp_d[2][tmp_d[1]]

    if not isinstance(events, xp.ndarray):
      mu = xp.asnumpy(mu_d)
    else:
      mu = mu_d
  elif xp.__name__ == 'numpy':
    tmp = xp.unique(events, axis = 0, return_counts = True, return_inverse = True)
    mu  = tmp[2][tmp[1]]

  return mu


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
  def __init__(self, xp, name = 'l2_l1'):
    self.name = name
    self._xp  = xp
 
    if not self.name in ['l2_l1', 'l2_sq']:
     raise NotImplementedError

  def eval(self, x):
    if self.name == 'l2_l1':
      n = self._xp.linalg.norm(x, axis = 0).sum()
    elif self.name == 'l2_sq':
      n = (x**2).sum()

    return n

  def prox_convex_dual(self, x, sigma = None):
    """ proximal operator of the convex dual of the norm
    """
    if self.name == 'l2_l1':
      gnorm = self._xp.linalg.norm(x, axis = 0)
      r = x/self._xp.clip(gnorm, 1, None)
    elif self.name == 'l2_sq':
      r = x/(1+sigma)

    return r

#------------------------------------------------------------------------------------------------------

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

  def __init__(self, xp, joint_grad_field = None):
    self._xp = xp

    # e is the normalized joint gradient field that
    # we are only interested in the gradient component
    # perpendicular to it
    self.e = None
    
    if joint_grad_field is not None:
      norm   = self._xp.linalg.norm(joint_grad_field, axis = 0)
      self.e = joint_grad_field / norm

  def fwd(self, x):
    g = []
    for i in range(x.ndim):
      g.append(self._xp.diff(x, axis = i, append = self._xp.take(x, [-1], i)))
    g = self._xp.array(g)

    if self.e is not None:
      g = g - (g*self.e).sum(0)*self.e

    return g

  def adjoint(self, y):
    d = self._xp.zeros(y[0,...].shape, dtype = y.dtype)

    if self.e is not None:
      y2 = y - (y*self.e).sum(0)*self.e
    else:
      y2 = y

    for i in range(y.shape[0]):
      d -= self._xp.diff(y2[i,...], axis = i, prepend = self._xp.take(y2[i,...], [0], i))

    return d

#------------------------------------------------------------------------------------------------------

class LMPETAcqModel:
  def __init__(self, proj, events, attn_list, sens_list, fwhm = 0):
    self.proj        = proj
    self.events      = events
    self.attn_list   = attn_list
    self.sens_list   = sens_list
    self.fwhm        = fwhm

    if isinstance(attn_list, np.ndarray):
      self._ndi = ndi
    else:
      self._ndi = ndi_cupy

  def fwd(self, img, isub = 0, nsubsets = 1):
    if np.any(fwhm > 0):
      img = self._ndi.gaussian_filter(img, fwhm/2.35)

    ss = slice(isub, None, nsubsets)
    img_fwd = self.sens_list[ss]*self.attn_list[ss]*self.proj.fwd_project_lm(img, self.events[ss])

    return img_fwd

  def back(self, values, isub = 0, nsubsets = 1):
    ss = slice(isub, None, nsubsets)
    back_img = self.proj.back_project_lm(self.sens_list[ss]*self.attn_list[ss]*values, self.events[ss])

    if np.any(fwhm > 0):
      back_img = self._ndi.gaussian_filter(back_img, fwhm/2.35)

    return back_img

#--------------------------------------------------------------------------------------------
def osem_lm(lm_acq_model, contam_list, sens_img, niter, nsubsets, xstart = None, verbose = True):

  if isinstance(lm_acq_model.events, np.ndarray):
    xp = np
  else:
    xp = cp

  img_shape  = tuple(proj.img_dim)

  # initialize recon
  if xstart is None:
    recon = xp.full(img_shape, 1., dtype = xp.float32)
  else:
    recon = xstart.copy()

  # run OSEM iterations
  for it in range(niter):
    for i in range(nsubsets):
      if verbose: print(f'iteration {it+1} subset {i+1}')
    
      exp_list = lm_acq_model.fwd(recon, i, nsubsets) + contam_list[i::nsubsets] 

      recon   *= (lm_acq_model.back(nsubsets/exp_list, i, nsubsets) / sens_img)

  return recon

#--------------------------------------------------------------------------------------------
def lm_spdhg(lm_acq_model, contam_list, grad_operator, grad_norm, 
             x0 = None, beta = 6, niter = 5, nsubsets = 56, gamma = 1., 
             rho = 0.999, rho_grad = 0.999, verbose = True):

  if isinstance(lm_acq_model.events, np.ndarray):
    xp = np
  else:
    xp = cp

  # count the "multiplicity" of every event in the list
  # if an event occurs n times in the list of events, the multiplicity is n
  mu = count_event_multiplicity(lm_acq_model.events, xp)
  
  img_shape = tuple(proj.img_dim)
  
  # setup the probabilities for doing a pet data or gradient update
  # p_g is the probablility for doing a gradient update
  # p_p is the probablility for doing a PET data subset update
  
  if beta == 0:
    p_g = 0
  else: 
    p_g = 0.5
    # norm of the gradient operator = sqrt(ndim*4)
    ndim  = len([x for x in img_shape if x > 1])
    grad_op_norm = xp.sqrt(ndim*4)
  
  p_p = (1 - p_g) / nsubsets
  
   # initialize variables
  if x0 is None:
    x = xp.zeros(img_shape, dtype = np.float32)
  else:
    x = x0.copy()
  
  # initialize y for data
  y = 1 - (mu / (lm_acq_model.fwd(x) + contam_list))
  
  z = sens_img + lm_acq_model.back((y - 1) / mu)
  zbar = z.copy()
  
  
  # calculate S for the gradient operator
  if p_g > 0:
    S_g = gamma*rho_grad/grad_op_norm
    T_g = p_g*rho_grad/(gamma*grad_op_norm)
  
  # calculate the "step sizes" S_i for the PET fwd operator
  S_i = []
  
  ones_img = xp.ones(img_shape, dtype = np.float32)
  
  for i in range(nsubsets):
    ss = slice(i,None,nsubsets)
    # clip inf values
    tmp = lm_acq_model.fwd(ones_img, i, nsubsets)
    tmp = np.clip(tmp, tmp[tmp > 0].min(), None)
    S_i.append(gamma*rho/tmp)
  
  
  # calculate the step size T
  T = xp.zeros_like(sens_img)
  i = xp.where(sens_img > 0)
  T[i] = nsubsets*p_p*rho / (gamma*sens_img[i])
  
  if p_g > 0:
    T = np.clip(T, None, T_g)
  
  # allocate arrays for gradient operations
  y_grad = xp.zeros((len(img_shape),) + img_shape, dtype = xp.float32)
  
  #--------------------------------------------------------------------------------------------
  # SPDHG iterations
  
  zero_sens_inds = xp.where(sens_img == 0)

  for it in range(niter):
    subset_sequence = np.random.permutation(np.arange(int(nsubsets/(1-p_g))))
  
    for iss in range(subset_sequence.shape[0]):
      x = np.clip(x - T*zbar, 0, None)
  
      # select a random subset
      i = subset_sequence[iss]

      x[zero_sens_inds]    = 0
      zbar[zero_sens_inds] = 0
      z[zero_sens_inds]    = 0
  
      if i < nsubsets:
        # PET subset update
        print(f'iteration {it + 1} step {iss} subset {i+1}')
  
        ss = slice(i,None,nsubsets)
  
        y_plus = y[ss] + S_i[i]*(lm_acq_model.fwd(x, i, nsubsets) + contam_list[ss])
  
        # apply the prox for the dual of the poisson logL
        y_plus = 0.5*(y_plus + 1 - np.sqrt((y_plus - 1)**2 + 4*S_i[i]*mu[ss]))
  
        dz = lm_acq_model.back((y_plus - y[ss])/mu[ss], i, nsubsets)
  
        # update variables
        z = z + dz
        y[ss] = y_plus.copy()
        zbar = z + dz/p_p
      else:
        print(f'iteration {it + 1} step {iss} gradient update')
        y_grad_plus = (y_grad + S_g*grad_operator.fwd(x))
  
        # apply the prox for the gradient norm
        y_grad_plus = beta*grad_norm.prox_convex_dual(y_grad_plus/beta, sigma = S_g/beta)
  
        dz = grad_operator.adjoint(y_grad_plus - y_grad)
        dz[zero_sens_inds] = 0
  
        # update variables
        z = z + dz
        y_grad = y_grad_plus.copy()
        zbar = z + dz/p_g

  return x



#--------------------------------------------------------------------------------------------

if __name__ == '__main__':
  on_gpu = True
  
  voxsize   = np.array([2., 2., 2.], dtype = np.float32)
  img_shape = (166,166,94)
  verbose   = True
  
  # FHHM for resolution model in voxel
  fwhm  = 4.5 / (2.35*voxsize)
  
  # prior strength
  beta     = 6
  niter    = 100
  nsubsets = 224
  
  np.random.seed(1)
  
  #--------------------------------------------------------------------------------------------
  
  if on_gpu:
    xp = cp
  else:
    xp = np
  
  #--------------------------------------------------------------------------------------------
  # setup scanner and projector
  
  # define the 4 ring DMI geometry
  scanner = ppp.RegularPolygonPETScanner(
                 R                    = 0.5*(744.1 + 2*8.51),
                 ncrystals_per_module = np.array([16,9]),
                 crystal_size         = np.array([4.03125,5.31556]),
                 nmodules             = np.array([34,4]),
                 module_gap_axial     = 2.8,
                 on_gpu               = on_gpu)
  
  
  # define sinogram parameter - also needed to setup the LM projector
  
  # speed of light in mm/ns
  speed_of_light = 300.
  
  # time resolution FWHM in ns
  time_res_FWHM = 0.385
  
  # sigma TOF in mm
  sigma_tof = (speed_of_light/2) * (time_res_FWHM/2.355)
  
  # the TOF bin width in mm is 13*0.01302ns times the speed of light (300mm/ns) divided by two
  sino_params = ppp.PETSinogramParameters(scanner, rtrim = 65, ntofbins = 29, 
                                          tofbin_width = 13*0.01302*speed_of_light/2)
  
  # define the projector
  proj = ppp.SinogramProjector(scanner, sino_params, img_shape,
                               voxsize = voxsize, tof = True, 
                               sigma_tof = sigma_tof, n_sigmas = 3.)
  
  
  #--------------------------------------------------------------------------------------------
  # calculate sensitivity image
  
  sens_img = np.load('../../lm-spdhg/python/data/dmi/NEMA_TV_beta_6_ss_224/sens_img.npy')
  
  #--------------------------------------------------------------------------------------------
  
  # read the actual LM data and the correction lists
  
  # read the LM data
  if verbose:
    print('Reading LM data')
  
  with h5py.File('../../lm-spdhg/python/data/dmi/lm_data.h5', 'r') as data:
    sens_list   =  data['correction_lists/sens'][:]
    atten_list  =  data['correction_lists/atten'][:]
    contam_list =  data['correction_lists/contam'][:]
    LM_file     =  Path(data['header/listfile'][0].decode("utf-8"))
  
  with h5py.File(LM_file, 'r') as data:
    events = data['MiceList/TofCoinc'][:]
  
  # swap axial and trans-axial crystals IDs
  events = events[:,[1,0,3,2,4]]
  
  # for the DMI the tof bins in the LM files are already meshed (only every 13th is populated)
  # so we divide the small tof bin number by 13 to get the bigger tof bins
  # the definition of the TOF bin sign is also reversed 
  
  events[:,-1] = -(events[:,-1]//13)
  
  nevents = events.shape[0]
  
  ## shuffle events since events come semi sorted
  if verbose: 
    print('shuffling LM data')
  ie = np.arange(nevents)
  np.random.shuffle(ie)
  events = events[ie,:]
  sens_list   = sens_list[ie]
  atten_list  = atten_list[ie]  
  contam_list = contam_list[ie]
  
  if on_gpu:
    print('copying data to GPU')
    events      = cp.asarray(events)
    sens_img    = cp.asarray(sens_img)
    sens_list   = cp.asarray(sens_list)
    atten_list  = cp.asarray(atten_list)
    contam_list = cp.asarray(contam_list)
  
  #--------------------------------------------------------------------------------------------
  # OSEM as intializer
  lm_acq_model = LMPETAcqModel(proj, events, atten_list, sens_list, fwhm = fwhm)
  
  print('OSEM')
  
  t0 = time()
  x0 = osem_lm(lm_acq_model, contam_list, sens_img, 2, 28)
  t1 = time()
  print(f'recon time: {(t1-t0):.2f}s')
  
  #--------------------------------------------------------------------------------------------
  # LM-SPDHG
  
  print('LM-SPDHG')
  grad_op   = GradientOperator(xp)
  grad_norm = GradientNorm(xp) 
  
  x = lm_spdhg(lm_acq_model, contam_list, grad_op, grad_norm, x0 = x0, beta = 6, 
               niter = 50, nsubsets = 112, gamma =  3. / ndi.gaussian_filter(cp.asnumpy(x0),2.).max())
  
  #----------------------------------------------------------------------------------------------
  
  import pymirc.viewer as pv
  vi = pv.ThreeAxisViewer(cp.asnumpy(x), imshow_kwargs = {'vmax':0.4})
