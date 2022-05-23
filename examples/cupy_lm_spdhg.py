import pyparallelproj as ppp
import h5py
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

import cupyx.scipy.ndimage as ndi_cupy
import scipy.ndimage as ndi

from pathlib import Path
from time import time

def cupy_unique_axis0(ar, return_index = False, return_inverse = False, return_counts = False):
  """ analogon of numpy's unique() for 2D arrays for axis = 0
  """

  if len(ar.shape) != 2:
    raise ValueError("Input array must be 2D.")

  perm     = cp.lexsort(ar.T[::-2])
  aux      = ar[perm]
  mask     = cp.empty(ar.shape[0], dtype = cp.bool_)
  mask[0]  = True
  mask[1:] = cp.any(aux[1:] != aux[:-1], axis=1)

  ret = aux[mask]
  if not return_index and not return_inverse and not return_counts:
    return ret

  ret = ret,

  if return_index:
    ret += perm[mask],
  if return_inverse:
    imask          = cp.cumsum(mask) - 1
    inv_idx        = cp.empty(mask.shape, dtype = cp.intp)
    inv_idx[perm]  = imask
    ret           += inv_idx,
  if return_counts:
    nonzero  = cp.nonzero(mask)[0]  # may synchronize
    idx      = cp.empty((nonzero.size + 1,), nonzero.dtype)
    idx[:-1] = nonzero
    idx[-1]  = mask.size
    ret     += idx[1:] - idx[:-1],

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
class LM_OSEM:
  def __init__(self, lm_acq_model, contam_list, verbose = True):
    self.lm_acq_model = lm_acq_model
    self.contam_list  = contam_list
    self.img_shape    = tuple(self.lm_acq_model.proj.img_dim)
    self.verbose      = True

    if isinstance(self.lm_acq_model.events, np.ndarray):
      self._xp = np
    else:
      self._xp = cp
  
  def init(self, sens_img, nsubsets, x = None):
    self.epoch_counter = 0
    self.sens_img      = sens_img
    self.nsubsets      = nsubsets

    if x is None:
      self.x = self._xp.full(img_shape, 1., dtype = self._xp.float32)
    else:
      self.x = x.copy()

  def run_update(self, isub):
    exp_list = self.lm_acq_model.fwd(self.x, isub, self.nsubsets) + self.contam_list[isub::self.nsubsets] 
    self.x  *= (self.lm_acq_model.back(self.nsubsets/exp_list, isub, self.nsubsets) / self.sens_img)

  def run(self, niter):
    for it in range(niter):
      for isub in range(self.nsubsets):
        if self.verbose: print(f'iteration {self.epoch_counter+1} subset {isub+1}')
        self.run_update(isub)
      self.epoch_counter += 1

#--------------------------------------------------------------------------------------------
class LM_SPDHG:
  def __init__(self, lm_acq_model, contam_list, grad_operator, grad_norm, beta):
    self.lm_acq_model  = lm_acq_model
    self.contam_list   = contam_list
    self.img_shape     = tuple(self.lm_acq_model.proj.img_dim)
    self.verbose       = True
    self.grad_operator = grad_operator
    self.grad_norm     = grad_norm
    self.beta          = beta

    if isinstance(self.lm_acq_model.events, np.ndarray):
      self._xp = np
    else:
      self._xp = cp

  def init(self, sens_img, nsubsets, x = None, gamma = 1, rho = 0.999, rho_grad = 0.999):
    self.epoch_counter = 0
    self.sens_img      = sens_img
    self.gamma         = gamma
    self.rho           = rho
    self.rho_grad      = rho_grad
    self.nsubsets      = nsubsets

    if x is None:
      self.x = self._xp.full(img_shape, 1., dtype = self._xp.float32)
    else:
      self.x = x.copy()

    self.mu = count_event_multiplicity(lm_acq_model.events, self._xp)

    if self.beta == 0:
      self.p_g = 0
    else: 
      self.p_g = 0.5
      # norm of the gradient operator = sqrt(ndim*4)
      ndim  = len([x for x in self.img_shape if x > 1])
      self.grad_op_norm = self._xp.sqrt(ndim*4)
    
    self.p_p = (1 - self.p_g) / self.nsubsets
  

    # initialize y for data, z and zbar
    self.y    = 1 - (self.mu / (self.lm_acq_model.fwd(self.x) + self.contam_list))
    self.z    = self.sens_img + self.lm_acq_model.back((self.y - 1) / self.mu)
    self.zbar = self.z.copy()

    # calculate S for the gradient operator
    if self.p_g > 0:
      self.S_g = self.gamma*self.rho_grad/self.grad_op_norm
      self.T_g = self.p_g*self.rho_grad/(self.gamma*self.grad_op_norm)

    # calculate the "step sizes" S_i for the PET fwd operator
    self.S_i = []
    ones_img = self._xp.ones(img_shape, dtype = self._xp.float32)
    
    for i in range(self.nsubsets):
      ss = slice(i, None, nsubsets)
      tmp = self.lm_acq_model.fwd(ones_img, i, self.nsubsets)
      tmp = np.clip(tmp, tmp[tmp > 0].min(), None) # clip 0 values before division
      self.S_i.append(self.gamma*self.rho/tmp)
 
    # calculate the step size T
    self.T = self._xp.zeros_like(sens_img)
    inds = self._xp.where(self.sens_img > 0)
    self.T[inds] = self.nsubsets*self.p_p*self.rho / (self.gamma*self.sens_img[inds])
    
    if self.p_g > 0:
      self.T = self._xp.clip(self.T, None, self.T_g)
    
    # allocate arrays for gradient operations
    self.y_grad = self._xp.zeros((len(img_shape),) + img_shape, dtype = self._xp.float32)
  
    # indices where sensitivity is 0 (e.g. outside ring)
    self.zero_sens_inds = self._xp.where(self.sens_img == 0)

    # intitial subset sequence
    self.subset_sequence = np.random.permutation(np.arange(int(self.nsubsets/(1-self.p_g))))


  def run_update(self, isub):
    self.x = np.clip(self.x - self.T*self.zbar, 0, None)
  
    # select a random subset
    i = self.subset_sequence[isub]

    self.x[self.zero_sens_inds]    = 0
    self.zbar[self.zero_sens_inds] = 0
    self.z[self.zero_sens_inds]    = 0
  
    if i < self.nsubsets:
      # PET subset update
      if self.verbose: print(f'iteration step {isub} subset {i+1}')
      ss = slice(i, None, self.nsubsets)
  
      y_plus = self.y[ss] + self.S_i[i]*(self.lm_acq_model.fwd(self.x, i, self.nsubsets) + self.contam_list[ss])
  
      # apply the prox for the dual of the poisson logL
      y_plus = 0.5*(y_plus + 1 - np.sqrt((y_plus - 1)**2 + 4*self.S_i[i]*self.mu[ss]))
      dz     = self.lm_acq_model.back((y_plus - self.y[ss])/self.mu[ss], i, self.nsubsets)
  
      # update variables
      self.z    += dz
      self.y[ss] = y_plus.copy()
      self.zbar  = self.z + dz/self.p_p
    else:
      print(f'step {isub} gradient update')
      y_grad_plus = (self.y_grad + self.S_g*self.grad_operator.fwd(self.x))
  
      # apply the prox for the gradient norm
      y_grad_plus = self.beta*self.grad_norm.prox_convex_dual(y_grad_plus/self.beta, sigma = self.S_g/self.beta)
  
      dz = self.grad_operator.adjoint(y_grad_plus - self.y_grad)
      dz[self.zero_sens_inds] = 0
  
      # update variables
      self.z     += dz
      self.y_grad = y_grad_plus.copy()
      self.zbar   = self.z + dz/self.p_g


  def run(self, niter):
    for it in range(niter):
      self.subset_sequence = np.random.permutation(np.arange(int(self.nsubsets/(1-self.p_g))))
      for isub in range(self.subset_sequence.shape[0]):
        if self.verbose: print(f'iteration {self.epoch_counter+1}', end = ' ')
        self.run_update(isub)
      self.epoch_counter += 1



#--------------------------------------------------------------------------------------------

if __name__ == '__main__':
  on_gpu    = True
  voxsize   = np.array([2., 2., 2.], dtype = np.float32)
  img_shape = (166,166,94)
  verbose   = True
  
  # FHHM for resolution model in voxel
  fwhm  = 4.5 / (2.35*voxsize)
  
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
 
  #ne = 10000000

  #sens_list = sens_list[:ne]
  #atten_list = atten_list[:ne]
  #contam_list = contam_list[:ne]*(ne/events.shape[0])
  #events = events[:ne,:]
 
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
  lm_osem = LM_OSEM(lm_acq_model, contam_list)
  lm_osem.init(sens_img, 28)
  lm_osem.run(2)
  t1 = time()
  print(f'recon time: {(t1-t0):.2f}s')
  
  #--------------------------------------------------------------------------------------------
  # LM-SPDHG
  
  print('LM-SPDHG')
  grad_op   = GradientOperator(xp)
  grad_norm = GradientNorm(xp) 

  img_norm = float(ndi_cupy.gaussian_filter(lm_osem.x,2).max())

  lm_spdhg = LM_SPDHG(lm_acq_model, contam_list, grad_op, grad_norm, 6)
  lm_spdhg.init(sens_img, 56, x = lm_osem.x, gamma = 30. / img_norm, rho = 1)

  niter = 50
  r = xp.zeros((niter,) + img_shape, dtype = xp.float32)

  for i in range(niter):
    lm_spdhg.run(1)
    r[i,...] = lm_spdhg.x.copy() 

  #----------------------------------------------------------------------------------------------
  
  import pymirc.viewer as pv
  vi = pv.ThreeAxisViewer(cp.asnumpy(r), imshow_kwargs = {'vmax':0.4})
