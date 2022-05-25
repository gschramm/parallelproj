import numpy as np

##------------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------------
#
#def osem_lm(events, attn_list, sens_list, contam_list, proj, sens_img, niter, nsubsets, 
#            fwhm = 0, verbose = False, xstart = None, callback = None, subset_callback = None,
#            callback_kwargs = None, subset_callback_kwargs = None):
#
#  img_shape  = tuple(proj.img_dim)
#
#  # initialize recon
#  if xstart is None:
#    recon = np.full(img_shape, events.shape[0] / np.prod(img_shape), dtype = np.float32)
#  else:
#    recon = xstart.copy()
#
#  # run OSEM iterations
#  for it in range(niter):
#    for i in range(nsubsets):
#      if verbose: print(f'iteration {it+1} subset {i+1}')
#    
#      exp_list = pet_fwd_model_lm(recon, proj, events[i::nsubsets,:], attn_list[i::nsubsets], 
#                                      sens_list[i::nsubsets], fwhm = fwhm) + contam_list[i::nsubsets]
#
#      recon *= np.divide(pet_back_model_lm(1/exp_list, proj, events[i::nsubsets,:], 
#                                           attn_list[i::nsubsets], 
#                                           sens_list[i::nsubsets], fwhm = fwhm)*nsubsets, 
#                         sens_img, out = np.zeros_like(sens_img), where = (sens_img != 0))
#
#      if subset_callback is not None:
#        subset_callback(recon, iteration = (it+1), subset = (i+1), **subset_callback_kwargs)
#
#    if callback is not None:
#      callback(recon, iteration = (it+1), subset = (i+1), **callback_kwargs)
#      
#  return recon
#
##------------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------------
#
#def spdhg(em_sino, attn_sino, sens_sino, contam_sino, proj, niter,
#          fwhm = 0, gamma = 1., rho = 0.999, rho_grad = 0.999, verbose = False, 
#          xstart = None, ystart = None, y_grad_start = None,
#          callback = None, subset_callback = None,
#          callback_kwargs = None, subset_callback_kwargs = None,
#          grad_operator = None, grad_norm = None, beta = 0, precond = True):
#  """ SPDHG for PET recon with gradient-based prior """
#
#  img_shape = tuple(proj.img_dim)
#  nsubsets  = proj.nsubsets
#
#  if grad_operator is None:
#    grad_operator = GradientOperator()
#
#  if grad_norm is None:
#    grad_norm = GradientNorm()
#
#  # setup the probabilities for doing a pet data or gradient update
#  # p_g is the probablility for doing a gradient update
#  # p_p is the probablility for doing a PET data subset update
#
#  if beta == 0:
#    p_g = 0
#  else: 
#    p_g = 0.5
#
#  # norm of the gradient operator = sqrt(ndim*4)
#  ndim  = len([x for x in img_shape if x > 1])
#  grad_op_norm = np.sqrt(ndim*4)
#
#  p_p = (1 - p_g) / nsubsets
#
#  # calculate S and T for the gradient operator
#  if p_g > 0:
#    S_g = gamma*rho_grad/grad_op_norm
#    T_g = rho_grad*p_g/(grad_op_norm*gamma)
#
#  # calculate the "step sizes" S_i for the PET fwd operator
#  S_i = []
#  ones_img = np.ones(img_shape, dtype = np.float32)
#
#  for i in range(nsubsets):
#    if precond:
#      ss = proj.subset_slices[i]
#      # get the slice for the current subset
#      tmp =  pet_fwd_model(ones_img, proj, attn_sino[ss], sens_sino[ss], i, fwhm = fwhm)
#      tmp = np.clip(tmp, tmp[tmp > 0].min(), None)
#      S_i.append(gamma*rho/tmp)
#    else:
#      S_i.append(gamma*rho/(grad_op_norm/np.sqrt(nsubsets)))
#
#  T_i = np.zeros((nsubsets,) + img_shape)
#
#  for i in range(nsubsets):
#    if precond:
#      # get the slice for the current subset
#      ss = proj.subset_slices[i]
#      # generate a subset sinogram full of ones
#      ones_sino = np.ones(proj.subset_sino_shapes[i] , dtype = np.float32)
#
#      tmp = pet_back_model(ones_sino, proj, attn_sino[ss], sens_sino[ss], i, fwhm = fwhm)
#      T_i[i,...] = rho*p_p/(gamma*tmp)
#    else:
#      T_i[i,...] = rho*p_p/(gamma*grad_op_norm/np.sqrt(nsubsets))
#
#  # take the element-wise min of the T_i's of all subsets
#  T = T_i.min(axis = 0)
#
#  del T_i
#
#  if p_g > 0:
#    T = np.clip(T, None, T_g)
#
#  #--------------------------------------------------------------------------------------------
#  # initialize variables
#  if xstart is None:
#    x = np.zeros(img_shape, dtype = np.float32)
#  else:
#    x = xstart.copy()
#
#  if ystart is None:
#    y = np.zeros(em_sino.shape, dtype = np.float32)
#  else:
#    y = ystart.copy()
#
#  z = np.zeros(img_shape, dtype = np.float32)
#  for i in range(nsubsets):
#    # get the slice for the current subset
#    ss = proj.subset_slices[i]
#    if np.any(y[ss] != 0):
#      z += pet_back_model(y[ss], proj, attn_sino[ss], sens_sino[ss], i, fwhm = fwhm)
#
#  # allocate arrays for gradient operations
#  if y_grad_start is None:
#    y_grad = np.zeros((x.ndim,) + img_shape, dtype = np.float32)
#  else:
#    y_grad = y_grad_start.copy()
#    z += grad_operator.adjoint(y_grad)
#
#  zbar = z.copy()
#
#  #--------------------------------------------------------------------------------------------
#  # SPDHG iterations
#
#  for it in range(niter):
#    subset_sequence = np.random.permutation(np.arange(int(nsubsets/(1-p_g))))
#
#    for iss in range(subset_sequence.shape[0]):
#      x = np.clip(x - T*zbar, 0, None)
#
#      # select a random subset
#      i = subset_sequence[iss]
#
#      if i < nsubsets:
#        # PET subset update
#        print(f'iteration {it + 1} step {iss} subset {i+1}')
#        # get the slice for the current subset
#        ss = proj.subset_slices[i]
#  
#        y_plus = y[ss] + S_i[i]*(pet_fwd_model(x, proj, attn_sino[ss], sens_sino[ss], i, 
#                                               fwhm = fwhm) + contam_sino[ss])
#  
#        # apply the prox for the dual of the poisson logL
#        y_plus = 0.5*(y_plus + 1 - np.sqrt((y_plus - 1)**2 + 4*S_i[i]*em_sino[ss]))
#  
#        dz = pet_back_model(y_plus - y[ss], proj, attn_sino[ss], sens_sino[ss], i, fwhm = fwhm)
#  
#        # update variables
#        z = z + dz
#        y[ss] = y_plus.copy()
#        zbar = z + dz/p_p
#      else:
#        print(f'iteration {it + 1} step {iss} gradient update')
#
#        y_grad_plus = (y_grad + S_g*grad_operator.fwd(x))
#
#        # apply the prox for the gradient norm
#        y_grad_plus = beta*grad_norm.prox_convex_dual(y_grad_plus/beta, sigma = S_g/beta)
#
#        dz = grad_operator.adjoint(y_grad_plus - y_grad)
#
#        # update variables
#        z = z + dz
#        y_grad = y_grad_plus.copy()
#        zbar = z + dz/p_g
#
#
#      if subset_callback is not None:
#        subset_callback(x, iteration = (it+1), subset = (i+1), **subset_callback_kwargs)
#
#    if callback is not None:
#      callback(x, y = y, y_grad = y_grad, iteration = (it+1), subset = (i+1), **callback_kwargs)
#
#  return x
#
##------------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------------
#
#def pdhg(em_sino, attn_sino, sens_sino, contam_sino, proj, niter,
#         fwhm = 0, gamma = 1., rho = 0.999, rho_grad = 0.999, verbose = False, 
#         xstart = None, ystart = None, y_grad_start = None,
#         callback = None, callback_kwargs = None,
#         grad_operator = None, grad_norm = None, beta = 0, precond = True):
#  """ PDHG for PET recon with gradient-based prior """
#
#  img_shape = tuple(proj.img_dim)
#  if proj.nsubsets != 1:
#    raise ValueError('For PDHG a projector with 1 subset is needed.')
#
#  if grad_operator is None:
#    grad_operator = GradientOperator()
#
#  if grad_norm is None:
#    grad_norm = GradientNorm()
#
#  # norm of the gradient operator = sqrt(ndim*4)
#  ndim  = len([x for x in img_shape if x > 1])
#  grad_op_norm = np.sqrt(ndim*4)
#
#  S_g = gamma*rho_grad/grad_op_norm
#  T_g = rho_grad/(grad_op_norm*gamma)
#
#  # calculate the "step sizes" S_i for the PET fwd operator
#  ones_img = np.ones(img_shape, dtype = np.float32)
#
#  if precond:
#    # get the slice for the current subset
#    tmp =  pet_fwd_model(ones_img, proj, attn_sino, sens_sino, fwhm = fwhm)
#    tmp = np.clip(tmp, tmp[tmp > 0].min(), None)
#    S = gamma*rho/tmp
#  else:
#    S = gamma*rho/grad_op_norm
#
#  if precond:
#    # generate a subset sinogram full of ones
#    ones_sino = np.ones(em_sino.shape , dtype = np.float32)
#
#    tmp = pet_back_model(ones_sino, proj, attn_sino, sens_sino, fwhm = fwhm)
#    T = rho/(gamma*tmp)
#  else:
#    T = rho/(gamma*grad_op_norm)
#
#  if beta > 0:
#    T = np.clip(T, None, T_g)
#
#  #--------------------------------------------------------------------------------------------
#  # initialize variables
#  if xstart is None:
#    x = np.zeros(img_shape, dtype = np.float32)
#  else:
#    x = xstart.copy()
#
#  if ystart is None:
#    y = np.zeros(em_sino.shape, dtype = np.float32)
#  else:
#    y = ystart.copy()
#
#  z = np.zeros(img_shape, dtype = np.float32)
#  if np.any(y != 0):
#    z = pet_back_model(y, proj, attn_sino, sens_sino, fwhm = fwhm)
#
#  # allocate arrays for gradient operations
#  if beta > 0:
#    if y_grad_start is None:
#      y_grad = np.zeros((x.ndim,) + img_shape, dtype = np.float32)
#    else:
#      y_grad = y_grad_start.copy()
#      z += grad_operator.adjoint(y_grad)
#
#  #--------------------------------------------------------------------------------------------
#  # PDHG iterations
#
#  for it in range(niter):
#    x = np.clip(x - T*z, 0, None)
#
#    # PET subset update
#    print(f'iteration {it + 1}')
#
#    # get the slice for the current subset
#    y_plus = y + S*(pet_fwd_model(x, proj, attn_sino, sens_sino, fwhm = fwhm) + contam_sino)
#  
#    # apply the prox for the dual of the poisson logL
#    y_plus = 0.5*(y_plus + 1 - np.sqrt((y_plus - 1)**2 + 4*S*em_sino))
#  
#    z += pet_back_model(y_plus - y, proj, attn_sino, sens_sino, fwhm = fwhm)
#  
#    # update variables
#    y = y_plus.copy()
#
#    if beta > 0:
#      print(f'iteration {it + 1} gradient update')
#      y_grad_plus = (y_grad + S_g*grad_operator.fwd(x))
#
#      # apply the prox for the gradient norm
#      y_grad_plus = beta*grad_norm.prox_convex_dual(y_grad_plus/beta, sigma = S_g/beta)
#
#      z += grad_operator.adjoint(y_grad_plus - y_grad)
#
#      # update variables
#      y_grad = y_grad_plus.copy()
#
#
#    if callback is not None:
#      callback(x, y = y, y_grad = y_grad, iteration = (it+1), **callback_kwargs)
#
#  return x
#
#
#
##----------------------------------------------------------------------------------------------------
#
#def pdhg_l2_denoise(img, grad_operator, grad_norm, 
#                    weights = 2e-2, niter = 200, cost = None, nonneg = False, verbose = False):
#  """
#  First-order primal dual image denoising with weighted L2 data fidelity term.
#  Solves the problem: argmax_x( \sum_i w_i*(x_i - img_i)**2 + norm(grad_operator x) )
#
#  Argumtents
#  ----------
#
#  img           ... an nd image image
#
#  grad_operator ... gradient operator with methods fwd() and adjoint()
#
#  grad_norm     ... gradient norm with methods eval() and prox_convex_dual()
#
#  Keyword arguments
#  -----------------
#
#  weights  ... (scalar or array) with weights for data fidelity term - default 2e-2
#
#  niter    ... (int) number of iterations to run - default 200
#
#  cost     ... (1d array) 1d output array for cost calcuation - default None
# 
#  nonneg   ... (bool) whether to clip negative values in solution - default False
#
#  verbose  ... (bool) whether to print some diagnostic output - default False
#  """
#
#  x    = img.copy().astype(np.float)
#  xbar = x.copy()
#  
#  ynew = np.zeros((x.ndim,) + x.shape)
#
#  if weights is np.array: gam = weights.min()  
#  else:                      gam = weights
#
#  tau    = 1./ gam
#  sig    = 1./(tau*4.*x.ndim)
#  
#  # start the iterations
#  for i in range(niter):
#    if verbose: print(i)
#
#    # (1) fwd model
#    ynew += sig*grad_operator.fwd(xbar)
#    
#    # (2) proximity operator
#    ynew = grad_norm.prox_convex_dual(ynew, sigma = sig)
#    
#    # (3) back model
#    xnew = x - tau*grad_operator.adjoint(ynew)
#    
#    # (4) apply proximity of G
#    xnew = (xnew + weights*img*tau) / (1. + weights*tau)
#    if nonneg: xnew = np.clip(xnew, 0, None)  
#    
#    # (5) calculate the new stepsizes
#    theta = 1.0 / np.sqrt(1 + 2*gam*tau)
#    tau   = tau*theta
#    sig   = sig/theta 
#    
#    # (6) update variables
#    xbar = xnew + theta*(xnew  - x)
#    x    = xnew.copy()
#  
#    # (0) store cost 
#    if cost is not None: 
#      cost[i] = 0.5*(weights*(x - img)**2).sum() + grad_norm.eval(grad_operator.fwd(x))
#      if verbose: print(cost[i])
#
#  return x
#
##----------------------------------------------------------------------------------------------------------
#
#def osem_lm_emtv(events, attn_list, sens_list, contam_list, proj, sens_img, niter, nsubsets, 
#                 fwhm = 0, verbose = False, xstart = None, callback = None, subset_callback = None,
#                 callback_kwargs = None, subset_callback_kwargs = None, niter_denoise = 20,
#                 grad_norm = None, grad_operator = None, beta = 0):
#
#  if grad_operator is None:
#    grad_operator = GradientOperator()
#
#  if grad_norm is None:
#    grad_norm = GradientNorm()
#
#  img_shape  = tuple(proj.img_dim)
#
#  # initialize recon
#  if xstart is None:
#    recon = np.full(img_shape, events.shape[0] / np.prod(img_shape), dtype = np.float32)
#  else:
#    recon = xstart.copy()
#
#  # construct a mask that shows which area is inside the scanner
#  # needed to avoid weights getting 0 which causes gam = inft in PDHG accel. denoising
#  x0 = (np.arange(img_shape[0]) - img_shape[0]/2 + 0.5)*proj.voxsize[0]
#  x1 = (np.arange(img_shape[1]) - img_shape[1]/2 + 0.5)*proj.voxsize[1]
#  X0,X1 = np.meshgrid(x0, x1, indexing = 'ij')
#  RHO   = np.sqrt(X0**2 + X1**2)
#
#  mask = RHO < 0.95*proj.scanner.R
#  mask = np.dstack([mask]*img_shape[2])
#
#  # run OSEM iterations
#  for it in range(niter):
#    for i in range(nsubsets):
#      if verbose: print(f'iteration {it+1} subset {i+1}')
#
#      # calculate the weights for weighted denoising problem that we have to solve
#      if beta > 0:
#        # wegihts post EM TV denoise step
#        weights = sens_img / (beta*np.clip(recon, recon[recon > 0].min(), None))
#        # clip also max of weights to avoid float overflow and division by 0
#        weights = np.clip(weights, weights[mask == 1].min(), 0.1*np.finfo(np.float32).max)
#         
#
#      # EM step
#      exp_list = pet_fwd_model_lm(recon, proj, events[i::nsubsets,:], attn_list[i::nsubsets], 
#                                      sens_list[i::nsubsets], fwhm = fwhm) + contam_list[i::nsubsets]
#
#      recon *= np.divide(pet_back_model_lm(1/exp_list, proj, events[i::nsubsets,:], 
#                                           attn_list[i::nsubsets], sens_list[i::nsubsets], 
#                                           fwhm = fwhm)*nsubsets, 
#                         sens_img, out = np.zeros_like(sens_img), where = (sens_img != 0))
#
#      # "TV" step (weighted denoising)
#      if beta > 0:
#        recon = pdhg_l2_denoise(recon, grad_operator, grad_norm, 
#                                weights = weights, niter = niter_denoise, nonneg = True)
#
#      if subset_callback is not None:
#        subset_callback(recon, iteration = (it+1), subset = (i+1), **subset_callback_kwargs)
#
#    if callback is not None:
#      callback(recon, iteration = (it+1), subset = (i+1), **callback_kwargs)
#      
#  return recon

#-----------------------------------------------------------------------------------------------------------------
class OSEM:
  def __init__(self, em_sino, acq_model, contam_sino, xp, verbose = True):
    self.em_sino     = em_sino
    self.acq_model   = acq_model
    self.contam_sino = contam_sino
    self.img_shape   = tuple(self.acq_model.proj.img_dim)
    self.verbose     = True
    self._xp         = xp
  
  def init(self, x = None):
    self.epoch_counter = 0
    self.cost = np.array([], dtype = np.float32)

    if x is None:
      self.x = self._xp.full(self.img_shape, 1., dtype = self._xp.float32)
    else:
      self.x = x.copy()

    # calculate the sensitivity images
    self.sens_imgs = self._xp.zeros((self.acq_model.proj.nsubsets,) + self.img_shape, dtype = self._xp.float32)

    for isub in range(self.acq_model.proj.nsubsets):
      if self.verbose: print(f'calculating sensitivity image {isub}')
      self.sens_imgs[isub,...] = self.acq_model.adjoint(self._xp.ones(self.acq_model.proj.subset_sino_shapes[isub]), 
                                                        isub = isub) 

  def run_update(self, isub):
    ss       = self.acq_model.proj.subset_slices[isub]
    exp_sino = self.acq_model.forward(self.x, isub) + self.contam_sino[ss] 
    self.x  *= (self.acq_model.adjoint(self.em_sino[ss]/exp_sino, isub) / self.sens_imgs[isub,...])

  def run(self, niter, calculate_cost = False):
    cost  = np.zeros(niter, dtype = np.float32)

    for it in range(niter):
      for isub in range(self.acq_model.proj.nsubsets):
        if self.verbose: print(f'iteration {self.epoch_counter+1} subset {isub+1}')
        self.run_update(isub)

      if calculate_cost:
        cost[it] = self.eval_neg_poisson_logL()
     
      self.epoch_counter += 1

    self.cost = np.concatenate((self.cost, cost)) 

  def eval_neg_poisson_logL(self):
    cost = 0
    for isub in range(self.acq_model.proj.nsubsets):
      ss       = self.acq_model.proj.subset_slices[isub]
      exp_sino = self.acq_model.forward(self.x, isub) + self.contam_sino[ss] 
      cost    += float((exp_sino - self.em_sino[ss]*self._xp.log(exp_sino)).sum())

    return cost


#-----------------------------------------------------------------------------------------------------------------
class LM_OSEM:
  def __init__(self, lm_acq_model, contam_list, xp, verbose = True):
    self.lm_acq_model = lm_acq_model
    self.contam_list  = contam_list
    self.img_shape    = tuple(self.lm_acq_model.proj.img_dim)
    self.verbose      = True
    self._xp          = xp
  
  def init(self, sens_img, nsubsets, x = None):
    self.epoch_counter = 0
    self.sens_img      = sens_img
    self.nsubsets      = nsubsets

    if x is None:
      self.x = self._xp.full(self.img_shape, 1., dtype = self._xp.float32)
    else:
      self.x = x.copy()

  def run_update(self, isub):
    exp_list = self.lm_acq_model.forward(self.x, isub, self.nsubsets) + self.contam_list[isub::self.nsubsets] 
    self.x  *= (self.lm_acq_model.adjoint(self.nsubsets/exp_list, isub, self.nsubsets) / self.sens_img)

  def run(self, niter):
    for it in range(niter):
      for isub in range(self.nsubsets):
        if self.verbose: print(f'iteration {self.epoch_counter+1} subset {isub+1}')
        self.run_update(isub)
      self.epoch_counter += 1

#--------------------------------------------------------------------------------------------
class LM_SPDHG:
  def __init__(self, lm_acq_model, contam_list, event_counter, prior, xp):
    self.lm_acq_model  = lm_acq_model
    self.contam_list   = contam_list
    self.event_counter = event_counter
    self.prior         = prior
    self._xp           = xp

    self.img_shape     = tuple(self.lm_acq_model.proj.img_dim)
    self.verbose       = True

  def init(self, sens_img, nsubsets, x = None, gamma = 1, rho = 0.999, rho_grad = 0.999):
    self.epoch_counter = 0
    self.sens_img      = sens_img
    self.gamma         = gamma
    self.rho           = rho
    self.rho_grad      = rho_grad
    self.nsubsets      = nsubsets

    if x is None:
      self.x = self._xp.full(self.img_shape, 1., dtype = self._xp.float32)
    else:
      self.x = x.copy()

    self.mu = self.event_counter.count(self.lm_acq_model.events)

    if self.prior.beta == 0:
      self.p_g = 0
    else: 
      self.p_g = 0.5
      # norm of the gradient operator = sqrt(ndim*4)
      ndim  = len([x for x in self.img_shape if x > 1])
      self.grad_op_norm = self._xp.sqrt(ndim*4)
    
    self.p_p = (1 - self.p_g) / self.nsubsets
  

    # initialize y for data, z and zbar
    self.y    = 1 - (self.mu / (self.lm_acq_model.forward(self.x) + self.contam_list))
    self.z    = self.sens_img + self.lm_acq_model.adjoint((self.y - 1) / self.mu)
    self.zbar = self.z.copy()

    # calculate S for the gradient operator
    if self.p_g > 0:
      self.S_g = self.gamma*self.rho_grad/self.grad_op_norm
      self.T_g = self.p_g*self.rho_grad/(self.gamma*self.grad_op_norm)

    # calculate the "step sizes" S_i for the PET fwd operator
    self.S_i = []
    ones_img = self._xp.ones(self.img_shape, dtype = self._xp.float32)
    
    for i in range(self.nsubsets):
      ss = slice(i, None, nsubsets)
      tmp = self.lm_acq_model.forward(ones_img, i, self.nsubsets)
      tmp = self._xp.clip(tmp, tmp[tmp > 0].min(), None) # clip 0 values before division
      self.S_i.append(self.gamma*self.rho/tmp)
 
    # calculate the step size T
    self.T = self._xp.zeros_like(sens_img)
    inds = self._xp.where(self.sens_img > 0)
    self.T[inds] = self.nsubsets*self.p_p*self.rho / (self.gamma*self.sens_img[inds])
    
    if self.p_g > 0:
      self.T = self._xp.clip(self.T, None, self.T_g)
    
    # allocate arrays for gradient operations
    self.y_grad = self._xp.zeros((len(self.img_shape),) + self.img_shape, dtype = self._xp.float32)
  
    # indices where sensitivity is 0 (e.g. outside ring)
    self.zero_sens_inds = self._xp.where(self.sens_img == 0)

    # intitial subset sequence
    self.subset_sequence = np.random.permutation(np.arange(int(self.nsubsets/(1-self.p_g))))


  def run_update(self, isub):
    self.x = self._xp.clip(self.x - self.T*self.zbar, 0, None)
  
    # select a random subset
    i = self.subset_sequence[isub]

    self.x[self.zero_sens_inds]    = 0
    self.zbar[self.zero_sens_inds] = 0
    self.z[self.zero_sens_inds]    = 0
  
    if i < self.nsubsets:
      # PET subset update
      if self.verbose: print(f'iteration step {isub} subset {i+1}')
      ss = slice(i, None, self.nsubsets)
  
      y_plus = self.y[ss] + self.S_i[i]*(self.lm_acq_model.forward(self.x, i, self.nsubsets) + self.contam_list[ss])
  
      # apply the prox for the dual of the poisson logL
      y_plus = 0.5*(y_plus + 1 - self._xp.sqrt((y_plus - 1)**2 + 4*self.S_i[i]*self.mu[ss]))
      dz     = self.lm_acq_model.adjoint((y_plus - self.y[ss])/self.mu[ss], i, self.nsubsets)
  
      # update variables
      self.z    += dz
      self.y[ss] = y_plus.copy()
      self.zbar  = self.z + dz/self.p_p
    else:
      print(f'step {isub} gradient update')
      y_grad_plus = (self.y_grad + self.S_g*self.prior.gradient_operator.forward(self.x))
  
      # apply the prox for the gradient norm
      y_grad_plus = self.prior.beta*self.prior.gradient_norm.prox_convex_dual(y_grad_plus/self.prior.beta, sigma = self.S_g/self.prior.beta)
  
      dz = self.prior.gradient_operator.adjoint(y_grad_plus - self.y_grad)
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
