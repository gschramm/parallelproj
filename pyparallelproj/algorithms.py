import numpy as np
from pyparallelproj.models import pet_fwd_model, pet_back_model, pet_fwd_model_lm, pet_back_model_lm
from pyparallelproj.utils import GradientOperator, GradientNorm

def osem(em_sino, attn_sino, sens_sino, contam_sino, proj, niter,
         fwhm = 0, verbose = False, xstart = None, 
         callback = None, subset_callback = None,
         callback_kwargs = None, subset_callback_kwargs = None):

  img_shape  = tuple(proj.img_dim)

  # calculate the sensitivity images for each subset
  sens_img  = np.zeros((proj.nsubsets,) + img_shape, dtype = np.float32)
 
  for i in range(proj.nsubsets):
    # get the slice for the current subset
    ss        = proj.subset_slices[i]
    # generate a subset sinogram full of ones
    ones_sino = np.ones(proj.subset_sino_shapes[i] , dtype = np.float32)
    sens_img[i,...] = pet_back_model(ones_sino, proj, attn_sino[ss], sens_sino[ss], i, fwhm = fwhm)
  
  # initialize recon
  if xstart is None:
    recon = np.full(img_shape, em_sino.sum() / np.prod(img_shape), dtype = np.float32)
  else:
    recon = xstart.copy()

  # run OSEM iterations
  for it in range(niter):
    for i in range(proj.nsubsets):
      if verbose: print(f'iteration {it+1} subset {i+1}')

      # get the slice for the current subset
      ss        = proj.subset_slices[i]

      exp_sino = pet_fwd_model(recon, proj, attn_sino[ss], sens_sino[ss], i, 
                               fwhm = fwhm) + contam_sino[ss]
      ratio  = em_sino[ss] / exp_sino
      recon *= np.divide(pet_back_model(ratio, proj, attn_sino[ss], sens_sino[ss], i, fwhm = fwhm), 
                         sens_img[i,...], out = np.zeros_like(sens_img[i,...]), 
                         where = (sens_img[i,...] != 0)) 
    
      if subset_callback is not None:
        subset_callback(recon, iteration = (it+1), subset = (i+1), **subset_callback_kwargs)

    if callback is not None:
      callback(recon, iteration = (it+1), subset = (i+1), **callback_kwargs)
      
  return recon

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

def osem_lm(events, attn_list, sens_list, contam_list, proj, sens_img, niter, nsubsets, 
            fwhm = 0, verbose = False, xstart = None, callback = None, subset_callback = None,
            callback_kwargs = None, subset_callback_kwargs = None):

  img_shape  = tuple(proj.img_dim)

  # initialize recon
  if xstart is None:
    recon = np.full(img_shape, events.shape[0] / np.prod(img_shape), dtype = np.float32)
  else:
    recon = xstart.copy()

  # run OSEM iterations
  for it in range(niter):
    for i in range(nsubsets):
      if verbose: print(f'iteration {it+1} subset {i+1}')
    
      exp_list = pet_fwd_model_lm(recon, proj, events[i::nsubsets,:], attn_list[i::nsubsets], 
                                      sens_list[i::nsubsets], fwhm = fwhm) + contam_list[i::nsubsets]

      recon *= np.divide(pet_back_model_lm(1/exp_list, proj, events[i::nsubsets,:], 
                                           attn_list[i::nsubsets], 
                                           sens_list[i::nsubsets], fwhm = fwhm)*nsubsets, 
                         sens_img, out = np.zeros_like(sens_img), where = (sens_img != 0))

      if subset_callback is not None:
        subset_callback(recon, iteration = (it+1), subset = (i+1), **subset_callback_kwargs)

    if callback is not None:
      callback(recon, iteration = (it+1), subset = (i+1), **callback_kwargs)
      
  return recon

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

def spdhg(em_sino, attn_sino, sens_sino, contam_sino, proj, niter,
          fwhm = 0, gamma = 1., rho = 0.999, verbose = False, 
          xstart = None, ystart = None, y_grad_start = None,
          callback = None, subset_callback = None,
          callback_kwargs = None, subset_callback_kwargs = None,
          grad_operator = None, grad_norm = None, beta = 0, precond = True):

  img_shape = tuple(proj.img_dim)
  nsubsets  = proj.nsubsets

  if grad_operator is None:
    grad_operator = GradientOperator()

  if grad_norm is None:
    grad_norm = GradientNorm()

  # setup the probabilities for doing a pet data or gradient update
  # p_g is the probablility for doing a gradient update
  # p_p is the probablility for doing a PET data subset update

  if beta == 0:
    p_g = 0
  else: 
    p_g = 0.5

  # norm of the gradient operator = sqrt(ndim*4)
  ndim  = len([x for x in img_shape if x > 1])
  grad_op_norm = np.sqrt(ndim*4)

  p_p = (1 - p_g) / nsubsets

  # calculate S and T for the gradient operator
  if p_g > 0:
    S_g = gamma*rho/grad_op_norm
    T_g = rho*p_g/(grad_op_norm*gamma)

  # calculate the "step sizes" S_i for the PET fwd operator
  S_i = []
  ones_img = np.ones(img_shape, dtype = np.float32)

  pet_op_norms = np.zeros(nsubsets)

  for i in range(nsubsets):
    if precond:
      ss = proj.subset_slices[i]
      # get the slice for the current subset
      tmp =  pet_fwd_model(ones_img, proj, attn_sino[ss], sens_sino[ss], i, fwhm = fwhm)
      tmp = np.clip(tmp, tmp[tmp > 0].min(), None)
      S_i.append(gamma*rho/tmp)
    else:
      S_i.append(gamma*rho/(grad_op_norm/np.sqrt(nsubsets)))
      #S_i.append(gamma*rho/(grad_op_norm/nsubsets))

  T_i = np.zeros((nsubsets,) + img_shape)

  for i in range(nsubsets):
    if precond:
      # get the slice for the current subset
      ss = proj.subset_slices[i]
      # generate a subset sinogram full of ones
      ones_sino = np.ones(proj.subset_sino_shapes[i] , dtype = np.float32)

      tmp = pet_back_model(ones_sino, proj, attn_sino[ss], sens_sino[ss], i, fwhm = fwhm)
      T_i[i,...] = rho*p_p/(gamma*tmp)
    else:
      T_i[i,...] = rho*p_p/(gamma*grad_op_norm/np.sqrt(nsubsets))
      #T_i[i,...] = rho*p_p/(gamma*grad_op_norm/nsubsets)

  # take the element-wise min of the T_i's of all subsets
  T = T_i.min(axis = 0)

  del T_i

  if p_g > 0:
    T = np.clip(T, None, T_g)

  #--------------------------------------------------------------------------------------------
  # initialize variables
  if xstart is None:
    x = np.zeros(img_shape, dtype = np.float32)
  else:
    x = xstart.copy()

  if ystart is None:
    y = np.zeros(em_sino.shape, dtype = np.float32)
  else:
    y = ystart.copy()

  z = np.zeros(img_shape, dtype = np.float32)
  for i in range(nsubsets):
    # get the slice for the current subset
    ss = proj.subset_slices[i]
    if np.any(y[ss] != 0):
      z += pet_back_model(y[ss], proj, attn_sino[ss], sens_sino[ss], i, fwhm = fwhm)

  # allocate arrays for gradient operations
  if y_grad_start is None:
    y_grad = np.zeros((x.ndim,) + img_shape, dtype = np.float32)
  else:
    y_grad = y_grad_start.copy()
    z += grad_operator.adjoint(y_grad)

  zbar = z.copy()

  #--------------------------------------------------------------------------------------------
  # SPDHG iterations

  for it in range(niter):
    subset_sequence = np.random.permutation(np.arange(int(nsubsets/(1-p_g))))

    for iss in range(subset_sequence.shape[0]):
      x = np.clip(x - T*zbar, 0, None)

      # select a random subset
      i = subset_sequence[iss]

      if i < nsubsets:
        # PET subset update
        print(f'iteration {it + 1} step {iss} subset {i+1}')
        # get the slice for the current subset
        ss = proj.subset_slices[i]
  
        y_plus = y[ss] + S_i[i]*(pet_fwd_model(x, proj, attn_sino[ss], sens_sino[ss], i, 
                                               fwhm = fwhm) + contam_sino[ss])
  
        # apply the prox for the dual of the poisson logL
        y_plus = 0.5*(y_plus + 1 - np.sqrt((y_plus - 1)**2 + 4*S_i[i]*em_sino[ss]))
  
        dz = pet_back_model(y_plus - y[ss], proj, attn_sino[ss], sens_sino[ss], i, fwhm = fwhm)
  
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

        # update variables
        z = z + dz
        y_grad = y_grad_plus.copy()
        zbar = z + dz/p_g


      if subset_callback is not None:
        subset_callback(x, iteration = (it+1), subset = (i+1), **subset_callback_kwargs)

    if callback is not None:
      callback(x, y = y, y_grad = y_grad, iteration = (it+1), subset = (i+1), **callback_kwargs)

  return x


#----------------------------------------------------------------------------------------------------

def pdhg_l2_denoise(img, grad_operator, grad_norm, 
                    weights = 2e-2, niter = 200, cost = None, nonneg = False, verbose = False):
  """
  First-order primal dual image denoising with weighted L2 data fidelity term.
  Solves the problem: argmax_x( \sum_i w_i*(x_i - img_i)**2 + norm(grad_operator x) )

  Argumtents
  ----------

  img           ... an nd image image

  grad_operator ... gradient operator with methods fwd() and adjoint()

  grad_norm     ... gradient norm with methods eval() and prox_convex_dual()

  Keyword arguments
  -----------------

  weights  ... (scalar or array) with weights for data fidelity term - default 2e-2

  niter    ... (int) number of iterations to run - default 200

  cost     ... (1d array) 1d output array for cost calcuation - default None
 
  nonneg   ... (bool) whether to clip negative values in solution - default False

  verbose  ... (bool) whether to print some diagnostic output - default False
  """

  x    = img.copy().astype(np.float)
  xbar = x.copy()
  
  ynew = np.zeros((x.ndim,) + x.shape)

  if weights is np.array: gam = weights.min()  
  else:                      gam = weights

  tau    = 1./ gam
  sig    = 1./(tau*4.*x.ndim)
  
  # start the iterations
  for i in range(niter):
    if verbose: print(i)

    # (1) fwd model
    ynew += sig*grad_operator.fwd(xbar)
    
    # (2) proximity operator
    ynew = grad_norm.prox_convex_dual(ynew, sigma = sig)
    
    # (3) back model
    xnew = x - tau*grad_operator.adjoint(ynew)
    
    # (4) apply proximity of G
    xnew = (xnew + weights*img*tau) / (1. + weights*tau)
    if nonneg: xnew = np.clip(xnew, 0, None)  
    
    # (5) calculate the new stepsizes
    theta = 1.0 / np.sqrt(1 + 2*gam*tau)
    tau   = tau*theta
    sig   = sig/theta 
    
    # (6) update variables
    xbar = xnew + theta*(xnew  - x)
    x    = xnew.copy()
  
    # (0) store cost 
    if cost is not None: 
      cost[i] = 0.5*(weights*(x - img)**2).sum() + grad_norm.eval(grad_operator.fwd(x))
      if verbose: print(cost[i])

  return x

#----------------------------------------------------------------------------------------------------------

def osem_lm_emtv(events, attn_list, sens_list, contam_list, proj, sens_img, niter, nsubsets, 
                 fwhm = 0, verbose = False, xstart = None, callback = None, subset_callback = None,
                 callback_kwargs = None, subset_callback_kwargs = None, niter_denoise = 20,
                 grad_norm = None, grad_operator = None, beta = 0):

  if grad_operator is None:
    grad_operator = GradientOperator()

  if grad_norm is None:
    grad_norm = GradientNorm()

  img_shape  = tuple(proj.img_dim)

  # initialize recon
  if xstart is None:
    recon = np.full(img_shape, events.shape[0] / np.prod(img_shape), dtype = np.float32)
  else:
    recon = xstart.copy()

  # construct a mask that shows which area is inside the scanner
  # needed to avoid weights getting 0 which causes gam = inft in PDHG accel. denoising
  x0 = (np.arange(img_shape[0]) - img_shape[0]/2 + 0.5)*proj.voxsize[0]
  x1 = (np.arange(img_shape[1]) - img_shape[1]/2 + 0.5)*proj.voxsize[1]
  X0,X1 = np.meshgrid(x0, x1, indexing = 'ij')
  RHO   = np.sqrt(X0**2 + X1**2)

  mask = RHO < 0.95*proj.scanner.R
  mask = np.dstack([mask]*img_shape[2])

  # run OSEM iterations
  for it in range(niter):
    for i in range(nsubsets):
      if verbose: print(f'iteration {it+1} subset {i+1}')

      # calculate the weights for weighted denoising problem that we have to solve
      if beta > 0:
        # wegihts post EM TV denoise step
        weights = sens_img / (beta*np.clip(recon, recon[recon > 0].min(), None))
        # clip also max of weights to avoid float overflow and division by 0
        weights = np.clip(weights, weights[mask == 1].min(), 0.1*np.finfo(np.float32).max)
         

      # EM step
      exp_list = pet_fwd_model_lm(recon, proj, events[i::nsubsets,:], attn_list[i::nsubsets], 
                                      sens_list[i::nsubsets], fwhm = fwhm) + contam_list[i::nsubsets]

      recon *= np.divide(pet_back_model_lm(1/exp_list, proj, events[i::nsubsets,:], 
                                           attn_list[i::nsubsets], sens_list[i::nsubsets], 
                                           fwhm = fwhm)*nsubsets, 
                         sens_img, out = np.zeros_like(sens_img), where = (sens_img != 0))

      # "TV" step (weighted denoising)
      if beta > 0:
        recon = pdhg_l2_denoise(recon, grad_operator, grad_norm, 
                                weights = weights, niter = niter_denoise, nonneg = True)

      if subset_callback is not None:
        subset_callback(recon, iteration = (it+1), subset = (i+1), **subset_callback_kwargs)

    if callback is not None:
      callback(recon, iteration = (it+1), subset = (i+1), **callback_kwargs)
      
  return recon
