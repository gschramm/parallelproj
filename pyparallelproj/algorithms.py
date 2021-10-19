import numpy as np
from pyparallelproj.models import pet_fwd_model, pet_back_model, pet_fwd_model_lm, pet_back_model_lm
from pyparallelproj.utils import GradientOperator

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
      recon *= (pet_back_model(ratio, proj, attn_sino[ss], sens_sino[ss], i, 
                               fwhm = fwhm) / sens_img[i,...]) 
    
      if subset_callback is not None:
        subset_callback(recon, iteration = (it+1), subset = (i+1), **subset_callback_kwargs)

    if callback is not None:
      callback(recon, iteration = (it+1), subset = (i+1), **callback_kwargs)
      
  return recon

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

def osem_lm(events, attn_list, sens_list, contam_list, lmproj, sens_img, niter, nsubsets, 
            fwhm = 0, verbose = False, xstart = None, callback = None, subset_callback = None,
            callback_kwargs = None, subset_callback_kwargs = None):

  img_shape  = tuple(lmproj.img_dim)

  # initialize recon
  if xstart is None:
    recon = np.full(img_shape, events.shape[0] / np.prod(img_shape), dtype = np.float32)
  else:
    recon = xstart.copy()

  # run OSEM iterations
  for it in range(niter):
    for i in range(nsubsets):
      if verbose: print(f'iteration {it+1} subset {i+1}')
    
      exp_list = pet_fwd_model_lm(recon, lmproj, events[i::nsubsets,:], attn_list[i::nsubsets], 
                                      sens_list[i::nsubsets], fwhm = fwhm) + contam_list[i::nsubsets]

      recon *= (pet_back_model_lm(1/exp_list, lmproj, events[i::nsubsets,:], attn_list[i::nsubsets], 
                                  sens_list[i::nsubsets], fwhm = fwhm)*nsubsets / sens_img)

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
          beta = 0, pet_operator_norms = None, grad_operator = None):
 
  img_shape = tuple(proj.img_dim)
  nsubsets  = proj.nsubsets

  if grad_operator is None:
    grad_operator = GradientOperator()

  # setup the probabilities for doing a pet data or gradient update
  # p_g is the probablility for doing a gradient update
  # p_p is the probablility for doing a PET data subset update

  if beta == 0:
    p_g = 0
  else: 
    p_g = 0.5
    # norm of the gradient operator = sqrt(ndim*4)
    ndim  = len([x for x in img_shape if x > 1])
    grad_norm = np.sqrt(ndim*4)

  p_p = (1 - p_g) / nsubsets

  # calculate S and T for the gradient operator
  if p_g > 0:
    S_g = (gamma*rho/grad_norm)
    T_g = rho*p_g/(gamma*grad_norm)

  # calculate the "step sizes" S_i for the PET fwd operator
  S_i = []
  if pet_operator_norms is None:
    ones_img = np.ones(img_shape, dtype = np.float32)

    for i in range(nsubsets):
      # get the slice for the current subset
      ss = proj.subset_slices[i]
      tmp =  pet_fwd_model(ones_img, proj, attn_sino[ss], sens_sino[ss], i, fwhm = fwhm)
      tmp = np.clip(tmp, tmp[tmp > 0].min(), None)
      S_i.append((gamma*rho) / tmp)
  else:
    for i in range(nsubsets):
      S_i.append((gamma*rho)/pet_operator_norms[i])


  T_i = []
  if pet_operator_norms is None:
    for i in range(nsubsets):
      # get the slice for the current subset
      ss = proj.subset_slices[i]
      # generate a subset sinogram full of ones
      ones_sino = np.ones(proj.subset_sino_shapes[i] , dtype = np.float32)

      tmp = pet_back_model(ones_sino, proj, attn_sino[ss], sens_sino[ss], i, fwhm = fwhm)
      T_i.append((rho*p_p/gamma) / tmp)
  else:
    for i in range(nsubsets):
      T_i.append((rho*p_p/gamma) / pet_operator_norms[i])


  if p_g > 0:
    if isinstance(T_i[0],np.ndarray):
      T_i.append(np.full(T_i[0].shape, T_g, dtype = T_i[0].dtype))
    else:
      T_i.append(T_g)

  T_i = np.array(T_i)
    
  # take the element-wise min of the T_i's of all subsets
  T = T_i.min(axis = 0)

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

  zbar = z.copy()

  # allocate arrays for gradient operations
  if y_grad_start is None:
    y_grad = np.zeros((x.ndim,) + img_shape, dtype = np.float32)
  else:
    y_grad = y_grad_start.copy()

  #--------------------------------------------------------------------------------------------
  # SPDHG iterations

  for it in range(niter):
    subset_sequence = np.random.permutation(np.arange(int(nsubsets/(1-p_g))))

    for iss in range(subset_sequence.shape[0]):
      
      # select a random subset
      i = subset_sequence[iss]

      if i < nsubsets:
        # PET subset update
        print(f'iteration {it + 1} step {iss} subset {i+1}')
        # get the slice for the current subset
        ss = proj.subset_slices[i]
  
        x = np.clip(x - T*zbar, 0, None)
  
        y_plus = y[ss] + S_i[i]*(pet_fwd_model(x, proj, attn_sino[ss], sens_sino[ss], i, 
                                               fwhm = fwhm) + contam_sino[ss])
  
        # apply the prox for the dual of the poisson logL
        y_plus = 0.5*(y_plus + 1 - np.sqrt((y_plus - 1)**2 + 4*S_i[i]*em_sino[ss]))
        y_plus[em_sino[ss] == 0] = 1
  
        dz = pet_back_model(y_plus - y[ss], proj, attn_sino[ss], sens_sino[ss], i, fwhm = fwhm)
  
        # update variables
        z = z + dz
        y[ss] = y_plus.copy()
        zbar = z + dz/p_p
      else:
        print(f'iteration {it + 1} step {iss} gradient update')

        x_grad = grad_operator.fwd(x)
        y_grad_plus = (y_grad + S_g*x_grad).reshape(x.ndim,-1)

        # proximity operator for dual of TV
        gnorm = np.linalg.norm(y_grad_plus, axis = 0)
        y_grad_plus /= np.maximum(np.ones(gnorm.shape, np.float32), gnorm / beta)
        y_grad_plus = y_grad_plus.reshape(x_grad.shape)

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
