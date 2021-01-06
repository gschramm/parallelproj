import numpy as np
from pyparallelproj.models import pet_fwd_model, pet_back_model, pet_fwd_model_lm, pet_back_model_lm

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
      if verbose: print(f'iteration {it + 1} subset {i+1}')

      # get the slice for the current subset
      ss        = proj.subset_slices[i]

      exp_sino = pet_fwd_model(recon, proj, attn_sino[ss], sens_sino[ss], i, 
                               fwhm = fwhm) + contam_sino[ss]
      ratio  = em_sino[ss] / exp_sino
      recon *= (pet_back_model(ratio, proj, attn_sino[ss], sens_sino[ss], i, 
                               fwhm = fwhm) / sens_img[i,...]) 
    
      if subset_callback is not None:
        subset_callback(recon, **subset_callback_kwargs)

    if callback is not None:
      callback(recon, **callback_kwargs)
      
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
    recon = np.full(img_shape, em_sino.sum() / np.prod(img_shape), dtype = np.float32)
  else:
    recon = xstart.copy()

  # run OSEM iterations
  for it in range(niter):
    for i in range(nsubsets):
      if verbose: print(f'iteration {it + 1} subset {i+1}')
    
      exp_list = pet_fwd_model_lm(recon, lmproj, events[i::nsubsets,:], attn_list[i::nsubsets], 
                                      sens_list[i::nsubsets], fwhm = fwhm) + contam_list[i::nsubsets]

      recon *= (pet_back_model_lm(1/exp_list, lmproj, events[i::nsubsets,:], attn_list[i::nsubsets], 
                                  sens_list[i::nsubsets], fwhm = fwhm)*nsubsets / sens_img)

      if subset_callback is not None:
        subset_callback(recon, **subset_callback_kwargs)

    if callback is not None:
      callback(recon)
      callback(recon, **callback_kwargs)
      
  return recon

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

def spdhg(em_sino, attn_sino, sens_sino, contam_sino, proj, niter,
          fwhm = 0, gamma = 1., rho = 0.999, verbose = False, 
          xstart = None, ystart = None,
          callback = None, subset_callback = None,
          callback_kwargs = None, subset_callback_kwargs = None):

  sino_shape = tuple(proj.sino_params.shape)
  img_shape  = tuple(proj.img_dim)

  msubsets = proj.nsubsets

  # calculate the "step sizes" S_i, T_i  for the projector
  S_i = np.zeros(em_sino.shape, dtype = np.float32)
  
  ones_img = np.ones(img_shape, dtype = np.float32)
  for i in range(nsubsets):
    S_i[i,...] = (gamma*rho) / pet_fwd_model(ones_img, proj, attn_sino[i,...], sens_sino[i,...], i, 
                                             fwhm = fwhm)
  # clip inf values
  S_i[S_i == np.inf] = S_i[S_i != np.inf].max()


  ones_sino = np.ones((sino_shape[0], sino_shape[1] // nsubsets, sino_shape[2], 
                       sino_shape[3]), dtype = np.float32)
  T_i = np.zeros((nsubsets,) + img_shape, dtype = np.float32)
  for i in range(nsubsets):
    tmp = pet_back_model(ones_sino, proj, attn_sino[i,...], sens_sino[i,...], i, fwhm = fwhm)
    T_i[i,...] = (rho/(nsubsets*gamma)) / tmp  
                                                         
  
  # take the element-wise min of the T_i's of all subsets
  T = T_i.min(axis = 0)


  # initialize variables
  if xstart is None:
    x = np.zeros(img_shape, dtype = np.float32)
  else:
    x = xstart.copy()

  if ystart is None:
    y = np.zeros(em_sino.shape, dtype = np.float32)
  else:
    y = ystart.copy()

  z  = np.zeros(img_shape, dtype = np.float32)

  for i in range(nsubsets):
    if np.any(y[i,...] != 0):
      z += pet_back_model(y[i,...], proj, attn_sino[i,...], sens_sino[i,...], i, fwhm = fwhm)

  zbar = z.copy()

  for it in range(niter):
    subset_sequence = np.random.permutation(np.arange(nsubsets))
    for ss in range(nsubsets):
      # select a random subset
      i = subset_sequence[ss]
      print(f'iteration {it + 1} step {ss} subset {i}')
  
      x = np.clip(x - T*zbar, 0, None)
  
      y_plus = y[i,...] + S_i[i,...]*(pet_fwd_model(x, proj, attn_sino[i,...], sens_sino[i,...], i, 
                                                    fwhm = fwhm) + contam_sino[i,...])
  
      # apply the prox for the dual of the poisson logL
      y_plus = 0.5*(y_plus + 1 - np.sqrt((y_plus - 1)**2 + 4*S_i[i,...]*em_sino[i,...]))
  
      dz = pet_back_model(y_plus - y[i,...], proj, attn_sino[i,...], sens_sino[i,...], i, fwhm = fwhm)
  
      # update variables
      z = z + dz
      y[i,...] = y_plus.copy()
      zbar = z + dz*nsubsets

      if subset_callback is not None:
        subset_callback(x, **subset_callback_kwargs)

    if callback is not None:
      callback(x, **callback_kwargs)

  return x
