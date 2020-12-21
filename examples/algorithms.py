import numpy as np
import pyparallelproj as ppp

def osem(em_sino, attn_sino, sens_sino, contam_sino, proj, niter, nsubsets, 
         fwhm = 0, cost = None, verbose = False, callback = None):

  sino_shape = tuple(proj.sino_params.shape)
  img_shape  = tuple(proj.img_dim)

  # calculate the sensitivity images for each subset
  sens_img  = np.zeros((nsubsets,) + img_shape, dtype = np.float32)
  ones_sino = np.ones((sino_shape[0], sino_shape[1] // nsubsets, sino_shape[2], 
                       sino_shape[3]), dtype = np.float32)
 
  for i in range(nsubsets):
    sens_img[i,...] = ppp.pet_back_model(ones_sino, proj, attn_sino, sens_sino, i, fwhm = fwhm)
  
  # initialize recon
  recon = np.full(img_shape, em_sino.sum() / np.prod(img_shape), dtype = np.float32)

  # run OSEM iterations
  for it in range(niter):
    for i in range(nsubsets):
      if verbose: print(f'iteration {it + 1} subset {i+1}')
      exp_sino = ppp.pet_fwd_model(recon, proj, attn_sino, sens_sino, i, fwhm = fwhm) + contam_sino[i,...]
      ratio  = em_sino[i,...] / exp_sino
      recon *= (ppp.pet_back_model(ratio, proj, attn_sino, sens_sino, i, fwhm = fwhm) / sens_img[i,...]) 
    
      if callback is not None:
        callback(recon)
      
    if cost is not None:
      exp = np.zeros(em_sino.shape, dtype = np.float32)
      for i in range(nsubsets):
        exp[i,...] = ppp.pet_fwd_model(recon, proj, attn_sino, sens_sino, i, fwhm = fwhm) + contam_sino[i,...]
      cost[it] = (exp - em_sino*np.log(exp)).sum()
      if verbose: print(f'cost {cost[it]}')


  return recon
