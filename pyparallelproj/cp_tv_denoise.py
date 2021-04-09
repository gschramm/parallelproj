import numpy
from   .utils   import grad, div, prox_tv

def cp_tv_denoise(img, weights = 2e-2, niter = 200, cost = None, nonneg = False, verbose = False):
  """
  First-order primal dual weighted TV denoising of an image.
  Solves the problem: argmax_x( \sum_i w_i*(x_i - img_i)**2 + TV(x) )

  Argumtents
  ----------

  img ... a 2d, 3d, or 4d image

  Keyword arguments
  -----------------

  weights  ... (scalar or array) with weights for data fidelity term - default 2e-2

  niter    ... (int) number of iterations to run - default 200

  cost     ... (1d array) 1d output array for cost calcuation - default None
 
  nonneg   ... (bool) whether to clip negative values in solution - default False

  verbose  ... (bool) whether to print some diagnostic output - default False
  """
  x       = img.copy().astype(numpy.float)
  xshape  = x.shape
  yshape  = (x.ndim,) + x.shape
  
  xbar     = x.copy()
  fwd_xbar = numpy.zeros(yshape)
  
  ynew = numpy.zeros(yshape)

  if weights is numpy.array: gam = weights.min()  
  else:                      gam = weights

  Lip_sq = 4.*x.ndim
  tau    = 1./ gam
  sig    = 1./(tau*Lip_sq)
  
  # allocate memory for fwd model in case cost needs to be calculated
  if cost is not None: 
    fwd_x = numpy.zeros(yshape)
  
  # start the iterations
  for i in range(niter):
    if verbose: print(i)

    # (1) fwd model
    grad(xbar, fwd_xbar)
    ynew += sig*fwd_xbar
    
    # (2) proximity operator
    prox_tv(ynew, 1.)
    
    # (3) back model
    xnew = x + tau*div(ynew)
    
    # (4) apply proximity of G
    xnew = (xnew + weights*img*tau) / (1. + weights*tau)
    if nonneg: xnew = numpy.clip(xnew, 0, None)  
    
    # (5) calculate the new stepsizes
    theta = 1.0 / numpy.sqrt(1 + 2*gam*tau)
    tau   = tau*theta
    sig   = sig/theta 
    
    # (6) update variables
    xbar = xnew + theta*(xnew  - x)
    x    = xnew.copy()
  
    # (0) store cost 
    if cost is not None: 
      grad(x, fwd_x)
      cost[i] = 0.5*(weights*(x - img)**2).sum() + numpy.sqrt((fwd_x**2).sum(axis = 0)).sum()
      if verbose: print(cost[i])

  return x
