import math
import numpy
from numba import njit

#-----------------------------------------------------------------------------------
@njit(parallel = True)
def prox_tv2d(x,beta):

  for i in range(x.shape[1]):
    for j in range(x.shape[2]):

        norm = math.sqrt(x[0,i,j]**2 + x[1,i,j]**2) / beta

        if norm > 1:       
          x[0,i,j] /= norm
          x[1,i,j] /= norm

#-----------------------------------------------------------------------------------
@njit(parallel = True)
def prox_tv3d(x,beta):

  for i in range(x.shape[1]):
    for j in range(x.shape[2]):
      for k in range(x.shape[3]):

        norm = math.sqrt(x[0,i,j,k]**2 + x[1,i,j,k]**2 + x[2,i,j,k]**2) / beta

        if norm > 1:       
          x[0,i,j,k] /= norm
          x[1,i,j,k] /= norm
          x[2,i,j,k] /= norm

#-----------------------------------------------------------------------------------
@njit(parallel = True)
def prox_tv4d(x,beta):

  for i in range(x.shape[1]):
    for j in range(x.shape[2]):
      for k in range(x.shape[3]):
        for l in range(x.shape[4]):

         norm = math.sqrt(x[0,i,j,k,l]**2 + x[1,i,j,k,l]**2 + x[2,i,j,k,l]**2 + x[3,i,j,k,l]**2) / beta

         if norm > 1:       
           x[0,i,j,k,l] /= norm
           x[1,i,j,k,l] /= norm
           x[2,i,j,k,l] /= norm
           x[3,i,j,k,l] /= norm


#-----------------------------------------------------------------------------------
def prox_tv(x, beta):
  """
  Proximity operator for convex dual of TV.
  If the Euclidean norm of every point in x in shrunk to beta.

  Arguments:
  x    ... gradient array of shape (ndim, image.dim)
  beta ... critical norm
  """
  ndim = x.shape[0]

  if   ndim == 2: prox_tv2d(x, beta)
  elif ndim == 3: prox_tv3d(x, beta)
  else          : raise TypeError('Invalid dimension of input') 

