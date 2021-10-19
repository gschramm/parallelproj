import math
import numpy
from numba import njit, stencil

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


#-----------------------------------------------------------------------------------
@stencil
def fwd_diff2d_0(x):
  return x[1,0] - x[0,0]

@stencil
def back_diff2d_0(x):
  return x[0,0] - x[-1,0]

@stencil
def fwd_diff2d_1(x):
  return x[0,1] - x[0,0]

@stencil
def back_diff2d_1(x):
  return x[0,0] - x[0,-1]

#-----------------------------------------------------------------------------

@stencil
def fwd_diff3d_0(x):
  return x[1,0,0] - x[0,0,0]

@stencil
def back_diff3d_0(x):
  return x[0,0,0] - x[-1,0,0]

@stencil
def fwd_diff3d_1(x):
  return x[0,1,0] - x[0,0,0]

@stencil
def back_diff3d_1(x):
  return x[0,0,0] - x[0,-1,0]

@stencil
def fwd_diff3d_2(x):
  return x[0,0,1] - x[0,0,0]

@stencil
def back_diff3d_2(x):
  return x[0,0,0] - x[0,0,-1]

#-----------------------------------------------------------------------------

@stencil
def fwd_diff4d_0(x):
  return x[1,0,0,0] - x[0,0,0,0]

@stencil
def back_diff4d_0(x):
  return x[0,0,0,0] - x[-1,0,0,0]

@stencil
def fwd_diff4d_1(x):
  return x[0,1,0,0] - x[0,0,0,0]

@stencil
def back_diff4d_1(x):
  return x[0,0,0,0] - x[0,-1,0,0]

@stencil
def fwd_diff4d_2(x):
  return x[0,0,1,0] - x[0,0,0,0]

@stencil
def back_diff4d_2(x):
  return x[0,0,0,0] - x[0,0,-1,0]

@stencil
def fwd_diff4d_3(x):
  return x[0,0,0,1] - x[0,0,0,0]

@stencil
def back_diff4d_3(x):
  return x[0,0,0,0] - x[0,0,0,-1]


#-----------------------------------------------------------------------------

@njit(parallel = True)
def grad2d(x, g):
  fwd_diff2d_0(x, out = g[0,]) 
  fwd_diff2d_1(x, out = g[1,]) 

@njit(parallel = True)
def grad3d(x, g):
  fwd_diff3d_0(x, out = g[0,]) 
  fwd_diff3d_1(x, out = g[1,]) 
  fwd_diff3d_2(x, out = g[2,]) 

@njit(parallel = True)
def grad4d(x, g):
  fwd_diff4d_0(x, out = g[0,]) 
  fwd_diff4d_1(x, out = g[1,]) 
  fwd_diff4d_2(x, out = g[2,]) 
  fwd_diff4d_3(x, out = g[3,]) 

def grad(x,g):
  """
  Calculate the gradient of 2d,3d, or 4d array via the finite forward diffence

  Arguments
  ---------
  
  x ... a 2d, 3d, or 4d numpy array
  g ... (output) array of size ((x.ndim,), x.shape) used to store the ouput

  Examples
  --------

  import numpy
  import pynucmed
  x = numpy.random.rand(20,20,20)
  g = numpy.zeros((x.ndim,) + x.shape) 
  pynucmed.misc.grad(x,g)
  y = pynucmed.misc.div(g) 

  Note
  ----

  This implementation uses the numba stencil decorators in combination with
  jit in parallel nopython mode

  """
  ndim = x.ndim
  if   ndim == 2: grad2d(x, g)
  elif ndim == 3: grad3d(x, g)
  elif ndim == 4: grad4d(x, g)
  else          : raise TypeError('Invalid dimension of input') 

#-----------------------------------------------------------------------------

def complex_grad(x,g):
  """
  Calculate the gradient of 2d,3d, or 4d complex array via the finite forward diffence

  Arguments
  ---------
  
  x ... a complex numpy arrays represented by 2 float arrays
        2D ... x.shape = [n0,n1,2]
        3D ... x.shape = [n0,n1,n2,2]
        4D ... x.shape = [n0,n1,n2,n3,2]

  g ... (output) array of size ((2*x[...,0].ndim,), x[...,0].shape) used to store the ouput

  Note
  ----

  This implementation uses the numba stencil decorators in combination with
  jit in parallel nopython mode.
  The gradient is calculated separately for the real and imag part and
  concatenated together.

  """
  ndim = x[...,0].ndim
  if   ndim == 2: 
    grad2d(x[...,0], g[:ndim,...])
    grad2d(x[...,1], g[ndim:,...])
  elif ndim == 3: 
    grad3d(x[...,0], g[:ndim,...])
    grad3d(x[...,1], g[ndim:,...])
  elif ndim == 4: 
    grad4d(x[...,0], g[:ndim,...])
    grad4d(x[...,1], g[ndim:,...])
  else          : raise TypeError('Invalid dimension of input') 

#-----------------------------------------------------------------------------

@njit(parallel = True)
def div2d(g):
  tmp = numpy.zeros(g.shape)
  back_diff2d_0(g[0,], out = tmp[0,]) 
  back_diff2d_1(g[1,], out = tmp[1,]) 

  return tmp[0,] + tmp[1,]

@njit(parallel = True)
def div3d(g):
  tmp = numpy.zeros(g.shape)
  back_diff3d_0(g[0,], out = tmp[0,]) 
  back_diff3d_1(g[1,], out = tmp[1,]) 
  back_diff3d_2(g[2,], out = tmp[2,]) 
 
  return tmp[0,] + tmp[1,] + tmp[2,]

@njit(parallel = True)
def div4d(g):
  tmp = numpy.zeros(g.shape)
  back_diff4d_0(g[0,], out = tmp[0,]) 
  back_diff4d_1(g[1,], out = tmp[1,]) 
  back_diff4d_2(g[2,], out = tmp[2,]) 
  back_diff4d_3(g[3,], out = tmp[3,]) 
 
  return tmp[0,] + tmp[1,] + tmp[2,] + tmp[3,]

def div(g):
  """
  Calculate the divergence of 2d, 3d, or 4d array via the finite backward diffence

  Arguments
  ---------
  
  g ... a gradient array of size ((x.ndim,), x.shape)

  Returns
  -------

  an array of size g.shape[1:]

  Examples
  --------

  import numpy
  import pynucmed
  x = numpy.random.rand(20,20,20)
  g = numpy.zeros((x.ndim,) + x.shape) 
  pynucmed.misc.grad(x,g)
  y = pynucmed.misc.div(g) 

  Note
  ----

  This implementation uses the numba stencil decorators in combination with
  jit in parallel nopython mode

  See also
  --------

  pynucmed.misc.grad
  """
  ndim = g.shape[0]
  if   ndim == 2: return div2d(g)
  elif ndim == 3: return div3d(g)
  elif ndim == 4: return div4d(g)
  else          : raise TypeError('Invalid dimension of input') 


def complex_div(g):
  """
  Calculate the divergence of 2d, 3d, or 4d "complex" array via the finite backward diffence

  Arguments
  ---------
  
  g ... a gradient array of size (2*(x.ndim,), x.shape)

  Returns
  -------

  a real array of shape (g.shape[1:] + (2,)) representing the complex array by 2 real arrays

  Note
  ----

  This implementation uses the numba stencil decorators in combination with
  jit in parallel nopython mode

  See also
  --------

  pynucmed.misc.grad
  """

  ndim = g.shape[0] // 2
  tmp  = numpy.zeros(g.shape[1:] + (2,))

  if ndim == 2: 
    tmp[...,0] = div2d(g[:ndim,...])
    tmp[...,1] = div2d(g[ndim:,...])
  elif ndim == 3: 
    tmp[...,0] = div3d(g[:ndim,...])
    tmp[...,1] = div3d(g[ndim:,...])
  elif ndim == 4: 
    tmp[...,0] = div4d(g[:ndim,...])
    tmp[...,1] = div4d(g[ndim:,...])
  else: raise TypeError('Invalid dimension of input') 

  return tmp

#----------------------------------------------------------------------------------------------------

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

  def __init__(self, joint_grad_field = None):

    # e is the normalized joint gradient field that
    # we are only interested in the gradient component
    # perpendicular to it
    self.e = None
    
    if joint_grad_field is not None:
      norm   = numpy.linalg.norm(joint_grad_field, axis = 0)
      self.e = numpy.divide(joint_grad_field, norm, out = numpy.zeros_like(joint_grad_field), 
                            where = (norm != 0)) 

  def fwd(self, x):
    g = numpy.zeros((x.ndim,) + x.shape)
    grad(x, g)

    if self.e is not None:
      g = g - (g*self.e).sum(0)*self.e

    return g

  def adjoint(self, y):
    if self.e is not None:
      return -div(y - (y*self.e).sum(0)*self.e)
    else:
      return -div(y)

#----------------------------------------------------------------------------------------------------

class GradientNorm:
  """ 
  norm of a gradient field

  Parameters
  ----------

  name : str
    name of the norm
    'l2_l1' ... mixed L2/L1 (sum of pointwise Euclidean norms in every voxel)

  beta : float
    factor multiplied to the norm (default 1)
  """
  def __init__(self, name = 'l2_l1', beta = 1):
    self.name = name
    self.beta = beta
 
    if not self.name in ['l2_l1']:
     raise NotImplementedError

  def eval(self,x):
    if self.name == 'l2_l1':
      n = numpy.linalg.norm(x, axis = 0).sum()

    if self.beta != 1: n *= self.beta

    return n

  def prox_convex_dual(self, x, sigma = None):
    """ proximal operator of the convex dual of the norm
    """
    if self.name == 'l2_l1':
      gnorm = numpy.linalg.norm(x, axis = 0)
      if self.beta != 1: gnorm /= self.beta
      x /= numpy.clip(gnorm, 1, None)
