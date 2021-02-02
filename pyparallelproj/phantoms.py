import numpy as np

def ellipse_inds(nx, ny, rx, ry = None, x0 = 0, y0 = 0):

  if ry == None:
    ry = rx

  x = np.arange(nx) - nx/2 + 0.5
  y = np.arange(ny) - ny/2 + 0.5
  
  X,Y = np.meshgrid(x,y, indexing = 'ij')
  
  return np.where((((X-x0)/rx)**2 + ((Y-y0)/ry)**2) <= 1)

#--------------------------------------------------------------
def ellipse_phantom(n = 256, c = 3):
  r    = n/6
  
  img = np.zeros((n,n), dtype = np.float32)
  i0  = ellipse_inds(n, n, n/4, n/2.2)
  i1  = ellipse_inds(n, n, n/32, n/32)
  
  phis = np.linspace(0,2*np.pi,9)[:-1]
  
  img[i0] = 1
  img[i1] = 0
  
  for i, phi in enumerate(phis):
    i = ellipse_inds(n, n, np.sqrt(i+1)*n/80, np.sqrt(i+1)*n/80, x0 = r*np.sin(phi), y0 = r*np.cos(phi))
    img[i] = c

  return img
