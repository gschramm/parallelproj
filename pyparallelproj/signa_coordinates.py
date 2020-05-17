import h5py
import numpy as np

def load_signa_coordinates(only2d   = False,
                           views    = np.arange(224),
                           Dscanner = 650,
                           Lscanner = 250,
                           planes   = np.arange(1981),
                           only2D   = False):
  if only2d:
    planes = np.arange(1)

  xstart = np.zeros((3,planes.shape[0],357,views.shape[0]), dtype = np.float32)
  xend   = np.zeros((3,planes.shape[0],357,views.shape[0]), dtype = np.float32)
  
  # setup the LOR coordinates
  with h5py.File('../data/proj_coords.h5', 'r') as data:
  
    for i, view in enumerate(views):
      xstart[...,i] = data['start/' + str(view)][:,planes,:]
      xend[...,i]   = data['end/' + str(view)][:,planes,:]
  
  # shift LOR coordinates such that iso center is at 0
  xstart[0,...] += -148.5
  xstart[1,...] += -148.5
  xstart[2,...] += -44
  xend[0,...]   += -148.5
  xend[1,...]   += -148.5
  xend[2,...]   += -44
  
  # scale to ring diameter and length in mm
  xstart[0,...] *= 0.5 * Dscanner / 158.94327
  xend[0,...]   *= 0.5 * Dscanner / 158.94327
  xstart[1,...] *= 0.5 * Dscanner / 158.94327
  xend[1,...]   *= 0.5 * Dscanner / 158.94327
  xstart[2,...] *= 0.5 * Lscanner / 43.555344
  xend[2,...]   *= 0.5 * Lscanner / 43.555344
  
  # in the 2D case we set the 3rd coordinate to 0
  if only2D:
    xstart[2,...] = 0
    xend[2,...]   = 0

  return np.swapaxes(np.swapaxes(xstart, 1, 3), 1, 2), np.swapaxes(np.swapaxes(xend, 1, 3), 1, 2)

#-----------------------------------------------------------------------------------------

if __name__ == '__main__':

  views = np.arange(15)*224//15
  xstart, xend = load_signa_coordinates(only2d = True, views = views)

  import matplotlib.pyplot as py
  
  fig, ax = py.subplots(3,5,figsize = (15,9))
  for i in range(357):
    for k in range(15):
      ax.flatten()[k].plot([xstart[0,:,k,0].flatten()[i], xend[0,:,k,0].flatten()[i]],
                           [xstart[1,:,k,0].flatten()[i], xend[1,:,k,0].flatten()[i]], 
                           color = 'b', lw = 0.3)
      ax.flatten()[k].set_xlim(-370,370)
      ax.flatten()[k].set_ylim(-370,370)
      ax.flatten()[k].set_title(f'view {views[k]}')
  
  fig.tight_layout()
  fig.show()
