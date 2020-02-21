import numpy as np
import h5py

def setup_testdata(nviews = 7):
  xstart = np.zeros((3,1981,357,nviews), dtype = np.float32)
  xend   = np.zeros((3,1981,357,nviews), dtype = np.float32)
  
  # setup the LOR coordinates
  with h5py.File('../data/proj_coords.h5', 'r') as data:
    img = np.swapaxes(data['images/cylinder'][:],0,2).astype(np.float32)

    img[img == 0] = 1e-6

    for i, view in enumerate((np.arange(nviews)*224/nviews).astype(int)):
      xstart[...,i] = data['start/' + str(view)][:]
      xend[...,i]   = data['end/' + str(view)][:]

  # shift LOR coordinates such that iso center is at 0
  xstart[0,...] += (-158.943269 + 10.443268)
  xstart[1,...] += (-158.943269 + 10.443268)
  xstart[2,...] += (-41.5649262 + 0.4446602)
  xend[0,...]   += (-158.943269 + 10.443268)
  xend[1,...]   += (-158.943269 + 10.443268)
  xend[2,...]   += (-41.5649262 + 0.4446602)

  voxsize    = np.array([0.73,0.73,0.8], dtype = np.float32)
  img_origin = (-np.array(img.shape, dtype = np.float32)/2 + 0.5) * voxsize

  return xstart, xend, img, img_origin, voxsize
