import numpy as np
import h5py

def setup_testdata(nviews = 8):
  xstart = np.zeros((3,1981,357,nviews), dtype = np.float32)
  xend   = np.zeros((3,1981,357,nviews), dtype = np.float32)
  
  # setup the LOR coordinates
  with h5py.File('../data/proj_coords.h5', 'r') as data:
    img = np.swapaxes(data['images/cylinder'][:],0,2).astype(np.float32)

    for i, view in enumerate((np.arange(nviews)*224/nviews).astype(int)):
      xstart[...,i] = data['start/' + str(view)][:]
      xend[...,i]   = data['end/' + str(view)][:]

  scanner_center   = np.array([150.38,158.50,45.99], dtype = np.float32)
  scanner_diameter = np.array([314.12,317.88,83.13], dtype = np.float32)

  #voxsize    = np.array([0.75,0.75,0.93], dtype = np.float32)
  voxsize    = np.array([0.7183755,0.7183755,0.9340433], dtype = np.float32)
  #img_origin = (scanner_center - 0.5*np.array(img.shape)*voxsize).astype(np.float32)
  img_origin = np.array([41.821224,41.821224,0.91168165], dtype = np.float32)

  return xstart, xend, img, img_origin, voxsize
