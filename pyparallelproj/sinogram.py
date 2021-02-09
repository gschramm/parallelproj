import numpy as np

class PETSinogramParameters:
  """ Sinogram parameter for a cylinderical PET scanner consisting of modules

  Paramters
  ---------
  scanner : RegularPolygonPETScanner
    an object containing the parameter of the cylindrical PET scanner

  span : int
    number specifying the meshing of axial planes (spanning).
    At the moment, only span 1 is supported (no axial mashing)
    Default: 1

  rtrim : int
    numer of LORs to trim at both sides in the radial direction
    Default: 46

  ntofbins : int
    number of TOF bins in the sinogram.
    For a non-TOF sinogram use 1.
    Default: 1

  tofbin_width : float
    (spatial) width of every TOF bin.
    Should be in the same units as the crystal coordinates of the scanner.
    Default: None

  spatial_dim_order : 1d numpy array with permutation of [0,1,2]
    order of the spatial dimensions
    (default) [0,1,2] -> (radial, angular, planes)
              [1,0,2] -> (angular, radial, planes)
              [2,1,0] -> (planes, angular, radial)
              ...

  tof_dim : int
    position of tof dimension
    -1 (default) ... TOF is last dimension
    other values are not supported yet
  """
  def __init__(self, scanner, span = 1, rtrim = 46, ntofbins = 1, tofbin_width = None,
                     spatial_dim_order = np.array([0,1,2]), tof_dim = -1):

    if (scanner.ncrystals_per_plane % 2) != 0:
      raise ValueError(f'Scanners with odd number of crystals per plan not supported yet.')

    if span != 1:
      raise ValueError(f'Span {span} not supported yet')
    else:
      self.scanner = scanner
      self.span    = span
      self.rtrim   = rtrim
 
      self.nrad     = (scanner.ncrystals_per_plane + 1) - 2*self.rtrim
      self.nviews   = scanner.ncrystals_per_plane//2
      self.nplanes  = scanner.ncrystals_axial**2

      self.spatial_dim_order = spatial_dim_order

      # TOF parameters
      self.ntofbins     = ntofbins
      self.tofbin_width = tofbin_width
      self.tof_dim      = tof_dim

      self.nLORs_per_view = self.nrad*self.nplanes 

      self.spatial_shape = tuple(np.array([self.nrad, self.nviews, self.nplanes])[self.spatial_dim_order])
      if self.tof_dim == -1:
        self.shape = self.spatial_shape + (self.ntofbins,) 
      else:
        raise ValueError('TOF dim must be -1')
                           
      self.istart_plane, self.iend_plane = self.get_plane_det_index()

  #-------------------------------------------------------------------
  def get_plane_det_index(self):
    start = np.arange(self.scanner.ncrystals_axial)
    end   = np.arange(self.scanner.ncrystals_axial)
  
    for i in range(1,self.scanner.ncrystals_axial):
      tmp1 = np.arange(self.scanner.ncrystals_axial-i)
      tmp2 = np.arange(self.scanner.ncrystals_axial-i) + i
  
      start = np.concatenate((start, tmp1, tmp2))
      end   = np.concatenate((end,   tmp2, tmp1))
  
    return start, end

  #-------------------------------------------------------------------
  def get_view_crystal_indices(self, views):
    """ get the trans-axial and axial crystal indices for a number of views of the sinogram

    Parameters
    ----------
    views : 1d numpy int array
      containing the views for which the crystals indices should be calculated

    Returns
    -------
    2 numpy arrays of shape (nradial, nviews, nplanes, 2) containing the indices of
    thes start and end detectors for the views.
    """
    i_tr_start = np.zeros((self.nrad, views.shape[0]), dtype = int)
    i_tr_end   = np.zeros((self.nrad, views.shape[0]), dtype = int)

    n = self.scanner.ncrystals_per_plane

    # the radial sampling is done in a zig zag pattern like
    # this is handy since then the number of "radial" bins per view is constant
    # for a scanner with an even number of detectors per plane
    # 0 -> -1
    # 0 -> -2
    # 1 -> -2
    # 1 -> -3
    # 2 -> -3
    # 2 -> -4

    for i, view in enumerate(views):
      i_tr_start[:,i] = (np.concatenate((np.repeat(np.arange(n//2),2),[n//2])) - view)[self.rtrim:-self.rtrim]
      i_tr_end[:,i]   = (np.concatenate(([-1],np.repeat(-np.arange(n//2) - 2,2))) - view)[self.rtrim:-self.rtrim]

    crystal_id_start = np.array(np.meshgrid(i_tr_start.flatten(), self.istart_plane)).T.reshape(-1,2) 
    crystal_id_end   = np.array(np.meshgrid(i_tr_end.flatten(),   self.iend_plane)).T.reshape(-1,2) 

    return (np.transpose(crystal_id_start.reshape((self.nrad, views.shape[0],self.nplanes,2)),
                         np.concatenate((self.spatial_dim_order,[3]))), 
            np.transpose(crystal_id_end.reshape((self.nrad, views.shape[0],self.nplanes,2)),
                         np.concatenate((self.spatial_dim_order,[3]))))

  #-------------------------------------------------------------------
  def sinogram_to_listmode(self, sinogram, return_multi_index = False,
                                 subset = 0, nsubsets = 1):
    """ Convert an unsigned int sinogram to a list of events (list-mode data)

    Parameters
    ----------
    sinogram : 4D numpy unsigned int array of shape (nrad, nviews, nplanes, ntofbins)
      containing the counts to be converted into listmode data

    return_multi_index : bool
      whether to return the sinogram multi-index for each LM event as well.
      This is useful for converting contamination and sensitivity sinograms to listmode data
      when simulated listmode data from sinograms.
      Default: False

    subset : int 
      sinogram subset number of input - default 0 (first subset)

    nsubsets : int
      number of sinogram subsets for input - defaut 1 (one subset = complete sinogram)

    Returns
    -------
    If return_multi_index is False
      A 2D numpy int16 array of shape (nevents, 5) where each listmode event is characterized
      by 5 integer number (2 start crystals IDs, 2 end crystal IDs, tof bin)
    If return_multi_index is True
      a 2D array with the sinogram multi-index of every event is returned as well
      (2 output arguments)
    """

    if not np.issubdtype(sinogram.flatten()[0], np.signedinteger):
      raise ValueError('Sinogram must be of type unsigned int for conversion to LM.')

    # events is a 2D array of all events
    # each event if characterize by 5 integers: 
    # [start_crystal_id_tr, start_crystal_id_ax, end_crystal_id_tr, end_crystal_id_ax, tofbin]
    istart, iend = self.get_view_crystal_indices(np.arange(self.nviews)[subset::nsubsets])

    events  = np.zeros((sinogram.sum(),5), dtype = np.int16)
    counter = 0
    
    it = np.nditer(sinogram, flags=['multi_index'])

    if return_multi_index:
      multi_index = np.zeros((events.shape[0],4), dtype = np.int16)

    for x in it:
      if x > 0:
        events[counter:(counter+x),0:2] = istart[it.multi_index[:-1]]
        events[counter:(counter+x):,2:4] = iend[it.multi_index[:-1]]
        # for the LM projector, the central tofbin is 0, so we have to shift
        # the tof index of the sinogram bu ntofbins // 2
        events[counter:(counter+x):,4]   = it.multi_index[-1] - self.ntofbins // 2

        if return_multi_index:
          multi_index[counter:(counter+x),:] = it.multi_index
    
        counter += x
  
    tmp = np.arange(events.shape[0])
    np.random.shuffle(tmp)

    if return_multi_index:
      return events[tmp,:], multi_index[tmp,:]
    else:
      return events[tmp,:]

#----------------------------------------------------------------------
if __name__ == '__main__':
 
  import matplotlib.pyplot as py
  from pet_scanners import RegularPolygonPETScanner
  scanner = RegularPolygonPETScanner(ncrystals_per_module = np.array([16,2]),
                                     nmodules             = np.array([28,1]))
  sino    = PETSinogramParameters(scanner)

  views = np.arange(15) * sino.nviews // 15

  istart, iend = sino.get_view_crystal_indices(views)

  xstart = scanner.get_crystal_coordinates(istart.reshape(-1,2)).reshape((sino.nrad,views.shape[0],sino.nplanes,3))
  xend   = scanner.get_crystal_coordinates(iend.reshape(-1,2)).reshape((sino.nrad,views.shape[0],sino.nplanes,3))

  fig,ax = py.subplots(3,5, figsize = (15,9))

  for k, view in enumerate(views):
    for i in range(xstart.shape[0]):
      ax.flatten()[k].plot([xstart[i,k,0,0], xend[i,k,0,0]], [xstart[i,k,0,1], xend[i,k,0,1]], 
                           color = 'b', lw = 0.1)
    ax.flatten()[k].set_xlim(-350,350)
    ax.flatten()[k].set_ylim(-350,350)
    ax.flatten()[k].set_title(f'view {view}')
  
  fig.tight_layout()
  fig.show()
