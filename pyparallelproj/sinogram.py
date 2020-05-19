import numpy as np

class PETSinogram:

  def __init__(self, scanner, data = None, span = 1, rtrim = 46):
    if span != 1:
      raise ValueError(f'Span {span} not supported yet')
    else:
      self.scanner = scanner
      self.data    = data
      self.span    = span
      self.rtrim   = rtrim
 
      self.nrad    = (scanner.ncrystals_per_plane + 1) - 2*self.rtrim
      self.nviews  = scanner.ncrystals_per_plane//2
      self.nplanes = scanner.ncrystals_axial**2

      self.nLORs_per_view = self.nrad*self.nplanes 

      self.shape = (self.nrad, self.nviews, self.nplanes)

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
    """ get an angular subset of the complete sinogram

    """
    i_tr_start = np.zeros((self.nrad, views.shape[0]), dtype = int)
    i_tr_end   = np.zeros((self.nrad, views.shape[0]), dtype = int)

    n = self.scanner.ncrystals_per_plane

    for i, view in enumerate(views):
      i_tr_start[:,i] = (np.concatenate((np.repeat(np.arange(n//2),2),[n//2])) - view)[self.rtrim:-self.rtrim]
      i_tr_end[:,i]   = (np.concatenate(([-1],np.repeat(-np.arange(n//2) - 2,2))) - view)[self.rtrim:-self.rtrim]

      crystal_id_start = np.array(np.meshgrid(i_tr_start.flatten(), self.istart_plane)).T.reshape(-1,2) 
      crystal_id_end   = np.array(np.meshgrid(i_tr_end.flatten(),   self.iend_plane)).T.reshape(-1,2) 

    return crystal_id_start, crystal_id_end


#----------------------------------------------------------------------
if __name__ == '__main__':
 
  import matplotlib.pyplot as py
  from pet_scanners import RegularPolygonPETScanner
  scanner = RegularPolygonPETScanner(ncrystals_per_module = np.array([16,1]),
                                     nmodules             = np.array([28,1]))
  sino    = PETSinogram(scanner)

  views = np.arange(15) * sino.nviews // 15

  istart, iend = sino.get_view_crystal_indices(views)
  xstart = scanner.get_crystal_coordinates(istart).reshape((sino.nrad,views.shape[0],sino.nplanes,3))
  xend   = scanner.get_crystal_coordinates(iend).reshape((sino.nrad,views.shape[0],sino.nplanes,3))

  fig,ax = py.subplots(3,5, figsize = (15,9))

  for k, view in enumerate(views):
    for i in range(xstart.shape[0]):
      ax.flatten()[k].plot([xstart[i,k,0,0], xend[i,k,0,0]], [xstart[i,k,0,1], xend[i,k,0,1]], 
                           color = 'b', lw = 0.1)
    ax.flatten()[k].set_xlim(-350,350)
    ax.flatten()[k].set_ylim(-350,350)
    ax.flatten()[k].set_title(f'view {view}')
    #ax.flatten()[k].set_aspect('equal')
  
  fig.tight_layout()
  fig.show()
