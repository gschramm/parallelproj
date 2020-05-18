import numpy as np
import matplotlib.pyplot as py

class RegularPolygonPETScanner:

  def __init__(self,
               R                   = 325,
               crystals_per_module = np.array([16,9]),
               crystal_size        = np.array([4.,5.]),
               nmodules            = np.array([28,5]),
               module_gap_axial    = 5.):

    self.R = R
    self.crystals_per_module = crystals_per_module
    self.crystal_size = crystal_size
    self.nmodules = nmodules
    self.module_gap_axial = module_gap_axial

    self.ncrystals_per_plane = nmodules[0]*crystals_per_module[0]
    self.ncrystals_axial     = nmodules[1]*crystals_per_module[1]

    self.calculate_crystal_coordinates()

  def calculate_crystal_coordinates(self):
    # the distance of a crystal to the center of the module
    d = (np.arange(self.crystals_per_module[0]) - self.crystals_per_module[0]/2 + 0.5)*self.crystal_size[0]
    
    # x0 and x1 crystal coordinates of one ring (single layer of crystals)
    self.xc0 = np.zeros(self.ncrystals_per_plane)
    self.xc1 = np.zeros(self.ncrystals_per_plane)
   
    self.alpha_module = np.linspace(0,2*np.pi, self.nmodules[0]+1)[:-1]

    for i,alpha in enumerate(self.alpha_module):
      self.xc0[i*self.crystals_per_module[0]:(i+1)*self.crystals_per_module[0]] = self.R*np.cos(alpha) - d*np.sin(alpha)
      self.xc1[i*self.crystals_per_module[0]:(i+1)*self.crystals_per_module[0]] = self.R*np.sin(alpha) + d*np.cos(alpha)
    
    self.xc2 = np.zeros(self.ncrystals_axial)
    
    for i in range(self.nmodules[1]):
      self.xc2[i*self.crystals_per_module[1]:(i+1)*self.crystals_per_module[1]] = (
          np.arange(self.crystals_per_module[1])*self.crystal_size[1] + 
          i*(self.crystals_per_module[1]*self.crystal_size[1] + self.module_gap_axial))
    
    # shift center in x2 direction to 0
    self.xc2 -= 0.5*self.xc2.max()

  def show_crystal_config(self, show_crystal_numbers = False):
    fig, ax = py.subplots(1, 2, figsize = (12,7))
    ax[0].plot(self.xc0, self.xc1, 'r.')

    ax[1].plot(self.xc2, np.full(self.ncrystals_axial, self.xc1.max()), 'r.')
    ax[1].plot(self.xc2, np.full(self.ncrystals_axial, self.xc1.min()), 'r.')

    ax[0].set_xlabel('xc0')
    ax[0].set_ylabel('xc1')
    ax[1].set_xlabel('xc2')
    ax[1].set_ylabel('xc1')

    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    for axx in ax.flatten():
      axx.grid(ls = ':')

    if show_crystal_numbers:
      for i in range(self.ncrystals_per_plane):
        ax[0].text(self.xc0[i], self.xc1[i], str(i))
      for i in range(self.ncrystals_axial):
        ax[1].text(self.xc2[i], self.xc1.max(), str(i))
        ax[1].text(self.xc2[i], self.xc1.min(), str(i))

    fig.tight_layout()
    fig.show()

    return fig, ax

#-----------------------------------------------------------------------------------------------

if __name__ == '__main__':
  scanner = RegularPolygonPETScanner()
  fig, ax = scanner.show_crystal_config(show_crystal_numbers = True)

## michelogram test
#from michelogram import get_michelogram_index
#naxial      = crystals_per_module[1]*nmodules[1]
#n0, n1, seg = get_michelogram_index(naxial,0)
#
#m = np.zeros((naxial,naxial), dtype = np.int)
#s = np.zeros((naxial,naxial), dtype = np.int)
#for i in range(naxial**2):
#  m[n0[i],n1[i]] = i
#  s[n0[i],n1[i]] = seg[i]


#-----------------------------------------------------------------------------------------------
#fig,ax = py.subplots(3,5, figsize = (15,9))
#
#for k, view in enumerate(np.arange(15)*(224//15)):
#  istart = np.concatenate((np.repeat(np.arange(ncrystals_per_plane//2 - 1),2), 
#                           [ncrystals_per_plane//2 -1])) - view
#  iend   = np.concatenate(([-1],np.repeat(-np.arange(ncrystals_per_plane//2 - 1) - 2,2))) - view
#  
#  xstart0 = xc0[istart[45:-45]]
#  xstart1 = xc1[istart[45:-45]]
#  xend0   = xc0[iend[45:-45]]
#  xend1   = xc1[iend[45:-45]]
#  
#  for i in range(xstart0.shape[0]):
#    ax.flatten()[k].plot([xstart0[i], xend0[i]], [xstart1[i], xend1[i]], color = 'b', lw = 0.1)
#    ax.flatten()[k].set_aspect('equal')
#
#fig.tight_layout()
#fig.show()
