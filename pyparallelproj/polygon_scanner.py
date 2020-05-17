import numpy as np
from michelogram import get_michelogram_index

R                   = 325
crystals_per_module = np.array([16,9])
crystal_size        = np.array([4.,5.])
nmodules            = np.array([28,5])
module_gap_axial    = 5.

# the distance of a crystal to the center of the module
d = (np.arange(crystals_per_module[0]) - crystals_per_module[0]/2 + 0.5)*crystal_size[0]

# x0 and x1 crystal coordinates of one ring (single layer of crystals)
ncrystals_per_plane = nmodules[0]*crystals_per_module[0]
xc0 = np.zeros(ncrystals_per_plane)
xc1 = np.zeros(ncrystals_per_plane)

for i,alpha in enumerate(np.linspace(0,2*np.pi,nmodules[0]+1)[:-1]):
  xc0[i*crystals_per_module[0]:(i+1)*crystals_per_module[0]] = R*np.cos(alpha) - d*np.sin(alpha)
  xc1[i*crystals_per_module[0]:(i+1)*crystals_per_module[0]] = R*np.sin(alpha) + d*np.cos(alpha)

xc2 = np.zeros(crystals_per_module[1]*nmodules[1])

for i in range(nmodules[1]):
  xc2[i*crystals_per_module[1]:(i+1)*crystals_per_module[1]] = (
      np.arange(crystals_per_module[1])*crystal_size[1] + 
      i*(crystals_per_module[1]*crystal_size[1] + module_gap_axial))

# shift center in x2 direction to 0
xc2 -= 0.5*xc2.max()

#-----------------------------------------------------------------------------------------------
# michelogram test
naxial      = crystals_per_module[1]*nmodules[1]
n0, n1, seg = get_michelogram_index(naxial,0)

m = np.zeros((naxial,naxial), dtype = np.int)
s = np.zeros((naxial,naxial), dtype = np.int)
for i in range(naxial**2):
  m[n0[i],n1[i]] = i
  s[n0[i],n1[i]] = seg[i]


#-----------------------------------------------------------------------------------------------
# import matplotlib.pyplot as py
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
