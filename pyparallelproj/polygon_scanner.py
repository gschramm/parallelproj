import numpy as np
import matplotlib.pyplot as py

R                   = 325
crystals_per_module = np.array([16,9])
crystal_size        = np.array([4,4])
nmodules            = 28

d = (np.arange(crystals_per_module[0]) - crystals_per_module[0]/2 + 0.5)*crystal_size[0]

# crystal coordinates of one ring (single layers of crystals
ncrystals_per_plane = nmodules*crystals_per_module[0]
xc0 = np.zeros(ncrystals_per_plane)
xc1 = np.zeros(ncrystals_per_plane)

for i,alpha in enumerate(np.linspace(0,2*np.pi,nmodules+1)[:-1]):
  xc0[i*crystals_per_module[0]:(i+1)*crystals_per_module[0]] = R*np.cos(alpha) - d*np.sin(alpha)
  xc1[i*crystals_per_module[0]:(i+1)*crystals_per_module[0]] = R*np.sin(alpha) + d*np.cos(alpha)


#-----------------------------------------------------------------------------------------------
fig,ax = py.subplots(3,5, figsize = (15,9))

for k, view in enumerate(np.arange(15)*(224//15)):
  istart = np.concatenate((np.repeat(np.arange(ncrystals_per_plane//2 - 1),2), 
                           [ncrystals_per_plane//2 -1])) - view
  iend   = np.concatenate(([-1],np.repeat(-np.arange(ncrystals_per_plane//2 - 1) - 2,2))) - view
  
  x0start = xc0[istart[45:-45]]
  x1start = xc1[istart[45:-45]]
  x0end   = xc0[iend[45:-45]]
  x1end   = xc1[iend[45:-45]]
  
  for i in range(x0start.shape[0]):
    ax.flatten()[k].plot([x0start[i], x0end[i]], [x1start[i], x1end[i]], color = 'b', lw = 0.3)

fig.tight_layout()
fig.show()
