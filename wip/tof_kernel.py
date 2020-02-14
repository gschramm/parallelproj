import math

xstart = [1.,2.,3.,4.,5.,6.]
xend   = [20.,19.,18.,17.,16.,15.]

# offset of central tof bin from center between xstart and xend
tofcenter_offset = [0.,0.]

# number of tof bins
n_tofbins = 27

# tof resolution (in spatial units
sigma_tof = [1.3,1.4]

# width of a tof bin (in spatial nits)
tofbin_width = (30 + 2*sigma_tof[0])/n_tofbins

# number of sigmas to consider to calculation of tof kernel
n_sigmas = 3

# the LOR number
i = 0

#---------------------------------------------------------------------------
# voxel independent stuff

# calculate the difference vector 
u0 = xend[i*3 + 0] - xstart[i*3 + 0]
u1 = xend[i*3 + 1] - xstart[i*3 + 1]
u2 = xend[i*3 + 2] - xstart[i*3 + 2]

u_norm = math.sqrt(u0*u0 + u1*u1 + u2*u2)

u0 /= u_norm
u1 /= u_norm
u2 /= u_norm

# calculate mid point of LOR
x_m0 = 0.5*(xstart[i*3 + 0] + xend[i*3 + 0])
x_m1 = 0.5*(xstart[i*3 + 1] + xend[i*3 + 1])
x_m2 = 0.5*(xstart[i*3 + 2] + xend[i*3 + 2])


#---------------------------------------------------------------------------
# voxel specific stuff

# generate dummy voxel coordinates
x_v0 = xstart[i*3 + 0] + 0.99*u0*u_norm
x_v1 = xstart[i*3 + 1] + 0.99*u1*u_norm
x_v2 = xstart[i*3 + 2] + 0.99*u2*u_norm


# calculate which TOF bins it are within +- n_sigmas
# it runs from -n_tofbins//2 ... 0 ... n_tofbins//2
i1 = (((x_v0 - x_m0)*u0 + (x_v1 - x_m1)*u1 + (x_v2 - x_m2)*u2) - tofcenter_offset[i] + n_sigmas*sigma_tof[i]) / tofbin_width
i2 = (((x_v0 - x_m0)*u0 + (x_v1 - x_m1)*u1 + (x_v2 - x_m2)*u2) - tofcenter_offset[i] - n_sigmas*sigma_tof[i]) / tofbin_width

if i1 <= i2:
  it1 = math.floor(i1)
  it2 = math.ceil(i2)
else:
  it1 = math.floor(i2)
  it2 = math.ceil(i1)

n_half = n_tofbins // 2

if it1 < -n_half:
  it1 = -n_half
if it2 < -n_half:
  it2 = -n_half
if it1 > n_half:
  it1 = n_half
if it2 > n_half:
  it2 = n_half

tof_weights = []
inds        = []

for it in range(it1, it2 + 1):
  # calculate the coordinates of the center for the TOF bin
  x_c0 = x_m0 + (it*tofbin_width + tofcenter_offset[i])*u0
  x_c1 = x_m1 + (it*tofbin_width + tofcenter_offset[i])*u1
  x_c2 = x_m2 + (it*tofbin_width + tofcenter_offset[i])*u2
  
  # calculate the distance between voxel and tof bin center
  d = math.sqrt((x_c0 - x_v0)**2 +  (x_c1 - x_v1)**2 + (x_c2 - x_v2)**2)
  
  d_far  = d + 0.5*tofbin_width
  d_near = d - 0.5*tofbin_width
  
  tof_weight = 0.5*(math.erf(d_far/(math.sqrt(2)*sigma_tof[i])) - math.erf(d_near/(math.sqrt(2)*sigma_tof[i])))

  inds.append(it)
  tof_weights.append(tof_weight)

  # the index of the tof bin in the projection array is it + n_half!

# results
print(sum(tof_weights))
import matplotlib.pyplot as py
fig, ax = py.subplots()
ax.plot(inds,tof_weights,'.-')
ax.set_xlim(-n_half, n_half)
fig.show()
