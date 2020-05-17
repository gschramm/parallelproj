import numpy as np

def get_michelogram_index(naxial, span):

  if span != 0:
    raise ValueError(f'Span {span} not supported')

  if span == 0:
    planes = np.arange(naxial**2)

  nseg   = naxial
  nplanes_per_seg = np.zeros(nseg, dtype = np.int)
  
  for i in range(nseg):
   if i == 0:
     nplanes_per_seg[i] = naxial
   else:
     nplanes_per_seg[i] = 2*(naxial-i)
  
  nplanes_per_seg_cum = np.cumsum(nplanes_per_seg)

  n0  = np.zeros(planes.shape[0], dtype = int)
  n1  = np.zeros(planes.shape[0], dtype = int)
  seg = np.zeros(planes.shape[0], dtype = int)

  for i, plane in enumerate(planes):
    if plane < naxial:
      seg[i] = 0
      n0[i]  = plane
      n1[i]  = plane
    else:
      seg[i] = np.where(nplanes_per_seg_cum > plane)[0][0]
     
      is_even_plane = (plane  % 2 == 0)
      is_even_det   = (naxial % 2 == 0)

      if is_even_det:
        if is_even_plane:
          tmp_plane = plane
        else:
          tmp_plane = plane - 1
      else:
        if is_even_plane:
          tmp_plane = plane - 1
        else:
          tmp_plane = plane
     
      offset = (tmp_plane - nplanes_per_seg_cum[seg[i]-1]) // 2
      tmp_n0     = seg[i] + offset
      tmp_n1     = offset
   
      if is_even_det:
        if is_even_plane:
          n0[i] = tmp_n0
          n1[i] = tmp_n1
        else:
          n0[i] = tmp_n1
          n1[i] = tmp_n0
      else:
        if is_even_plane:
          n0[i] = tmp_n1
          n1[i] = tmp_n0
        else:
          n0[i] = tmp_n0
          n1[i] = tmp_n1

  return n0, n1, seg


