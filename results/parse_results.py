import pandas as pd

with open('p100_linux.txt', 'r') as file1:
  lines = file1.read().splitlines()

mode = 'LM'
counts = 4e5

if mode == 'sino':
  print(f'{mode}')
elif mode == 'LM':
  print(f'{mode} {counts:.1e} counts')

for ngpus in [1,2,4]:
  print(f'ngpus {ngpus}')
  df   = {}
  
  for nontof in [True, False]:
    for fov in ['brain','wb']:
      pattern = f"ngpus:{ngpus} counts:{counts} nsubsets:28 n:5 tpb:64 nontof:{nontof} img_mem_order:C sino_dim_order:['0', '1', '2'] fov:{fov} voxsize:['2', '2', '2']"  
      
      istart = lines.index(pattern)
      
      sino_fwd  = lines[istart+8]
      sino_back = lines[istart+9]
      
      lm_fwd  = lines[istart+19]
      lm_back = lines[istart+20]
  
      if mode == 'sino':
        fwd  = f'{sino_fwd[10:16]} +- {sino_fwd[30:36]}'
        back = f'{sino_back[10:16]} +- {sino_back[30:36]}'
      elif mode == 'LM':
        fwd  = f'{lm_fwd[9:15]} +- {lm_fwd[29:35]}'
        back = f'{lm_back[9:15]} +- {lm_back[29:35]}'
  
      if nontof:
        key = f'Non-TOF {fov.upper()} (s)'
      else:
        key = f'TOF {fov.upper()} (s)'
  
      df[key] = [fwd,back]
  
  df = pd.DataFrame(df, index = ['fwd', 'back'])
  print(df.to_markdown())
  print('')
