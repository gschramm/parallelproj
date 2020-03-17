import seaborn as sns
import pandas as pd
import matplotlib.pyplot as py
import numpy as np

mode = 'lm'

if mode == 'sino':
  df   = pd.read_csv('fermi_sino.txt', delim_whitespace = True)
  df   = df.append(pd.read_csv('genius_sino.txt', delim_whitespace = True))
  ymax = 5
elif mode == 'lm':
  df   = pd.read_csv('fermi_lm.txt', delim_whitespace = True)
  df   = df.append(pd.read_csv('genius_lm.txt', delim_whitespace = True))
  df['size'] /= 1e6
  ymax = 0.18


df['time/size'] = df['time']/df['size']

df_fwd   = df.loc[df.dir == 'fwd']
df_back  = df.loc[df.dir == 'back']
df_back2 = df.loc[df.dir == 'back2']

# print the results
res = df.groupby(['dir','size','processor'], sort=False)['time'].apply(np.median)
print(res)


fig, ax = py.subplots(2,1, figsize = (10,7), dpi = 50)
sns.swarmplot('size', 'time/size', hue='processor', data=df_fwd,   ax = ax[0], dodge = True, size = 5)
sns.swarmplot('size', 'time/size', hue='processor', data=df_back,  ax = ax[1], dodge = True, size = 5)
ax[0].grid(ls = ':')
ax[1].grid(ls = ':')
ax[0].set_title('fwd projection')
ax[1].set_title('back projection')
ax[0].set_ylim(0, ymax)
ax[1].set_ylim(0, ymax)
fig.tight_layout()
fig.savefig('/home/georg/Nextcloud/presentations/2003_parallelproj/' + mode + '_' + str(ymax) + '.png', 
            dpi = 300)
fig.show()
