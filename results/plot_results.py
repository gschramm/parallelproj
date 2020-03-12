import seaborn as sns
import pandas as pd
import matplotlib.pyplot as py

df = pd.read_csv('genius_lm.txt', delim_whitespace = True)
df['time/size'] = df['time']/df['size']

df_fwd  = df.loc[df.dir == 'fwd']
df_back = df.loc[df.dir == 'back']

fig, ax = py.subplots(2,1, figsize = (14,8))
sns.swarmplot('size', 'time/size', hue='processor',data=df_fwd,  ax = ax[0], dodge = True)
sns.swarmplot('size', 'time/size', hue='processor',data=df_back, ax = ax[1], dodge = True)
ax[0].grid(ls = ':')
ax[1].grid(ls = ':')
ax[0].set_title('fwd projection')
ax[1].set_title('back projection')
ax[0].set_ylim(0,df_fwd['time/size'].max())
ax[1].set_ylim(0,df_back['time/size'].max())
fig.tight_layout()
fig.show()
