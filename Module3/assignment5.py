import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from pandas.tools.plotting import andrews_curves

# Look pretty...
matplotlib.style.use('ggplot')


# %%
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
df = pd.read_csv('./Datasets/wheat.data')



#%%
# TODO: Drop the 'id' feature
# 
df.drop(labels=['id'], axis=1, inplace=True)


# %%
# TODO: Plot a Andrew curves chart grouped by
# the 'wheat_type' feature. Be sure to set the optional
# display parameter alpha to 0.4
# 
andrews_curves(df, 'wheat_type', alpha=.4)


plt.show()

#%%
df.drop(labels=['area', 'perimeter'], axis=1, inplace=True)
andrews_curves(df, 'wheat_type', alpha=.4)


plt.show()
