import pandas as pd
import matplotlib.pyplot as plt


#%%
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
df = pd.read_csv('./Datasets/wheat.data')


# %%
# TODO: Drop the 'id' feature
# 
df.drop(labels='id', axis=1, inplace=True)


#%%
corr_mat = df.corr()


#%%
plt.imshow(corr_mat)
cols = list(df.columns)
plt.xticks(range(len(cols)), cols)
plt.yticks(range(len(cols)), cols)
plt.colorbar()


plt.show()


