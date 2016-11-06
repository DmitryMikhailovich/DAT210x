import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Look pretty...
matplotlib.style.use('ggplot')


# %%
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
df = pd.read_csv('./Datasets/wheat.data')


# %%
# TODO: Create a 2d scatter plot that graphs the
# area and perimeter features
# 
plt.scatter(df['area'], df['perimeter'], alpha=.75, marker='^')
plt.xlabel('area')
plt.ylabel('perimeter')
plt.show()


# %%
# TODO: Create a 2d scatter plot that graphs the
# groove and asymmetry features
# 
plt.scatter(df['groove'], df['asymmetry'], alpha=.75, marker='.')
plt.xlabel('groove')
plt.ylabel('asymmetry')
plt.show()


# %%
# TODO: Create a 2d scatter plot that graphs the
# compactness and width features
# 
plt.scatter(df['compactness'], df['width'], alpha=.75, marker='o')
plt.xlabel('compactness')
plt.ylabel('width')
plt.show()


#%%
# BONUS TODO: 
# After completing the above, go ahead and run your program
# Check out the results, and see what happens when you add
# in the optional display parameter marker with values of
# either '^', '.', or 'o'.





