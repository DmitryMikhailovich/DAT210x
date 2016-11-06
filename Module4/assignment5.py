#%%
import pandas as pd
from sklearn.manifold import Isomap
from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import pathlib
# Look pretty...
matplotlib.style.use('ggplot')


#%%
# TODO: Start by creating a regular old, plain, "vanilla"
# python list. You can call it 'samples'.
#
samples = []
colors = []

#%%
# TODO: Write a for-loop that iterates over the images in the
# Module4/Datasets/ALOI/32/ folder, appending each of them to
# your list. Each .PNG image should first be loaded into a
# temporary NDArray, just as shown in the Feature
# Representation reading.
#
# Optional: Resample the image down by a factor of two if you
# have a slower computer. You can also convert the image from
# 0-255  to  0.0-1.0  if you'd like, but that will have no
# effect on the algorithm's results.
#
path = pathlib.Path('./Datasets/ALOI/32/')
for file in path.iterdir():
    img = misc.imread(file)
    samples.append(img.reshape(-1))
    colors.append('b')




#%%
# TODO: Once you're done answering the first three questions,
# right before you converted your list to a dataframe, add in
# additional code which also appends to your list the images
# in the Module4/Datasets/ALOI/32_i directory. Re-run your
# assignment and answer the final question below.
#

path_i = pathlib.Path('./Datasets/ALOI/32i/')
for file in path_i.iterdir():
    img = misc.imread(file)
    samples.append(img.reshape(-1))
    colors.append('r')

# %%
# TODO: Convert the list to a dataframe
#
df = pd.DataFrame.from_records(samples)



#%%
# TODO: Implement Isomap here. Reduce the dataframe df down
# to three components, using K=6 for your neighborhood size
#
iso = Isomap(n_neighbors=6, n_components=3)
iso.fit(df)
T = iso.transform(df)


#%%
# TODO: Create a 2D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker. Graph the first two
# isomap components
#
%matplotlib inline
plt.scatter(T[:, 0], T[:, 1], marker='o', c=colors)
plt.xlabel('0 component')
plt.ylabel('1 component')
plt.show()


%matplotlib tk

#%%
# TODO: Create a 3D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker:
#

ax = plt.subplot(111, projection='3d')
ax.scatter(T[:, 0], T[:, 1], T[:, 2], c=colors)

plt.show()

