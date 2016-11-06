import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.manifold import Isomap
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
# If you'd like to try this lab with PCA instead of Isomap,
# as the dimensionality reduction technique:
Test_PCA = False


def plotDecisionBoundary(model, X, y):
  print("Plotting...")
  import matplotlib.pyplot as plt
  import matplotlib
  matplotlib.style.use('ggplot') # Look Pretty

  fig = plt.figure()
  ax = fig.add_subplot(111)

  padding = 0.1
  resolution = 0.1

  #(2 for benign, 4 for malignant)
  colors = {2:'royalblue',4:'lightsalmon'} 

  
  # Calculate the boundaris
  x_min, x_max = X[:, 0].min(), X[:, 0].max()
  y_min, y_max = X[:, 1].min(), X[:, 1].max()
  x_range = x_max - x_min
  y_range = y_max - y_min
  x_min -= x_range * padding
  y_min -= y_range * padding
  x_max += x_range * padding
  y_max += y_range * padding

  # Create a 2D Grid Matrix. The values stored in the matrix
  # are the predictions of the class at at said location
  import numpy as np
  xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                       np.arange(y_min, y_max, resolution))

  # What class does the classifier say?
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  # Plot the contour map
  plt.contourf(xx, yy, Z, cmap=plt.cm.seismic)
  plt.axis('tight')

  # Plot your testing points as well...
  for label in np.unique(y):
    indices = np.where(y == label)
    plt.scatter(X[indices, 0], X[indices, 1], c=colors[label], alpha=0.8)

  p = model.get_params()
  plt.title('K = ' + str(p['n_neighbors']))
  plt.legend(loc='best')
  plt.show()


# 
# TODO: Load in the dataset, identify nans, and set proper headers.
# Be sure to verify the rows line up by looking at the file in a text editor.
#
df = pd.read_csv('./Datasets/breast-cancer-wisconsin.data', header=None,  
                 names=['sample', 
'thickness', 'size', 'shape', 'adhesion', 
'epithelial', 'nuclei', 'chromatin', 'nucleoli', 'mitoses', 'status'], 
na_values=['?'])



# 
# TODO: Copy out the status column into a slice, then drop it from the main
# dataframe. You can also drop the sample column, since that doesn't provide
# us with any machine learning power.
#
y = df['status']
df.drop(['sample', 'status'], inplace=True, axis=1)



#
# TODO: With the labels safely extracted from the dataset, replace any nan values
# with the mean feature / column value
#
df = df.fillna(df.mean(axis=0))



#
# TODO: Do train_test_split. Use the same variable names as on the EdX platform in
# the reading material, but set the random_state=7 for reproduceability, and keep
# the test_size at 0.5 (50%).
#
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=.5, random_state=7)




#
# TODO: Experiment with the basic SKLearn preprocessing scalers. We know that
# the features consist of different units mixed in together, so it might be
# reasonable to assume feature scaling is necessary. Print out a description
# of the dataset, post transformation.
#
scaler = None
#scaler = preprocessing.RobustScaler()
#scaler = preprocessing.Normalizer()
#scaler = preprocessing.MinMaxScaler()

#scaler.fit(X_train)
#X_train_s = scaler.transform(X_train)
#X_test_s = scaler.transform(X_test)
X_train_s = X_train
X_test_s = X_test

#
# PCA and Isomap are your new best friends
model = None
if Test_PCA:
  print("Computing 2D Principle Components")
  #
  # TODO: Implement PCA here. save your model into the variable 'model'.
  # You should reduce down to two dimensions.
  #
  model = RandomizedPCA(n_components=2)
  

else:
  print("Computing 2D Isomap Manifold")
  #
  # TODO: Implement Isomap here. save your model into the variable 'model'
  # Experiment with K values from 5-10.
  # You should reduce down to two dimensions.
  #
  model = Isomap(n_components=2, n_neighbors=5)
  



#
# TODO: Train your model against data_train, then transform both
# data_train and data_test using your model. You can save the results right
# back into the variables themselves.
#
X_train_s = model.fit_transform(X_train_s)
X_test_s = model.transform(X_test_s)



# 
# TODO: Implement and train KNeighborsClassifier on your projected 2D
# training data here. You can use any K value from 1 - 15, so play around
# with it and see what results you can come up. Your goal is to find a
# good balance where you aren't too specific (low-K), nor are you too
# general (high-K). You should also experiment with how changing the weights
# parameter affects the results.
#
scores = []
for n in range(1, 16):
    knn = KNeighborsClassifier(n_neighbors=n, weights='uniform')
    knn.fit(X_train_s, y_train)


#
# INFO: Be sure to always keep the domain of the problem in mind! It's
# WAY more important to errantly classify a benign tumor as malignant,
# and have it removed, than to incorrectly leave a malignant tumor, believing
# it to be benign, and then having the patient progress in cancer. Since the UDF
# weights don't give you any class information, the only way to introduce this
# data into SKLearn's KNN Classifier is by "baking" it into your data. For
# example, randomly reducing the ratio of benign samples compared to malignant
# samples from the training set.



#
# TODO: Calculate + Print the accuracy of the testing set
#   
    score = knn.score(X_test_s, y_test)
    print('Score of KNN with K=', n, ' is ', score)
    scores.append(score)

    plotDecisionBoundary(knn, X_test_s, y_test)
print('Average score is', np.array(scores).mean())