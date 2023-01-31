from clustervisualizer.ClusterAnalyzer import ClusterAnalyzer as clust

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.datasets import load_iris
from sklearn.cluster import KMeans



#### Load iris data set
X, y = load_iris(as_frame=True, return_X_y=True)



#### Cluster data using hard clustering
kmeans_obj = KMeans(n_clusters=3, random_state=1)
preds = kmeans_obj.fit_predict(X)



####  Perform cluster analysis
names = { 'setosa': ('petal width (cm)', lambda x: -1*np.mean(x)),
          'virginica': 'petal length (cm)',
          'versicolor': ('sepal width (cm)', lambda x: -1*np.mean(x)) }
## Make cluster analysis instance and name clusters
clust_inst = clust(X, preds,
                   name_clusters = names)
## Create histograms
clust_inst.hist1d(bins=15)
clust_inst.hist2d(bins=(10,10))