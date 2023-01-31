from clustervisualizer.ClusterAnalyzer import ClusterAnalyzer as clust

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.datasets import load_iris
from sklearn.cluster import KMeans



#### Load iris data set
X, y = load_iris(as_frame=True, return_X_y=True)
# ... and add flower type to X dataframe
type_dict = { 0 : 'setosa',
              1 : 'versicolor',
              2 : 'virginica' }
X['type'] = [ type_dict[elem] for elem in y ]



#### Cluster data using hard clustering
kmeans_obj = KMeans(n_clusters=3, random_state=1)
fit_params = ['sepal length (cm)', 'sepal width (cm)',
              'petal length (cm)', 'petal width (cm)']
preds = kmeans_obj.fit_predict( X[fit_params] )



####  Perform cluster analysis
clust1 = clust(
            X,   # dataframe to analyze
            preds   # 1d array of predictions (in this case)
              )
# Create histograms
hist_vars = [ *fit_params, 'type' ]
clust1.hist1d(hist_vars=hist_vars,
              bins=15)