from clustervisualizer.ClusterAnalyzer import ClusterAnalyzer as clust

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture



#### Load iris data set
X, y = load_iris(as_frame=True, return_X_y=True)
# ... and add flower type to X dataframe
type_dict = { 0 : 'setosa',
              1 : 'versicolor',
              2 : 'virginica' }
X['type'] = [ type_dict[elem] for elem in y ]



#### Cluster data using soft clustering
fit_params = ['sepal length (cm)', 'sepal width (cm)',
              'petal length (cm)', 'petal width (cm)']
gm = GaussianMixture(n_components=3).fit( X[fit_params] )
preds = gm.predict_proba( X[fit_params] )



####  Perform cluster analysis
## Make cluster analysis instance and name clusters
clust_inst = clust(
                X,
                preds,
                name_clusters={
                    'setosa': ('petal width (cm)', 
                               lambda x: -1*np.mean(x)),
                    'virginica': 'petal length (cm)',
                    'versicolor': ('sepal width (cm)',
                                   lambda x: -1*np.mean(x))
                                }
                    )
## Make histograms
hist_vars = [ *fit_params, 'type' ]
clust_inst.hist1d(bins=15, hist_vars=hist_vars)
clust_inst.hist2d(bins=(10,10),
                  hist_var='probability')