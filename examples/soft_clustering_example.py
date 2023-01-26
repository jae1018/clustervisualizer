from clustervisualizer.ClusterAnalyzer import ClusterAnalyzer as clust

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture




X, y = load_iris(as_frame=True, return_X_y=True)



gm = GaussianMixture(n_components=3).fit(X)
preds = gm.predict_proba(X)



fldr = '/home/jedmond/Documents/testing_clustering_download/soft'
#fldr = None
clust_inst = clust(X, preds, output_folder=fldr)
clust_inst.hist1d(bins=15)
clust_inst.hist2d(bins=(10,10), hist_var='probability')