from clustervisualizer.ClusterAnalyzer import ClusterAnalyzer as clust
from clustervisualizer.CrossingAnalyzer import CrossingAnalyzer as cross

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.cluster import KMeans




#### Generate data
nl = 100
nh = 100
## simulate "low" temperature measurements
lt = np.random.normal(loc=10, scale=2, size=nl)
## simulate "high" temperature" measurements
ht = np.random.normal(loc=100, scale=10, size=nh)
X = pd.DataFrame({
    'temp': np.hstack( [lt, ht] ),
    # Make artifical time measurements
    'time': pd.date_range(start='2008-01-01',
                          freq='1 min',
                          periods=nl+nh)
                 })
y = pd.DataFrame({
    'src': np.hstack( [ np.full(nl,0), np.full(nh,1) ] )
                 })



#### Show time series
plt.scatter(X['time'], X['temp'], s=5)
## delineate cluster crossing
halfway_time = X['time'][ int(X.shape[0]/2) ] 
plt.axvline(halfway_time, c='black', linestyle='dashed')



#### Do simple clustering over temperature data
kmeans_obj = KMeans(n_clusters=2)
preds = kmeans_obj.fit_predict(X['temp'].values.reshape(-1,1))



####  Perform cluster analysis
## Make cluster analysis instance and name clusters
clust_inst = clust(X, preds,
                   name_clusters={'L': ('temp', lambda x: -1*np.mean(x)),
                                  'H': 'temp'})
## Create 1d histograms
clust_inst.hist1d(bins=15,
                  hist_vars='temp')
## Compute crossings between clusters (only one occurs here)
cross_dict = clust_inst.compute_crossings(
            time_var='time',
            min_crossing_duration = pd.Timedelta('1 min'),
            max_crossing_duration = pd.Timedelta('5 min'),
            min_beyond_crossing_duration = pd.Timedelta('5 min'),
            max_beyond_crossing_duration = pd.Timedelta('10 min'),
            min_cluster_frac = 0.8,
            order_matters = False
                                         )


#### Perform crossing analysis
## Make crossing analysis instance and call the 'L' to 'H' (or vice-versa)
## cluster boundary 'bd'
cross_inst = cross(cross_dict,
                   crossing_names={ ('L','H') : 'bd'},
                   keep_empty = False)
## Plot cluster crossing by viewing times series of temperature
## for very close-to-crossing points ('immediate')
cross_inst.plot_crossings(crossing='bd',
                          x='time',
                          y='temp',
                          point_type = 'immediate')
## Now for wider time interval ('normal')
cross_inst.plot_crossings(crossing='bd',
                          x='time',
                          y='temp',
                          point_type = 'normal')
## And for largest saved interval ('extended')
cross_inst.plot_crossings(crossing='bd',
                          x='time',
                          y='temp',
                          point_type = 'extended')