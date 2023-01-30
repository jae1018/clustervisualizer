from clustervisualizer.ClusterAnalyzer import ClusterAnalyzer as clust
from clustervisualizer.CrossingAnalyzer import CrossingAnalyzer as cross
from simulate_detector import simulate_detector

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans





#### Simulate a detector moving between two different temperature gases
detector_sim = simulate_detector(
                init_posit = (0.1, 0.5),
                init_veloc = (0.0678, 0.0351),
                box_bnds = (0,2,0,2),
                max_iter = 1000,
                temp_boundary = 1.0
                                )
detector_sim.start()
df = detector_sim.get_data()
# add artifical time series to df
df['time'] = pd.date_range(start='2008-01-01',
                           freq='1 min',
                           periods=df.shape[0])
# show trajectory of detector
detector_sim.plot_trajectory()
# show full time series
plt.scatter(df['time'], df['temp'], s=5)



#### Do simple clustering over temperature data
kmeans_obj = KMeans(n_clusters=2)
preds = kmeans_obj.fit_predict(df['temp'].values.reshape(-1,1))



####  Perform cluster analysis
## Make cluster analysis instance and name clusters
clust_inst = clust(df, preds,
                   name_clusters={'L': ('temp', lambda x: -1*np.mean(x)),
                                  'H': 'temp'})
## Create 1d histograms
clust_inst.hist1d(bins=15,
                  hist_vars=['temp','x','y'])
clust_inst.hist2d(histxy = ('x','y'),
                  hist_var = 'temp',
                  bins = (10,10))
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
## Plot cluster crossings by viewing x vs y plot 
cross_inst.plot_crossings(crossing='bd',
                          x='x',
                          y='y',
                          point_type = 'immediate')
## Retrieve crossing data as single dataframe
cross_df = cross_inst.get_crossing_dfs_as_list(crossing='bd',
                                               point_type = 'immediate',
                                               cluster='H',
                                               single_df=True)
## Plot crossings on scale of box
fig, axes = plt.subplots(1,1)
axes.set_xlim(0,2)
axes.set_ylim(0,2)
axes.scatter(cross_df['x'], cross_df['y'], s=1)
plt.show()