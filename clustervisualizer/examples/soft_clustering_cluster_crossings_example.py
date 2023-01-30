from clustervisualizer.ClusterAnalyzer import ClusterAnalyzer as clust
from clustervisualizer.CrossingAnalyzer import CrossingAnalyzer as cross
from simulate_detector import simulate_detector

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture





def to_epoch(pd_timestamp):
    if isinstance(pd_timestamp,list):
        return [ (elem - pd.to_datetime("1970-01-01")) / pd.Timedelta("1s") \
                 for elem in pd_timestamp ]
    else:
        return (pd_timestamp - pd.to_datetime("1970-01-01")) / pd.Timedelta("1s")





def from_epoch(epoch_val):
    return pd.to_datetime(epoch_val, unit="s")






"""
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
"""



#### Simulate a detector moving between two different temperature gases
detector_sim = simulate_detector(
                init_posit = (0.1, 0.5),
                init_veloc = (0.0678, 0.0351),
                box_bnds = (0,2,0,2),
                max_iter = 1000,
                temp_boundary = [0.8,1.2]
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
gm = GaussianMixture(n_components=2).fit(df['temp'].values.reshape(-1,1))
preds = gm.predict_proba(df['temp'].values.reshape(-1,1))



####  Perform cluster analysis
## Make cluster analysis instance and name clusters
clust_inst = clust(df, preds,
                   name_clusters={'L': ('temp', lambda x: -1*np.mean(x)),
                                  'H': 'temp'})
## Create histograms
clust_inst.hist1d(bins=15,
                  hist_vars=['temp','x','y'])
clust_inst.hist2d(histxy = ('x','y'),
                  hist_var = 'probability',
                  bins = (15,15))
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