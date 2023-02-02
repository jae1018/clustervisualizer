import pytest
import numpy as np
import pandas as pd

#from src import clustervisualizer
#import src

from clustervisualizer.ClusterAnalyzer import ClusterAnalyzer as clust





def test_ClusterAnalyzer_init():
    
    df = pd.DataFrame( {'a':np.arange(10), 'b':np.arange(10)*5.5} )
    preds = np.hstack( [np.full(10,0), np.full(10,1)] )
    _clust = clust(df, preds)

    pd.testing.assert_frame_equal(_clust.df, df)
    np.testing.assert_array_equal(_clust.pred_arr, preds)





def test_ClusterAnalyzer_build_constraints_list():
    
    df = pd.DataFrame({
        'cat': np.hstack(
            [ np.full(10,elem) for elem in ['a','b','c'] ]
                        )
                      })
    preds = np.full(df.shape[0], 0)
    _clust = clust(df, preds)
    
    algo_result = _clust._build_constraints_list_of_dicts('cat')
    expected_result = [ {'cat':'a'},
                        {'cat':'b'},
                        {'cat':'c'} ]
    assert expected_result == algo_result
    
    
    
    
    
    
if __name__ == "__main__":
    pytest.main()