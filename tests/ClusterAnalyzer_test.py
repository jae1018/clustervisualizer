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










class Test_ClusterAnalyzer_build_constraints_list:
    
    #self._int_type = np.arange(1).dtype.type
    _empty_int_arr = np.array([]).astype( np.arange(1).dtype.type )
    
    
    
    def compare_expected_vs_algo_result(self, algo, expec):
        
        # check length
        assert len(algo) == len(expec)
        
        #try:
        for i in range(len(algo)):
            assert algo[i][0] == expec[i][0]
            np.testing.assert_array_equal(algo[i][1], expec[i][1])
            




    def test_build_constraints_list_one_categ_multiple_values(self):
        
        df = pd.DataFrame({
            'cat': np.hstack(
                [ np.full(10,elem) for elem in ['a','b','c'] ]
                            )
                          })
        preds = np.full(df.shape[0], 0)
        _clust = clust(df, preds)
        
        algo_result = _clust._build_constraints_list_of_dicts('cat')
        expected_result = [
            ( {'cat':'a'}, np.arange(10) ),
            ( {'cat':'b'}, np.arange(10)+10 ),
            ( {'cat':'c'}, np.arange(10)+20 )
                          ]
        self.compare_expected_vs_algo_result(algo_result, expected_result)
        #assert expected_result == algo_result





    def test_build_constraints_list_multiple_categ_multiple_values(self):
        
        df = pd.DataFrame({
            'X': np.hstack(
                [ np.full(10,elem) for elem in ['a','b','c'] ]
                            ),
            'Y': np.hstack(
                [ np.full(15,elem) for elem in ['1','2'] ]
                            )
                          })
        preds = np.full(df.shape[0], 0)
        _clust = clust(df, preds)
        
        algo_result = _clust._build_constraints_list_of_dicts(['X','Y'])
        expected_result = [
            ( {'X':'a', 'Y':'1'}, np.arange(10) ),
            ( {'X':'a', 'Y':'2'}, self._empty_int_arr ),
            ( {'X':'b', 'Y':'1'}, np.arange(5)+10 ),
            ( {'X':'b', 'Y':'2'}, np.arange(5)+15 ),
            ( {'X':'c', 'Y':'1'}, self._empty_int_arr ),
            ( {'X':'c', 'Y':'2'}, np.arange(10)+20 )
                          ]
        self.compare_expected_vs_algo_result(algo_result, expected_result)
        #assert expected_result == algo_result    
    
    
    
    
    
if __name__ == "__main__":
    pytest.main()