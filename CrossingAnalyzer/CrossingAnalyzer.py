#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:02:53 2022

@author: jedmond
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import warnings
import copy
import os
import sys
from matplotlib.dates import AutoDateLocator, AutoDateFormatter, date2num
#from matplotlib.ticker import AutoMajorLocator



module_loc = "/home/jedmond/Documents/ML_Research/Source/customplotlib"
sys.path.insert(0, module_loc)
import customplotlib as cpl





class CrossingAnalyzer:
    
    
    #### Constants
    CROSSING_NUM       = "crossing_num"
    CROSSING_PT_TYPE   = "crossing_points"
    _PT_TYPE_EXTENDED  = "extended"
    _PT_TYPE_NORMAL    = "normal"
    _PT_TYPE_IMMEDIATE = "immediate"
    PT_TYPE = {
        _PT_TYPE_EXTENDED  : 0,
        _PT_TYPE_NORMAL    : 1,
        _PT_TYPE_IMMEDIATE : 2
              }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def __init__(self, crossing_dict,
                       crossing_names = None,
                       keep_empty     = None,
                       output_folder  = None):
    
        """
        
        """



        #### Make DEEP-COPY of crossing-dict for class
        self.cross_dict = copy.deepcopy( crossing_dict )
        
        
        
        #### Make output folder
        self.out_fldr = self._init_output_folder(output_folder)
        
        
        
        #### Assign crossing names
        ## If none, make dict with original keys from cross dict being both
        ## key and value 
        if crossing_names is None:
            crossing_names = { key : key for key in crossing_dict }
        ## Replace old names in class crossing dict with new names
        ## given
        for old_name in crossing_names:
            new_name = crossing_names[old_name]
            self.cross_dict[new_name] = self.cross_dict[old_name]
            del self.cross_dict[old_name]
                
            
                
        #### Determine cluster names
        self.cluster_names = np.unique( 
            [ tuple_elem for _tuple in list(crossing_dict.keys()) \
              for tuple_elem in _tuple ]
                                      ).tolist()
            


        #### If dataframe are empty, can choose to not include them
        #### in class dict (if specified by user)
        if keep_empty is None: keep_empty = True
        if not keep_empty:
            
            ## Iterate over dict and track keys where dataframes are empty
            keys_to_del = []
            for cross_name in self.cross_dict:
                if self.cross_dict[cross_name].shape[0] == 0:
                    keys_to_del.append( cross_name )
                    
            ## Delete each dict entry with empty dataframe
            for key in keys_to_del:
                del self.cross_dict[key]
        
        
        
        #### Determine the crossing numbers per key 
        num_per_key = {}
        for key in self.cross_dict:
            num_per_key[key] = np.unique(
                    self.cross_dict[key][CrossingAnalyzer.CROSSING_NUM]
                                        )
        self.cross_per_key = num_per_key
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def _init_output_folder(self, output_folder):
        
        """
        asdf
        """
        
        if output_folder is None: output_folder = os.getcwd()
        out_folder_full_path = os.path.join(output_folder,
                                            "CrossingAnalysis")
        os.makedirs(out_folder_full_path, exist_ok=True)
        return out_folder_full_path
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def _make_rel_subdir(self, subdir_path):
        
        """
        Makes relative subdirectory under file structure rooted
        at class out_fldr if it does not exist.
        
        
        PARAMETERS
        ----------
        subdir_path: str
            Path (global or relative) of folder to make
            
            
        RETURNS
        -------
        None
        """
    
        # Get global path to subdir_path
        total_path = os.path.join(self.out_fldr, subdir_path)
        # Make dir(s) if DNE
        if not os.path.exists(total_path):
            os.makedirs(total_path)
        # Return global path
        return total_path
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def _get_crossing_by_num(self, crossing, crossing_num):
        
        """
        
        Retrieves a crossing dataframe for the class crossing dict
        given by crossing key and crossing number
        
        
        PARAMETERS
        ---------
        crossing: str (or 2-elem tuple of strs)
            key to access subset of crossings
            
        crossing_num: int
            Number used to access particular crossing
            
            
        RETURNS
        -------
        Pandas dataframe of data related to single crossing
        
        """
        
        ## might be faster, currently unsuure; try again if too slow
        """start_ind = np.searchsorted( 
                self.cross_dict[crossing][CrossingAnalyzer.CROSSING_NUM_STR]
                                    )
        end_ind = np.searchsorted( 
                self.cross_dict[crossing][CrossingAnalyzer.CROSSING_NUM_STR],
                side="right"
                                    )
        return self.cross_dict[crossing].iloc[start_ind:end_ind,:]"""
    
        return self.cross_dict[crossing][ 
            self.cross_dict[crossing][CrossingAnalyzer.CROSSING_NUM] == 
                                    crossing_num
                                        ]
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    def _get_data_via_pt_type(self, cross_df, pt_type):
        
        """
        
        Retrieves sub-dataframe from a crossing dataframe given a pt_type
        
        
        PARAMETERS
        ----------
        cross_arr: pandas dataframe
            A pandas df for a single crossing
        
        pt_type: str
            Handles which subset of data is retrieved according to
            crossing point type
            
        
        RETURNS
        -------
        Pandas df with data satisyfing pt_type
        
        """
        
        
        int_pt_type = CrossingAnalyzer.PT_TYPE[pt_type]
        return cross_df[
            cross_df[CrossingAnalyzer.CROSSING_PT_TYPE] >= int_pt_type
                        ]










    def _get_data_via_cluster(self, cross_df, cluster):
        
        """
        
        Retrieves sub-dataframe from a crossing dataframe given a cluster
        
        
        PARAMETERS
        ----------
        cross_arr: pandas dataframe
            A pandas df for a single crossing
        
        cluster: str
            Handles which subset of data is retrieved according to cluster type
            
        
        RETURNS
        -------
        Pandas df with data satisyfing cluster_type
        
        """
        
        
        
        #### Get the row indices where the point_type is 'immediate'
        ## Get int representation of immediate pt type
        _immed_pt_type_int = \
                CrossingAnalyzer.PT_TYPE[CrossingAnalyzer._PT_TYPE_IMMEDIATE] 
        ## Find row inds
        immed_row_inds = np.where(
            cross_df[CrossingAnalyzer.CROSSING_PT_TYPE] >= _immed_pt_type_int
                                 )[0]# + cross_df.index[0]
        #print("CROSS:",cross_df, immed_row_inds)
        
        
        # make plot to debug
        """print(cross_df)
        import matplotlib.pyplot as plt
        xvals = np.arange(cross_df.shape[0])
        for prob_label in ['magsheath','solar_wind']:
            plt.scatter(xvals, cross_df[prob_label], s=2)
        inds = np.where(cross_df['crossing_points'] == 2)[0]
        for ind in inds:
            plt.axvline( xvals[ind], linewidth=1 ) 
        plt.show()
        plt.close()"""
        
        
        #### Determine what is the first (earlier) and second (later) cluster
        #### by seeing which cluster had the dominant probability at the
        #### FIRST and LAST of the immediate points
        immed_dom_clusters = \
            cross_df[self.cluster_names].\
                    idxmax(axis='columns').iloc[immed_row_inds].values
        #print(immed_dom_clusters)
        first_cluster = immed_dom_clusters[0]
        second_cluster = immed_dom_clusters[-1]
        
        
        
        #### Build index range of in-cluster indices that can be used to
        #### select the in-cluster data from cross_df (and raise error
        #### if given cluster param is not among them!)
        in_cluster_inds = None
        if first_cluster == cluster:
            #in_cluster_inds = np.arange(cross_df.index[0], immed_row_inds[0]+1)
            in_cluster_inds = np.arange(immed_row_inds[0]+1)
        elif second_cluster == cluster:
            in_cluster_inds = np.arange(immed_row_inds[-1], cross_df.shape[0])
        else:
            #raise ValueError("Given cluster param \"" + cluster
            #                 + "\" not among dominant clusters: [ "
            #                 + " , ".join(immed_dom_clusters) + " ]")
            warnings.warn("Given cluster param \"" + cluster
                             + "\" not among dominant clusters: [ "
                             + " , ".join(immed_dom_clusters) + " ];"
                             + " returning empty dataframe")
            in_cluster_inds = np.zeros(0, dtype=np.int32)
        #print(in_cluster_inds)
        #print("SUBSET:",cross_df.iloc[in_cluster_inds])
        
        
        
        #### Return subset of cross-df of in-cluster indices 
        return cross_df.iloc[in_cluster_inds]











    ########## "PUBLIC" FUNCTIONS ##########













    def get_crossing_df(self, crossing):
        
        """
        
        Retrieve the dataframe(s) saved under crossing(s)
        
        Use this, followed by set_crossing_df, to calculate new variables
        in the dataset
        
        
        PARAMETERS
        ----------
        crossing: str, or list of str
            Strs that act as the crossing_names given when this class
            is instantiated
            
        
        RETURNS
        -------
        Dict with crossing_names as key and values as pandas dataframe
        
        """
        
        if isinstance(crossing, str): crossing = [ crossing ]
        return { cross_name : self.cross_dict[cross_name] \
                 for cross_name in crossing }
        














    def set_crossing_df(self, crossing_dict):
    
        """
        
        Used to update the crossing dataframe
        
        Will fail if key in given cross_dict does not match previous crossing
        key!
        
        
        PARAMETERS
        ----------
        crossing_dict: dict(str : pandas dataframe)
            Dict with previously-used crossing name as key and new pandas
            dataframe as value
            
        
        RETURNS
        -------
        None
        
        """
    
        for cross_name in crossing_dict:
            if cross_name not in list(self.cross_dict):
                
                ## Raise error if crossing name not among original keys
                raise ValueError(
                    "Only previously used crossing names supported: [ "
                    + " , ".join( list(self.cross_dict.keys()) ) + " ]. "
                    "Given crossing \"" + cross_name + "\" not recognized."
                                )
                
                ## Otherwise, replace old dataframe with given
                self.cross_dict[cross_name] = crossing_dict[cross_name]
        















    def get_crossing_dfs_as_list(self, crossing    = None,
                                       point_type  = None,
                                       cluster     = None,
                                       labels      = None,
                                       filter_dict = None):
        
        """
        
        Retrieves data from each crossing dataframe as a list of dataframes
        
        
        PARAMETERS
        ----------
        crossing: str (or 2-elem tuple of strs), (optional, default is first key)
            key to access subset of crossings
            
        pt_type: str (optional, default is None)
            Handles which subset of data is retrieved according to
            crossing point type
            
        cluster: str (optional, default is None)
            Handles which subset of data is retrieved according to cluster type
            
        labels: str / list of strs (optional, default is all labels)
            Labels to access from crossing dataframes
            
        filter_dict: dict(str-label, function)
            Dict with dataframe labels as keys and functions as values.
            Data for which the function returns True are kept and others are
            discarded.

        
        RETURNS
        -------
        list of crossing dataframes
        
        """
        


        #### Set default args
        if crossing is None: crossing = list(self.crossing_names.keys())[0]
        if labels is None: labels = list(self.cross_dict[crossing])
        if isinstance(labels,str): labels = [ labels ]
        
        
        
        #### Get all crossing_nums for crossing
        crossing_nums = self.cross_per_key[crossing]
        crossing_dfs = []
        for crossing_num in crossing_nums:
            
            ## Retrieve particular crossing df via crossing number
            cross_df = self._get_crossing_by_num(crossing,
                                                 crossing_num)
            
            ## Filter crossing df to only get desired sub-interval of points
            if point_type is not None:
                cross_df = self._get_data_via_pt_type(cross_df,
                                                      point_type)
            
            ## Further filter crossing df to get desired cluster of data
            ## out (e.g. a data transitions from c0 to c1 cluster and only
            ## c0 data is desired)
            if cluster is not None:
                cross_df = self._get_data_via_cluster(cross_df,
                                                      cluster)
                
            ## If filter_dict is not None, check that crossing_df 
            ## (its 'immediate' point_type pts) passes the func check
            ignore_this_cross = False
            if filter_dict is not None:
                for filter_key in filter_dict:
                    # Get subset of data that is in immediate interval
                    immed_only_cross_df = self._get_data_via_pt_type(
                                            cross_df,
                                            CrossingAnalyzer._PT_TYPE_IMMEDIATE
                                                                    )
                    # Get func, data for func, and apply func
                    _func = filter_dict[filter_key]
                    _data_for_func = immed_only_cross_df[filter_key]
                    # continue loop (e.g. DON'T save data) if not all True
                    if not np.all( _func(_data_for_func) ):
                        ignore_this_cross = True
                        break
            
            ## Apply function to x and y data for this dataframe if specified
            if not ignore_this_cross:
                crossing_dfs.append( cross_df[labels] )
        

        return crossing_dfs










    def get_crossing_nums(self, crossing=None,
                                filter_dict=None):
        
        """
        
        Return crossing numbers that satisfy constraints in filter_dict
        
        (filter_dict same as in get_crossing_dfs_as_list)
        
        """
        
        if crossing is None: crossing = list(self.crossing_names.keys())[0]
        if filter_dict is None: filter_dict = {}
        
        dfs_list = self.get_crossing_dfs_as_list(
                            crossing = crossing,
                            filter_dict = filter_dict
                                                )
        
        return [ dfs_list[i][CrossingAnalyzer.CROSSING_NUM].values[0] \
                 for i in range(len(dfs_list)) ]
        





    







    def plot_single_crossing(self, crossing=None,
                                   crossing_num=None,
                                   x = None,
                                   y = None,
                                   point_type = None):
        
        if crossing is None: crossing = list(self.crossing_names.keys())[0]
        if x is None: x = list(self.cross_dict[crossing])[0]
        if y is None: y = list(self.cross_dict[crossing])[1]
        if isinstance(y,str): y = [ y ]
        # -- write good init for crossing num here --
        # -- write good init for point_type here --
        
        
        #### get cross_df from crossing number
        cross_df = self._get_crossing_by_num(crossing, crossing_num)
        cross_df = self._get_data_via_pt_type(cross_df, point_type)
        
        fig, axes = plt.subplots(1,1)
        
        for _y in y:
            plt.scatter( cross_df[x], cross_df[_y], s=2 )

        plt.show()
        plt.close()        
        












    def full_plot_single_crossing(self, crossing = None,
                                        crossing_num = None):
        
        if not isinstance(crossing_num,list):
            if not isinstance(crossing_num, type(np.array([]))):
                raise ValueError("gib as container plz")
        
        for num in crossing_num:
            
            fig = plt.figure(figsize=(8,8))
            s2g_grid = (5,4)
            # first 5 plots are v vector, b vector, log dens, log temp, probs
            v_ax    = plt.subplot2grid(s2g_grid, (0, 0), colspan=2)
            b_ax    = plt.subplot2grid(s2g_grid, (1, 0), colspan=2)
            dens_ax = plt.subplot2grid(s2g_grid, (2, 0), colspan=2)
            temp_ax = plt.subplot2grid(s2g_grid, (3, 0), colspan=2)
            prob_ax = plt.subplot2grid(s2g_grid, (4, 0), colspan=2)
            # then spatial plots
            xy_ax   = plt.subplot2grid(s2g_grid, (0, 2), colspan=2, rowspan=2)
            xz_ax   = plt.subplot2grid(s2g_grid, (2, 2), colspan=2, rowspan=2)
            xx_ax   = plt.subplot2grid(s2g_grid, (4, 2), colspan=2, rowspan=1)
    
            cross_df = self._get_crossing_by_num(crossing, num)
            
            locator = AutoDateLocator()
            formatter = AutoDateFormatter(locator)
            
    
            times = date2num( cross_df['time'] )
            v_ax.scatter( times, cross_df['VX'], s=2, label='VX' )
            v_ax.scatter( times, cross_df['VY'], s=2, label='VY' )
            v_ax.scatter( times, cross_df['VZ'], s=2, label='VZ' )
            v_ax.xaxis.set_major_locator(locator)
            v_ax.xaxis.set_major_formatter( AutoDateFormatter(locator) )
            v_ax.set(xticklabels=[])
            v_ax.legend()
            
            b_ax.scatter( times, cross_df['BX'], s=2, label='BX' )
            b_ax.scatter( times, cross_df['BY'], s=2, label='BY' )
            b_ax.scatter( times, cross_df['BZ'], s=2, label='BZ' )
            b_ax.xaxis.set_major_locator(locator)
            b_ax.xaxis.set_major_formatter( AutoDateFormatter(locator) )
            b_ax.set(xticklabels=[])
            b_ax.legend()
            
            dens_ax.scatter( times, cross_df['density'], s=2, label='density' )
            dens_ax.xaxis.set_major_locator(locator)
            dens_ax.xaxis.set_major_formatter( AutoDateFormatter(locator) )
            dens_ax.set(xticklabels=[])
            dens_ax.set_yscale('log')
            dens_ax.legend()
            
            temp_ax.scatter( times, cross_df['temperature'], s=2, label='temp' )
            temp_ax.xaxis.set_major_locator(locator)
            temp_ax.xaxis.set_major_formatter( AutoDateFormatter(locator) )
            temp_ax.set(xticklabels=[])
            temp_ax.set_yscale('log')
            temp_ax.legend()
            
            prob_ax.scatter( times, cross_df['solar_wind'], s=2, label='solar_wind' )
            prob_ax.scatter( times, cross_df['magsheath'], s=2, label='magsheath' )
            prob_ax.scatter( times, cross_df['magsphere'], s=2, label='magsphere' )
            prob_ax.xaxis.set_major_locator(locator)
            prob_ax.xaxis.set_major_formatter( AutoDateFormatter(locator) )
            prob_ax.legend()
            
            # indicate pt type
            linewidth = 2
            immed_c = "black"
            norm_c = "red"
            normal_df = self._get_data_via_pt_type(cross_df, 'normal')
            immed_df = self._get_data_via_pt_type(cross_df, 'immediate')
            prob_ax.axvline(x = date2num(normal_df['time'])[0],
                            linewidth=linewidth, color=norm_c)
            prob_ax.axvline(x = date2num(normal_df['time'])[-1],
                            linewidth=linewidth, color=norm_c)
            prob_ax.axvline(x = date2num(immed_df['time'])[0],
                            linewidth=linewidth, color=immed_c)
            prob_ax.axvline(x = date2num(immed_df['time'])[-1],
                            linewidth=linewidth, color=immed_c)
            
            
            xy_ax.scatter( cross_df['X'], cross_df['Y'], s=2 )
            #xy_ax.set(xticklabels=[])
            xz_ax.scatter( cross_df['X'], cross_df['Z'], s=2 )
            #xz_ax.set(xticklabels=xz_ax.xaxis.get_ticklabels())
            #print( xz_ax.xaxis.get_majorticklabels() )
            #xz_ax.xaxis.set_major_locator( xz_ax.xaxis.get_major_locator() )
            #xz_ax.xaxis
            xx_ax.scatter( cross_df['X'], cross_df['X'], s=2 )
            
            
            fig.autofmt_xdate()
            
            
            fig.tight_layout()
            fig.savefig('/home/jedmond/Desktop/cross_plots/'
                        + str(num) + '.png')
            plt.close()
        
    
    
    
    
    
    
    
    
    
    
    
    
    def plot_crossings(self, crossing        = None,
                             x               = None,
                             y               = None,
                             point_type      = None,
                             subplot_kwargs  = None,
                             #point_type_func = None,
                             cluster         = None,
                             filter_dict     = None,
                             heatmap         = None,
                             heatmap_hist    = None,
                             cmap            = None,
                             heatmap_func    = None,
                             #heatmap_kwargs = None
                             sep_crossings   = None,
                             figname         = None,
                             heatmap_bounds  = None,
                             return_plot     = None):
        
        """
        
        ...
        
        heatmap_hist is kwargs dict of params you'd supply to 
        matplotlib.pyplot.hist
        
        ...
        
        """
        
        
        
        #### Set default args
        if crossing is None: crossing = list(self.crossing_names.keys())[0]
        if x is None: x = list(self.cross_dict[crossing])[0]
        if y is None: y = list(self.cross_dict[crossing])[1]
        if point_type is None: point_type = "immediate"
        if sep_crossings is None: sep_crossings = False
        if cmap is None: cmap = "winter"
        if subplot_kwargs is None: subplot_kwargs = {}
        if return_plot is None: return_plot = False
        #if figsize is None:
        #    figsize =  matplotlib.pyplot.rcParams['figure.figsize']
        
        
        
        #### Check possible errors related to heatmap
        ## raise error given heatmap and told to separate crossings
        if (heatmap is not None) and sep_crossings:
            raise ValueError("Cannot plot crossings separately AND use"
                             + " heatmap; one or the other.")
        ## raise error if given heatmap_hist and not heatmap
        if (heatmap_hist is not None) and (heatmap is None):
            raise ValueError("heatmap arg must be set if using heatmap_hist.")
        ## Have to specify heatmap if using heatmap_func
        if (heatmap_func is not None) and (heatmap is None):
            raise ValueError("heatmap arg must be set if using heatmap_func.")
        ## if heatmap_bounds is not None and not a 2-element container
        ## (list, tuple, array, etc.), then raise error
        if (heatmap_bounds is not None) and (len(heatmap_bounds) != 2):
                raise ValueError("Have to supply 2 elements (min, max)"
                                 + " as heatmap_bounds.")
            
            
            
        #### Set default heatmap_hist args
        default_heatmap_hist = {'bins':30,
                                'orientation':'horizontal'}
        if heatmap_hist is not None:
            ## If given bool (as True), then use default params
            if (type(heatmap_hist) == bool) and heatmap_hist:
                heatmap_hist = {}
            heatmap_hist = { **default_heatmap_hist, **heatmap_hist }
            
        
        
        #### Retrieve crossing dataframes that have been filtered
        #### according to desired crossing, point_type, and cluster
        crossing_dfs = self.get_crossing_dfs_as_list(crossing = crossing,
                                                     point_type = point_type,
                                                     cluster = cluster,
                                                     filter_dict = filter_dict)
        ## Get the desired x and y pandas series from each dataframe
        _x = [ elem[x] for elem in crossing_dfs ]
        _y = [ elem[y] for elem in crossing_dfs ]
        ## Get desired heatmap data if specified
        if heatmap is not None:
            _heatmap = [ elem[heatmap] for elem in crossing_dfs ]
        title = "Num crossings: " + str(len(crossing_dfs))
            
        
        
        #### Prepare fig and axes objects
        fig, scat_ax, hist_ax = None, None, None
        if heatmap_hist is not None:
            fig, axes = plt.subplots(1, 2, **subplot_kwargs)#, dpi=120)
            hist_ax, scat_ax = axes
        else:
            fig, scat_ax = plt.subplots(1, 1, **subplot_kwargs)#, dpi=120)
            
        
        
        #### Plot collected crossing data
        ## First check that _x and _y are non-empty!
        empty_xy = (len(_x) == 0) or (len(_y) == 0)
        if empty_xy:
            warnings.warn("No data under crossing \"" + crossing + "\", so"
                          + " no plot to make.")
            
        ## If crossings are to be plotted separated, then call scatter for
        ## *each* pair of x and y
        if sep_crossings and not empty_xy:
            for i in range(len(_x)):
                scat_ax.scatter(_x[i], _y[i], s=2)
                
        ## Otherwise, scatter-plot x / y data all at once
        if not sep_crossings and not empty_xy:
            # Reorganize plot data into 1d arrays
            _x = np.hstack(_x)
            _y = np.hstack(_y)
            # If no heatmap, then basic scatter plot
            if heatmap is None:
                scat_ax.scatter(_x, _y, s=2)
            # Otherwise, incorporate heatmap-related plot(s)
            else:
                # . prepare heatmap data and apply func if given
                _heatmap = np.hstack(_heatmap)
                if heatmap_func is not None:
                    _heatmap = heatmap_func(_heatmap)
                # . keep only data with heatmap values in heatmap bounds if given
                # . (if not given, then take all values as being in bounds)
                inds_in_bounds = np.arange(_heatmap.shape[0])
                if heatmap_bounds is not None:
                    hmin, hmax = heatmap_bounds
                    inds_in_bounds = np.where(
                            (_heatmap >= hmin) & (_heatmap <= hmax)
                                            )[0]
                    num_excluded = np.setdiff1d( np.arange(_x.shape[0]),
                                                 inds_in_bounds ).shape[0]
                    title += " | num excluded: " + str(num_excluded)
                _heatmap_in_bds = _heatmap[inds_in_bounds]
                _x_in_bds = _x[inds_in_bounds]
                _y_in_bds = _y[inds_in_bounds]
                inds_beyond_bounds = np.setdiff1d( np.arange(_x.shape[0]),
                                                   inds_in_bounds )
                #$$ _heatmap_beyond_bds = _heatmap[inds_beyond_bounds]
                _x_beyond_bds = _x[inds_beyond_bounds]
                _y_beyond_bds = _y[inds_beyond_bounds]
                # . then plot with heatmap and add colorbar
                _ = scat_ax.scatter(_x_beyond_bds, _y_beyond_bds, s=10,
                                    c="black",
                                    marker="x")
                plot_res = scat_ax.scatter(_x_in_bds, _y_in_bds, s=2,
                                           c=_heatmap_in_bds,
                                           cmap=cmap)
                fig.colorbar(plot_res,
                             label=heatmap,
                             orientation="vertical")
                # . Make heatmap hist if specified
                if heatmap_hist is not None:
                    # .. warn user is using non-immediate point type - 
                    # .. could have duplicated values for heatmap, changing
                    # .. the counts!!
                    if point_type != CrossingAnalyzer._PT_TYPE_IMMEDIATE:
                        warnings.warn("Using non-immediate point type, which"
                                      + " could mean duplicated hist values.")
                    hist_ax.hist(_heatmap, **heatmap_hist )
                    if heatmap_bounds is not None:
                        hist_ax.axhline(y=hmin, linewidth=1, c="black")
                        hist_ax.axhline(y=hmax, linewidth=1, c="black")
                
                
        
        #### Make title and save fig (just show and close otherwise)
        ## Make title
        if (scat_ax is not None) and (hist_ax is not None):
            fig.suptitle(title)
            fig.tight_layout()
        else:
            plt.title(title)
        ## Save  / show fig
        if figname is not None:
            print(self.out_fldr)
            figpath = os.path.join(self.out_fldr,figname+".png")
            fig.savefig(figpath)
        else:
            plt.show()
            plt.close()
            
            
        if not return_plot:
            plt.close()
        else:
            return fig, { 'hist': hist_ax, 'scat': scat_ax }
        
        
        
        #return _x, _y
            
            
        











    



    



