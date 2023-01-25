#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLASS VARS:
    
    input_df: 2D pandas dataframe
    cluster_affil_arr: 1D (hard-clustering) or 2D (soft) array with integers / 
                        probabilities representing cluster affiliation. If
                        integers, cluster ints range from 0 to n-1 for n clusters;
                        if probabilties, then each data points probs should sum
                        to 1.

@author: jedmond
"""



import os
#import shutil, copy, math, sys
import warnings
import itertools
#import time, gc
import numpy as np
import pandas as pd
#from scipy.optimize import minimize_scalar
import scipy

# Graphical Imports
import matplotlib.pyplot as plt
#import matplotlib.transforms as transforms
#from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.ticker import AutoMinorLocator
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
#import matplotlib.dates as mdates

# Import from same package
from . import utils





class ClusterAnalyzer:
    
    """
    class descrip here
    """
    
    
    IMAGE_TYPE = ".png"
    MULTIPLOT_SIZE = (16,8)
    SINGLEPLOT_SIZE = (8,8)
    ROW_TABLE_LIMIT = 7
    DEFAULT_COLORS = ["blue", "orange", "red", "green", "chocolate",
                      "lime", "teal", "orchid", "cyan", "yellowgreen"]
    BINS_1D = 100
    BINS_2D = [50,50]
    ALL_CLUSTERS_INT = -1
    SOFT_CLSTR_STR = "soft"
    HARD_CLSTR_STR = "hard"
    PROBAB_STR = "probability"
    OCCUP_STR = "occupancy"
    
    
    def __init__(self, df,
                       pred_arr,
                       output_folder = None,
                       name_clusters = None):
        
        """
        Creates a cluster analysis object.
        
        
        Parameters
        ----------
        df: Pandas dataframe (N rows)
            df of data that was clustered
            
        pred_arr: 1d numpy arr (N rows of ints) or 2d numpy arr
                  (N rows x K columns of floats)
            Numpy array used to indicate cluster affiliation.
            
            1d ARRAY <==> HARD CLUSTERING
                A 1d array of ints means hard clustering was used 
                (i.e. a point belongs to a cluster - or not). The
                number of clusters is inferred from the number of
                unique integers counting from 0 and up.
                
            2d ARRAY <==> SOFT CLUSTERING
                A 2d array of floats means soft clustering was used
                (i.e. for clustering a point among K clusters, the point
                 is assigned K probabilities that sum to 1).
                
        output_folder: str (default None)
            Path to output folder to store results in.
            If None, a subfolder 'Cluster_Analysis' is made at the
            current working directory.
            
        name_clusters: 
        
        
        """
        
        # Make output folder
        self.out_fldr = self._init_output_folder(output_folder)
        
        # Save provided df and assignment arr
        self.df = df
        self.pred_arr = pred_arr
        self.n_clusters = self._compute_num_clusters()
        
        # Determine what labels (strings) to apply to each cluster based on
        # which clusters maximize functions
        self.cluster_names = self._assign_cluster_names(name_clusters)
        

    







    def _init_output_folder(self, output_folder):
        
        """
        Creates subfolder at address given by output_folder arg
        
        Parameters
        ----------
        output_folder: str (or None)
            If not none, path to desired subfolder to create
            If None, path becomes cwd/Cluster_Analysis
        
        Returns
        -------
        out_folder_full_path: str
            global address to created subfolder
        """
        
        if output_folder is None: output_folder = os.getcwd()
        out_folder_full_path = os.path.join(output_folder,
                                            "Cluster_Analysis")
        os.makedirs(out_folder_full_path, exist_ok=True)
        return out_folder_full_path
    
    
    
    
    
    
    
    
    
    
    def _make_rel_subdir(self, subdir_path):
        
        """
        Makes relative subdirectory under file structure rooted
        at class attribute out_fldr if it does not exist.
        
        PARAMETERS
        ----------
        subdir_path: str
            Path (global or relative) of folder to make
            
        RETURNS
        -------
        total_path: str
            Global address to created subdirectory
        """
    
        # Get global path to subdir_path
        total_path = os.path.join(self.out_fldr, subdir_path)
        # Make dir(s) if DNE
        if not os.path.exists(total_path):
            os.makedirs(total_path)
        # Return global path
        return total_path
    
    
    
    
    
    
    
    
    
    
    def _is_categ_var(self, label):
        
        """
        Checks if data in class dataframe given by label is a
        categorical variable or not. This is done by seeing if
        the array has type that is NOT a subset of np.number
        
        PARAMETERS
        ----------
        label: str (or 1-element list of str)
            str used to access data in class dataframe
            
        RETURNS
        -------
        boolean (True if categorical variable, False if not)
        """
        
        if isinstance(label,list): label = label[0]        
        return not np.issubdtype( self.df[label].dtype.type, np.number )
    
    
    
    
    





    def _assign_cluster_names(self, cluster_names_dict):
        
        """
        Names the clusters according to strings in cluster_names_dict.
        The resulting naming is representing using a dict (str:int), 
        where keys are the cluster names and the ints refer to either
        (1) the unique ints for hard clustering, or (2) the column index
        for soft clustering.
        
        If dict given is None, just names the clusters as 'c0', 'c1', ...
        
        Currently, for soft clustering, the top 20% of data is taken
        and functions are applied to that subset.
            
        PARAMETERS
        ----------
        cluster_names_dict: dict (name_str : value), or None
            Str keys will become the cluster names based on the value given.
            The values can be either ...
                (1) a str, where said str is a label for data. The cluster
                    that has the highest average for this label will
                    inherit name_str as that cluster's name.
                (2) or 2-element tuples, where the 1st element is a label
                    str like in option (1), but the 2nd element is a
                    function. The cluster that maximizes this function
                    using label data will inherit name_str is the cluster
                    name
                
        RETURNS
        -------
        dict with keys as cluster names (strs) and and values as cluster ints
        """
        
        # Init dict for labels to cluster ints
        names2clusterint_dict = {}
        
        
        #### If cluster_names_dict is None, then no funcs given; just
        #### make the cluster name the str equivalent of the cluster int
        if cluster_names_dict is None:
            for n in range(self.n_clusters):
                names2clusterint_dict["c"+str(n)] = n
            return names2clusterint_dict
        
        
        
        #### Ensure that values in cluster_names_dict are lists / tuples
        #### and that, if no func is given, provide default func
        default_func = np.average
        for key in cluster_names_dict:
            # If only given a single value (i.e. not a tuple / list),
            # then convert to 1-element tuple
            if not isinstance(cluster_names_dict[key],(tuple,list)):
                cluster_names_dict[key] = (cluster_names_dict[key],)
            # If len is 1, then change to 2-element tuple with 2nd elem
            # as default func
            if len( cluster_names_dict[key] ) == 1:
                cluster_names_dict[key] = (cluster_names_dict[key][0],
                                           default_func)
        
        
        
        # Otherwise, cluster names will be assigned based on which cluster
        # maximizes functions in cluster_names_funcs_dict
        
        #### For each proposed cluster name, apply the given function
        #### to each cluster and pick out the cluster that has the max
        #### val
        for cluster_name in cluster_names_dict:
            label, cluster_func = cluster_names_dict[cluster_name]
            vals = {}
            
            # Compute avg for each key,value pair using label given in
            # value; cluster with maximal avg will inherit key as name    
            for n in range(self.n_clusters):
                
                ## Get data in cluster by label
                label_data_in_cluster = None
                
                # If hard, then just retrieve all data by label that's
                # predicted to be in cluster
                if self._clustering_type() == ClusterAnalyzer.HARD_CLSTR_STR:
                    label_data_in_cluster = self._get_data(
                                            cluster=n,
                                            label=label
                                                          )
                
                # If soft, have to approach more cautiously. Retrieve data by
                # label in class df that has class prob >= 80%
                if self._clustering_type() == ClusterAnalyzer.SOFT_CLSTR_STR:
                    label_data_in_cluster = self._get_data(
                                            cluster=n,
                                            label=label,
                                            min_prob=0.8
                                                            )
                
                # Apply cluster_func to data and save with cluster_int as
                # key in vals dict
                vals[ cluster_func(label_data_in_cluster) ] = n
                #print("for label",label,"and cluster",n, "got func result",
                #      cluster_func(label_data_in_cluster) )
                
                
            ## Now assign cluster_name to cluster_int with highest val
            # Get max val from keys
            max_val = np.max( list(vals.keys()) )
            # Try to save to dict
            cluster_int_for_name = vals[max_val]
            names2clusterint_dict[ cluster_name ] = cluster_int_for_name
            
            
        #### Raise error if multiple names mapped to same int
        vals = [ names2clusterint_dict[key] for key in names2clusterint_dict ]
        if len(np.unique(vals)) != len(cluster_names_dict):
            error_mssg = ("Given cluster names mapped to non-unique cluster"
                          + " ints: " + str(names2clusterint_dict))
            raise ValueError(error_mssg)
        
        return names2clusterint_dict
                
    
    
    
    
    
    
    
    

    def _compute_num_clusters(self):
        
        """
        Compute number of clusters.
        
        PARAMETERS
        ----------
        None
        
        RETURNS
        -------
        number of clusters (int)
        """
        
        # If soft clustering, return number of columns of pred_arr
        if self._clustering_type() == ClusterAnalyzer.SOFT_CLSTR_STR:
            return self.pred_arr.shape[1]
        # If hard clustering, return number of unique ints in pred_arr
        if self._clustering_type() == ClusterAnalyzer.HARD_CLSTR_STR:
            return np.unique(self.pred_arr).shape[0]
    
    
    
    
    
    
    
    
    
        
    # Returns str indicating soft or hard clustering
    def _clustering_type(self):
        
        """
        Indicates if clustering is hard or soft
        
        PARAMETERS
        ----------
        None
        
        RETURNS
        -------
        str stating if clustering if hard or soft
        """
        
        # Grab fundamental numpy type from pred_arr
        arr_type = self.pred_arr.dtype.type
        # If of type int, then hard clustering
        if np.issubdtype(arr_type, np.integer):
            return ClusterAnalyzer.HARD_CLSTR_STR
        # If of type float, then soft clustering
        elif np.issubdtype(arr_type, np.floating):
            return ClusterAnalyzer.SOFT_CLSTR_STR
        # Otherwise, unrecognized type and raise Value Error
        else:
            error_mssg = ("Unrecognized type in cluster array")
            raise ValueError(error_mssg)
    
    
    
    
    
    
    
    
    
    
    def _cluster_name2int(self, str_name):
        
        """
        Once clusters are named, can convert from cluster_name to
        cluster_int using this.
        
        Parameters
        ----------
        str_name: str
            str that should be key in cluster_names class dict
        
        Returns
        -------
        int: cluster int
        """
        
        return self.cluster_names[str_name]
    
    
    
    
    
    
    
    
    
    
    def _cluster_int2name(self, cluster_int):
        
        """
        Once clusters are named, can convert from cluster int to
        cluster name using this
        
        Parameters
        ----------
        cluster_int: int
            int used to represent a cluster (unique cluster int if
            hard clustering or column index if soft clustering)
            
        Returns
        -------
        str: cluster name corresponding to give cluster int
        """
        
        return { self.cluster_names[key] : key \
                 for key in self.cluster_names }[cluster_int]
    
    
    
    
    
    
    
    
    
    
    def _is_str(val):
        
        """
        checks if val is str (either vanilla or numpy)
        """
        
        return isinstance(val, (str, np.str_))
    
    
    
    
    
    
    
    
    
    
    # Retrieves all data belonging to cluster given by cluster_int (e.g. 1, 2,
    # 3, etc); Note that cluster counting starts from 1!
    # all data given by label can be retrieve by feeding cluster in the
    # class constant ClusterAnalyzer.ALL_CLUSTERS_INT.
    # If hard clustering used, an array of row indices for the input df of
    # points belonging to that cluster arr returned (e.g. if a df has 10 rows
    # and rows 1,4, and 7 belong to cluster 1, then _get_cluster_data(1) would
    # return array([1,4,7]) ).
    # If soft clustering is used, then the probabilities of a row belonging to
    # such a cluster a returned (for prev example, return would be
    # array([some 10 floats b/w 0 and 1 here]) ).
    # If soft clustering is None and min_prob is set, then only points with
    # min_prob in the desired cluster are returned. Note that the original
    # points themselves are returned - NOT a weighted average!
    def _get_data(self, cluster  = None,
                        label    = None,
                        min_prob = None):
        
        """
        Retrieves data in cluster (or from all). The *context* of that data
        varies with the parameters specified, as indicated below.



        PARAMETERS
        -----------
        Note that for all params below:
            
            *) cluster: int or str
                 Default is class constant int used to represent all clusters
                 Can only be str if was given name_clusters dict at
                 class initialization.
                 
            *) label: str or list of str
                 Must be labels of class dataframe
                 
            *) min_prob: number
                 Num between 0 and 1 inclusive. Used to select data in soft
                 clustering whose probability exceeds this value.
                 
            *) If a param is not explicitly stated below, it means it
                 was NOT passed!
        
        
                
        IF NO PARAMS GIVEN:
            
            Array of entire class df is returned
        
        
        
        HARD CLUSTERING...
        
            1) cluster:
                 Returns row indices of data in class df that are
                 within cluster.
                 
            2) cluster, label:
                 Returns array of data under label(s) that are within cluster
                 
            3) label:
                 Returns array of data under label(s) across ALL clusters.
                 
            *) min_prob is not used
    
        
    
        SOFT CLUSTERING...
        
            1) cluster:
                 Returns array or probabilities associated with that cluster
                 
            2) cluster, label:
                 Returns the EXPECTATION VALUES of the data. For numerical
                 labels, this is a scalar; for categorical, this is a dict
                 with expectation values as values and categ items as keys.
                 
            3) cluster, min_prob:
                 Returns row indices of class df whose cluster probability
                 exceeds min_prob
                 
            4) cluster, label, min_prob:
                 Returns class df data given by label if the cluster
                 probability exceeds min_prob

    
        
        RETURNS
        -------
        numpy array / dict (various possibilities based on params)
        """
        
        
        
        #### Setup default params
        if cluster is None: cluster = ClusterAnalyzer.ALL_CLUSTERS_INT
        #if label is None: label = list(self.df)[0]
        
        
        #### If given cluster is str, convert to int equivalent
        if ClusterAnalyzer._is_str(cluster):
            cluster = self.cluster_names[cluster]
        
        
        
        #### If label is str, convert it to list of str
        if ClusterAnalyzer._is_str(label): label = [ label ]
        
        
        
        #### Check if cluster is legal
        # If cluster is int, check if in range of n_clusters
        if np.issubdtype(cluster, np.integer):
        #if ClusterAnalyzer._is_int(cluster):
            # Have to account for typical range, but also the 
            # designated class int indicating all clusters
            legal_cluster_ints = np.arange(self.n_clusters).tolist()
            legal_cluster_ints.append( self.ALL_CLUSTERS_INT )
            # check if cluster int is legal
            if cluster not in legal_cluster_ints:
                error_mssg = ("Int given not recognized among cluster ints"
                              + " where num_clusters = " + str(self.n_clusters))
                raise ValueError(error_mssg)
        # If str, then check if in list of names
        elif ClusterAnalyzer._is_str(cluster):
            if cluster not in self.cluster_names:
                error_mssg = ("Given name \"" + cluster + "\" not among"
                              + " cluster names: [" + ",".join(self.cluster_names)
                              + "]")
                raise ValueError(error_mssg)
        # Otherwise, throw error due to unrecognized type 
        else:
            error_mssg = "Unrecognized type for cluster param"
            raise ValueError(error_mssg)



        #### If all_clusters are specified, then just return all data
        #### under label, regardless of clustering type
        if cluster == ClusterAnalyzer.ALL_CLUSTERS_INT:
            # if no label given, get ALL data in df
            if label is None: label = list(self.df)
            if len(label)==1:
                return self.df[label].values.flatten()
            else:
                return self.df[label].values
        
        
        
        #### Handle hard clustering
        if self._clustering_type() == ClusterAnalyzer.HARD_CLSTR_STR:
            # Get inds of data points that are in cluster
            row_inds = np.where(self.pred_arr == cluster)[0]
            # If label is None, just return the inds ...
            if label is None:
                return row_inds
            # ... otherwise, return the data corresponding to those inds
            else:
                return self.df[label][ row_inds ].values
        
        
        
        #### Handle soft clustering
        if self._clustering_type() == ClusterAnalyzer.SOFT_CLSTR_STR:
            # get cluster probabilities
            cluster_probs = self.pred_arr[:,cluster]
            
            
            ## NO LABEL GIVEN
            # if label is none and no min_prob, just return probs
            if ((label is None) and (min_prob is None)):
                return cluster_probs
            # If label is none and min_prob is NOT none, return row inds
            # of df that have cluster prob >= min_prob
            if ((label is None) and (min_prob is not None)):
                return np.where(cluster_probs >= min_prob)[0]
            
            
            ## LABEL GIVEN, MIN_PROB GIVEN 
            # if label is NOT none and min_prob is NOT none, return
            # vals with prob exceeding min_prob
            if ((label is not None) and (min_prob is not None)):
                return self.df[label][cluster_probs >= min_prob].values
            
            
            ## LABEL GIVEN and MIN_PROB NONE
            # if label is not none AND min-prob is none, then
            # compute expectation value!
            if ((label is not None) and (min_prob is None)):
                
                # Make dict tracking expec value for each label in label
                # (if its a list)
                expec_vals_dict = {}
                for single_label in label:
                
                    # numeric var, expec val is easy
                    if not self._is_categ_var(label):
                        expec_vals_dict[single_label] = \
                                cluster_probs * self.df[label].values
                    
                    # categ var, need to get counts then get expec val
                    else:
                        categ_dict = {}
                        categ_label_data = self.df[single_label].values
                        unique_categ_vals = np.unique( categ_label_data )
                        # Track each unique val in the categ data
                        for a_val in unique_categ_vals:
                            inds = np.where(categ_label_data == a_val)[0]
                            categ_dict[a_val] = np.sum( cluster_probs[inds] )
                        expec_vals_dict[ single_label ] = categ_dict
                    
                # If label has length 1, then just return the val
                if len(label) == 1:
                    return expec_vals_dict[ label[0] ]
                # If multiple labels, return whole dict
                else:
                    return expec_vals_dict

            
                
       
    
    
    
    
    
    # Determine number of rows and columns of subplots in a figure based on
    # total number of subplots
    def _num_rows_and_cols_for_subplots(self,num_plots):
        
        """
        Decides the number of rows and columns for a matplotlib
        multi plot figure based on number of plots to make
        
        Parameters
        ----------
        num_plots: int
            Number of plots to make
            
        Returns
        -------
        2-element list of number rows x number columns
        
        """
        
        num_subplot_rows, num_subplot_cols = None, None
        
        if num_plots == 1:
            num_subplot_cols = 1
        
        elif num_plots > 1 and num_plots <= 4:
            num_subplot_cols = 2
            
        elif num_plots > 4 and num_plots <= 9:
            num_subplot_cols = 3
            
        elif num_plots > 9 and num_plots <= 16:
            num_subplot_cols = 4
            
        elif num_plots > 16 and num_plots <= 25:
            num_subplot_cols = 5
            
        
        if num_plots <= 2: num_subplot_rows = 1
        else: num_subplot_rows = int( (num_plots - 1) / num_subplot_cols ) + 1
            
        return [num_subplot_rows,num_subplot_cols]

        
        
    
    
    
    
    
    
    
    def _num2str(self, num,
                       scale       = None,
                       decimal_pts = None):
        
        """
        Formatter for converting large numbers to string representation
        (e.g. 9*10**10 --> '90B', where B is billion)
        
        Parameters
        ----------
        num: number
            number to format
            
        scale: str (default None)
            Used to set scale of number. If None, will convert to most
            convenient scale.
            
        decimal_pts: int (default None)
            Number of places to right of decimal to keep.
            
        Returns
        -------
        String-formatted num
        """
        
        if scale is None: scale = "auto"
        if decimal_pts is None: decimal_pts = 2
        
        
        ### If auto, scale based on how large it is
        if scale == "auto":
            scale = ""
            # < thousands
            if (abs(num) <= 100): scale = ""
            # Thousands
            elif ((abs(num) > 10**2) and (abs(num) <= 10**5)): scale = "K"
            # Millions
            elif ((abs(num) > 10**5) and (abs(num) <= 10**8)): scale = "M"
            # Billions
            elif ((abs(num) > 10**8) and (abs(num) <= 10**11)): scale = "B"
            # Trillions
            elif ((abs(num) > 10**11) and (abs(num) <= 10**15)): scale = "T"
            # Like, really big (>1 Quadrillion)
            else:
                warnings.warn("Number to format is very large (>10**15)")#,
                #              category=UserWarning,
                #              module=user_ns.get("__name__"))
            


        ### format num with desired number of decimal pts
        # Format thousands
        if scale == "":
            num2str = ("{:." + str(decimal_pts) + "f}").format( num )
        if scale == "K":
            num2str = ("{:." + str(decimal_pts) + "f}").format( num/10**3 )
        # Format millions
        if scale == "M":
            num2str = ("{:." + str(decimal_pts) + "f}").format( num/10**6 )
        # Format billions
        if scale == "B":
            num2str = ("{:." + str(decimal_pts) + "f}").format( num/10**9 )
        # Format trillions
        if scale == "T":
            num2str = ("{:." + str(decimal_pts) + "f}").format( num/10**12 )
            
            
        return num2str + scale
        
    
    
    
    
    
    
    
    
    
    def _make_legal_filename(self, figname):
        
        """
        Make a legal filename based on the current operating system
        (Currently planned only for Unix systems)
        
        Parameters
        ----------
        figname: str
            figname to make legal
            
        Returns
        -------
        str: Legalized figname
        """
        
        # p means "per" as in X/Y = X per Y
        new_name = figname.replace("/","p")
        new_name = new_name.replace("_","")
        new_name = new_name.replace(" - ","_")
        new_name = new_name.replace(" ","_")
        return new_name
    
    
    

    
    
            
                
    
    
    
    
    
    
    
    
    
    
    def _hist1d_cluster_and_label(self, axis,
                                        cluster       = None,
                                        label         = None,
                                        logx          = None,
                                        logy          = None,
                                        subplot_title = None,
                                        num_bins      = None):
        
        """
        Computes histogram for a particular variable of a cluster
        on a matplotlib axis object. Hard clustering histograms show the 
        in-cluster distributon on top of the full distribution; soft plots 
        the probability over the histogram instead.
        
        
        PARAMETERS
        ----------
        axis: Matplotlib axis object
            The axis in which the data is to be plotted
            
        cluster: int (optional, default is class int indicating all clusters)
            The int representing what cluster data is being taken from
            
        label: str (optional, default ???)
            Label indicating column of data in class df to be plotted
            
        logx: bool (optional, default False)
            If True, data for hist is converted to log10 scale BEFORE hist
            
        logy: bool (optional, default False)
            If True, histogram *results* are plotted on log scale
            
        subplot_title: str (optional, default is label)
            Str used as the title for the subplot
            
        num_bins: int (optional, default is class int for default number of bins)
            Number of bins used for making the histogram
            
            
        RETURNS
        -------
        None
        """
        
        if cluster is None: cluster = ClusterAnalyzer.ALL_CLUSTERS_INT
        if label is None: label = "uninit"
        if logx is None: logx = False
        if logy is None: logy = False
        if subplot_title is None: subplot_title = label
        if num_bins is None: num_bins = ClusterAnalyzer.BINS_1D
        
        
        # regardless of clustering type, set up data for background
        label_data = self._get_data(label=label)
        if logx: label_data = np.log10(label_data)
        
        
        
        
        # 1111111111 Histograms with Numeric Data 1111111111
        
        if not self._is_categ_var(label):
        
            
            #### enforce same bins across all data and in-cluster data
            min_val, max_val = np.min(label_data), np.max(label_data)
            # n+1 edges for n bins
            max_val_for_hist = ((max_val - min_val) / num_bins) + max_val
            label_bins = np.linspace(min_val,max_val_for_hist,num=num_bins)
            
            
            ## Make background hist
            axis.hist(label_data,
                      bins = label_bins,
                      alpha = 0.5,
                      color = "red",
                      log = logy)
            
            
            # grid for background data
            axis.grid(True)
            
            
            #### NUMERIC HARD CLUSTERING:
            ####   show hist of in-cluster data in front
            if self._clustering_type() == ClusterAnalyzer.HARD_CLSTR_STR:
                
                # Retrieve in-cluster data
                in_cluster_data = self._get_data(cluster=cluster,
                                                 label=label)
                
                # make in-cluster hist
                axis.hist(in_cluster_data,
                          alpha = 0.5,
                          color = "blue",
                          bins = label_bins,
                          log = logy)
            
            
            
            #### NUMERIC SOFT CLUSTERING;
            ####   show distribution of probs OVER SAME BINS
            if self._clustering_type() == ClusterAnalyzer.SOFT_CLSTR_STR:
                
                # Retreive all probs for cluster
                in_cluster_probs = self._get_data(cluster=cluster)
                
                # Use scipy binned stat to make histogram over label data AND
                # get average probs of data within those bins
                """print(label_data, label_data.shape, 
                      in_cluster_probs, in_cluster_probs.shape,
                      label_bins, label_bins.shape)"""
                stat_results = scipy.stats.binned_statistic(
                                                label_data,
                                                in_cluster_probs,
                                                statistic="mean",
                                                bins=label_bins
                                                            )
                
                ## Make hist of prob
                # Go from n+1 to n bin_edges by taking mean of edges
                mean_bin_vals = np.array(
                    [ (stat_results.bin_edges[i] + stat_results.bin_edges[i+1])/2
                      for i in range(stat_results.bin_edges.shape[0]-1) ]
                                         )
                # Make ax copy with same x axis
                axis_probs = axis.twinx()
                axis_probs.scatter(mean_bin_vals,
                                   stat_results.statistic,
                                   s=2.0,
                                   c="purple")
                # Enforce same y scale for all prob plots
                axis_probs.set_ylim([-0.05,1.05])
                
                ## Make special gridlines for PROB Y-AXIS
                # Major gridlines
                axis_probs.grid(True, which="major",
                                **{'linestyle':'dashed',
                                   'color':'black',
                                   'alpha':0.7})
                # minor gridlines
                axis_probs.yaxis.set_minor_locator(AutoMinorLocator())
                axis_probs.grid(True, which="minor",
                                **{'linestyle':'dotted',
                                   'color':'black',
                                   'alpha':0.7,
                                   'linewidth':1})
                
            
            ## Setup minor X-AXIS ticks for *orig* (not probs) data
            axis.xaxis.set_minor_locator(AutoMinorLocator())
            axis.xaxis.grid(True, which='minor',
                            **{'linestyle':'dotted'})
            
            
            ## Copy major and minor x-axis ticks to top of plot
            axis.tick_params(which="both",
                             top=True, labeltop=False,
                             bottom=True, labelbottom=True)
            
        
        # 11111111111111111111111111111111111111111111111111
        
        
        
        
        
        # 2222222222 Histograms with Categorical Data 2222222222
        
        else:
            
            # get values and counts of categorical var; then make dict with
            # categ values as keys and counts as values
            label_vals, label_counts = np.unique(label_data,
                                                 return_counts=True)
            label_categ_dict = {}
            for key, value in zip(label_vals, label_counts):
                label_categ_dict[key] = value
            
            # Make bar chart
            barchart_label_data = axis.bar(
                                    label_categ_dict.keys(),
                                    label_categ_dict.values(),
                                    log=logy
                                           )

            # Get cluster data to count frequencies
            # NOTE: in-cluster data is only *retrieved* in the 
            #       if statements below; it's then plotted in the
            #       same way, regardless of clustering type.
            cluster_data = self._get_data(cluster=cluster,
                                        label=label)
            cluster_categ_dict = None
            
            
            #### CATEG HARD CLUSTERING:
            ####   get counts of unique vars of in-cluster data
            if self._clustering_type() == ClusterAnalyzer.HARD_CLSTR_STR:
                
                # For hard clustering, cluster_data is just a 1d array of
                # categorical vals. So determine vals and counts then make 
                # dict to show counts per categ value
                cluster_vals, cluster_counts = np.unique(cluster_data,
                                                         return_counts=True)
                cluster_categ_dict = {}
                for key, value in zip(cluster_vals, cluster_counts):
                    cluster_categ_dict[key] = value
            
            
            
            #### CATEG SOFT CLUSTERING:
            ####   get expectation value of occupancy for each unique value
            ####   in var
            if self._clustering_type() == ClusterAnalyzer.SOFT_CLSTR_STR:
                
                #self._get_data(cluster=cluster, label=label)
                
                # For soft clustering, cluster_data is expected occupancy
                # (e.g. sum of probs for pts with same categ value) and 
                # presented as a dict
                # ( e.g. {"categA":50.3, "categB":71.9, etc} )
                cluster_categ_dict = cluster_data
            
            
            
            # Plot dict of categ vals of cluster data
            barchart_cluster_data = axis.bar(
                                    cluster_categ_dict.keys(),
                                    cluster_categ_dict.values(),
                                    log=logy
                                             )
            
            
            # Show percentage on bar plot!
            for i in range(len(label_vals)):
                orig_bar = barchart_label_data[i]
                cluster_bar = barchart_cluster_data[i]
                cluster_height = cluster_bar.get_height()
                perc_val = 100 * cluster_height / orig_bar.get_height()
                
                # for more info on text, see page
                # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html
                axis.text(x=cluster_bar.get_x() + cluster_bar.get_width() / 2,
                          y=cluster_height+.10,
                          s="{:.1f}%".format(perc_val),
                          ha='center',
                          fontsize="small",#"x-small",
                          c="black")
            """for p in pps:
                height = p.get_height()
                perc_val = height #if not log_y_axis else np.log10(height)
                axis.text(x=p.get_x() + p.get_width() / 2,
                        y=height+.10,
                        s="{:.1f}%".format(perc_val),
                        ha='center')"""
            
            # Rotate labels a little!
            for a_label in axis.get_xticklabels():
                a_label.set_rotation(40)
                a_label.set_horizontalalignment('right')

        # 222222222222222222222222222222222222222222222222222222
        
        
    
        
        # Set title for subplot
        axis.set_title(subplot_title)
        
        # Tack on grid
        axis.yaxis.grid(True, which="major")
    
    
    
    
    
    
    
    
    
    
    def _make_hist2d_patch_collection(self, xbins,
                                            ybins,
                                            rect_vals,
                                            nan_patch_dict   = None,
                                            color_patch_dict = None):
            
        """
        Creates two "patch collections" for 2d histogram data: one collection,
        the "color_patch", for 2d bins in which a value gauges its color from
        a color bar, and another collection, the "nan_patch" for 2d bins in
        which the color bar does not apply.
        
        For more on matplotlib collections, see
        https://matplotlib.org/stable/api/collections_api.html
        
        
        
        PARAMETERS
        ----------
        xbins: 1d numpy array
            Array of x-axis bin values for a 2d histogram
        
        ybins: 1d numpy array
            Array of y-axis bin values for a 2d histogram
                
        rect_vals: 1d numpy array
            Array of values used to indicate what color the patch should be
            from a color bar; if the value is np.nan, the nan_patch will be
            used to color it instead.
            
        nan_patch_dict: kwargs dict (optional)
            Kwargs used to create the nan_patch collection
            
        color_patch_dict: kwargs dict (optional)
            Kwargs used to create the color_patch collection
            
            
            
        RETURNS
        -------
        2-element list of matplotlib nan and color patch collections, resp.
        """
        
        #### Create rectangle from coords for each bin-pair and save it
        #### to nan or not-nan list
        nan_rect_list = []
        color_rect_list = []
        for i in range(xbins.shape[0]-1):
            for q in range(ybins.shape[0]-1):
                
                # get bin values
                x_bin_start = xbins[i]
                x_bin_end = xbins[i+1]
                y_bin_start = ybins[q]
                y_bin_end = ybins[q+1]
                
                # Build rectangle
                rect_corner_pt = (x_bin_start, y_bin_start)
                width = x_bin_end - x_bin_start
                height = y_bin_end - y_bin_start
                rect = Rectangle(rect_corner_pt, width, height)
                
                # Save stat value 
                if np.isnan(rect_vals[i,q]):
                    nan_rect_list.append( rect )
                else:
                    color_rect_list.append( rect )
                    
        
        
        #### Make PatchCollection of Color Rectangles
        # Setup default heatmap patch dict
        default_color_patch_kwargs = { "alpha": 0.5,
                                       "cmap": plt.get_cmap('viridis'),
                                       "edgecolor": "black",
                                       "linewidth": 0.125 }
        # If not set, just assign to default and keep param unmodified by user
        if color_patch_dict is None:
            color_patch_dict = default_color_patch_kwargs
        color_patch_dict = {**default_color_patch_kwargs,
                              **color_patch_dict}
        # Check if user gave cmap as str, which means we need to
        # manually import the cmap object itself from matplotlib
        if ClusterAnalyzer._is_str( color_patch_dict['cmap'] ):
            color_patch_dict['cmap'] = plt.get_cmap(color_patch_dict['cmap'])
        # Make the color patch collection
        color_rect_coll = PatchCollection(color_rect_list,
                                          **color_patch_dict)
        
        
        
        #### Make PatchCollection of Nan Rectangles
        # Setup default nan patch dict
        default_nan_patch_kwargs = { "alpha": 1.0,
                                     "facecolor": "grey",
                                     "edgecolor": "black",
                                     "linewidth": 0.125 }
        # If not set, just assign to default and keep param unmodified by user
        if nan_patch_dict is None:
            nan_patch_dict = default_nan_patch_kwargs
        nan_patch_dict = {**default_nan_patch_kwargs,
                          **nan_patch_dict}
        # Make the nan patch collection
        nan_rect_coll = PatchCollection(nan_rect_list,
                                        **nan_patch_dict)
        
        
        
        return nan_rect_coll, color_rect_coll
    
    
    
    
    
    
    
    
    def _hist2d_single_cluster(self, fig,
                                     axis,
                                     cluster          = None,
                                     histxy           = None,
                                     hist_var         = None,
                                     hist_stat        = None,
                                     bins             = None,
                                     logx             = None,
                                     logy             = None,
                                     log_hist_var     = None,
                                     log_hist_stat    = None,
                                     cbar_bounds      = None,
                                     color_patch_dict = None,
                                     nan_patch_dict   = None,
                                     subplot_title    = None):
        
        """
        Create a 2d histogram for a pair of variables with the 
        
        
        
        Computes histogram for a particular variable of a cluster
        on a matplotlib axis object. Hard clustering histograms show the 
        in-cluster distributon on top of the full distribution; soft plots 
        the probability over the histogram instead.
        
        
        TODO:
            weighted avgs of labels in bins!
            sum up label_vals * probs / occupancy where occupancy = sum probs
        """
        
        
        
        ### set default params
        if cluster is None: cluster = ClusterAnalyzer.ALL_CLUSTERS_INT
        if histxy is None: histxy = list(self.df)[:2]
        if hist_var is None: hist_var = list(self.df)[2]
        if hist_stat is None: hist_stat = "mean"
        if bins is None: bins = ClusterAnalyzer.BINS_2D
        if logx is None: logx = False
        if logy is None: logy = False
        if log_hist_var is None: log_hist_var = False
        if log_hist_stat is None: log_hist_stat = False
        if subplot_title is None: subplot_title = str(cluster)
        
        
        
        ### get histxy data
        histx_data = self._get_data(label=histxy[0])
        histy_data = self._get_data(label=histxy[1])
        
        
        
        ### Determine xbins and ybins based on bins (and type thereof)
        # If given ints, then compute lin-spaced bins
        #if ClusterAnalyzer._is_int(bins[0]):
        if np.issubdtype(bins[0], np.integer):
            # compute xbins
            xbins = np.linspace(np.min(histx_data),
                                np.max(histx_data),
                                bins[0]+1)
            # compute ybins
            ybins = np.linspace(np.min(histy_data),
                                np.max(histy_data),
                                bins[1]+1)
        # Otherwise, take x/y bins as arrays
        else:
            xbins = bins[0]
            ybins = bins[1]
        
        
        
        ### get hist var data
        hist_var_data = None
        # If plotting prob or occupancy, then have to handle data specially
        if ((hist_var == ClusterAnalyzer.PROBAB_STR) or \
            (hist_var == ClusterAnalyzer.OCCUP_STR)):
            # If all clusters specified, then just array of ones will work
            if cluster == ClusterAnalyzer.ALL_CLUSTERS_INT:
                hist_var_data = np.ones(self.df.shape[0])
            # Otherwise, retrieve probs belonging to particular cluster
            else:
                hist_var_data = self._get_data(cluster=cluster)
        # Can just get data via label otherwise
        if hist_var in list(self.df):
            hist_var_data = self._get_data(label=hist_var)
        
        
        
        ### Handle special calculations
        # occupancy cannot have a statistic applied to them
        if hist_var == ClusterAnalyzer.OCCUP_STR:
            if hist_stat != "sum":
                warnings.warn("Any statistic apart from \"sum\" cannot be"
                              + " used with hist_var = \""
                              + ClusterAnalyzer.OCCUP_STR + "\"; forcing "
                              + "hist_stat to \"sum\"...")
                hist_stat = "sum"
        # If soft clustering and hist_stat is "count", then
        # result is non-sensical; raise error and tell user to use "occupancy"
        if self._clustering_type() == ClusterAnalyzer.SOFT_CLSTR_STR:
            if hist_stat == "count":
                raise ValueError("Can't use statistic \"count\" with soft"
                                 + " clustering; use hist_var = \""
                                 + ClusterAnalyzer.OCCUP_STR + "\" with "
                                 + "hist_stat = \"sum\" instead")
           
        
        
        ### convert data to log if desired 
        if logx: histx_data = np.log10( histx_data )
        if logy: histy_data = np.log10( histy_data )
        if log_hist_var: hist_var_data = np.log10( hist_var_data )
        
        
        
        ### compute 2d hist
        stat_results = scipy.stats.binned_statistic_2d(
                                            histx_data,
                                            histy_data,
                                            hist_var_data,
                                            hist_stat,
                                            bins = [xbins,ybins]
                                                        )
        # Get hist data from scipy result
        hist_var_stats = stat_results.statistic
        
        
        
        ### convert hist2d *results* into log-scale if specified
        if log_hist_stat: hist_var_stats = np.log10(hist_var_stats)
            
        
            
        ### Make separate Patch Collections of Rectangles that have defined /
        ### undefined (i.e. number vs np.nan) values.
        nan_patchcoll, color_patchcoll = \
                self._make_hist2d_patch_collection(
                            xbins,
                            ybins,
                            hist_var_stats,
                            nan_patch_dict = nan_patch_dict,
                            color_patch_dict = color_patch_dict
                                            )
        
        
        
        ### Get bounds for colorbar to apply to other plots if desired
        if cbar_bounds is None:
            if hist_var == ClusterAnalyzer.PROBAB_STR:
                cbar_bounds = (0,1)
            else:
                minval = np.min(hist_var_data)
                maxval = np.max(hist_var_data)
                cbar_bounds = ( minval - (maxval - minval) * 0.05,
                                (maxval - minval) * 0.05 + maxval )
        
        
        
        ### Plot color patches onto axis
        axis.add_collection(color_patchcoll)
        # get non nans from heatmap vals (2d -> 1d via row major ordering!)
        color_patchcoll.set_array( 
                hist_var_stats.flatten()[
                    ~np.isnan( hist_var_stats.flatten() )
                                        ]
                                  )
        # set cbar bounds
        color_patchcoll.set_clim(cbar_bounds)
        # set colorbar
        fig.colorbar(color_patchcoll, ax=axis)
        
        
        
        ### Apply nan-colored patches
        axis.add_collection(nan_patchcoll)
        
        
        
        ### Make x,y labels
        axis.set_xlabel(histxy[0])
        axis.set_ylabel(histxy[1])
        
        
        
        ### Make subtitle
        axis.set_title(subplot_title)
        
        
        
        ### Return tuple for later cbar changes if desired
        return (color_patchcoll, hist_var_stats)
        
    
    
    
    
    
    
    def _calc_mahal_dist(self, data, mu, cov):
        
        """
        Calculates the mahalanobis distance (i.e. multivariate analog of
        'how many standard deviations away if point P away from gaussian G?')
        
        
        PARAMETERS
        ----------
        data: 1d or 2d numpy array
            Data in the form that the gaussian mixture model was trained on
        
        mu: scalar or 1d numpy vector
            The mean of a gaussian
            
        cov: scalar or 2d numpy matrix
            The covariance of the gaussian. Must be positive semi-definite
            if matrix!
        
        
        RETURNS
        -------
        1d numpy vector (the Mahalanobis distances of the data given)
        """
        
        ### if the mu shifted data is defined as A (e.g. 
        ### A = data_matrix - mu_vector), then we're trying to compute the 
        ### (x_vector - mu_vector).T inv_cov_matrix
        mu_shifted_data = data - mu
        return np.sqrt(
                 np.einsum('ij,ji->i', 
                 np.matmul(mu_shifted_data, np.linalg.inv(cov)),
                 mu_shifted_data.T)
                      )
        
    
    
    
    
    
    def _build_constraints_list_of_dicts(self, constraints):
        
        """
        
        Build list of dicts of constraints from initial list.
        Each dict represents a possible combination of unique values for
        the variable names given in constraints
        
        EXAMPLE
        -------
        Let the class df have variables A (with values 1 and 2) and B
        (with values x, y, z).
        Then _build_constraints_list_of_dicts(["A","B"]) would yield
        [ {"A":1, "B":"x"},
          {"A":1, "B":"y"},
          {"A":1, "B":"z"},
          {"A":2, "B":"x"},
          {"A":2, "B":"y"},
          {"A":2, "B":"z"} ]
        
        PARAMETERS
        ----------
        constraints: str / list of strs / dicts
                     (if dict, ONLY key is variable name in class df)
        
        RETURNS
        -------
        list of dicts; each dict has one str per key and indicates the 
        allowed value for that variable for that combination
        
        """
        
        
        
        #### Convert input from str -> list (if str)
        #### and list -> dict with values being None (if list)
        if isinstance(constraints, str): constraints = [ constraints ]
        if isinstance(constraints, list):
            constraints = { elem : None for elem in constraints }
        
        #### Determine unique values for each variable given
        constraints_dict = {}
        for elem in constraints:
            
            ## If None, then determine all unique values in variable
            if constraints[elem] is None:
                label_data = self._get_data(label=elem)
                # If variable is not categorical, throw error
                if not self._is_categ_var(elem):
                    raise ValueError("Separating data based on the number of"
                                     + " unique values in a NUMERIC variable"
                                     + " is not supported; categorical only!")
                # Otherwise, get unique values and save into dict
                constraints_dict[elem] = np.unique(label_data).tolist()
            
            ## Otherwise, copy value from original dict to new dict
            else:
                constraints_dict[elem] = constraints[elem]
        
        
        #### Build constraint list
        ## Get keys of constraints dict for consistent order
        constraints_keys = list(constraints_dict.keys())
        ## Get combos of list values
        constraint_combos = list(itertools.product(
                        *[constraints_dict[key] for key in constraints_keys]
                                                ))
        ## Convert list of lists into list of dicts with variables as key names
        constraint_combos_dict_list = []
        for combo in constraint_combos:
          
            # turn each combo-tuple into a dict with the constraints_keys
            # as the keys for the new dict
            combo_as_dict = {}
            for i in range(len(constraints_keys)):
                combo_as_dict[ constraints_keys[i] ] = combo[i]
            constraint_combos_dict_list.append( combo_as_dict )
        
        return constraint_combos_dict_list
    
    
    
    
    
    
    
    
    
    
    # ==================== "PUBLIC" FUNCTIONS ====================
    
    
    
    
    
    
    def hist1d(self, hist_vars      = None,
                     num_bins       = None,
                     figsize        = None,
                     logx           = None,
                     logy           = None,
                     subplot_titles = None):
        
        """
        
        Create a multi-subplot figure of histograms for each cluster. A hist
        is made for each variable in hist_vars
        
        
        
        PARAMETERS
        ----------
        hist_vars: list of str (optional, default all labels in class df)
        
        """
        
        if hist_vars is None: hist_vars = list(self.df)
        if num_bins is None: num_bins = ClusterAnalyzer.BINS_1D
        if figsize is None: figsize = ClusterAnalyzer.MULTIPLOT_SIZE
        if subplot_titles is None:
            subplot_titles = { label : label for label in hist_vars }
        # fignames unsupported for now!
        
        
        ## Prepare logx
        if logx is None: logx = False
        if logx == False: logx = []
        if logx == True: logx = hist_vars
        
        
        ## Prepare logy
        if logy is None: logy = False
        if logy == False: logy = []
        if logy == True: logy = hist_vars
        
        
        # Make output hist1d folder
        hist_path = self._make_rel_subdir("Hist1d")
        
        
        # Make DIFFERENT figure for EACH cluster
        for cluster in range(self.n_clusters):
            
            
            # Create fig params
            num_plots = len(hist_vars)
            num_plot_rows, num_plot_cols = \
                self._num_rows_and_cols_for_subplots(num_plots)
            fig, axes = plt.subplots(num_plot_rows, num_plot_cols,
                                     dpi=120, figsize=figsize)
            
            
            # Turn axes from 2d array into 1d for easier iteration
            axes_1d = axes.reshape(num_plot_rows * num_plot_cols)
            
            
            # For each overall plot, make individual subplots
            for i in range(len(hist_vars)):
                label = hist_vars[i]
                ax = axes_1d[i]
                
                
                #### Call func to make each subplot
                self._hist1d_cluster_and_label(
                            ax,
                            cluster=cluster,
                            label=label,
                            logx = label in logx,
                            logy = label in logy,
                            num_bins = num_bins,
                            subplot_title = subplot_titles[label]
                                              )
            
            
            # Delete unsused subplots
            for i in range(len(hist_vars),len(axes_1d)): axes_1d[i].axis("off")
            
            
            
            #### Make suptitle and tighten-up subplots
            
            ## compute cluster occupancy
            cluster_data = self._get_data(cluster=cluster)
            cluster_occ = None
            
            # If hard clustering, then cluster data is row inds;
            # can track # pts in cluster by comparing array sizes
            if self._clustering_type() == ClusterAnalyzer.HARD_CLSTR_STR:
                cluster_occ = cluster_data.shape[0]
                
            # If soft clustering, the cluster data is array of probabilities;
            # compute occupancy by summing up probs in cluster
            if self._clustering_type() == ClusterAnalyzer.SOFT_CLSTR_STR: 
                cluster_occ = np.sum( cluster_data )
                
            # convert cluster_occ to str
            cluster_occ_str = self._num2str(cluster_occ)
            
            # compute num pts in df and % occupancy
            num_pts_str = self._num2str(self.df.shape[0])
            perc_occup_str = "{:.2f}".format(
                            100 * cluster_occ / self.df.shape[0]
                                            )
            
            ## Make suptitle
            cluster_name = self._cluster_int2name(cluster)
            suptitle1 = cluster_name + " with " + str(num_bins) + " bins"
            suptitle2 = ("Cluster Occupancy: " + cluster_occ_str + " pts out "
                         + "of " + num_pts_str + " (" + perc_occup_str + "%)")
            suptitle = suptitle1 + "\n" + suptitle2
            # Set suptitle
            fig.suptitle(suptitle,fontsize=16)
            fig.tight_layout()
            
            
            
            #### Save cluster histogram
            cluster_figname = cluster_name + "_cluster" + self.IMAGE_TYPE
            image_path = os.path.join(hist_path,
                                      cluster_figname)
            plt.savefig(image_path)
            plt.close()
    
        
    
    
    
    
    
    
    
    
    def hist2d(self, histxy           = None,
                     hist_var         = None,
                     hist_stat        = None,
                     bins             = None,
                     figsize          = None,
                     logx             = None,
                     logy             = None,
                     log_hist_var     = None,
                     log_hist_stat    = None,
                     cbar_bounds      = None,
                     color_patch_dict = None,
                     nan_patch_dict   = None,
                     figname          = None):
        
        """
        
        Supported hist_heatmap: any label in class df / prob
        
        heatmap_stat: "probability" or any stat listed in scipy docs below
        
        note: if heatmap_stat set to "count", hist_heatmap is irrelevant!!!
        
        for supported heatmap_stat values, check out scipy's hist2d docs
        EXECPT FOR "count":
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic_2d.html
        
        """
        
        if histxy is None: histxy = list(self.df)[:2]
        if hist_var is None: hist_var = list(self.df)[2]
        if hist_stat is None: hist_stat = "mean"
        if bins is None: bins = ClusterAnalyzer.BINS_2D
        if figsize is None: figsize = ClusterAnalyzer.MULTIPLOT_SIZE
        if logx is None: logx = False
        if logy is None: logy = False
        if log_hist_var is None: log_hist_var = False
        if log_hist_stat is None: log_hist_stat = False
        #if subplot
        
        
        
                
                
        ## If user set hist_heatmap to prob or count, raise error if user
        ## also tries to set heatmap_stat
        """if ((hist_heatmap == ClusterAnalyzer.PROBAB_STR) \
            or (hist_heatmap == "count")):
            if heatmap_stat is not None:
                raise ValueError(
                    "heatmap_stat cannot be set if hist_heatmap is set."
                                )"""
            
                
        
        
        # Make output hist2d folder
        hist_path = self._make_rel_subdir("Hist2d")
        
        
        # Make fig and axes for subplots
        num_plots = self.n_clusters + 1
        num_plot_rows, num_plot_cols = \
                self._num_rows_and_cols_for_subplots(num_plots)
        fig, axes = plt.subplots(num_plot_rows, num_plot_cols, figsize=figsize)
        axes_1d = axes.reshape(num_plot_rows * num_plot_cols)
        
        
        ### Depending on statistic for scipy's 2d hist, there might be
        ### bins with UNDEFINED VALUES (e.g. for statistic = 'count', bins
        ### with no values would have count = 0,... but for
        ### statistic = 'median', such bins would have no result). Scipy
        ### defines these bins to have np.nan values!
        
        
        # do check here to see if want consistent bins AND cbar bounds, then
        # supply rsult to func below
        histx_data = self._get_data(label=histxy[0])
        histy_data = self._get_data(label=histxy[1])
        ### compute xbins and ybins based on num_bins
        # compute xbins
        xbins = np.linspace(np.min(histx_data),
                            np.max(histx_data),
                            bins[0]+1)
        # compute ybins
        ybins = np.linspace(np.min(histy_data),
                            np.max(histy_data),
                            bins[1]+1)
        
        
        ### Init dict for tracking colorbars and histvals as
        ### functions of subplot used
        subplot_dict = {}
        
        
        ### create 2d hist for covering ALL CLUSTERS
        subplot_dict[ axes_1d[0] ] = \
            self._hist2d_single_cluster(
                            fig,
                            axes_1d[0],
                            cluster = ClusterAnalyzer.ALL_CLUSTERS_INT,
                            histxy = histxy,
                            hist_var = hist_var,
                            hist_stat = hist_stat,
                            bins = [xbins,ybins],
                            logx = logx,
                            logy = logy,
                            log_hist_var = log_hist_var,
                            log_hist_stat = log_hist_stat,
                            cbar_bounds = cbar_bounds,
                            subplot_title = "All Clusters",
                            color_patch_dict = color_patch_dict,
                            nan_patch_dict = nan_patch_dict
                                         )
        
        
        
        
        ### Make subplot of overall figure for EACH cluster
        possible_clusters = np.arange(self.n_clusters)
        for i in range(len(possible_clusters)):
            axis = axes_1d[i+1]
            cluster = possible_clusters[i]
            
            # Get cluster name as str
            cluster_name = self._cluster_int2name(cluster)
            
            # create 2d hist for each cluster
            subplot_dict[axis] = \
                self._hist2d_single_cluster(
                                    fig,
                                    axis,
                                    cluster = cluster,
                                    histxy = histxy,
                                    hist_var = hist_var,
                                    hist_stat = hist_stat,
                                    bins = bins,
                                    logx = logx,
                                    logy = logy,
                                    log_hist_var = log_hist_var,
                                    log_hist_stat = log_hist_stat,
                                    cbar_bounds = cbar_bounds,
                                    subplot_title = cluster_name,
                                    color_patch_dict = color_patch_dict,
                                    nan_patch_dict = nan_patch_dict
                                            )
            
            
        
        ### Ensure all subplots have same x,y lims and all rectangles
        ### can be seen
        for ax in axes_1d:
            ax.set_xlim(xbins[0],xbins[-1])
            ax.set_ylim(ybins[0],ybins[-1])
        
        
        
        ### If cbar_bounds is True, then enforce same cbar_bounds over
        ### all subplots
        if cbar_bounds == True:
            min_hist_vals = []
            max_hist_vals = []
            # Get list of min / max vals per hist_vals in each subplot
            for ax in subplot_dict:
                min_hist_vals.append(
                    np.min( subplot_dict[ax][1] )
                                    )
                max_hist_vals.append(
                    np.max( subplot_dict[ax][1] )
                                    )
            # Determine true min / max and make new bounds
            minval = min(min_hist_vals)
            maxval = max(max_hist_vals)
            new_bounds = ( minval - (maxval - minval) * 0.05,
                           maxval + (maxval - minval) * 0.05 )
            # Modify patch collection bounds for each subplot
            for ax in subplot_dict:
                subplot_dict[ax][0].set_clim( new_bounds )
                
        
        
        ### Delete unsused subplots
        for i in range(num_plots,len(axes_1d)): axes_1d[i].axis("off")
        
        
        
        ### Make suptitle
        # If log was used *before* stats, then slap LOG on hist_var str
        hist_var_for_sup = hist_var
        if log_hist_var: hist_var_for_sup = "LOG-" + hist_var_for_sup
        suptitle = ("Hist2D: heatmap \"" + hist_var_for_sup
                    + "\" | bins " + str(bins))
        # If log was used *after* stats, put LOG at front of suptitle
        if log_hist_stat: suptitle = "LogScale-" + suptitle
        # Set suptitle
        fig.suptitle(suptitle,fontsize=16)
        fig.tight_layout()
        
        
        
        #### Save cluster histogram
        if figname is None: figname = "hist2d_" + hist_var
        image_path = os.path.join(hist_path, figname + self.IMAGE_TYPE)
        plt.savefig(image_path)
        plt.close()
    
    
    
    
    
    




    def mahal_hist(self, trans_data = None,
                         gmm        = None,
                         bins       = None,
                         figsize    = None,
                         logx       = None,
                         logy       = None,
                         figname    = None):
        
        """
        
        Creates histograms of the mahalanobis distance (i.e. multivariate
        gaussian equivalent of z-score) 
        
        
        """
        
        # Need to provide training data and scikit-learn mixture model
        # instance to run!
        if ((trans_data is None) or (gmm is None)):
            raise ValueError(
                "Both \"training_data\" and \"gmm\" default args must be set!"
                            )
        
        # Set typical default args
        if bins is None: bins = ClusterAnalyzer.BINS_1D
        if figsize is None: figsize = ClusterAnalyzer.MULTIPLOT_SIZE
        if logx is None: logx = False
        if logy is None: logy = False
        

        # Make output hist2d folder
        hist_path = self._make_rel_subdir("Hist1d")
        
        
        # Make fig and axes for subplots
        num_plots = self.n_clusters
        num_plot_rows, num_plot_cols = \
                self._num_rows_and_cols_for_subplots(num_plots)
        fig, axes = plt.subplots(num_plot_rows, num_plot_cols, figsize=figsize)
        axes_1d = axes.reshape(num_plot_rows * num_plot_cols)
        
        
        ### Make subplot of overall figure for EACH cluster
        possible_clusters = np.arange(self.n_clusters)
        for i in range(len(possible_clusters)):
            hist_ax = axes_1d[i]
            cluster = possible_clusters[i]
            
            ### Compute mahanabolis distances based on gaussian for cluster
            mean = gmm.means_[i]
            # have to check covariance type! some types allow for single
            # matrices instead of full, independent matrices per gaussian
            if gmm.covariance_type != 'full':
                covmat = gmm.covariances_
            else:
                covmat = gmm.covariances_[i]
            
            mahal_dist = self._calc_mahal_dist(trans_data,
                                               mean,
                                               covmat)
            if logx: mahal_dist = np.log10(mahal_dist)
            
            # compute bin edges manually so as to provide same bins between
            # hist for mahal distances and for prob averaging
            bin_edges = None
            if isinstance(bins, int):
                min_val, max_val = np.min(mahal_dist), np.max(mahal_dist)
                # n+1 edges for n bins
                max_val_for_hist = ((max_val - min_val) / bins) + max_val
                bin_edges = np.linspace(min_val, max_val_for_hist, num=bins)
            else:
                bin_edges = bins
                
            
            
            
            hist_ax.hist(mahal_dist,
                         bins=bin_edges,
                         log=logy,
                         color="red",
                         alpha=0.5)
            
            #hist_ax.yaxis.grid(True, which="major")
            
            
            
            ### compute binned prob for cluster
            prob_ax = hist_ax.twinx()
            # Retreive all probs for cluster
            in_cluster_probs = self._get_data(cluster=cluster)
            # Use scipy binned stat to make histogram over label data AND
            # get average probs of data within those bins
            stat_results = scipy.stats.binned_statistic(
                                            mahal_dist,
                                            in_cluster_probs,
                                            statistic="mean",
                                            bins=bin_edges
                                                        )
            # get mean bin value between bin edges
            mean_bin_vals = np.array(
                [ (stat_results.bin_edges[i] + stat_results.bin_edges[i+1])/2
                  for i in range(stat_results.bin_edges.shape[0]-1) ]
                                     )
            # plots probs
            prob_ax.scatter(mean_bin_vals,
                            stat_results.statistic,
                            s=4.0,
                            c="purple")
            # Enforce same y scale for all prob plots
            prob_ax.set_ylim([-0.05,1.05])
            ## Make special gridlines for PROB Y-AXIS
            # Major gridlines
            prob_ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
            prob_ax.set_yticks([0.1,0.3,0.5,0.7,0.9], minor=True)
            prob_ax.grid(True, which="major",
                         **{'linestyle':'dashed',
                            'color':'black',
                            'alpha':0.7})
            # minor gridlines
            #prob_ax.yaxis.set_minor_locator(AutoMinorLocator())
            prob_ax.grid(True, which="minor",
                         **{'linestyle':'dotted',
                            'color':'black',
                            'alpha':0.7,
                            'linewidth':1})
            
            
            
            ### Make subtitle
            cluster_name = self._cluster_int2name(cluster)
            hist_ax.set_title(cluster_name + ' cluster')
        
        
        
        ### Delete unsused subplots
        for i in range(num_plots,len(axes_1d)): axes_1d[i].axis("off")
        
            
        
        ### Make suptitle
        # Get str of integer number of bins
        strbins = None
        if isinstance(bins, int):
            strbins = str(bins)
        else:
            strbins = str(len(bins))
        suptitle = 'Mahalanobis Dist Hists per cluster | ' + strbins + ' bins'
        # Set suptitle
        fig.suptitle(suptitle,fontsize=16)
        fig.tight_layout()
        
        
        
        #### Save cluster histogram
        if figname is None: figname = "mahal_hist"
        image_path = os.path.join(hist_path, figname + self.IMAGE_TYPE)
        plt.savefig(image_path)
        plt.close()
            
            
            
            
            
    
        
    
    def compute_crossings(self, time_var                     = None,
                                constraints                  = None,
                                min_prob                     = None,
                                min_crossing_duration        = None,
                                max_crossing_duration        = None,
                                min_beyond_crossing_duration = None,
                                max_beyond_crossing_duration = None,
                                min_cluster_frac             = None,
                                order_matters                = None,
                                overlap_preference           = None,
                                save_crossings               = None):

        
        """
        
        Determine crossings in clustering
        
        """
        
        
        
        #### Handle basic default args
        if time_var is None: time_var = list(self.df)[0]
        if min_prob is None: min_prob = 0.8
        if order_matters is None: order_matters = True
        if save_crossings is None: save_crossings = False
        if overlap_preference is None:
            overlap_preference = "best"
        
        
        
        #### Handle duration-related params
        kwargs_to_check = [ min_crossing_duration,
                            max_crossing_duration,
                            min_beyond_crossing_duration,
                            max_beyond_crossing_duration ]
        if max_beyond_crossing_duration is None:
            max_beyond_crossing_duration = min_beyond_crossing_duration
        ## Require all duration-related params to be set!
        if (None in kwargs_to_check):
            raise ValueError("all duration-related kwargs have to be set!")
        ## Also require that said params be pandas timedelta objs
        pd_tdelta_type = pd._libs.tslibs.timedeltas.Timedelta
        for a_kwarg in kwargs_to_check:
            if not isinstance(a_kwarg, pd_tdelta_type):
                raise ValueError("all duration-related kwargs have to be"
                                 + " Pandas Timedelta instances")
        
        
        
        #### Retrieve time data and preds
        times = self._get_data(label=time_var)
        cluster_preds = self.pred_arr
        
        
        
        #### Build dict with keys as cluster-name tuples and values as lists.
        #### Crossings will be saved to list with key matching the
        #### departure and arrival cluster names (derived form cluster ints)
        cluster_names_permutations = list(itertools.product(
                                list(self.cluster_names.keys()),
                                repeat=2
                                                           ))
        ## Make sure to incorporate same cluster crossings
        ## (e.g. dips in probability for soft clustering)
        cluster_names_permutations.extend(
            [ (c_name, c_name) \
              for c_name in list(self.cluster_names.keys()) ]
                                         )
        ## Make the dict
        crossing_dict = { name_tuple : []
                          for name_tuple in cluster_names_permutations }
        
        
        
        #### Build constraints dict
        constraint_combos_list = \
                self._build_constraints_list_of_dicts(constraints)
        
                
        
        #### Get particular combo of constraints, get subset of data that
        #### satisfy them, and check for crossings in that subset
        #crossing_dfs_and_constraints = []
        print("Computing crossings under constraints:")
        total_saved_crossings = 0
        for constraint_combo in constraint_combos_list:
            print("  ",constraint_combo)
            
            
            ## Find row inds that satisfy ALL constraints in constraint_combo
            row_inds = np.arange(self.df.shape[0])
            for var_key in constraint_combo:
                row_inds = np.intersect1d(
                    row_inds,
                    np.where(self.df[var_key] == constraint_combo[var_key])[0]
                                         )
                
                
            ## Get subset that satsify those constraints and sort w/r/t time
            subdf = self.df.iloc[ row_inds, : ]
            # Get argsort for sorted times
            sort_inds = np.argsort(subdf[time_var])
            # Get sorted subdf *and* cluster predictions with sort_inds
            sorted_subdf = subdf.iloc[sort_inds,:]
            sorted_preds = cluster_preds[ sort_inds ]
            # Get times of subdf
            times = sorted_subdf[time_var].values
        
        
            ## Perform clustering based on soft or hard
            if self._clustering_type() == ClusterAnalyzer.HARD_CLSTR_STR:
                
                print("not yet implemented!")
                """self.compute_cluster_crossings_hard(
                            times,
                            sorted_preds,
                            min_crossing_duration=min_crossing_duration,
                            max_crossing_duration=max_crossing_duration
                                                    )"""
            
            if self._clustering_type() == ClusterAnalyzer.SOFT_CLSTR_STR:
                
                crossing_table_list = utils.compute_crossings_soft(
                        times,
                        sorted_preds,
                        min_crossing_duration = min_crossing_duration,
                        max_crossing_duration = max_crossing_duration,
                        min_prob = min_prob,
                        min_cluster_frac = min_cluster_frac,
                        min_beyond_crossing_duration = \
                                    min_beyond_crossing_duration,
                        max_beyond_crossing_duration = \
                                    max_beyond_crossing_duration,
                        overlap_preference = overlap_preference
                                                                    )
              
                
            ## combine crossing_tables into list of dfs assigned to
            ## dict with keys being leaving and arriving cluster ints
            ## (e.g. (0,1) for leaving cluster 0, arriving at cluster 1)
            for i in range(len(crossing_table_list)):
                crossing_table = crossing_table_list[i]
                
                # Get single crossing data from subdf via crossing inds
                inds = crossing_table[:,0]
                a_crossing_in_subdf = subdf.iloc[inds,:].copy(deep=True)
                a_crossing_in_subdf.reset_index(drop=True, inplace=True)
                
                # Save crossing data about which crossing is being
                # processed and what type of crossing pts
                a_crossing_in_subdf["crossing_points"] = crossing_table[:,1]
                a_crossing_in_subdf["crossing_num"] = \
                        i + total_saved_crossings
                
                # Save data related to cluster probs
                if self._clustering_type() == ClusterAnalyzer.SOFT_CLSTR_STR:
                    for cluster_name in self.cluster_names:
                        cluster_int = self.cluster_names[cluster_name]
                        a_crossing_in_subdf[cluster_name] = \
                                sorted_preds[inds,cluster_int]
                if self._clustering_type() == ClusterAnalyzer.HARD_CLSTR_STR:
                    print(" not implemented yet!")
                    
                # Determine the cluster that's being left and the one
                # that's arriving
                in_cross_inds = np.where(crossing_table[:,1] == 2)[0]
                earliest_ind = crossing_table[:,0][
                                        np.min(in_cross_inds)
                                                  ]
                leaving_cluster_int = np.argmax( sorted_preds[earliest_ind,:] )
                leaving_cluster_name = \
                        self._cluster_int2name(leaving_cluster_int)
                latest_ind = crossing_table[:,0][
                                        np.max(in_cross_inds)
                                                  ]
                incoming_cluster_int = np.argmax( sorted_preds[latest_ind,:] )
                incoming_cluster_name = \
                        self._cluster_int2name(incoming_cluster_int)
                
                # Knowing what the leaving and incoming clusters are,
                # save the crossing_df into the right list in the dict
                crossing_dict[
                    (leaving_cluster_name, incoming_cluster_name)
                             ].append( 
                                 a_crossing_in_subdf
                                     )
                
                                 
            ## increment total number of saved crossings
            total_saved_crossings += len(crossing_table_list)

        
        
        #### If order matters, then return crossings dict as it
        #### (where going from cluster "a" to "b" is distinct from going
        #### from cluster "b" to "a")
        if order_matters:
            for key in crossing_dict:
                if len(crossing_dict[key]) > 0:
                    crossing_dict[key] = pd.concat( crossing_dict[key] )
        
        
        
        #### Otherwise, combine crossing datasets so that they're
        #### time agnostic (e.g. going from 0 to 1 or 1 to 0 just
        #### becomes a 0/1 crossing)
        else:
            
            
            ## Convert cluster ints to cluster names and get all combinations
            ## of those name of lenth 2
            cluster_names_combinations = list(itertools.combinations(
                                    list(self.cluster_names.keys()),
                                    2
                                                                    ))
            # Also make sure to add names for transition from one cluster
            # to itself (e.g. dip in c0 probability below min_prob)
            cluster_names_combinations.extend(
                [ (c_name, c_name) \
                  for c_name in list(self.cluster_names.keys()) ]
                                             )
            
            
            ## Iterate over each cluster name combination and amalgamate
            ## the crossings from both ordering into the same list
            unordered_crossing_dict = {}
            for crossing_name_tuple in cluster_names_combinations:
                combined_crossings = []
                
                # grab crossing for original ordering of tuple (e.g. ("a","b"))
                # (and delete original list to save space)
                combined_crossings.extend( 
                    crossing_dict[crossing_name_tuple]
                                         )
                del crossing_dict[crossing_name_tuple]
                
                # grab crossings from *reversed* ordering (e.g. ("b","a"))
                # (and delete original to save space)
                # but skip if leaving and arrival cluster are the same!
                if crossing_name_tuple[0] != crossing_name_tuple[1]:
                    combined_crossings.extend(
                        crossing_dict[ crossing_name_tuple[::-1] ]
                                            )
                    del crossing_dict[ crossing_name_tuple[::-1] ]
                
                # Sort w/r/t for earliest pt in each df in combined_crossings,
                # and then stack into vertically stacked df
                sort_inds = np.argsort(
                        [ df[time_var][0] for df in combined_crossings ]
                                        )
                if len(combined_crossings) > 0:
                    unordered_crossing_dict[crossing_name_tuple] = \
                        pd.concat(
                            [ combined_crossings[i] for i in sort_inds ]
                                 )
                else:
                    unordered_crossing_dict[crossing_name_tuple] = []
                    
            ## Change ref back to crossing_dict for same variable name
            crossing_dict = unordered_crossing_dict
            
            
            
        #### Last, check over any values in crossing_dict for empty lists.
        #### Replace the lists with empty dataframes possessing the same
        #### columns as other dfs in crossing_dict
        keys_of_empty_lists = [ key for key in crossing_dict \
                                if isinstance(crossing_dict[key],list) ]
        df_type = pd.core.frame.DataFrame
        keys_of_dfs = [ key for key in crossing_dict \
                        if isinstance(crossing_dict[key],df_type) ]
        if len(keys_of_empty_lists) > 0:
            ## will fail if empty
            cols_used = list(crossing_dict[ keys_of_dfs[0] ])
            for empty_list_key in keys_of_empty_lists:
                crossing_dict[empty_list_key] = pd.DataFrame(columns=cols_used)
        ## return empty dict if none found (work more on this later!!!)
        else:
            return {}
                
                
                
        #### Make output folder to save crossings if specified
        if save_crossings is not None:
            out_fldr_path = self._make_rel_subdir("crossing_csvs")
            ## Save each crossing dataframe with (cluster1)_(cluster2)
            ## filename structure
            for key in crossing_dict:
                file_path = os.path.join(out_fldr_path,
                                         '_'.join(key) + '.csv')
                crossing_dict[key].to_csv(file_path, index=False)
        
        
        
        return crossing_dict
                
    
            
    
    
    
    
    
            

    def check_accuracy_gmm(self, trans_data   = None,
                                 gmm          = None,
                                 total_bins   = None,
                                 cluster_bins = None,
                                 figsize      = None,
                                 hist_logx    = None,
                                 hist_logy    = None,
                                 chi2_pval    = None,
                                 prob_stat    = None):
    
    
        """
        
        For clustering via gmm, make 2 plots per cluster:
            1) Hist of the Mahalanobis distances per gaussian
            2) The qq plot of the chi-square for each gaussian
    
    
        NOTE: prob can vary significantly across small changes in mahal dist
              (b/c we're compressing m-dimensional data into 1 - so many
               opportunities for ...), so it's recommended to use a large
              number of bins to resolve the probabilities well, e.g. >= 500
              
              
        PARAMETERS
        ----------
        trans_data: ...
            ...
            
        gmm: sckit-learn Gaussian Mixture Model instance
            Scikit-Learn GMM that has already been fitted
            
            
        
        RETURNS
        -------
        None
        
        """
        
        
        # Need to provide training data and scikit-learn mixture model
        # instance to run!
        if ((trans_data is None) or (gmm is None)):
            raise ValueError(
                "Both \"training_data\" and \"gmm\" default args must be set!"
                            )
            
        # convert trans data from pandas df to numpy arr if given the former
        if isinstance(trans_data, pd.core.frame.DataFrame):
            trans_data = trans_data.values
        
        # Set typical default args
        if total_bins is None: total_bins = ClusterAnalyzer.BINS_1D
        if cluster_bins is None: cluster_bins = ClusterAnalyzer.BINS_1D
        if figsize is None: figsize = ClusterAnalyzer.MULTIPLOT_SIZE
        if hist_logx is None: hist_logx = False
        if hist_logy is None: hist_logy = False
        if prob_stat is None: prob_stat = "mean"
        if isinstance(prob_stat,str): prob_stat = [ prob_stat ]
        
        
        # setup colors for prob stat (have to be dark to be visible)
        prob_stat_colors = ["cornflowerblue", "forestgreen",
                            "lime", "darkviolet"]
        
    
        # Make output hist2d folder
        gmm_path = self._make_rel_subdir("GMM_Accuracy")

        
        # Make dict with inds of outliers
        outlier_dict = {}

        
        #### Make figure with mahal dist / prob hist and qq chi-square
        possible_clusters = np.arange(self.n_clusters)
        for i in range(len(possible_clusters)):
            cluster = possible_clusters[i]
            
            
            # Make fig and axes for subplots
            num_plot_rows, num_plot_cols = 2, 2
            fig, axes = plt.subplots(num_plot_rows, num_plot_cols,
                                     figsize=figsize)
            axes_1d = axes.reshape(num_plot_rows * num_plot_cols)
            mahal_hist_ax, prob_hist_ax, qq_ax, chi2_hist = axes_1d
            
            
            
            ### NOTE:
            ### Making 3 plots (denoted with headers / footers as integers)
            ### 1 is mahalanobis distance histogram
            ### 2 is prob scatter plot with avg prob plotted using scipy's
            ###    binned_statistic with bins from plot 1 - this is plotted
            ###    on the same axis as 1 with same x-axis but different y-axis
            ### 3 is quantile-quantile (qq) chi-square plot of distances
            ###    with m degrees of freedom
            
            
            
            # 1111111111 Make mahalanobis dist histogram 1111111111
            
            ### Compute mahalanobis distances given gmm params for cluster
            # Get mean
            mean = gmm.means_[i]
            # Get cov matrix by checking for covariance type; if not full,
            # then components will have shared covariance matrix (only one!)
            if gmm.covariance_type != 'full':
                covmat = gmm.covariances_
            else:
                covmat = gmm.covariances_[i]
            # Compute mahal distances given mean and cov matrix
            mahal_dist = self._calc_mahal_dist(trans_data,
                                               mean,
                                               covmat)
            # Convert distances to log scale if specified
            if hist_logx: mahal_dist = np.log10(mahal_dist)
            
            
            ### Manually compute bin edges for histogram
            ### This way, can share same bins between hist and prob plotting
            total_bin_edges = None
            # If int, create bin edges based on min / max val and # of bins
            if isinstance(total_bins, int):
                min_val, max_val = np.min(mahal_dist), np.max(mahal_dist)
                # n+1 edges for n bins
                max_val_for_hist = ((max_val - min_val) / total_bins) + max_val
                total_bin_edges = np.linspace(min_val, max_val_for_hist,
                                              num=total_bins)
            # otherwise, take bin_edges as given
            else:
                total_bin_edges = total_bins
            
            
            ### Make histogram 
            mahal_hist_ax.hist(mahal_dist,
                               bins=total_bin_edges,
                               log=hist_logy,
                               color="red",
                               alpha=0.5)
            
            
            ### Make sup title for hist_ax
            # Get str of integer number of bins
            strbins = None
            if isinstance(total_bins, int):
                strbins = str(total_bins)
            else:
                strbins = str(len(total_bins))
            subtitle = ('Mahalanobis Dist Hist (rel to cluster) over ALL data | '
                        + strbins + ' bins')
            mahal_hist_ax.set_title(subtitle)
            
            # 11111111111111111111111111111111111111111111111111111
            
            
            
            # 2222222222 Create scatter plot of avg prob 2222222222
            
            ### Get mean bin value between bin edges
            # get mean bin value between bin edges
            #mean_bin_vals = np.array(
            #    [ (stat_results.bin_edges[i] + stat_results.bin_edges[i+1])/2
            #      for i in range(stat_results.bin_edges.shape[0]-1) ]
            #                         )
            mean_bin_vals = np.array(
                [ (total_bin_edges[i] + total_bin_edges[i+1])/2
                  for i in range(total_bin_edges.shape[0]-1) ]
                                    )
            
            
            ### make scatter plot x-axis same as hist x-axis
            prob_ax = mahal_hist_ax.twinx()
            
            
            ### Retreive all probs for cluster
            in_cluster_probs = self._get_data(cluster=cluster)
            
            
            ### plot binned stat for probs for each stat given
            for c in range(len(prob_stat)):
                # Use scipy binned stat to make histogram over label data AND
                # get average probs of data within those bins
                stat_results = scipy.stats.binned_statistic(
                                                mahal_dist,
                                                in_cluster_probs,
                                                statistic=prob_stat[c],
                                                bins=total_bin_edges
                                                            )
                # plots avg prob
                scat_color = prob_stat_colors[c] if c < len(prob_stat_colors) \
                             else None
                prob_ax.scatter(mean_bin_vals,
                                stat_results.statistic,
                                s=4.0,
                                label=prob_stat[c],
                                c=scat_color)
            
            
            ### Enforce same y scale for all prob plots
            prob_ax.set_ylim([-0.05,1.05])
            
            
            ### Setup legend
            prob_ax.legend(loc="upper right",
                           markerscale=1.75,
                           framealpha=0.5)
            
            
            ### Make special gridlines for PROB Y-AXIS
            # Major gridlines
            prob_ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
            prob_ax.grid(True, which="major",
                         **{'linestyle':'dashed',
                            'color':'black',
                            'alpha':0.7})
            # Minor gridlines
            prob_ax.set_yticks([0.1,0.3,0.5,0.7,0.9], minor=True)
            prob_ax.grid(True, which="minor",
                         **{'linestyle':'dotted',
                            'color':'black',
                            'alpha':0.7,
                            'linewidth':1})
        
            # 22222222222222222222222222222222222222222222222222222
            
            
            
            # 3333333333 Make qqplot for chi-square 3333333333
            
            ### recompute mahalanobis distances for data IN CLUSTER
            ### This is determined by whatever cluster owns the max prob
            ### for that point
            # Predict on data s.t. we get cluster labels back
            data_predictions = gmm.predict(trans_data)
            # Determine points in cluster
            cluster_inds = np.where(data_predictions == cluster)[0]
            # Recompute mahal dist
            mahal_dist_in_cluster = self._calc_mahal_dist(
                                        trans_data[cluster_inds,:],
                                        mean,
                                        covmat
                                                         )
            
            
            ### Create probplot
            chi2_dof = trans_data.shape[1]
            scipy.stats.probplot(mahal_dist_in_cluster**2,
                                 dist=scipy.stats.chi2(df=chi2_dof),
                                 plot=qq_ax)
            
            
            ### set title for plot
            qq_title = ('QQ-Plot for MD2 vs Chi2 with dof = ' + str(chi2_dof))
            qq_ax.set_title(qq_title)
            
            # 333333333333333333333333333333333333333333333333
            
            
            
            # 4444444444 Make hist for probs 44444444
            
            # Make hist of probs
            prob_hist_ax.hist(in_cluster_probs,
                              bins = cluster_bins,
                              log = True)
            
            # 444444444444444444444444444444444444444
            
            
            
            # 5555555555 Make chi-sq HIST with in-cluster hist overlaid 5555555555
            
            ### Sample a chi sqaure dist with num samples == num pts in cluster
            n = cluster_inds.shape[0]
            chi2_rands = scipy.stats.chi2.rvs(df=chi2_dof, size=n)
            
            
            ### Compute chi2 comparison only if cluster is NON-empty
            if n == 0:
                outlier_dict[cluster] = np.array([], dtype=np.int32)
                
            else:
            
                ### determine bins
                min_val = np.min(
                    [chi2_rands.min(), (mahal_dist_in_cluster**2).min()]
                                )
                max_val = np.max(
                    [chi2_rands.max(), (mahal_dist_in_cluster**2).max()]
                                )
                max_val_for_chi2_hist = ((max_val - min_val) / cluster_bins) + max_val
                chi2_bin_edges = np.linspace(min_val, max_val_for_chi2_hist,
                                             num=cluster_bins)
                
                
                ### THIS IS A CHEAP HACK - FIX LATER
                chi2_hist.hist(chi2_rands,
                               bins = chi2_bin_edges,
                               log = True,
                               color = 'red',
                               alpha = 0.5)
                
                
                ### add hist of mahal_dist^2
                chi2_hist.hist(mahal_dist_in_cluster**2,
                               bins = chi2_bin_edges,
                               log = True,
                               color = "purple",
                               alpha = 0.5)
                
                
                ### Make sub title for plot
                #subtitle = ("Chi2 / In-cluster MD2 hists | bins "
                #            + str(cluster_bins) )
                subtitle = ("Chi2 / In-cluster MD2 hists ")
                # If specified, indicate how much data has significance
                # level exceeded by pvalue
                if chi2_pval is not None:
                    
                    ## Determine max chi2 val to check data against
                    # If chi2_pval > 0, then compute p value the typical way
                    max_chi2_val = None
                    chi2_val_substr = None
                    if chi2_pval > 0:
                        # Get max x val where pval of chi2 data has accumulated prior
                        max_chi2_val = scipy.stats.chi2.ppf(1-chi2_pval,df=chi2_dof)
                        chi2_val_substr = ("p-value "
                                           + "{:.2f}".format(100 * chi2_pval) + "%")
                    # If chi2_pval == 0 EXACTLY, then compute max chi2 val
                    # generated from chi2 rands
                    elif chi2_pval == 0:
                        max_chi2_val = np.max(chi2_rands)
                        chi2_val_substr = "max(chi2_samples)"
                    else:
                        raise ValueError("Unrecognized value for chi2_pal")
                        
                    ## Find data exceeding max_chi2_val
                    # track inds with MD^2 beyond this x val
                    inds_beyond_max = np.where(
                            mahal_dist_in_cluster**2 >= max_chi2_val
                                              )[0]
                    # save frac of pts inds rel to total data size
                    frac_beyond_max = "{:.3f}".format(
                            100 * inds_beyond_max.shape[0] / n
                                                     )
                    # save full subtitle
                    subtitle = (subtitle + " | " + frac_beyond_max
                                + "% of " + self._num2str(n) + " >= "
                                + chi2_val_substr)
                    
                    ## Show vertical line of chi2 critical value corresponding
                    ## to p value
                    chi2_hist.axvline(x=max_chi2_val)
                    
                    ## Save inds that are beyond max_chi2_val as outliers
                    ## (inds relative to *original* df)
                    outlier_dict[cluster] = \
                        np.arange(self.df.shape[0])[ cluster_inds ][ inds_beyond_max ]

                ## Add subtitle to plot
                chi2_hist.set_title(subtitle)
            
            # 555555555555555555555555555555555555555555555555555555
            
            
            
            ### Make suptitle for cluster figure
            cluster_name = self._cluster_int2name(cluster)
            suptitle = cluster_name + ' cluster'
            fig.suptitle(suptitle,fontsize=16)
            fig.tight_layout()
            
            
            
            #### Save cluster histogram
            cluster_figname = cluster_name + "_cluster" + self.IMAGE_TYPE
            image_path = os.path.join(gmm_path, cluster_figname)
            plt.savefig(image_path)
            plt.close()
            
            
            
        #### Return dict of outliers inds per cluster if given pval
        if chi2_pval is not None:
            return outlier_dict
    
    
    
    
    
    
    
    
    
    
                
        
        
