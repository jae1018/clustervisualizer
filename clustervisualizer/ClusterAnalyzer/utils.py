#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 12:36:10 2022

@author: jedmond
"""

import numpy as np
import pandas as pd
import decorator




@decorator.decorator
def confirm_kwargs_not_none(func, *args, **kwargs):
    """
    Function to be used as decorator to ensure that named arguments (kwargs)
    are not None.
    """
    if None in [ kwargs[key] for key in kwargs ]:
        raise ValueError("All named arguments must be non-None")
    return func(*args, **kwargs)















class CrossingTable:
    
    
    """
    Small class of constants used to index columns of crossing table
    
    Constructs the Nx2 array of results expected from the
    compute_crossings_soft() function.
    
    (more descrip here about x - y inds)
    
    If any kwargs are none, this function will raise an error!!!

    
    
    DIAGRAM
    -------
    prob-axis
       || * *   * * * * * * * * |       | o o   o o o o o o | 
       || |   *     |           *       o     o   |         o 
       ||-|---------|-----------|-------|---------|---------|- min_prob
       || |         |           | * * o |         |         |
       || |         |           | o o * |         |         |
       || |   o     |           o       *     *   |         * 
       || o o   o o o o o o o o |       | * *   * * * * * * | 
     ==||=|=========|===========|=======|=========|=========|= time-axis
       || X         |           A       B         |         Y
       ||           Q                             W
           * - cluster 1 probability
           o - cluster 2 probability
    
    
    PARAMETERS
    ----------
    start_ind / end_ind: int
        The starting / ending indices for the first column of the array
        (e.g. start_ind / end_ind = 5, 10 => [5, 6, 7, 8, 9, 10])
        
    All (?)_ind params: int
        The indices of the x, q, a, b, w, y values in the diagram above
    
    
    RETURNS
    -------
    N x 2 numpy array where N is the number of pts in any crossings
    
    """

    
    # Constants for accessing cols of a table
    INDS_COL = 0
    CROSSING_PT_COL = 1
    
    # Constants indicating pt-type in a table
    CROSSING_PT_XY = 0
    CROSSING_PT_QW = 1
    CROSSING_PT_AB = 2
    
    
    @confirm_kwargs_not_none
    def __init__(self, *, x_ind, y_ind, q_ind, w_ind, a_ind, b_ind):
        
        ## Save indices as class members
        self.x_ind = x_ind
        self.y_ind = y_ind
        self.q_ind = q_ind
        self.w_ind = w_ind
        self.a_ind = a_ind
        self.b_ind = b_ind
        
        ## Build the individual arrays
        inds_arr = np.arange(self.x_ind, self.y_ind+1)
        crossing_pt_types = np.full(inds_arr.shape[0],
                                    CrossingTable.CROSSING_PT_XY)
        crossing_pt_types[ np.arange(self._rel_ind(q_ind),
                                     self._rel_ind(w_ind+1)) ] = \
                CrossingTable.CROSSING_PT_QW
        #crossing_pt_types[ np.arange(q_ind-x_ind, w_ind+1-x_ind) ] = 1
        crossing_pt_types[ np.arange(self._rel_ind(a_ind),
                                     self._rel_ind(b_ind+1)) ] = \
                CrossingTable.CROSSING_PT_AB
        #crossing_pt_types[ np.arange(a_ind-x_ind, b_ind+1-x_ind) ] = 2
        
        ## Then combine into N x 2 array
        self.crossing_table = np.vstack( [inds_arr, crossing_pt_types] ).T
    
    
    def _rel_ind(self, ind):
        """
        Used to compute the relative index of the given ind relative to the
        x ind
        """
        return ind - self.x_ind
        
        
    def geq_min_crossing_duration(self, times, min_crossing_duration):
        """
        Checks that the crossing duration (the (A,B) interval is >=
        the min_crossing_duration given)
        """
        return times[self.b_ind] - times[self.a_ind] >= min_crossing_duration
    
    
    def leq_max_crossing_duration(self, times, max_crossing_duration):
        """
        Checks that the crossing duration (the (A,B) interval is <=
        the max_crossing_duration given)
        """
        return times[self.b_ind] - times[self.a_ind] <= max_crossing_duration
        
    
    def has_overlapping_qw_interval(self, times, crossing_table_inst):
        """
        Checks if the (Q,W) interval overlaps with the (Q,W) interval of
        the CrossingTable object given)
        """
        # if crossing_table_inst is None, then just return False
        if crossing_table_inst is None: return False
        # get qw bounds for current object
        current_qw_bounds = self.get_qw_interval(times)
        current_q_time = current_qw_bounds[0]
        current_w_time = current_qw_bounds[1]
        # get qw bounds for given object
        next_qw_bounds = crossing_table_inst.get_qw_interval(times)
        next_q_time = next_qw_bounds[0]
        next_w_time = next_qw_bounds[1]
        if current_q_time < next_q_time:
            return next_q_time < current_w_time
        elif current_q_time < next_q_time:
            return True
        else:
            return next_w_time > current_q_time

    
    def get_qw_interval(self, times):
        """
        Returns a 2-element pair of times that are the times corresponding
        to the (Q,W) interval bounds
        """
        return (times[self.q_ind], times[self.w_ind])
    
    
    def get_earliest_time(self, times):
        """
        Returns the earliest times from the crossing table, which corresponds
        to the time a x_ind
        """
        return times[self.x_ind]
    
    
    def get_qa_inds(self):
        """
        Return inds from q_ind to a_ind of crossing table
        """
        return np.arange(self.q_ind, self.a_ind+1)
    
    
    def get_bw_inds(self):
        """
        Return inds from b_ind to w_ind of crossing table
        """
        return np.arange(self.b_ind, self.w_ind+1)
    
    
    def get_in_cluster_frac_for_qa(self, preds):
        """
        Compute fraction of pts in (Q,A) interval whose dominant cluster
        match the dominant cluster at A pt
        """
        qa_inds = self.get_qa_inds()
        dom_clusters_over_qa = np.argmax( preds[qa_inds,:], axis=1 )
        num_in_cluster = np.sum(
                dom_clusters_over_qa[:-1] == dom_clusters_over_qa[-1]
                                )
        return num_in_cluster / (dom_clusters_over_qa.shape[0] - 1)
    
    
    def get_in_cluster_frac_for_bw(self, preds):
        """
        Compute fraction of pts in (Q,A) interval whose dominant cluster
        match the dominant cluster at A pt
        """
        bw_inds = self.get_bw_inds()
        dom_clusters_over_bw = np.argmax( preds[bw_inds,:], axis=1 )
        num_in_cluster = np.sum(
                dom_clusters_over_bw[1:] == dom_clusters_over_bw[0]
                                )
        return num_in_cluster / (dom_clusters_over_bw.shape[0] - 1)
    
    
    def departure_cluster(self, preds):
        """
        Determine what cluster is the crossing is leaving
        """
        return np.argmax( preds[self.a_ind,:], axis=1 )
    
    
    def arrival_cluster(self, preds):
        """
        Determine what cluster is the crossing is leaving
        """
        return np.argmax( preds[self.b_ind,:], axis=1 )
    
    
    def get_array(self):
        return self.crossing_table
    
    
    def __str__(self):
        return self.get_array().__str__()
    
    
    def __repr__(self):
        return self.__str__()

    






## break crossings into 3 types:
##   instantaneous
##     (no need to check for min / max crossing duration)
##   gradual different
##     only need to check for max_crossing_duration, not min!
##   gradual same
##     need to check for BOTH min / max crossing duration!











def find_possible_crossing_intervals_soft(preds, min_prob):

    """
    
    Finds soft crossings that are saved as Nx2 array where N is the number
    of possible crossings. Each row is a possible crossing and has two
    integers, the first being the row index of the start of the crossing
    and the second the index of the end of the crossing.
    
    These indices work in the style of numpy indexing, so a start / end
    ind of 13 and 14 is indexed like arr[13:14].
    
    
    PARAMETERS
    ----------
    preds: Nxk numpy array of probabilities (each row sums to 1)
        N is the number of data points, k is the number of clusters
    min_prob: float b/w 0 and 1 
    
    
    RETURNS:
    --------
    Nx2 numpy array of starting and ending indices of crossings in preds
    
    
    CROSSING DETAILS
    ----------------
    Finds three forms of crossings:
    (for below, let min_prob be ~0.8, '*' be cluster 1 probability, and
     'o' be cluster 2 probability)
    
    INSTANTANEOUS (there IS NO GAP in high-probability b/w a change
                   in starting and ending clusters)
        # prob-axis
        #  | * * *   * * * | o o o o   o o
        #  |       *     | o         o
        #  | ------------|-|-------------- min_prob
        #  |             | |
        #  |             | |
        #  |       o     | *         *
        #  | o o o   o o o | * * * *   * *
        #--|-------------|-|------------------ (x-axis)
    
    
    GRADUAL DIFFERENT (there IS A GAP in high-probability and a change
                       b/w starting and ending clusters)
        # prob-axis
        #  | * * * |       | o o o o   o o
        #  |       *       o         o
        #  | ------|-------|---------------- min_prob
        #  |       | * * o |
        #  |       | o o * |
        #  |       o       *         *
        #  | o o o |       | * * * *   * *
        #--|---------------------------------- (x-axis)
        #          A       B
        
        
    GRADUAL SAME (there IS A GAP in high-probability and NO change
                  b/w starting and ending clusters)
        # prob-axis
        #  | * * * |           |   *   * *
        #  |       *           * *   *
        #  | ------|-----------|------------- min_prob
        #  |       | * * o * * |
        #  |       | o o * o o |
        #  |       o           o o   o  
        #  | o o o |           |   o   o o
        #--|---------------------------------- (x-axis)
        #          A           B
    
    """
        
    
    ## Find all crossings from one cluster to a different one that happen
    ## ABOVE the min_prob threshold
    # Get all inds with single-cluster probability >= min_prob ...
    high_prob_inds = np.where( np.max(preds, axis=1) >= min_prob )[0]
    # ... of those inds, determine which cluster generated that probability ...
    max_cluster_rel_high_prob_inds = np.argmax(preds[high_prob_inds], axis=1)
    # ... and track when there's a change in the dominant cluster
    crossing_start_inds_rel_high_prob_inds = \
            np.where( np.diff(max_cluster_rel_high_prob_inds) != 0 )[0]
    # Determine the starting and ending indices of the crossings
    cluster_change_crossing_start_inds = \
            high_prob_inds[crossing_start_inds_rel_high_prob_inds]
    cluster_change_crossing_end_inds = \
            high_prob_inds[crossing_start_inds_rel_high_prob_inds+1]
    
    
    ## Find all instantaneous crossings as a subset of the crossings
    ## found above
    # First, find all starting *and* ending indices where there is a
    # change in the dominant cluster
    max_cluster_change_start_inds = \
            np.where( np.diff( np.argmax(preds, axis=1) ) )[0]
    # Then express these inds as a 2d arr of ind pairs where the 0th column
    # is the starting possible instantaneous crossing index and the 1st col
    # is the ending index
    max_cluster_change_ind_pairs = np.vstack(
                [max_cluster_change_start_inds,
                 max_cluster_change_start_inds+1]
                                            ).T
    # Next use np.isin to see if any of those indices are among the
    # high prob indices found earlier - do this for BOTH starting and ending
    # indices - and then use np.where and np.all to find the pairs
    # where both starting and ending indices were among them high prob inds
    viable_instant_crossing_ind_pairs = np.where(
                np.all(
                    np.isin(max_cluster_change_ind_pairs, high_prob_inds),
                    axis=1
                      )
                                                )[0]
    # Then save the subset of indices that were found to be legitimate
    # instantaneous crossing pairs
    instant_crossing_inds = \
            max_cluster_change_ind_pairs[viable_instant_crossing_ind_pairs, :]
 
    
    ## Find all gradual-different crossings as the complementary subset
    gradual_diff_crossing_start_inds = \
            np.setdiff1d(cluster_change_crossing_start_inds,
                         instant_crossing_inds[:,0])
    gradual_diff_crossing_end_inds = \
            np.setdiff1d(cluster_change_crossing_end_inds,
                         instant_crossing_inds[:,1])
    gradual_diff_crossing_inds = np.vstack([gradual_diff_crossing_start_inds,
                                            gradual_diff_crossing_end_inds]).T
    
    
    
    ## Find all gradual-same crossings as a subset
    # Get all inds with a gap in the probability (e.g. a non-consecutive
    # (A,B) interval seen in the gradual-different and gradual-same diagrams) ...
    inds_before_high_prob_gap_rel_high_prob_inds = \
            np.where( np.diff(high_prob_inds) > 1 )[0]
    inds_before_high_prob_gap = \
            high_prob_inds[inds_before_high_prob_gap_rel_high_prob_inds]
    # ... and set-subtract from them the inds already found earlier
    gradual_same_crossing_start_inds = \
            np.setdiff1d(inds_before_high_prob_gap,
                         cluster_change_crossing_start_inds)
    # do the same as above to find the ending indices
    inds_after_high_prob_gap = \
            high_prob_inds[inds_before_high_prob_gap_rel_high_prob_inds+1]
    gradual_same_crossing_end_inds = \
            np.setdiff1d(inds_after_high_prob_gap,
                         cluster_change_crossing_end_inds)
    gradual_same_crossing_inds = np.vstack([gradual_same_crossing_start_inds,
                                            gradual_same_crossing_end_inds]).T


    return [instant_crossing_inds,
            gradual_diff_crossing_inds, 
            gradual_same_crossing_inds]
    
    










def build_crossing_table_list(ind_pairs,
                              times,
                              *,
                              min_crossing_duration,
                              max_crossing_duration,
                              min_beyond_crossing_duration,
                              max_beyond_crossing_duration):
    
    """
    Given a N x 2 numpy arr of ind pairs, build list of crossing table
    objects
    """
    
    
    crossing_tables = []
    for i in range(ind_pairs.shape[0]):
        
        ## Get currently considered crossing interval
        an_ind_pair = ind_pairs[i,:]
        
        ## Get starting and ending crossing interval indices
        a_ind = an_ind_pair[0]
        b_ind = an_ind_pair[1]
        interval_start_time = times[a_ind]
        interval_end_time = times[b_ind]
           
        
        ## Find Q-W indices
        # Find index of Q for current interval
        q_ind = np.searchsorted(
                        times,
                        interval_start_time - min_beyond_crossing_duration
                                )
        # Find index of W for current interval
        w_ind = np.searchsorted(
                        times,
                        interval_end_time + min_beyond_crossing_duration,
                        #side="right"
                                )
        # decrement w_ind if assigned to last possible index via searchsorted
        if (w_ind == times.shape[0]):
            w_ind = w_ind - 1
        # if w_ind set to time beyond Q,W interval, then decrement ind
        if times[w_ind] > interval_end_time + min_beyond_crossing_duration:
            w_ind = w_ind - 1
       
        
        ## Find X-Y indices
        # find *potential* index of X for current interval
        x_ind = np.searchsorted(
                        times,
                        interval_start_time - max_beyond_crossing_duration
                                )
        # find *potential* index of Y for current interval
        y_ind = np.searchsorted(
                        times,
                        interval_end_time + max_beyond_crossing_duration#,
                        #side="right"
                                )
        # decrement y_ind if assigned to last possible index via searchsorted
        if (y_ind == times.shape[0]):
            y_ind = y_ind - 1
        # if y_ind set to time beyond X,Y interval, then decrement ind
        if times[y_ind] > interval_end_time + max_beyond_crossing_duration:
            y_ind = y_ind - 1
        
        
        ## Create crossing table saving original indices, the crossing number,
        ## and the inds of special pts in the crossing
        crossing_tables.append(
            CrossingTable(x_ind=x_ind, y_ind=y_ind,
                          q_ind=q_ind, w_ind=w_ind,
                          a_ind=a_ind, b_ind=b_ind)
                              )
    
    
    return crossing_tables













@confirm_kwargs_not_none
def compute_crossings_soft(times,
                           preds,
                           *,
                           min_prob,
                           min_crossing_duration,
                           max_crossing_duration,
                           min_cluster_frac,
                           min_beyond_crossing_duration,
                           max_beyond_crossing_duration,
                           overlap_preference):
    
    """
    
    
    Note: The * in the args means that the following arguments are named
          (e.g. min_prob is passed using min_prob = val)
          
    This function will fail if the name params are not set!
    
    
    
   prob-axis
      ||  * *   * * * * * * * * |       | o o   o o o o o o | o
      ||  |   *     |           *       o     o   |         o
      ||--|---------|-----------|-------|---------|---------|---- min_prob
      ||  |         |           | * * o |         |         |
      ||  |         |           | o o * |         |         |
      ||  |   o     |           o       *     *   |         *
      ||  o o   o o o o o o o o |       | * *   * * * * * * | *
    ==||==|=========|===========|=======|=========|=========|==== time-axis
      ||  X         |           A       B         |         Y
      ||            Q                             W
          * - cluster 1 probability
          o - cluster 2 probability
          
     max_crossing_duration (max) means that length(A,B) <= max
     
     min_beyond_crossing_duration (min) means that Q, A, B, and W exist such
     that both length(Q,A) and length(W,B) >= min_beyond_crossing_duration
     
     max_beyond_crossing_duration (max_cc) means that data will be saved at
     times X to Y (X <= Q, Y >= W, and length(X,Y) <= max_cc) if this
     interval is found to be a crossing.
    
    
    
    REQUIRE 3 THINGS:
        1) length(A,B) <= max_crossing_duration
        2) length (A,B) >= min_crossing_duration
             (however, rule 2 only applies for )
        3) Interval (Q,W) doesn't overlap with the (Q,W) interval of an
           adjcent crossing (if it does, both are discarded)

             
             
    WHAT'S ALLOWED (IN FIGURES)
    ----------------------------
    
    Crossing intervals with space in between (A):
        
                  |--|                         |--|
            |-----|  |-----|             |-----|  |-----|
        |----------------------|     |----------------------|
            |-----|  |-----|             |-----|  |-----|
                  |--|                         |--|
        X   Q     A  B    W    Y     X'  Q'    A' B'    W'  Y'
    
    Crossing intervals with X-Y overlap (B):
    
                                        |--|
                                  |-----|  |-----|     
                              |----------------------|    
                              /   |-----|  |-----|             
                              /         |--|               
                              X'  Q'    A' B'    W'  Y'    
                  |--|        / 
            |-----|  |-----|  / 
        |----------------------|
            |-----|  |-----|   
                  |--|         
        X   Q     A  B    W    Y 
    
    Crossing intervals with X-W overlap (but no Q-W overlap) (C):
    
                                   |--|
                             |-----|  |-----|     
                         |----------------------|    
                         /   |-----|  |-----|             
                         /   +     |--|               
                         X'  Q'    A' B'    W'  Y'    
                  |--|   /   +
            |-----|  |-----| + 
        |----------------------|
            |-----|  |-----|   
                  |--|         
        X   Q     A  B    W    Y 
        
        
    
    WHAT'S **NOT** ALLOWED (IN FIGURES)
    ---------------------------------
        
    Crossing intervals with Q-W overlap (or any increasing overlap!)
                                |--|
                          |-----|  |-----|     
                      |----------------------|    
                      /   |-----|  |-----|             
                      /   +     |--|               
                      X'  Q'    A' B'    W'  Y'    
                  |--|/   +
            |-----|  |-----|
        |----------------------|
            |-----|  |-----|   
                  |--|         
        X   Q     A  B    W    Y 
        
        

    RETURNS
    -------
    Z x 2 numpy array where Z is the number of crossing-involved indices
    and 2 different columns
        col 1: indices of times / preds that are involved in a crossing
        col 2: integers indicating what pts of a crossing are relative to the
               figure above; pts in (Q,W) have value 1 exempting those in the
               (A,B) interval, pts in (A,B) have value 2, and all other pts
               in that crossing have value 0 (X and Y can be inferred as
               the first / last points of that crossing)
    
    """
     
    
    
    #### Convert times / preds to np array if given as list
    if isinstance(times, list): times = np.array(times)
    if isinstance(preds, list): preds = np.array(preds)
    
    
    
    #### Convert times to pandas datetime if it's not already
    times = pd.to_datetime(times)
    
    
    
    #### Ensure that times and pred_arr have same number of rows
    if times.shape[0] != preds.shape[0]:
        raise ValueError("Number of rows between times and preds do not match.")
    
    
    
    #### Get array of possible crossings
    instant_crossings, gradual_diff_crossings, gradual_same_crossings = \
            find_possible_crossing_intervals_soft(preds, min_prob)
    #possible_crossing_intervals = \
    
        
    ####
    # Before checking for overlap, we need to confirm that all possible
    # crossings found have acceptable crossing duration for that type
    # of crossing. This means that we'll first confirm that ...
    #   Instantaneous Crossings and Gradual-Different Crossings
    #     satisfy rule 1 (rule 2 is irrelevant)
    #   Gradual-Same Crossings satisfy both Rules 1 and 2
    # The crossings that pass the rules above are then checked to see if
    # they have (Q,W) overlap; if any do, the earliest one is selected
    # ((haven't implemented earliest q,w overlap selection yet!
    #  currently it's just removal of all crossings with overlap))
    ####
    

    
    #### Build crossing tables based on possible crossing intervals
    ## Build for instant crossings
    instant_crossings_tables = build_crossing_table_list(
            instant_crossings,
            times,
            min_crossing_duration = min_crossing_duration,
            max_crossing_duration = max_crossing_duration,
            min_beyond_crossing_duration = min_beyond_crossing_duration,
            max_beyond_crossing_duration = max_beyond_crossing_duration
                                                         )
    ## Build for gradual-different crossings
    gradual_diff_crossings_tables = build_crossing_table_list(
            gradual_diff_crossings,
            times,
            min_crossing_duration = min_crossing_duration,
            max_crossing_duration = max_crossing_duration,
            min_beyond_crossing_duration = min_beyond_crossing_duration,
            max_beyond_crossing_duration = max_beyond_crossing_duration
                                                             )
    ## Build for gradual-same crossings
    gradual_same_crossings_tables = build_crossing_table_list(
            gradual_same_crossings,
            times,
            min_crossing_duration = min_crossing_duration,
            max_crossing_duration = max_crossing_duration,
            min_beyond_crossing_duration = min_beyond_crossing_duration,
            max_beyond_crossing_duration = max_beyond_crossing_duration
                                                             )

    
    
    #### Checks min / max crossing duration rules for the crossings
    ## For Instant crossings, only need to worry about the max rule
    instant_crossings_tables = check_crossing_duration_rules(
            instant_crossings_tables,
            times,
            #min_crossing_duration = min_crossing_duration,
            max_crossing_duration = max_crossing_duration
                                                            )
    ## For Gradual-Different crossings, only need to worry about the max rule
    gradual_diff_crossings_tables = check_crossing_duration_rules(
            gradual_diff_crossings_tables,
            times,
            max_crossing_duration = max_crossing_duration
                                                            )
    ## For Gradual-Same crossings, need to check BOTH min / max rule
    gradual_same_crossings_tables = check_crossing_duration_rules(
            gradual_same_crossings_tables,
            times,
            min_crossing_duration = min_crossing_duration,
            max_crossing_duration = max_crossing_duration
                                                            )
    


    #### Combine all crossing tables into single list, sorted by earliest
    #### time in the tables
    # *** get rid of same crossings for now ...
    # *** need to come up with a way to choose other types of crossings
    # *** over gradual-same if they show up ... but until then, just toss em
    all_crossing_tables = [ *instant_crossings_tables,
                            *gradual_diff_crossings_tables]#,
                            #*gradual_same_crossings_tables ]
    crossing_tables_earliest_times = [ a_table.get_earliest_time(times) \
                                       for a_table in all_crossing_tables ]
    sort_inds = np.argsort(crossing_tables_earliest_times)
    all_crossing_tables = [ all_crossing_tables[i] \
                            for i in sort_inds ]
    
    
    
    #### Check consec_cluster_frac rules for each list of crossings
    all_crossing_tables = check_cluster_frac_rules(
            all_crossing_tables,
            preds,
            min_cluster_frac = min_cluster_frac
                                                    )
        
        
    
    #### With a single crossing_table_list, check for (Q,W) overlap
    good_crossing_tables = check_qw_overlap_for_crossing_table_list(
                all_crossing_tables,
                times,
                preds,
                min_beyond_crossing_duration = min_beyond_crossing_duration,
                max_beyond_crossing_duration = max_beyond_crossing_duration,
                overlap_preference = overlap_preference
                                                                    )
    


    #### Now convert from list of crossing tables to stacked N x 2 array
    if len(good_crossing_tables) > 0:
        return [ a_table.get_array() for a_table in good_crossing_tables ]
    else:
        return [ ]
    










@confirm_kwargs_not_none
def check_cluster_frac_rules(crossing_tables_list,
                             preds,
                             *,
                             min_cluster_frac):
    
    """
    
    Confirm that the crossings satisfy the min_cluster_frac rule
    
    """
    
    inds_to_keep = []
    for i in range(len(crossing_tables_list)):
        
        ## Get in-cluster fraction for the (Q,A) interval
        qa_in_cluster_frac = \
                crossing_tables_list[i].get_in_cluster_frac_for_qa(preds)
        
        ## Get in-cluster fraction for the (B,W) interval
        bw_in_cluster_frac = \
                crossing_tables_list[i].get_in_cluster_frac_for_bw(preds)
                
        ## If both fractions are >= min_cluster_frac, then keep the crossing
        #print("for table",crossing_tables_list[i],
        #      "\n got fracs:",qa_in_cluster_frac, bw_in_cluster_frac)
        if ((qa_in_cluster_frac >= min_cluster_frac) and
            (bw_in_cluster_frac >= min_cluster_frac)):
            inds_to_keep.append( i )
    
    return [ crossing_tables_list[i] for i in inds_to_keep ]
        
        















@confirm_kwargs_not_none
def check_qw_overlap_for_crossing_table_list(crossing_table_list,
                                             times,
                                             preds,
                                             *,
                                             min_beyond_crossing_duration,
                                             max_beyond_crossing_duration,
                                             overlap_preference):
    
    """
    
    Given (sorted!) crossing_table_list, iterate saved crossings that
    have no (Q,W) overlap. Crossings that do have such overlap are
    saved (or not) depending on the overlap_preference param
    
    Returns a list of crossing_tables that satisfy the rules
    """
    
    
    ## declare str constants for overlap checking
    EARLIEST_STR = "earliest"
    LATEST_STR = "latest"
    REMOVE_STR = "remove"
    BEST_STR = "best"
    allowed_overlap_prefs = [ EARLIEST_STR,
                              LATEST_STR,
                              REMOVE_STR,
                              BEST_STR ]
    
    
    ## confirm that overlap_preference kwarg matches list of options
    if overlap_preference not in allowed_overlap_prefs:
        raise ValueError("Given value for overlap_preference not among",
                         "accepted values:",allowed_overlap_prefs)
    
    
    
    
    
    def choose_from_overlapping_crossing_tables(crossing_table_list,
                                                qw_overlap_list,
                                                overlap_preference,
                                                preds):
        
        """
        given a crossing table list and a list of inds of crossing_tables
        in said list that have (Q,W) overlap, choose among them which
        crossing_table is desired
        The index in qw_overlap_list of the desired table is returned
        (e.g. 3 tables have overlap_list = [4,5,6] and 5 is best, so
        5 is returned)
        """
        
        ## These are already sorted, so just return last one for latest
        if overlap_preference == LATEST_STR:
            return qw_overlap_list[-1]
        
        if overlap_preference == EARLIEST_STR:
            return qw_overlap_list[0]
        
        if overlap_preference == REMOVE_STR:
            return None
        
        ## Return table with best SUM OF in_cluster fracs
        ## between (Q,A) and (B,W) intervals
        if overlap_preference == BEST_STR:
            
            frac_sums = []
            for crossing_table_ind in qw_overlap_list:
                crossing_table = crossing_table_list[crossing_table_ind]
                ## Get in-cluster fraction for the (Q,A) interval
                qa_in_cluster_frac = \
                        crossing_table.get_in_cluster_frac_for_qa(preds)
                ## Get in-cluster fraction for the (B,W) interval
                bw_in_cluster_frac = \
                        crossing_table.get_in_cluster_frac_for_bw(preds)
                ## save sum of fracs in list
                frac_sums.append( qa_in_cluster_frac + bw_in_cluster_frac )
            
            return qw_overlap_list[ np.argmax( frac_sums ) ]

    
    
    
    
    
    inds_to_keep = []    
    qw_overlap_list = []
    for i in range(len(crossing_table_list)):        
        current_crossing_table = crossing_table_list[i]
        
        
        ## Retreive next crossing table if possible
        next_crossing_table = None
        if i + 1 < len(crossing_table_list):    
            next_crossing_table = crossing_table_list[i+1]
        
        
        ## Check if overlap between current and next crossing table
        ## (if next is None, then func will return False)
        if current_crossing_table.has_overlapping_qw_interval(
                                            times,
                                            next_crossing_table
                                                              ):
            qw_overlap_list.append(i)
            
        
        ## If no overlap found AS OF CURRENT TABLE, then multiple options
        ## to consider ...
        else:
            
            # If no overlap found with prior table either, then simply save
            # ind as good
            if len(qw_overlap_list) == 0:
                inds_to_keep.append( i )
            
            
            # Otherwise, then overlap WAS found earlier and has only just
            # found a table that DIDN'T overlap.
            if len(qw_overlap_list) > 0:
                    
                # Check to make sure that we didn't just hit the last index
                # (which would return False for None as next_crossing_table)
                if i < len(crossing_table_list):
                    qw_overlap_list.append( i )
                    
                # See which ind was preferred based on overlap_preference
                chosen_ind = choose_from_overlapping_crossing_tables(
                                                crossing_table_list,
                                                qw_overlap_list,
                                                overlap_preference,
                                                preds
                                                                    )
                
                # If not given None, then save ind(s) given back
                # (either as int if single or as list if multiple)
                if chosen_ind is not None:
                    if isinstance(chosen_ind, list):
                        inds_to_keep.extend(chosen_ind)
                    else:
                        inds_to_keep.append(chosen_ind)
                
                # Re-set overlap list to empty
                qw_overlap_list = []
        
        
        
    return [ crossing_table_list[q] for q in inds_to_keep ]
        
        
    










def check_crossing_duration_rules(crossing_table_list,
                                  times,
                                  min_crossing_duration = None,
                                  max_crossing_duration = None):
    
    """
    
    Check if crossings in crossing_table_list satisfy rules
    
    HOWEVER
    
    Only check for rules related to params given
    
    """
    
    good_crossing_inds = []
    for i in range(len(crossing_table_list)):
        min_crossing_duration_good = True
        max_crossing_duration_good = True
        
        
        ## check min_crossing_duration rule if not None
        if min_crossing_duration is not None:
            min_crossing_duration_good = \
                crossing_table_list[i].geq_min_crossing_duration(
                                                    times,
                                                    min_crossing_duration
                                                                )
        
        
        ## check max_crossing_duration rule if not None
        if max_crossing_duration is not None:
            max_crossing_duration_good = \
                crossing_table_list[i].leq_max_crossing_duration(
                                                    times,
                                                    max_crossing_duration
                                                                )
                
                
        ## If rules checked are satisfied, save index
        if ((min_crossing_duration_good) and  (max_crossing_duration_good)):
            good_crossing_inds.append( i )
        
    
    ## return list of crossing tables that satisfied the rules
    return [ crossing_table_list[i] for i in good_crossing_inds ]






















def _test_find_possible_crossing_intervals_soft():
    
    """
    
    Tests find_possible_crossing_intervals_soft()
    
    For the doc strings below, 
        * - cluster 1 probability
        o - cluster 2 probability
    
    """
    
    
    ## defining some empty arrays here for convenience
    empty_arr = np.zeros((0,2), dtype=np.int64)
    
    
    
    def compare_expected_vs_algo_result(preds,
                                        min_prob,
                                        expected_result):
        
        """
        
        Tests the algorithm's results with what's expected
        
        """
        
        
        algo_result = find_possible_crossing_intervals_soft(preds, min_prob)
        
        
        try:
            num_elements = max( [len(expected_result), len(algo_result)] )
            arrs_equal_list = \
                    [ np.array_equal(expected_result[i], algo_result[i])
                      for i in range(num_elements) ]
            assert( np.all(arrs_equal_list) )
            
            
        except AssertionError:
            print("Failed!")
            print("Using preds",preds,"with min_prob",min_prob,
                  "expected result:",expected_result,"...")
            print("but got result:",algo_result)
            raise
            
            
        ## Will get index error if number of elements don't match
        except IndexError:
            print("Number of elements do not match between expected result ("
                  + str(len(expected_result)) + ") and algorithm result ("
                  + str(len(algo_result)) + ").")
            print("--expected result:\n",expected_result)
            print("--but got--:\n",algo_result)
            print(type(expected_result),type(algo_result))
            raise
    
    
    
    
    
    def no_crossing_test():
        
        """
        
          | * * * * * * * * * * * * *
          |                     
          | ------------------------- min_prob
          |           
          |           
          |           
          | o o o o o o o o o o o o o
        --|-------------------------- (x-axis)
        
        """
        
        preds_first_cluster = np.array( [0.8, 0.8, 0.8, 0.9, 1.0, 0.8] )
        preds = np.vstack( [preds_first_cluster,
                            1 - preds_first_cluster] ).T
        min_prob = 0.75
        #expected_result = np.zeros((0,2), dtype=np.int64)
        expected_result = [ empty_arr, empty_arr, empty_arr ]
        compare_expected_vs_algo_result(preds,
                                        min_prob,
                                        expected_result)
    
    
    
    
    
    def single_immediate_crossing_test():
        
        """
        
          | * * * * * * o o o o o o o
          |           | |          
          | ----------|-|------------ min_prob
          |           | |
          |           | |
          |           | |        
          | o o o o o o * * * * * * *
        --|-----------|-|------------ (x-axis)
                      A B
        
        """
        
        preds = np.array([ [0.8, 0.8, 0.8, 0.2, 0.2, 0.2],
                           [0.2, 0.2, 0.2, 0.8, 0.8, 0.8] ]).T
        min_prob = 0.75
        expected_result = [ np.array( [[2,3]] ),
                            empty_arr,
                            empty_arr ] 
        compare_expected_vs_algo_result(preds,
                                        min_prob,
                                        expected_result)
    
    
    
    
    
    def multiple_immediate_crossings_test():
        
        """
        
          | * * * * * * o o o o * * *
          |           | |     | |
          | ----------|-|-----|-|---- min_prob
          |           | |     | |
          |           | |     | |
          |           | |     | | 
          | o o o o o o * * * * o o o
        --|-----------|-|-----|-|---- (x-axis)
                      A B     A B
                      
        """
        
        preds_first_cluster = np.array( 
                [0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8] 
                                      )
        preds = np.vstack( [preds_first_cluster,
                            1 - preds_first_cluster] ).T
        min_prob = 0.75
        expected_result = [ np.array( [[3,4], [8,9]] ),
                            empty_arr,
                            empty_arr ]
        compare_expected_vs_algo_result(preds,
                                        min_prob,
                                        expected_result)
    
    
    
    
    
    def single_immediate_crossing_at_start_test():
        
        """
        
          | * o o o o o o 
          | | |     
          |-|-|------------ min_prob
          | | |
          | | |
          | | |
          | o * * * * * * 
        --|-|-|------------ (x-axis)
            A B     
                      
        """
        
        preds_first_cluster = np.array( 
                [0.8, 0.2, 0.2, 0.2, 0.2 ] 
                                      )
        preds = np.vstack( [preds_first_cluster,
                            1 - preds_first_cluster] ).T
        min_prob = 0.75
        expected_result = [ np.array( [[0,1]] ),
                            empty_arr,
                            empty_arr ]
        compare_expected_vs_algo_result(preds,
                                        min_prob,
                                        expected_result)
    
    
    
    
    
    def single_immediate_crossing_at_end_test():
        
        """
        
          | * * * * * * o 
          |           | |     
          |-----------|-|- min_prob
          |           | |
          |           | |
          |           | |
          | o o o o o o * 
        --|-----------|-|- (x-axis)
                      A B     
                      
        """
        
        preds_first_cluster = np.array( 
                [0.8, 0.8, 0.8, 0.8, 0.2 ] 
                                      )
        preds = np.vstack( [preds_first_cluster,
                            1 - preds_first_cluster] ).T
        min_prob = 0.75
        expected_result = [ np.array( [[3,4]] ),
                            empty_arr,
                            empty_arr ]
        compare_expected_vs_algo_result(preds,
                                        min_prob,
                                        expected_result)
    
    
    
    
    
    def single_gradual_crossing_diff_start_end_cluster_test():
        
        """
        
         prob-axis
            | * * * |       | o o o o   o o
            |       *       o         o
            | ------|-------|---------------- min_prob
            |       | * * o |
            |       | o o * |
            |       o       *         *
            | o o o |       | * * * *   * *
          --|-------------------------------- (x-axis)
                    A       B
                      
        """
        
        preds_first_cluster = np.array( 
                [1.0, 0.9, 0.9, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1 ] 
                                      )
        preds = np.vstack( [preds_first_cluster,
                            1 - preds_first_cluster] ).T
        min_prob = 0.75
        expected_result = [ empty_arr,
                            np.array( [[2,7]] ),
                            empty_arr ]
        compare_expected_vs_algo_result(preds,
                                        min_prob,
                                        expected_result)
    
    
    
    
    
    def multiple_gradual_crossings_diff_dominant_cluster_test():
        
        """
        
         prob-axis
            | * * * |       | o o o o |     | * *
            |       *       o         o     *
            | ------|-------|---------|-----|----- min_prob
            |       | * * o |         | o * |
            |       | o o * |         | * o |
            |       o       *         *     o
            | o o o |       | * * * * |     | o o
          --|------------------------------------- (x-axis)
                    A       B         A     B
                      
        """
        
        preds_first_cluster = np.array( 
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.2, 0.15, 0.2, 0.5, 0.7, 0.8, 0.9, 1.0 ] 
                                      )
        preds = np.vstack( [preds_first_cluster,
                            1 - preds_first_cluster] ).T
        min_prob = 0.75
        expected_result = [ empty_arr,
                            np.array( [[2,5], [7,10]] ),
                            empty_arr ]
        compare_expected_vs_algo_result(preds,
                                        min_prob,
                                        expected_result)
    
    
    
    
    
    def single_gradual_crossing_same_start_end_cluster_test():
        
        """
        
         prob-axis
            | * * * |         | * * * *   * *
            |       *         *         *
            | ------|---------|---------------- min_prob
            |       | * * o * |
            |       | o o * o |
            |       o         o         o
            | o o o |         | o o o o   o o
          --|-------------------------------- (x-axis)
                    A       B
                      
        """
        
        preds_first_cluster = np.array( 
                [1.0, 0.9, 0.9, 0.7, 0.6, 0.5, 0.4, 0.55, 0.7, 0.8, 0.9, 1.0 ] 
                                      )
        preds = np.vstack( [preds_first_cluster,
                            1 - preds_first_cluster] ).T
        min_prob = 0.75
        expected_result = [ empty_arr,
                            empty_arr,
                            np.array( [[2,9]] ) ]
        compare_expected_vs_algo_result(preds,
                                        min_prob,
                                        expected_result)
    
    
    
    
    
    def multiple_gradual_crossings_same_dominant_cluster_test():
        
        """
        
         prob-axis
            | * * * |         | * * * * |     | * *
            |       *         *         *     *
            | ------|---------|---------|-----|----- min_prob
            |       | * * o * |         | * * |
            |       | o o * o |         | o o |
            |       o         o         o     o
            | o o o |         | o o o o |     | o o
          --|--------------------------------------- (x-axis)
                    A         B         A     B
                      
        """
        
        preds_first_cluster = np.array( 
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.7, 0.8, 0.8, 0.7, 0.6, 0.8, 1.0 ] 
                                      )
        preds = np.vstack( [preds_first_cluster,
                            1 - preds_first_cluster] ).T
        min_prob = 0.75
        expected_result = [ empty_arr,
                            empty_arr,
                            np.array( [[2,7], [8,11]] ) ]
        compare_expected_vs_algo_result(preds,
                                        min_prob,
                                        expected_result)
    
    
    
    
    
    def single_immediate_single_gradual_crossing_test():
        
        """
        
         prob-axis
            | * * * | | o o o o o   |     | * *
            |       * o           o |     * 
            | ------|-|-------------|-----|--------- min_prob
            |       | |             o * * |   
            |       | |             * o o |
            |       o *             |     o
            | o o o | | * * * * * * |     | o o
          --|--------------------------------------- (x-axis)
                    A B             A     B
                      
        """
        
        preds_first_cluster = np.array( 
            [1.0, 0.9, 0.8, 0.2, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.9, 1.0 ] 
                                      )
        preds = np.vstack( [preds_first_cluster,
                            1 - preds_first_cluster] ).T
        min_prob = 0.75
        expected_result = [ np.array( [[2,3]] ),
                            np.array( [[5,9]] ),
                            empty_arr ]
        compare_expected_vs_algo_result(preds,
                                        min_prob,
                                        expected_result)
    
    
    
    
    
    def single_gradual_single_immediate_crossing_test():
        
        """
        
         prob-axis
            | * * * |         | o o o o | | * * * 
            |       *         o         o *   
            | ------|---------|---------|----------- min_prob
            |       | * * o o |         | |
            |       | o o * * |         | |  
            |       o         *         * o   
            | o o o |         | * * * * | | o o o
          --|--------------------------------------- (x-axis)
                    A         B         A B   
                      
        """
        
        preds_first_cluster = np.array( 
            [1.0, 0.9, 0.8, 0.7, 0.5, 0.4, 0.2, 0.1, 0.2, 0.8, 0.9, 1.0 ] 
                                      )
        preds = np.vstack( [preds_first_cluster,
                            1 - preds_first_cluster] ).T
        min_prob = 0.75
        expected_result = [ np.array( [[8,9]] ),
                            np.array( [[2,6]] ),
                            empty_arr ]
        compare_expected_vs_algo_result(preds,
                                        min_prob,
                                        expected_result)
    
    
    
    
    
    def gradual_immediate_gradual_crossing_test():
        
        """
        
         prob-axis
            | o o o |         | * * * * | | o o |     | * *
            |       o         *         * o     o     *
            | ------|---------|---------|-------|-----|----- min_prob
            |       | o o o * |         | |     | o * |
            |       | * * * o |         | |     | * o |
            |       *         o         o *     *     o
            | * * * |         | o o o o | | * * |     | o o
          --|----------------------------------------------- (x-axis)
                    A         B         A B     A     B
                      
        """
        
        preds_first_cluster = np.array( 
                [
                1.0, 0.9, 0.8, 0.7, 0.5, 0.4, 0.2,  # gradual
                0.1, 0.2, 0.8,  # immediate
                0.9, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1  # gradual
                ] 
                                      )
        preds = np.vstack( [preds_first_cluster,
                            1 - preds_first_cluster] ).T
        min_prob = 0.75
        expected_result = [ np.array( [[8,9]] ),
                            np.array( [[2,6], [12,16]] ),
                            empty_arr ]
        compare_expected_vs_algo_result(preds,
                                        min_prob,
                                        expected_result)
    
    
    
    
    
    def immediate_gradual_immediate_crossing_test():
        
        """
        
         prob-axis
            | o o | | * * * |         | o o o o | | * 
            |     o *       *         o         o *   
            |-----|-|-------|---------|---------|-|--- min_prob
            |     | |       | * * o o |         | |   
            |     | |       | o o * * |         | |   
            |     * o       o         *         * *   
            | * * | | o o o |         | * * * * | | * 
          --|------------------------------------------ (x-axis)
                  A B       A         B         A B   
                      
        """
        
        preds_first_cluster = np.array( 
                [
                1.0, 0.9, 0.8, 0.2, 0.1,  # immediate
                0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.85,  # gradual
                0.9, 1.0, 0.8, 0.2, 0.1  # immediate
                ] 
                                      )
        preds = np.vstack( [preds_first_cluster,
                            1 - preds_first_cluster] ).T
        min_prob = 0.75
        expected_result = [ np.array( [[2,3], [14,15]] ),
                            np.array( [[6,11]] ),
                            empty_arr ]
        compare_expected_vs_algo_result(preds,
                                        min_prob,
                                        expected_result)
    
    
    
    
    
    no_crossing_test()
    single_immediate_crossing_test()
    multiple_immediate_crossings_test()
    single_immediate_crossing_at_start_test()
    single_immediate_crossing_at_end_test()
    single_gradual_crossing_diff_start_end_cluster_test()
    single_gradual_crossing_same_start_end_cluster_test()
    multiple_gradual_crossings_diff_dominant_cluster_test()
    multiple_gradual_crossings_same_dominant_cluster_test()
    single_immediate_single_gradual_crossing_test()
    single_gradual_single_immediate_crossing_test()
    gradual_immediate_gradual_crossing_test()
    immediate_gradual_immediate_crossing_test()
    print("All find_possible_crossing_intervals_soft tests passed!")










def _test_compute_crossings_soft():
        
    """
    
    Tests compute_crossings_soft()
    
    For the doc strings below, 
        * - cluster 1 probability
        o - cluster 2 probability
    
    """
    
    
    
    
    
    def compare_expected_vs_algo_result(expected_result,
                                        *,
                                        times,
                                        preds,
                                        min_prob,
                                        min_crossing_duration,
                                        max_crossing_duration,
                                        min_cluster_frac,
                                        min_beyond_crossing_duration,
                                        max_beyond_crossing_duration,
                                        overlap_preference):
        
        """
        
        Tests the algorithm's results with what's expected
        
        """
        
        ## get algo result
        algo_result = compute_crossings_soft(
                times,
                preds,
                min_prob = min_prob,
                min_crossing_duration = min_crossing_duration,
                max_crossing_duration = max_crossing_duration,
                min_cluster_frac = min_cluster_frac,
                min_beyond_crossing_duration = min_beyond_crossing_duration,
                max_beyond_crossing_duration = max_beyond_crossing_duration,
                overlap_preference = overlap_preference
                                      )
        
        
        ## check algo result against expected result
        try:
            num_elements = max( [len(expected_result), len(algo_result)] )
            arrs_equal_list = \
                    [ np.array_equal(expected_result[i], algo_result[i])
                      for i in range(num_elements) ]
            assert( np.all(arrs_equal_list) )
            
        
        ## Will get if number elements agree, but constituent arrays don't match
        except AssertionError:
            print("Failed!")
            print("Using ...")
            # coerce preds and corresponding inds to giant strings sep by newlines
            preds_strs = "\n".join(
                [ "      " + str(i) + ": " + str(preds[i]) + "  " + \
                  times[i].strftime("%Y-%d-%m %H:%M:%S")
                  for i in range(preds.shape[0]) ]
                                  )
            print("  --preds & times--:\n" + preds_strs)
            # coerce times to giant strings separated by newlines
            #times_strs = "\n".join( [ a_time \
            #                          for a_time in times ] )
            #print("  --times--:",times_strs)
            print("  --min_prob--:",min_prob)
            print("  --min_beyond_crossing_duration--:",
                  min_beyond_crossing_duration)
            print("  --max_beyond_crossing_duration--:",
                  max_beyond_crossing_duration)
            print("  --overlap_pref--:",overlap_preference)
            print("  --min_cluster_frac--:",min_cluster_frac)
            print("--expected result--:\n",expected_result)
            print("--but got--:\n",algo_result)
            raise
        
        
        ## Will get index error if number of elements don't match
        except IndexError:
            print("Number of elements do not match between expected result ("
                  + str(len(expected_result)) + ") and algorithm result ("
                  + str(len(algo_result)) + ").")
            print("--expected result:\n",expected_result)
            print("--but got--:\n",algo_result)
            print(type(expected_result),type(algo_result))
            raise





    def no_crossings_test():
        
        """
        
        | * * * * * * * * * * * * *
        |                     
        | ------------------------- min_prob
        |           
        |           
        |           
        | o o o o o o o o o o o o o
      --|-------------------------- (x-axis)
        
        """
        
        # Make preds
        preds_first_cluster = np.array( 
                [ 1.0, 0.9, 0.8, 0.9, 0.9, 0.95, 0.8, 1.0 ] 
                                      )
        preds = np.vstack( [preds_first_cluster,
                            1 - preds_first_cluster] ).T
        # Make times
        pd_delta = pd.Timedelta(60, unit="s")
        times = [ pd.Timestamp("2001-01-01") + i * pd_delta \
                  for i in range(preds.shape[0]) ]
        # setup expected_result
        expected_result = []
        # check if algo matches expected result
        compare_expected_vs_algo_result(
                expected_result,
                times = times,
                preds = preds,
                min_prob = 0.75,
                min_cluster_frac = 1.0,
                min_crossing_duration = pd.Timedelta(5, unit="s"),
                max_crossing_duration = pd.Timedelta(30, unit="s"),
                min_beyond_crossing_duration = pd.Timedelta(60, unit="s"),
                max_beyond_crossing_duration = 3 * pd.Timedelta(60, unit="s"),
                overlap_preference = "remove"
                                        )
    
    
    
    
    
    def single_immediate_crossing_all_pts_in_crossing_test():
        
        """
        
        prob-axis
           ||  * *   * * * * * * * * |  | o o   o o o o o o |
           ||  |   *     |           *  o     o   |         o
           ||--|---------|-----------|--|---------|---------|---- min_prob
           ||  |         |           |  |         |         |
           ||  |         |           |  |         |         |
           ||  |   o     |           o  *     *   |         *
           ||  o o   o o o o o o o o |  | * *   * * * * * * |
         ==||==|=========|===========|==|=========|=========|==== time-axis
           ||  X         |           A  B         |         Y
           ||            Q                        W
               * - cluster 1 probability
               o - cluster 2 probability
               
        """
        
        # Make preds
        preds_first_cluster = np.array( 
            [
            1.0, 0.9, 0.8, 0.9, 0.8, 0.9,  # pts <= A
            0.2, 0.1, 0.15, 0.2, 0.2, 0.05  # pts >= B
            ] 
                                      )
        preds = np.vstack( [preds_first_cluster,
                            1 - preds_first_cluster] ).T
        # Make times
        pd_delta = pd.Timedelta(60, unit="s")
        times = [ pd.Timestamp("2001-01-01") + i * pd_delta \
                  for i in range(preds.shape[0]) ]
        # setup expected_result
        expected_result = [
                CrossingTable(x_ind=0, y_ind=11,
                              q_ind=2, w_ind=9,
                              a_ind=5, b_ind=6).get_array()
                          ]
        # check if algo matches expected result
        compare_expected_vs_algo_result(
                expected_result,
                times = times,
                preds = preds,
                min_prob = 0.75,
                min_cluster_frac = 1.0,
                min_crossing_duration = pd.Timedelta(5, unit="s"),
                max_crossing_duration = pd.Timedelta(90, unit="s"),
                min_beyond_crossing_duration = 3 * pd.Timedelta(60, unit="s"),
                max_beyond_crossing_duration = 5 * pd.Timedelta(60, unit="s"),
                overlap_preference = "remove"
                                        )
    
    
    
    
    
    def single_immediate_crossing_some_pts_in_crossing_test():
        
        """
        
        prob-axis
           || * * * *   * * * * * * * * |  | o o   o o o o o o | o o
           ||     |   *     |           *  o     o   |         o
           ||-----|---------|-----------|--|---------|---------|------ min_prob
           ||     |         |           |  |         |         |
           ||     |         |           |  |         |         |
           ||     |   o     |           o  *     *   |         *
           || o o o o   o o o o o o o o |  | * *   * * * * * * | * *
         ==||======|=========|===========|==|=========|=========|===== time-axis
           ||      X         |           A  B         |         Y
           ||                Q                        W
               * - cluster 1 probability
               o - cluster 2 probability
               
        """
        
        # Make preds
        preds_first_cluster = np.array( 
            [ 
            1.0, 0.9, 0.8, 0.9, 0.8, 0.9, 0.85, 0.8, 0.9,  # pts <= A
            0.2, 0.1, 0.15, 0.2, 0.2, 0.05, 0.1, 0.05, 0.2  # pts >= B
            ] 
                                      )
        preds = np.vstack( [preds_first_cluster,
                            1 - preds_first_cluster] ).T
        # Make times
        pd_delta = pd.Timedelta(60, unit="s")
        times = [ pd.Timestamp("2001-01-01") + i * pd_delta \
                  for i in range(preds.shape[0]) ]
        # setup expected_result
        expected_result = [
                CrossingTable(x_ind=3, y_ind=14,
                              q_ind=5, w_ind=12,
                              a_ind=8, b_ind=9).get_array()
                          ]
        # check if algo matches expected result
        compare_expected_vs_algo_result(
                expected_result,
                times = times,
                preds = preds,
                min_prob = 0.75,
                min_cluster_frac = 1.0,
                min_crossing_duration = pd.Timedelta(5, unit="s"),
                max_crossing_duration = pd.Timedelta(90, unit="s"),
                min_beyond_crossing_duration = 3 * pd.Timedelta(60, unit="s"),
                max_beyond_crossing_duration = 5 * pd.Timedelta(60, unit="s"),
                overlap_preference = "remove"
                                        )
    
    
    
    
    
    def single_gradual_crossing_all_pts_in_crossing_test():
        
        """
        
        prob-axis
           || * *   * * * * * * * * |       | o o   o o o o o o |
           || |   *     |           *       o     o   |         o
           ||-|---------|-----------|-------|---------|---------|-- min_prob
           || |         |           | * * o |         |         |
           || |         |           | o o * |         |         |
           || |   o     |           o       *     *   |         *
           || o o   o o o o o o o o |       | * *   * * * * * * |
         ==||=|=========|===========|=======|=========|=========|== time-axis
           || X         |           A       B         |         Y
           ||           Q                             W
               * - cluster 1 probability
               o - cluster 2 probability
               
        """
        
        # Make preds
        preds_first_cluster = np.array( 
            [ 
            0.9, 0.8, 0.9, 0.85, 0.8, 0.9,  # pts <= A
            0.7, 0.5, 0.4, 0.3,  # pts b/w A and B
            0.2, 0.2, 0.05, 0.1, 0.05, 0.2  # pts >= B
            ] 
                                      )
        preds = np.vstack( [preds_first_cluster,
                            1 - preds_first_cluster] ).T
        # Make times
        pd_delta = pd.Timedelta(60, unit="s")
        times = [ pd.Timestamp("2001-01-01") + i * pd_delta \
                  for i in range(preds.shape[0]) ]
        # setup expected_result
        expected_result = [
                CrossingTable(x_ind=0, y_ind=15,
                              q_ind=2, w_ind=13,
                              a_ind=5, b_ind=10).get_array()
                          ]
        # check if algo matches expected result
        compare_expected_vs_algo_result(
                expected_result,
                times = times,
                preds = preds,
                min_prob = 0.75,
                min_cluster_frac = 1.0,
                min_crossing_duration = pd.Timedelta(5, unit="s"),
                max_crossing_duration = pd.Timedelta(1000, unit="s"),
                min_beyond_crossing_duration = 3 * pd.Timedelta(60, unit="s"),
                max_beyond_crossing_duration = 5 * pd.Timedelta(60, unit="s"),
                overlap_preference = "remove"
                                        )
    
    
    
    
    
    def single_gradual_crossing_some_pts_in_crossing_test():
        
        """
        
        prob-axis
           || * * * *   * * * * * * * * |       | o o   o o o o o o | o
           ||     |   *     |           *       o     o   |         o   o
           ||-----|---------|-----------|-------|---------|---------|---- min_prob
           ||     |         |           | * * o |         |         |
           ||     |         |           | o o * |         |         |
           ||     |   o     |           o       *     *   |         *   *
           || o o o o   o o o o o o o o |       | * *   * * * * * * | *
         ==||=====|=========|===========|=======|=========|=========|==== time-axis
           ||     X         |           A       B         |         Y
           ||               Q                             W
               * - cluster 1 probability
               o - cluster 2 probability
               
        """
        
        # Make preds
        preds_first_cluster = np.array( 
            [
            0.9, 0.8, 0.8,  # pts < X
            0.9, 0.8, 0.9, 0.85, 0.8, 0.9,  # pts <= A
            0.7, 0.5, 0.4, 0.3,  # pts b/w A and B
            0.2, 0.2, 0.05, 0.1, 0.05, 0.2,  # pts >= B
            0.15, 0.2, 0.1  # pts > Y
            ] 
                                      )
        preds = np.vstack( [preds_first_cluster,
                            1 - preds_first_cluster] ).T
        # Make times
        pd_delta = pd.Timedelta(60, unit="s")
        times = [ pd.Timestamp("2001-01-01") + i * pd_delta \
                  for i in range(preds.shape[0]) ]
        # setup expected_result
        expected_result = [
                CrossingTable(x_ind=3, y_ind=18,
                              q_ind=5, w_ind=16,
                              a_ind=8, b_ind=13).get_array()
                          ]
        # check if algo matches expected result
        compare_expected_vs_algo_result(
                expected_result,
                times = times,
                preds = preds,
                min_prob = 0.75,
                min_cluster_frac = 1.0,
                min_crossing_duration = pd.Timedelta(5, unit="s"),
                max_crossing_duration = pd.Timedelta(1000, unit="s"),
                min_beyond_crossing_duration = 3 * pd.Timedelta(60, unit="s"),
                max_beyond_crossing_duration = 5 * pd.Timedelta(60, unit="s"),
                overlap_preference = "remove"
                                        )
    
    
    
    
    
    def gradual_then_immediate_no_overlap():
        
        """
        
        gradual crossing, then pts in neither, then immediate
        (e.g. gradual Y < immediate X in diagrams)
               
        """
        
        
        #### Setup data for first crossing (gradual)
        ## Make preds
        preds_first_crossing = np.array( 
            [
            0.9, 0.8, 0.8,  # pts < X
            0.9, 0.8, 0.9, 0.85, 0.8, 0.9,  # pts <= A
            0.7, 0.5, 0.4, 0.3,  # pts b/w A and B
            0.2, 0.2, 0.05, 0.1, 0.05, 0.2,  # pts >= B
            0.15, 0.2, 0.1,  # pts > Y
            ] 
                                      )
        preds_first_crossing = np.vstack( [preds_first_crossing,
                                           1 - preds_first_crossing] ).T
        ## setup expected result for first crossing (gradual)
        first_crossing_expected_result = CrossingTable(
                                            x_ind=3, y_ind=18,
                                            q_ind=5, w_ind=16,
                                            a_ind=8, b_ind=13
                                                      ).get_array()
        
        
        #### setup expected result for pts not in a crossing
        preds_in_between = np.array( [ 0.1, 0.2, 0.2, 0.1, 0.15 ] )
        preds_in_between = np.vstack( [preds_in_between,
                                       1 - preds_in_between] ).T
        
        
        #### setup expected result for second crossing (immediate)
        ## Make preds
        preds_second_crossing = np.array(
            [
            0.15, 0.2, 0.1,  # pts < X
            0.0, 0.1, 0.2, 0.2, 0.15, 0.9,  # pts <= A
            0.8, 0.85, 0.78, 0.88, 0.9, 1.0, # pts >= B
            0.9, 0.8, 0.79  # pts > Y    
            ]
                                        )
        preds_second_crossing = np.vstack( [preds_second_crossing,
                                            1 - preds_second_crossing] ).T
        ## setup expected result for escond crossing (immediate)
        second_crossing_expected_result = CrossingTable(
                                            x_ind=3, y_ind=14,
                                            q_ind=5, w_ind=12,
                                            a_ind=8, b_ind=9
                                                        ).get_array()
        ## increment inds in second_crossing to synchronize with full array
        second_crossing_expected_result[:,0] = \
                    (second_crossing_expected_result[:,0]
                     + preds_first_crossing.shape[0]
                     + preds_in_between.shape[0]
                     - 1)
        
        
        #### Combines crossings into single array
        ## Combine preds
        preds = np.vstack( [preds_first_crossing,
                            preds_in_between,
                            preds_second_crossing] )
        ## combine individual expected results into single large result
        expected_result = [ first_crossing_expected_result,
                            second_crossing_expected_result ]
        
        
        #### Supply times and get algo result
        pd_delta = pd.Timedelta(60, unit="s")
        times = [ pd.Timestamp("2001-01-01") + i * pd_delta \
                  for i in range(preds.shape[0]) ]
        # check if algo matches expected result
        compare_expected_vs_algo_result(
                expected_result,
                times = times,
                preds = preds,
                min_prob = 0.75,
                min_cluster_frac = 1.0,
                min_crossing_duration = pd.Timedelta(5, unit="s"),
                max_crossing_duration = pd.Timedelta(1000, unit="s"),
                min_beyond_crossing_duration = 3 * pd.Timedelta(60, unit="s"),
                max_beyond_crossing_duration = 5 * pd.Timedelta(60, unit="s"),
                overlap_preference = "remove"
                                        )
    
    
    
    
    
    def gradual_then_immediate_xy_overlap():
        
        """
        
        gradual crossing then immediate with permissible overlap - diagram B
        in depicted allowed crossings in the compute_crossings_soft
        doc-string.
        (e.g. gradual Y > immediate X, and gradual W < immediate X )
               
        """
        
        
        #### Setup data for first crossing (gradual)
        ## Make preds
        preds = np.array( 
            [
            ## first crossing (gradual)
            0.9, 0.8, 0.8,  # pts < gradual X
            0.9, 0.8, 0.9, 0.85, 0.8, 0.9,  # pts <= gradual A
            0.7, 0.5, 0.4, 0.3,  # pts b/w gradual A and gradual B
            0.2, 0.2, 0.05, 0.1,  # pts >= gradual B and not overlapping
            
            ## pts in x,y overlap
            0.05, 0.2,
            
            ## second crossing (immediate)
            0.1, 0.1, 0.2, 0.15,  # pts <= immediate A and not overlapping
            0.9, 0.8, 0.77, 0.85, 0.9, 0.8,  # pts >= immediate B 
            0.9, 1.0  # pts > immediate Y
            ]
                        )
        preds = np.vstack( [preds,
                            1 - preds] ).T
        ## setup expected result for first crossing (gradual)
        first_crossing_expected_result = CrossingTable(
                                            x_ind=3, y_ind=18,
                                            q_ind=5, w_ind=16,
                                            a_ind=8, b_ind=13
                                                      ).get_array()
        ## setup expected result for second crossing (immediate w/ overlap)
        second_crossing_expected_result = CrossingTable(
                                            x_ind=0, y_ind=11,
                                            q_ind=2, w_ind=9,
                                            a_ind=5, b_ind=6
                                                       ).get_array()
        # reindex index column of second crossing
        second_crossing_expected_result[:,0] = \
                    (second_crossing_expected_result[:,0]
                     + 17)
        
        
        #### Combines crossings into single array
        expected_result = [ first_crossing_expected_result,
                            second_crossing_expected_result ]
        
        
        #### Supply times and get algo result
        pd_delta = pd.Timedelta(60, unit="s")
        times = [ pd.Timestamp("2001-01-01") + i * pd_delta \
                  for i in range(preds.shape[0]) ]
        # check if algo matches expected result
        compare_expected_vs_algo_result(
                expected_result,
                times = times,
                preds = preds,
                min_prob = 0.75,
                min_cluster_frac = 1.0,
                min_crossing_duration = pd.Timedelta(5, unit="s"),
                max_crossing_duration = pd.Timedelta(1000, unit="s"),
                min_beyond_crossing_duration = 3 * pd.Timedelta(60, unit="s"),
                max_beyond_crossing_duration = 5 * pd.Timedelta(60, unit="s"),
                overlap_preference = "remove"
                                        )
    
    
    
    
    
    def gradual_then_immediate_xw_overlap():
        
        """
        
        gradual crossing then immediate with permissible overlap - diagram C
        in depicted allowed crossings in the compute_crossings_soft
        doc-string.
        (e.g. gradual Y > immediate X, and gradual W < immediate Q )
               
        """
        
        
        #### Setup data for first crossing (gradual)
        ## Make preds
        preds = np.array( 
            [
            ## first crossing (gradual)
            0.9, 0.8, 0.8,  # pts < gradual X
            0.9, 0.8, 0.9, 0.85, 0.8, 0.9,  # pts <= gradual A
            0.7, 0.5, 0.4, 0.3,  # pts b/w gradual A and gradual B
            0.2, 0.2, 0.05,  # pts >= gradual B and not overlapping
            
            ## pts in x,y overlap
            0.1, 0.05, 0.2,  # last ind 18
            
            ## second crossing (immediate)
            0.1, 0.2, 0.15,  # pts <= immediate A and not overlapping
            0.9, 0.8, 0.77, 0.85, 0.9, 0.8,  # pts >= immediate B 
            0.9, 1.0  # pts > immediate Y
            ]
                        )
        preds = np.vstack( [preds,
                            1 - preds] ).T
        ## setup expected result for first crossing (gradual)
        first_crossing_expected_result = CrossingTable(
                                            x_ind=3, y_ind=18,
                                            q_ind=5, w_ind=16,
                                            a_ind=8, b_ind=13
                                                      ).get_array()
        ## setup expected result for second crossing (immediate w/ overlap)
        second_crossing_expected_result = CrossingTable(
                                            x_ind=16, y_ind=27,
                                            q_ind=18, w_ind=25,
                                            a_ind=21, b_ind=22
                                                       ).get_array()
        
        
        #### Combines crossings into single array
        expected_result = [ first_crossing_expected_result,
                            second_crossing_expected_result ]
        
        
        #### Supply times and get algo result
        pd_delta = pd.Timedelta(60, unit="s")
        times = [ pd.Timestamp("2001-01-01") + i * pd_delta \
                  for i in range(preds.shape[0]) ]
        # check if algo matches expected result
        compare_expected_vs_algo_result(
                expected_result,
                times = times,
                preds = preds,
                min_prob = 0.75,
                min_cluster_frac = 1.0,
                min_crossing_duration = pd.Timedelta(5, unit="s"),
                max_crossing_duration = pd.Timedelta(1000, unit="s"),
                min_beyond_crossing_duration = 3 * pd.Timedelta(60, unit="s"),
                max_beyond_crossing_duration = 5 * pd.Timedelta(60, unit="s"),
                overlap_preference = "remove"
                                        )
    
    
    
    
    
    def gradual_then_immediate_qw_overlap():
        
        """
        
        gradual crossing then immediate with unallowed overlap - the lone
        diagram in what crossings *aren't* allowed in the 
        compute_crossings_soft doc-string.
        (e.g. gradual W > immediate Q in diagrams)
        
        Should not produce any crossings!
        
        """
        
        
        #### Setup data
        ## Make preds
        preds = np.array( 
            [
            ## first crossing (gradual)
            0.9, 0.8, 0.8,  # pts < gradual X
            0.9, 0.8, 0.9, 0.85, 0.8, 0.9,  # pts <= gradual A
            0.7, 0.5, 0.4, 0.3,  # pts b/w gradual A and gradual B
            0.2, 0.2,  # pts >= gradual B and not overlapping (DOES NOT REACH W)
            
            ## pts in overlap
            0.05, 0.2,
            
            ## second crossing (immediate)
            0.2, 0.15,  # pts <= immediate A and not overlapping
            0.9, 0.8, 0.77, 0.85, 0.9, 0.8,  # pts >= immediate B 
            0.9, 1.0  # pts > immediate Y
            ]
                        )
        preds = np.vstack( [preds,
                            1 - preds] ).T
        ## setup expected result
        expected_result = []
        
        
        
        #### Supply times and get algo result
        pd_delta = pd.Timedelta(60, unit="s")
        times = [ pd.Timestamp("2001-01-01") + i * pd_delta \
                  for i in range(preds.shape[0]) ]
        ## check if algo matches expected result
        compare_expected_vs_algo_result(
                expected_result,
                times = times,
                preds = preds,
                min_prob = 0.75,
                min_cluster_frac = 1.0,
                min_crossing_duration = pd.Timedelta(5, unit="s"),
                max_crossing_duration = pd.Timedelta(1000, unit="s"),
                min_beyond_crossing_duration = 3 * pd.Timedelta(60, unit="s"),
                max_beyond_crossing_duration = 5 * pd.Timedelta(60, unit="s"),
                overlap_preference = "remove"
                                        )
    
    
    
    
    
    def single_immediate_too_large_ab_gap():
        
        """
        
        immediate crossing with too-large of a crossing
        (e.g. length(A,B) > max_crossing_duration in diagrams)
        
        Should not produce any crossings!
        
        """
        
        
        
        #### Setup data
        ## Make preds
        preds_first_cluster = np.array( 
            [ 
            1.0, 0.9, 0.8, 0.9, 0.8, 0.9, 0.85, 0.8, 0.9,  # pts <= A
            0.2, 0.1, 0.15, 0.2, 0.2, 0.05, 0.1, 0.05, 0.2  # pts >= B
            ] 
                                      )
        preds = np.vstack( [preds_first_cluster,
                            1 - preds_first_cluster] ).T
        ## setup expected result
        expected_result = []
        
        
        
        #### Build times
        ## times leading up to A
        pd_delta = pd.Timedelta(60, unit="s")
        times_A = [ pd.Timestamp("2001-01-01") + i * pd_delta \
                    for i in range(8+1) ]
        ## times at and after B
        times_B = [ pd.Timestamp("2002-01-01") + i * pd_delta \
                    for i in range(8+1) ]
        ## combine times
        times = [ *times_A, *times_B ]
        
        
        
        
        ## check if algo matches expected result
        compare_expected_vs_algo_result(
                expected_result,
                times = times,
                preds = preds,
                min_prob = 0.75,
                min_cluster_frac = 1.0,
                min_crossing_duration = pd.Timedelta(5, unit="s"),
                max_crossing_duration = pd.Timedelta(100, unit="s"),
                min_beyond_crossing_duration = 3 * pd.Timedelta(60, unit="s"),
                max_beyond_crossing_duration = 5 * pd.Timedelta(60, unit="s"),
                overlap_preference = "remove"
                                        )
    
    
    
    
    
    def single_gradual_too_large_ab_gap():
        
        """
        
        gradual crossing with too-large of a crossing
        (e.g. length(A,B) > max_crossing_duration in diagrams)
        
        Should not produce any crossings!
        
        """
        
        
        
        #### Setup preds
        preds_first_cluster = np.array( 
            [ 
            0.9, 0.8, 0.9, 0.85, 0.8, 0.9,  # pts <= A
            0.7, 0.5,  # pts b/w A and B (pre-time gap)
            0.4, 0.3,  # pts b/w A and B (post-time gap)
            0.2, 0.2, 0.05, 0.1, 0.05, 0.2  # pts >= B
            ] 
                                      )
        preds = np.vstack( [preds_first_cluster,
                            1 - preds_first_cluster] ).T
      
        
        
        #### Build times
        ## times leading up to time gap
        pd_delta = pd.Timedelta(60, unit="s")
        times_A = [ pd.Timestamp("2001-01-01") + i * pd_delta \
                    for i in range(7+1) ]
        ## times after time gap
        times_B = [ pd.Timestamp("2002-01-01") + i * pd_delta \
                    for i in range(7+1) ]
        ## combine times
        times = [ *times_A, *times_B ]
        
        
        
        
        #### check if algo matches expected result
        expected_result = []
        compare_expected_vs_algo_result(
                expected_result,
                times = times,
                preds = preds,
                min_prob = 0.75,
                min_cluster_frac = 1.0,
                min_crossing_duration = pd.Timedelta(5, unit="s"),
                max_crossing_duration = pd.Timedelta(100, unit="s"),
                min_beyond_crossing_duration = 3 * pd.Timedelta(60, unit="s"),
                max_beyond_crossing_duration = 5 * pd.Timedelta(60, unit="s"),
                overlap_preference = "remove"
                                        )
    
    
    
    
    
    
    def single_immediate_timegap_in_qw_right():
        
        """
        
        gradual crossing with time gap in Q,W interval after crossing
        
        """
        
        
        
        #### Setup preds
        preds_first_cluster = np.array( 
            [ 
            0.9, 0.8, 0.9, 0.85, 0.8, 0.9,  # pts <= A
            0.2, 0.2, 0.05,  # pts >= B (pre-time gap)
            0.1, 0.05, 0.2  # pts post-time gap
            ] 
                                      )
        preds = np.vstack( [preds_first_cluster,
                            1 - preds_first_cluster] ).T
      
        
        
        #### Build times
        ## times leading up to time gap
        pd_delta = pd.Timedelta(60, unit="s")
        times_A = [ pd.Timestamp("2001-01-01") + i * pd_delta \
                    for i in range(8+1) ]
        ## times after time gap
        times_B = [ pd.Timestamp("2002-01-01") + i * pd_delta \
                    for i in range(2+1) ]
        ## combine times
        times = [ *times_A, *times_B ]
        
        
        
        #### Build expected result
        expected_result = [
                CrossingTable(x_ind=0, y_ind=8,
                              q_ind=2, w_ind=8,
                              a_ind=5, b_ind=6).get_array()
                          ]
        
        
        
        #### check if algo matches expected result
        compare_expected_vs_algo_result(
                expected_result,
                times = times,
                preds = preds,
                min_prob = 0.75,
                min_cluster_frac = 1.0,
                min_crossing_duration = pd.Timedelta(5, unit="s"),
                max_crossing_duration = pd.Timedelta(100, unit="s"),
                min_beyond_crossing_duration = 3 * pd.Timedelta(60, unit="s"),
                max_beyond_crossing_duration = 5 * pd.Timedelta(60, unit="s"),
                overlap_preference = "remove"
                                        )
    
    
    
    
    
    def single_immediate_timegap_in_qw_left():
        
        """
        
        gradual crossing with time gap in Q,W interval before crossing
        
        """
        
        
        
        #### Setup preds
        preds_first_cluster = np.array( 
            [ 
            0.9, 0.8, 0.9,  # pts <= A and pre-time gap
            0.85, 0.8, 0.9,  # pts <= A and w/in Q,W interval
            0.2, 0.2, 0.05, 0.1, 0.05, 0.2  # pts >= B (pre-time gap)
            ] 
                                      )
        preds = np.vstack( [preds_first_cluster,
                            1 - preds_first_cluster] ).T
      
        
        
        #### Build times
        ## times leading up to time gap
        pd_delta = pd.Timedelta(60, unit="s")
        times_A = [ pd.Timestamp("2001-01-01") + i * pd_delta \
                    for i in range(2+1) ]
        ## times after time gap
        times_B = [ pd.Timestamp("2002-01-01") + i * pd_delta \
                    for i in range(8+1) ]
        ## combine times
        times = [ *times_A, *times_B ]
        
        
        
        #### Build expected result
        expected_result = [
                CrossingTable(x_ind=3, y_ind=11,
                              q_ind=3, w_ind=9,
                              a_ind=5, b_ind=6).get_array()
                          ]
        
        
        
        #### check if algo matches expected result
        compare_expected_vs_algo_result(
                expected_result,
                times = times,
                preds = preds,
                min_prob = 0.75,
                min_cluster_frac = 1.0,
                min_crossing_duration = pd.Timedelta(5, unit="s"),
                max_crossing_duration = pd.Timedelta(100, unit="s"),
                min_beyond_crossing_duration = 3 * pd.Timedelta(60, unit="s"),
                max_beyond_crossing_duration = 5 * pd.Timedelta(60, unit="s"),
                overlap_preference = "remove"
                                        )
    
    
    
    
    
    def single_immediate_timegap_in_xy_right():
        
        """
        
        gradual crossing with time gap in X,Y interval after crossing
        (beyond the Q,W interval)
        
        """
        
        
        
        #### Setup preds
        preds_first_cluster = np.array( 
            [ 
            0.9, 0.8, 0.9, 0.85, 0.8, 0.9,  # pts <= A
            0.2, 0.2, 0.05, 0.1, 0.05,  # pts >= B (pre-time gap)
            0.2  # pts >= B (post-time gap)
            ] 
                                      )
        preds = np.vstack( [preds_first_cluster,
                            1 - preds_first_cluster] ).T
      
        
        
        #### Build times
        ## times leading up to time gap
        pd_delta = pd.Timedelta(60, unit="s")
        times_A = [ pd.Timestamp("2001-01-01") + i * pd_delta \
                    for i in range(10+1) ]
        ## times after time gap
        times_B = [ pd.Timestamp("2002-01-01") + i * pd_delta \
                    for i in range(0+1) ]
        ## combine times
        times = [ *times_A, *times_B ]
        
        
        
        #### Build expected result
        expected_result = [
                CrossingTable(x_ind=0, y_ind=10,
                              q_ind=2, w_ind=9,
                              a_ind=5, b_ind=6).get_array()
                          ]
        
        
        
        #### check if algo matches expected result
        compare_expected_vs_algo_result(
                expected_result,
                times = times,
                preds = preds,
                min_prob = 0.75,
                min_cluster_frac = 1.0,
                min_crossing_duration = pd.Timedelta(5, unit="s"),
                max_crossing_duration = pd.Timedelta(100, unit="s"),
                min_beyond_crossing_duration = 3 * pd.Timedelta(60, unit="s"),
                max_beyond_crossing_duration = 5 * pd.Timedelta(60, unit="s"),
                overlap_preference = "remove"
                                        )
    
    
    
    
    
    
    def single_immediate_timegap_in_xy_left():
        
        """
        
        gradual crossing with time gap in X,Y interval before crossing
        (outside of the Q,W interval)
        
        """
        
        
        
        #### Setup preds
        preds_first_cluster = np.array( 
            [ 
            0.9,  # pts <= A (pre-time gap)
            0.8, 0.9, 0.85, 0.8, 0.9,  # pts <= A (post-time gap)
            0.2, 0.2, 0.05, 0.1, 0.05, 0.2  # pts >= B 
            ] 
                                      )
        preds = np.vstack( [preds_first_cluster,
                            1 - preds_first_cluster] ).T
      
        
        
        #### Build times
        ## times leading up to time gap
        pd_delta = pd.Timedelta(60, unit="s")
        times_A = [ pd.Timestamp("2001-01-01") + i * pd_delta \
                    for i in range(0+1) ]
        ## times after time gap
        times_B = [ pd.Timestamp("2002-01-01") + i * pd_delta \
                    for i in range(10+1) ]
        ## combine times
        times = [ *times_A, *times_B ]
        
        
        
        #### Build expected result
        expected_result = [
                CrossingTable(x_ind=1, y_ind=11,
                              q_ind=2, w_ind=9,
                              a_ind=5, b_ind=6).get_array()
                          ]
        
        
        
        #### check if algo matches expected result
        compare_expected_vs_algo_result(
                expected_result,
                times = times,
                preds = preds,
                min_prob = 0.75,
                min_cluster_frac = 1.0,
                min_crossing_duration = pd.Timedelta(5, unit="s"),
                max_crossing_duration = pd.Timedelta(100, unit="s"),
                min_beyond_crossing_duration = 3 * pd.Timedelta(60, unit="s"),
                max_beyond_crossing_duration = 5 * pd.Timedelta(60, unit="s"),
                overlap_preference = "remove"
                                        )
    
    
    
    
    
    def single_immediate_too_small_ab_gap():
        
        """
        
        gradual crossing with too-small of a time gap in A,B interval
        
        Should not produce any crossings!
        
        """
        
        
        
        #### Setup preds
        preds_first_cluster = np.array( 
            [ 
            0.8, 0.9, 0.85, 0.8, 0.9,  # pts <= A (post-time gap)
            0.7,  # pts b/w crossings
            0.8, 0.85, 0.9, 1.0  # pts >= B 
            ] 
                                      )
        preds = np.vstack( [preds_first_cluster,
                            1 - preds_first_cluster] ).T
      
        
        
        #### Build times
        pd_delta = pd.Timedelta(60, unit="s")
        times = [ pd.Timestamp("2001-01-01") + i * pd_delta \
                    for i in range(preds.shape[0]) ]
        
        
        
        #### Build expected result
        expected_result = []
        
        
        
        #### check if algo matches expected result
        compare_expected_vs_algo_result(
                expected_result,
                times = times,
                preds = preds,
                min_prob = 0.75,
                min_cluster_frac = 1.0,
                min_crossing_duration = pd.Timedelta(200, unit="s"),
                max_crossing_duration = pd.Timedelta(1000, unit="s"),
                min_beyond_crossing_duration = 3 * pd.Timedelta(60, unit="s"),
                max_beyond_crossing_duration = 5 * pd.Timedelta(60, unit="s"),
                overlap_preference = "remove"
                                        )
        
        
        
        
        
        
        
        
        
    def multi_interval_test_from_tha_data():
        
        """
        
        taken from tha data, (time bounds here)
        
        """
        
        preds = np.array([
            [1.00000000e+00, 4.08735769e-12, 8.36288094e-42],
            [9.99999999e-01, 8.17769592e-10, 2.72443005e-36],
            [9.99999376e-01, 6.24135337e-07, 3.31942604e-31],
            [2.88127379e-11, 1.00000000e+00, 5.18392195e-22],
            [1.00000000e+00, 9.72163702e-12, 3.30470747e-46],
            [9.99999994e-01, 6.06330718e-09, 9.02943227e-38],
            [1.00000000e+00, 6.63558005e-11, 5.34213192e-39],
            [1.00000000e+00, 1.61640998e-10, 1.36050874e-40],
            [1.00000000e+00, 1.58006881e-11, 7.97587499e-42],
            [1.00000000e+00, 6.66725214e-13, 2.81244900e-43],
            [1.00000000e+00, 6.40176363e-12, 5.77429182e-42],
            [1.00000000e+00, 1.16277818e-10, 4.91955887e-38],
            [1.00000000e+00, 4.63530642e-12, 2.92550374e-41],
            [1.00000000e+00, 3.79052967e-12, 1.71972372e-42],
            [1.00000000e+00, 1.03132660e-11, 1.76987861e-40],
            [1.00000000e+00, 6.19924377e-11, 1.88558864e-39],
            [1.00000000e+00, 3.55895317e-10, 5.22598481e-38],
            [1.00000000e+00, 1.71106645e-12, 2.18273139e-43],
            [1.00000000e+00, 3.64802951e-11, 2.51144860e-39],
            [9.99999640e-01, 3.59678065e-07, 1.02256435e-33],
            [9.99999967e-01, 3.27597136e-08, 2.22429880e-38],
            [1.00000000e+00, 3.14943837e-10, 2.43004914e-40],
            [1.00000000e+00, 1.85572636e-10, 2.85343028e-40],
            [9.99999999e-01, 8.39816491e-10, 1.84944898e-38],
            [1.00000000e+00, 2.05630909e-10, 7.05820825e-41],
            [1.00000000e+00, 8.51314794e-11, 4.56527916e-40],
            [9.99996908e-01, 3.09244169e-06, 3.23039301e-34],
            [3.77421318e-07, 9.99999623e-01, 5.99877817e-23],
            [9.96589983e-01, 3.41001689e-03, 3.16924964e-28],
            [2.47749677e-01, 7.52250323e-01, 3.50021087e-21],
            [2.23985573e-08, 9.99999978e-01, 1.11811694e-22],
            [7.95148817e-11, 1.00000000e+00, 1.61582678e-20],
            [1.77780477e-11, 1.00000000e+00, 3.24421490e-25],
            [2.62750290e-14, 1.00000000e+00, 4.12585891e-22],
            [5.13180065e-15, 1.00000000e+00, 2.53066703e-21],
            [1.12876333e-13, 1.00000000e+00, 1.19619664e-22],
            [1.69089510e-15, 1.00000000e+00, 1.66670226e-20],
            [6.52190823e-14, 1.00000000e+00, 1.11896273e-21],
            [1.97622838e-15, 1.00000000e+00, 1.64599336e-21],
            [2.26064725e-15, 1.00000000e+00, 8.60091338e-21],
            [4.16245727e-14, 1.00000000e+00, 4.71584783e-22],
            [5.64481197e-14, 1.00000000e+00, 1.89703598e-22],
            [2.09189032e-14, 1.00000000e+00, 4.65965047e-22],
            [1.23839885e-13, 1.00000000e+00, 1.00510566e-21],
            [2.56757339e-11, 1.00000000e+00, 1.46484716e-21],
            [4.28126107e-12, 1.00000000e+00, 9.14265116e-22],
            [7.78268199e-11, 1.00000000e+00, 3.35262096e-21],
            [1.29475763e-11, 1.00000000e+00, 1.78388466e-21],
            [2.10541740e-11, 1.00000000e+00, 3.94751892e-21],
            [1.79003831e-10, 1.00000000e+00, 3.66572518e-21],
            [2.75339203e-13, 1.00000000e+00, 1.11702612e-21],
            [8.17096080e-11, 1.00000000e+00, 2.79486435e-21],
            [2.40296216e-11, 1.00000000e+00, 4.86056671e-22],
            [2.13966643e-13, 1.00000000e+00, 4.46709157e-22],
            [5.74952470e-13, 1.00000000e+00, 1.57661757e-21],
            [1.94543380e-13, 1.00000000e+00, 2.20929452e-21],
            [1.66145067e-13, 1.00000000e+00, 2.21435381e-21],
            [2.98168084e-14, 1.00000000e+00, 3.47895160e-21],
            [2.53403908e-13, 1.00000000e+00, 8.43334094e-22],
            [2.76295096e-13, 1.00000000e+00, 1.05103066e-20],
            [1.96120079e-13, 1.00000000e+00, 6.58317665e-21],
            [1.58334348e-13, 1.00000000e+00, 4.11367561e-22],
            [4.65271044e-14, 1.00000000e+00, 1.07585256e-21],
            [8.26614386e-15, 1.00000000e+00, 8.40587745e-21],
            [7.78067023e-13, 1.00000000e+00, 1.65979888e-20],
            [6.51765361e-12, 1.00000000e+00, 9.03144851e-22],
            [5.96886993e-11, 1.00000000e+00, 2.82508757e-21],
            [1.19116166e-11, 1.00000000e+00, 8.58015549e-21],
            [9.59824996e-11, 1.00000000e+00, 1.68779143e-20],
            [7.11524643e-11, 1.00000000e+00, 8.21498921e-21],
            [8.31411727e-12, 1.00000000e+00, 1.24063456e-20],
            [1.15839016e-10, 1.00000000e+00, 1.65510974e-20],
            [1.67707636e-10, 1.00000000e+00, 1.05365835e-19],
            [6.21335790e-11, 1.00000000e+00, 5.38226976e-20],
            [5.25345902e-09, 9.99999995e-01, 2.03507571e-17],
            [1.28616784e-09, 9.99999999e-01, 4.86516390e-18],
            [6.69127523e-07, 9.99999331e-01, 3.78085124e-21],
            [4.12442365e-07, 9.99999588e-01, 9.61091304e-17],
            [9.70120665e-08, 9.99999903e-01, 4.66104933e-16],
            [2.13237866e-10, 1.00000000e+00, 3.15103642e-13],
            [1.78807872e-09, 9.99999998e-01, 1.51177670e-14],
            [7.85072445e-10, 9.99999999e-01, 6.93493280e-15],
            [1.93204858e-10, 1.00000000e+00, 7.59667923e-17],
            [1.60923205e-09, 9.99999998e-01, 7.55898882e-16],
            [5.61013286e-10, 9.99999999e-01, 5.03533501e-14],
            [6.06484730e-10, 9.99999999e-01, 6.32712741e-14],
            [1.00000000e+00, 1.39834015e-11, 4.70523500e-43],
            [9.99999990e-01, 9.55294787e-09, 8.43227143e-39],
            [9.99999611e-01, 3.88729961e-07, 2.14999196e-38],
            [1.00000000e+00, 5.49184335e-11, 1.52999291e-40],
            [9.99999999e-01, 7.05324122e-10, 1.10432442e-38],
            [1.00000000e+00, 9.78238182e-11, 1.65281580e-42],
            [9.99999996e-01, 4.21994219e-09, 2.39448254e-42],
            [1.00000000e+00, 4.79232321e-12, 4.97088307e-46],
            [9.99999997e-01, 2.93079171e-09, 6.79028090e-37],
            [9.99999969e-01, 3.05515304e-08, 4.08284059e-38],
            [9.99391075e-01, 6.08924573e-04, 1.44043643e-28],
            [9.88138779e-01, 1.18612213e-02, 7.66815946e-29],
            [9.44628352e-01, 5.53716475e-02, 2.88615930e-24],
            [9.99999457e-01, 5.42931105e-07, 4.57665221e-39],
            [9.94777551e-01, 5.22244890e-03, 2.48088869e-32],
            [9.99999988e-01, 1.23849040e-08, 7.67636470e-39],
            [9.99999993e-01, 7.16361986e-09, 1.12091270e-39],
            [9.98267424e-01, 1.73257643e-03, 5.38649782e-33],
            [9.99970567e-01, 2.94328695e-05, 7.06986986e-35],
            [9.99972116e-01, 2.78844301e-05, 1.89076052e-34],
            [9.99955373e-01, 4.46274452e-05, 2.50787177e-34],
            [9.99997547e-01, 2.45318050e-06, 3.25346071e-36],
            [9.99849486e-01, 1.50513885e-04, 2.11495147e-33],
            [9.99999574e-01, 4.26024156e-07, 4.34677209e-37],
            [9.93867645e-01, 6.13235461e-03, 1.28180855e-31],
            [9.99999165e-01, 8.34843061e-07, 1.80787007e-36],
            [9.99786979e-01, 2.13021285e-04, 1.02726462e-33],
            [1.00000000e+00, 2.12852908e-12, 4.78495433e-46],
            [9.94349763e-01, 5.65023675e-03, 5.58340471e-32],
            [1.00000000e+00, 4.14684645e-14, 1.42007055e-46],
            [9.66742057e-01, 3.32579428e-02, 3.70874333e-31],
            [9.99999999e-01, 8.01115053e-10, 5.87387861e-39],
            [1.00000000e+00, 1.36230460e-15, 7.49797983e-48],
            [1.00000000e+00, 2.64442930e-12, 6.93639670e-44],
            [9.86955193e-01, 1.30448066e-02, 2.07620977e-31],
            [1.00000000e+00, 2.34028164e-12, 9.70104713e-43],
            [9.44920671e-01, 5.50793293e-02, 1.67186462e-31],
            [1.00000000e+00, 5.67013711e-11, 1.41033435e-44],
            [1.00000000e+00, 1.91500769e-14, 8.99921645e-46],
            [1.00000000e+00, 4.61792281e-14, 3.09452792e-44],
            [1.00000000e+00, 4.08884534e-14, 2.81327995e-45],
            [1.00000000e+00, 1.71549685e-14, 5.04067702e-45],
            [1.00000000e+00, 1.03421571e-14, 5.28735221e-47],
            [1.00000000e+00, 2.53264200e-14, 1.61157556e-46],
            [1.00000000e+00, 2.43254286e-14, 3.80704700e-47],
            [1.00000000e+00, 3.06849657e-13, 4.43787701e-46],
            [1.00000000e+00, 2.03040464e-13, 4.49557477e-42],
            [1.00000000e+00, 1.04686477e-14, 2.22315130e-47],
            [1.00000000e+00, 1.01371311e-14, 4.99805749e-47],
            [1.00000000e+00, 2.22323005e-14, 2.47257721e-46],
            [1.00000000e+00, 7.61300621e-14, 3.45645316e-45],
            [1.00000000e+00, 1.51083602e-14, 6.36746277e-47],
            [1.00000000e+00, 1.21921215e-14, 9.05070786e-48],
            [1.00000000e+00, 4.45813605e-14, 1.53785580e-46],
            [1.00000000e+00, 1.28343351e-13, 1.42011381e-45],
            [1.00000000e+00, 1.53941066e-13, 1.14630143e-45],
            [1.00000000e+00, 1.65116948e-14, 2.23382328e-46],
            [1.00000000e+00, 3.29298173e-12, 5.86717680e-43],
            [1.00000000e+00, 2.66641367e-13, 3.26497114e-45],
            [1.00000000e+00, 3.34584017e-12, 1.82023337e-41],
            [1.00000000e+00, 5.98996974e-14, 2.73810122e-46],
            [1.00000000e+00, 9.44724181e-15, 1.34936734e-48],
            [1.00000000e+00, 6.29167206e-14, 5.25621776e-45],
            [1.00000000e+00, 1.20978841e-14, 9.58677615e-48]
                            ])
        
        
        times = pd.to_datetime( 
                        np.array([
            '2008-06-25T15:40:23.320494080', '2008-06-25T15:41:59.385106432',
            '2008-06-25T15:43:35.449711616', '2008-06-25T15:45:11.514323712',
            '2008-06-25T15:46:47.578928896', '2008-06-25T15:48:23.643540736',
            '2008-06-25T15:49:59.708146176', '2008-06-25T15:51:35.772758272',
            '2008-06-25T15:53:11.837370112', '2008-06-25T15:54:47.901975552',
            '2008-06-25T15:56:23.966580992', '2008-06-25T16:01:12.160374528',
            '2008-06-25T16:02:48.224963840', '2008-06-25T16:06:00.354139392',
            '2008-06-25T16:07:36.418725376', '2008-06-25T16:09:12.483314944',
            '2008-06-25T16:10:48.547900672', '2008-06-25T16:12:24.612490496',
            '2008-06-25T16:14:00.677072384', '2008-06-25T16:15:36.741661952',
            '2008-06-25T16:17:12.806247680', '2008-06-25T16:18:48.870840576',
            '2008-06-25T16:20:24.935422464', '2008-06-25T16:22:01.000007936',
            '2008-06-25T16:23:37.064588544', '2008-06-25T16:25:13.129169664',
            '2008-06-25T16:26:49.193754880', '2008-06-25T16:28:25.258340096',
            '2008-06-25T16:30:01.322925312', '2008-06-25T16:31:37.387506432',
            '2008-06-25T16:33:13.452091648', '2008-06-25T16:34:49.516676864',
            '2008-06-25T16:36:25.581262336', '2008-06-25T16:38:01.645847552',
            '2008-06-25T16:39:37.710420480', '2008-06-25T16:41:13.775009280',
            '2008-06-25T16:42:49.839593216', '2008-06-25T16:44:25.904176896',
            '2008-06-25T16:46:01.968765952', '2008-06-25T16:47:38.033355008',
            '2008-06-25T16:49:14.097938688', '2008-06-25T16:50:50.162522624',
            '2008-06-25T16:52:26.227111680', '2008-06-25T16:54:02.291695360',
            '2008-06-25T16:55:38.356284416', '2008-06-25T16:57:14.420873472',
            '2008-06-25T16:58:50.485460480', '2008-06-25T17:00:26.550050304',
            '2008-06-25T17:02:02.614640384', '2008-06-25T17:03:38.679230464',
            '2008-06-25T17:05:14.743820544', '2008-06-25T17:06:50.808405760',
            '2008-06-25T17:08:26.872995584', '2008-06-25T17:10:02.937585664',
            '2008-06-25T17:11:39.002175488', '2008-06-25T17:13:15.066760704',
            '2008-06-25T17:14:51.131350784', '2008-06-25T17:16:27.195936000',
            '2008-06-25T17:18:03.260525824', '2008-06-25T17:19:39.325106688',
            '2008-06-25T17:21:15.389688576', '2008-06-25T17:22:51.454275072',
            '2008-06-25T17:24:27.518856960', '2008-06-25T17:26:03.583438592',
            '2008-06-25T17:27:39.648020736', '2008-06-25T17:29:15.712607488',
            '2008-06-25T17:30:51.777189120', '2008-06-25T17:32:27.841771008',
            '2008-06-25T17:34:03.906352640', '2008-06-25T17:35:39.970934272',
            '2008-06-25T17:37:16.035516160', '2008-06-25T17:38:52.100089856',
            '2008-06-25T17:40:28.164684544', '2008-06-25T17:42:04.229261056',
            '2008-06-25T17:43:40.293837824', '2008-06-25T17:45:16.358414336',
            '2008-06-25T17:46:52.422991104', '2008-06-25T17:48:28.487586048',
            '2008-06-25T17:50:04.552144384', '2008-06-25T17:51:40.616739328',
            '2008-06-25T17:53:16.681315840', '2008-06-25T17:54:52.745892608',
            '2008-06-25T17:56:28.810469120', '2008-06-25T17:58:04.875045888',
            '2008-06-25T17:59:40.939617024', '2008-06-25T18:01:17.004198400',
            '2008-06-25T18:02:53.068775680', '2008-06-25T18:04:29.133352960',
            '2008-06-25T18:06:05.197934336', '2008-06-25T18:07:41.262511872',
            '2008-06-25T18:09:17.327089152', '2008-06-25T18:10:53.391666432',
            '2008-06-25T18:12:29.456243712', '2008-06-25T18:14:05.520820992',
            '2008-06-25T18:25:00.000000000', '2008-06-25T18:30:00.000000000',
            '2008-06-25T18:35:00.000000000', '2008-06-25T18:40:00.000000000',
            '2008-06-25T18:42:54.683246336', '2008-06-25T18:46:06.812418048',
            '2008-06-25T18:47:42.877004032', '2008-06-25T18:49:18.941589760',
            '2008-06-25T18:50:55.006175744', '2008-06-25T18:54:07.135347712',
            '2008-06-25T18:55:43.199933440', '2008-06-25T18:58:55.329111808',
            '2008-06-25T19:02:07.458284032', '2008-06-25T19:03:43.522873856',
            '2008-06-25T19:05:19.587459840', '2008-06-25T19:06:55.652045824',
            '2008-06-25T19:10:07.781221376', '2008-06-25T19:11:43.845807360',
            '2008-06-25T19:13:19.910396928', '2008-06-25T19:14:55.974982912',
            '2008-06-25T19:16:32.039572736', '2008-06-25T19:18:08.104162560',
            '2008-06-25T19:19:44.168758272', '2008-06-25T19:21:20.233345280',
            '2008-06-25T19:22:56.297932288', '2008-06-25T19:24:32.362522880',
            '2008-06-25T19:27:44.491700480', '2008-06-25T19:29:20.556287744',
            '2008-06-25T19:30:56.620878336', '2008-06-25T19:32:32.685468672',
            '2008-06-25T19:34:08.750059008', '2008-06-25T19:35:44.814649600',
            '2008-06-25T19:37:20.879239936', '2008-06-25T19:38:56.943824896',
            '2008-06-25T19:40:32.649315072', '2008-06-25T19:42:08.713924864',
            '2008-06-25T19:43:44.778534656', '2008-06-25T19:46:56.907750144',
            '2008-06-25T19:48:32.972359936', '2008-06-25T19:50:09.036965888',
            '2008-06-25T19:51:45.101575424', '2008-06-25T19:53:21.166189056',
            '2008-06-25T19:54:57.230798848', '2008-06-25T19:56:33.295408384',
            '2008-06-25T19:58:09.360018176', '2008-06-25T19:59:45.424631808',
            '2008-06-25T20:01:21.489236480', '2008-06-25T20:02:57.553853696',
            '2008-06-25T20:04:33.618468352', '2008-06-25T20:06:09.683079424',
            '2008-06-25T20:07:45.747687936', '2008-06-25T20:09:21.812292864',
            '2008-06-25T20:10:57.876894464', '2008-06-25T20:12:33.941492992',
            '2008-06-25T20:14:10.006088448', '2008-06-25T20:15:46.070680832'
                        ], dtype='datetime64[ns]')
                            )
        
        
        first_crossing_table = CrossingTable(x_ind=11, y_ind=45,
                                             q_ind=17, w_ind=36,
                                             a_ind=26, b_ind=27).get_array()
        
        second_crossing_table = CrossingTable(x_ind=67, y_ind=95,
                                              q_ind=76, w_ind=93,
                                              a_ind=85, b_ind=86).get_array()
        
        expected_result = [ first_crossing_table,
                            second_crossing_table ]
        
        
        
        #### check if algo matches expected result
        compare_expected_vs_algo_result(
                expected_result,
                times = times,
                preds = preds,
                min_prob = 0.8,
                min_cluster_frac = 0.7,
                min_crossing_duration = pd.Timedelta(1, unit="min"),
                max_crossing_duration = pd.Timedelta(10, unit="min"),
                min_beyond_crossing_duration = pd.Timedelta(15, unit="min"),
                max_beyond_crossing_duration = pd.Timedelta(30, unit="min"),
                overlap_preference = "best"
                                        )
    
    
    
    
    
    
    
    no_crossings_test()
    single_immediate_crossing_all_pts_in_crossing_test()
    single_immediate_crossing_some_pts_in_crossing_test()
    single_gradual_crossing_all_pts_in_crossing_test()
    single_gradual_crossing_some_pts_in_crossing_test()
    gradual_then_immediate_no_overlap()
    gradual_then_immediate_xy_overlap()
    gradual_then_immediate_xw_overlap()
    gradual_then_immediate_qw_overlap()
    single_immediate_too_large_ab_gap()
    single_gradual_too_large_ab_gap()
    single_immediate_timegap_in_qw_right()
    single_immediate_timegap_in_qw_left()
    single_immediate_timegap_in_xy_right()
    single_immediate_timegap_in_xy_left()
    single_immediate_too_small_ab_gap()
    multi_interval_test_from_tha_data()
    ## make test where there are no xy, only qw and ab
    ## make test where there are no xy OR qw, only ab
    print("All compute_crossings_soft tests passed!")















def _test_build_crossing_table():
    
    """
    
    Tests the build_crossing_table function
    
    """
    
    
    
    
    def basic_test():
        
        """
        
        Simple test
        
        """
        
        ## Build expected result
        q_ind, w_ind = 2, 8
        a_ind, b_ind = 5, 6
        inds_arr = np.arange(10+1)
        crossing_pts_arr = np.full(inds_arr.shape[0], 0)
        crossing_pts_arr[ np.arange(q_ind,w_ind+1) ] = 1
        crossing_pts_arr[ np.arange(a_ind,b_ind+1) ] = 2
        expected_result = np.vstack( [inds_arr,
                                      crossing_pts_arr] ).T
        
        ## Get algo result
        the_table = CrossingTable(
                            x_ind=0, y_ind=inds_arr.shape[0]-1,
                            q_ind=q_ind, w_ind=w_ind,
                            a_ind=a_ind, b_ind=b_ind
                                 )
        algo_result = the_table.get_array()
        
        ## Compare algo vs expected result
        try:
            assert( np.array_equal(expected_result, algo_result) )
        except AssertionError:
            print("Failed!")
            print("Using ...")
            print("  --q_ind--:",q_ind)
            print("  --w_ind--:",w_ind)
            print("  --a_ind--:",a_ind)
            print("  --b_ind--:",b_ind)
            print("--expected result--:",expected_result)
            print("--but got--:",algo_result)
            raise
        
        
        
        
    
    basic_test()
    print("All build_crossing_table tests passed!")
    
                                           







if __name__ == "__main__":
    print("Running utils.py tests...")
    
    print("    Testing find_possible_crossing_intervals_soft: ", end="")
    _test_find_possible_crossing_intervals_soft()
    
    print("    Testing build_crossing_table: ", end="")
    _test_build_crossing_table()
    
    print("    Testing compute_crossings_soft: ", end="")
    _test_compute_crossings_soft()
    
    