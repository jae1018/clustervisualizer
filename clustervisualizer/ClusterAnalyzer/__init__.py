#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 13:42:27 2022

@author: jedmond
"""

## from (folder).(python_file) import (class)  - I think this is right?

##from ClusterAnalyzer import ClusterAnalyzer
##from ClusterAnalyzer.ClusterAnalyzer import ClusterAnalyzer
from . import ClusterAnalyzer
#from SCDataProc.sc_constants import sc_constants
#from SCDataProc.sc_defs import sc_defs
#import SCDataProc.utils as utils
##import ClusterAnalyzer.utils as utils
from . import utils
__all__ = [ "ClusterAnalyzer", "utils"]