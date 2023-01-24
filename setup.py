#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 15:45:08 2021

@author: jedmond
"""

from setuptools import setup, find_packages

#### TO DO


setup(
      name = "clustervisualizer",
      #version = __version__,
      author = "James Edmond",
      author_email = "edmondandy795@gmail.com",
      description = "Visualize clustered data.",
      #long_description = long_description,
      #long_description_content_type = "text/markdown",
      url = "https://github.com/jae1018/clustervisualizer",
      license = "MIT",
      packages = ["clustervisualizer",
		  "clustervisualizer.ClusterAnalyzer",
                  "clustervisualizer.CrossingAnalyzer"],
      install_requires = ["numpy>=1.8",
                          "scipy>=1.4.1", 
                          "matplotlib>=3.1.1",
                          "pandas>=1.4"],
      python_requires = '>=3.6',
      version = '0.1'
	)

#setup(
#    name='Unsupervised_THEMIS_Clustering_Project',
#    version="1.0",
#    packages=find_packages(),
#)
