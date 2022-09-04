Beam System Analysis
====================

A semi-automated scan-to-bim toolkit for reconstructing beam-and-column systems from point clouds. 

Requirements
------------
Tested using an Anaconda environment with the following packages:

alphashape  
matplotlib  
networkx  
numpy  
open3d v12+  
opencv  
progressbar  
scikit-image  
scipy  
shapely  
tkinter


Usage
-----
The tool expects a point cloud in a Open3d readable format (.xyz, .pts, .ply, .pcd), scaled in mm units, aligned to the world axes.

On running, the tool will open a dialog to select the point cloud. By default it will look in the last directory chosen.

The tool will create a directory with the same name as the input file to store output and diagram files.

Settings for analysis method, displayed geometry, and exported files are stored in the user_settings.ini file. Default analysis thresholds are as described in the publication. 