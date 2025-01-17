# searise-coseismic
First attempt at calculating uplift for the national seismic hazard model.

## Scripts
Scripts are all in the `subduction_discretize` directory. There are three main scripts that should be run in the following order:
1. `subduction_discretize/discretize_subduction.py` This script reads in the rectangular patches from the NSHM and 
   matches each patch to triangles in the finer mesh.
2. choose where to calculate greens functions and dispalcements:
   1. `subduction_discretize/discretized_gfs_grid.py` Calculates greens functions on a grid
   2. `subduction_discretize/discretized_gfs_coastline.py` Calculates greens functions along the coastline
   3. `subduction_discretize/discretized_gfs_sites.py` Calculates greens functions at specific coordinates
