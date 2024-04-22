# searise-coseismic
First attempt at calculating uplift for the national seismic hazard model. Requires Wellington area fault mesh files.

## Scripts
Scripts are all in the `occ-coseismic` directory. The main scripts should be run in the following 
order:
A. Prep crustal fault meshes and greens functions. The results are put into a new folder in the crustal directory
1. `crustal/subset_fault_sections_by_mesh.py` This script extracts the fault sections from the NSHM that are 
   located 
   along meshes in the Wellington area. Uses keywords that you have to input. Outputs a geojson with filtered traces.
2. `crustal/discretize_crustal.py` This script reads in fault traces from the NSHM, turns them into rectangular 
   patches with metadata, and matches each patch to triangles in the finer mesh. Inputs trace shapefiles from 
   subset_scenarios_by_mesh.py or from all  traces from the NSHM.
3. choose one or more Green's function scripts:
   1. `crustal/crustal_discretized_grid_gfs.py` Calculates greens functions from the discretized fault patches on a 
      regional grid and writes dictionary to file.
   2. `crustal/crustal_discretized_coastal_gfs.py` Calculates greens functions from the discretized fault patches 
      on a points along the coastline and writes dictionary to file.
   3. `crustal/crustal_discretized_sites_gfs.py` Calculates greens functions from the discretized fault patches at 
      specific coordinates and writes dictionary to file.
   
B. subduction zone file prep. The results are put in a new folder in the subduction zone directory
4. `subduction/discretize_subduction.py` This script reads in fault traces from the NSHM, turns them into 
   rectangular patches with metadata, and matches each patch to triangles in the mesh. Outputs a dictionary.
5. Choose where to calculate greens functions and displacements:
   1. `subduction_discretize/discretized_gfs_grid.py` Calculates greens functions on a grid
   2. `subduction_discretize/discretized_gfs_coast.py` Calculates greens functions along the coastline
   3. `subduction_discretize/discretized_gfs_sites.py` Calculates greens functions at specific coordinates

calculate scenario displacements and probabilities
6. `crustal_sz_shared_scripts/run_displacements_and_probabilities.py` Calculates displacements for each scenario and writes 
   pickle to results directory. Can run multiple branches at the same time IF they are the same fault type and 
   Greeen's function type (e.g., sites, grid)
   

