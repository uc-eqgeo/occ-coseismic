# searise-coseismic
First attempt at calculating uplift for the national seismic hazard model. Requires Wellington area fault mesh files.

## Scripts
Scripts are all in the `searise-coseismic` directory. The main scripts that should be run in the following 
order:
1. `crustal/subset_fault_sections_by_mesh_v3.py` This script extracts the fault sections from the NSHM that are located 
   along meshes in the Wellington area. Uses keywords that you have to input. Outputs a geojson with filtered traces.
2. `crustal/discretize_crustal_remeshed.py` This script reads in fault traces from the NSHM, turns them into retangular 
   patches with metadata, and matches each patch to triangles in the finer mesh. Inputs trace shapefiles with from 
   subset_scenarios_by_mesh.py or from all  traces from the NSHM.
3. choose one or more greens function scripts:
   1. `crustal/crustal_discretized_grid_gfs.py` Calculates greens functions from the discretized fault patches on a 
      regional grid and writes dictionary to file.
   2. `crustal/crustal_discretized_coastal_gfs.py` Calculates greens functions from the discretized fault patches on a 
      points along the coastline and writes dictionary to file.
   3. `crustal/crustal_discretized_sites_gfs.py` Calculates greens functions from the discretized fault patches at 
      specific coordinates (sites) and writes dictionary to file.
 
The following scripts are in the "crustal_sz_shared_scripts" directory. 

4. `crustal_sz_shared_scripts/run_displacements_and_probabilities.py` Calculates surface displacements using existing 
Green's functions, calculates probabilities of exceedence, and writes to pickle files. Options within that script to 
   make plots for displacement and probabilities.

Files/folders you need to run the above scripts:
1. `data/crustal_solutions/{NSHM solution directory}` Solution and crustal fault data from the NSHM.
2. `data/wellington_alt_geom/meshes{model_extension}/STL_remeshed` Meshes for the Wellington area.
3. `data/wellington_alt_geom/alt_geom_rakes.csv` Rakes for crustal fault meshes

Optional scripts:
6. `crustal_sz_shared_scripts/run_compare_branches.py` 
7. `crustal_sz_shared_scripts/run_aggregate_weighted_branches.py` Calculates the weighted means of all the branches 
   in the logic tree. Options to make plots.
