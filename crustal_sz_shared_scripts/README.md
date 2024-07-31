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
6. `crustal_sz_shared_scripts/run_aggregate_weighted_branches.py` Calculates the displacements for each branch of the 
   NSHM decision tree. Once this has been done for each branch, it can then combine branches (either all the crustal
   faults, all the subduction interface (Hik or Puy), or all crustal and either/or hik/puy, or all crustal, hik, and py).
   Designed to be run either locally or on NESI as task arrays.
   Preparing single branches:
   If running locally (nesi = False):
   a) calculate_fault_model_PPE = True (Will create scenarios for each branch individually)
   If running with nesi task arrays
   a1) nesi = True, nesi_step = prep, calculate_fault_model_PPE = True (Will calculate scenarios at each site for each branch individually)
   a2) nesi = True, nesi_step = combine, calculate_fault_model_PPE = True (Will combine all site scenarios into a single branch h5 file)
   a3) nesi = False, calculate_fault_model_PPE = True (Create all_branch_PPE_dict (needed for weighted))

   Creating weighted means
   If running locally
   b1) calculate_weighted_mean_PPE = True, paired_crustal_sz = False (Will create combined weighted scenarios for single fault type)
   b2) calculate_weighted_mean_PPE = True, paired_crustal_sz = True (Will create combined weighted scenarios for crustal + sub fault types)

   If running with nesi task arrays
   b1a) nesi = True, nesi_step = prep, calculate_weighted_mean_PPE = True, paired_crustal_sz = False (Create weighted means at for each site for one fault type)
   b1b) nesi = True, nesi_step = combine, calculate_weighted_mean_PPE = True, paired_crustal_sz = False (Combine all site weighted means into one fault type dictionary)
   b2a) nesi = True, nesi_step = prep, calculate_fault_model_PPE = True, paired_crustal_sz = True (Create weighted means at for each site for for crustal + sub fault types)
   b2a) nesi = True, nesi_step = combine, calculate_fault_model_PPE = True, paired_crustal_sz = True (Combine all site weighted means into one fault type dictionary)

