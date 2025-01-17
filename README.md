## OCC Coseismic
THIS IS THE MARKDOWN
This software is designed to generate a Probabilistic Coseismic Displacement Hazard Model (PCDHM) using the outputs of the New Zealand National Seismic Hazard Model (NZ NSHM). 
This is a national scaling of the Wellington case study (Delano et al., In Prep).
Seismicity rates are based off different fault models (crustal faults, hikurangi-kermadec subduction interface, puysegur subduction interface).

## Required Data
Fault and NSHM data must be in the 'Data' Folder

- ./Data/
    - branch_weight_data.xlsx : Spreadsheet, with different sheets for each fault model. Must contain columns containing N, b, C, S, def_model, time_dependence, and total weight RN
    - NSHM fault stuff

Site data
- ./{sites}.csv : CSV containing points of every site to be processed

## Processing

# Data Preparation
Initial processing is to identify the relevant faults and fault ruptures, and create site-specific greens functions for the vertical displacement associated with each rupture.
Crustal and subduction fault models must first be processed independently from within their respective directories

- Crustal Faults
    1) `./crustal/subset_fault_sections_by_mesh.py` This script extracts the fault sections from the NSHM that are located within a given search area.
        Specific faults can be requested or excluded.
        Outputs a geojson with filtered traces.
    2) `./crustal/discretize_crustal.py` This script reads in fault traces from the NSHM, turns them into rectangular patches with metadata, and matches each patch to triangles in the finer mesh. 
    Inputs trace shapefiles from subset_scenarios_by_mesh.py or from all  traces from the NSHM.
    3) `./crustal/crustal_discretized_sites_gfs.py` Calculates greens functions from the discretized fault patches at specific coordinates and writes dictionary to file.

- Subduction Interface
    1)  `./subduction/discretize_subduction.py` This script reads in fault traces from the NSHM, turns them into rectangular patches with metadata, and matches each patch to triangles in the mesh.
    Outputs a dictionary.
    2) `./subduction_discretize/discretized_gfs_sites.py` Calculates greens functions at specific coordinates.

# Calculate Probabilities
    1) `./crustal_sz_shared_scripts/run_displacements_and_probabilities.py` Calculates displacements for each scenario and writes pickle to results directory. Can run multiple branches at the same time IF they are the same fault type
    2) `./crustal_sz_shared_scripts/run_aggregate_weighted_branches.py` Calculates weighted mean PCDHM from the displacements of all scenarios