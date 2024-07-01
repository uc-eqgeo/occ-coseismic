import random
import geopandas as gpd
import pandas as pd
import os
import itertools
import pickle as pkl
import matplotlib.ticker as mticker
import rasterio
from rasterio.transform import Affine

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from helper_scripts import get_figure_bounds, make_qualitative_colormap, tol_cset, get_probability_color
from matplotlib.patches import Rectangle
from weighted_mean_plotting_scripts import get_mean_prob_barchart_data, get_mean_disp_barchart_data

matplotlib.rcParams['pdf.fonttype'] = 42

def prep_cumu_PPE_NESI(model_version_results_directory, branch_site_disp_dict, extension1, 
                       hours : int = 0, mins: int= 15, mem: int= 2, cpus: int= 1, account: str= 'uc03610'):
    """
    Must first run get_site_disp_dict to get the dictionary of displacements and rates

    inputs: runs for one logic tree branch
    Time_interval is in years

    function: calculates the poissonian probability of exceedance for each site for each displacement threshold value

    outputs: pickle file with probability dictionary (probs, disps, site_coords)

    CAVEATS/choices:
    - need to decide on number of 100-yr simulations to run (n_samples = 1000000)
    """

    sites_of_interest = list(branch_site_disp_dict.keys())

    os.makedirs(f"../{model_version_results_directory}/{extension1}/site_cumu_exceed/nesi_scripts", exist_ok=True)

    with open(f"../{model_version_results_directory}/{extension1}/site_cumu_exceed/site_name_list.txt", "w") as f:
        for site in sites_of_interest:
            f.write(f"{site}\n")
    
    for site_of_interest in sites_of_interest:
        with open(f"../{model_version_results_directory}/{extension1}/site_cumu_exceed/nesi_scripts/{site_of_interest}.sl", "wb") as f:
            f.write(b"#!/bin/bash -e\n")
            f.write(b"#SBATCH", f"--job-name=cutdeA100Mw8 # job name (shows up in the queue)\n")
            f.write(b"#SBATCH", f"--time={hours:02}:{mins:02}:00      # Walltime (HH:MM:SS)\n")
            f.write(b"#SBATCH", f"--mem-per-cpu={mem}GB\n")
            f.write(b"#SBATCH", f"--cpus-per-task={cpus}\n")
            f.write(b"#SBATCH", f"--account={account}\n")

            f.write(b"#SBATCH", f" -o logs/{site_of_interest}_%j.out\n")
            f.write(b"#SBATCH", f" -e logs/{site_of_interest}_%j.err\n")

            f.write(b"# Activate the conda environment\n")
            f.write(f"mkdir -p logs\n")
            f.write(f"module purge && module load Miniconda3\n")
            f.write(f"module load Python/3.11.3-gimkl-2022a\n")

            f.write(f"python NESI_displacements.py  --grdDir /nesi/nobackup/uc03610/jack/aotearoa/grds_5km --eventCSV whole_nz_8Mw.csv\n")

            f.write(b"# to call:\n")
            f.write(b"# sbatch slurm_example.sl\n")