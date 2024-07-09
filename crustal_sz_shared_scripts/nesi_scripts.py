import os
import argparse
import pickle as pkl
import numpy as np
import pandas as pd
from time import time

def prep_nesi_site_list(model_version_results_directory, branch_site_disp_dict, extension1, S=""):
    """
    Must first run get_site_disp_dict to get the dictionary of displacements and rates

    inputs: runs for one logic tree branch
    Time_interval is in years

    function: calculates the poissonian probability of exceedance for each site for each displacement threshold value

    outputs: pickle file with probability dictionary (probs, disps, site_coords)

    CAVEATS/choices:
    - need to decide on number of 100-yr simulations to run (n_samples = 100000)
    """

    sites_of_interest = list(branch_site_disp_dict.keys())

    branchdir = f"{model_version_results_directory}/{extension1}"

    os.makedirs(f"{branchdir}/site_cumu_exceed{S}", exist_ok=True)

    site_file = f"../{model_version_results_directory}/site_name_list.txt"

    if S == "":
        S='_'

    # Append site information for this branch to the main list
    with open(site_file, "a") as f:
        for site in sites_of_interest:
            f.write(f"{site} {branchdir} {S}\n")

def prep_SLURM_submission(model_version_results_directory, tasks_per_array, n_tasks,
                          hours: int = 0, mins: int= 3, mem: int= 45, cpus: int= 1, account: str= 'uc03610',
                          time_interval: int = 100, n_samples: int = 1000000, sd: float = 0.4):
    """
    Must first run get_site_disp_dict to get the dictionary of displacements and rates

    inputs: runs for one logic tree branch
    Time_interval is in years

    function: calculates the poissonian probability of exceedance for each site for each displacement threshold value

    outputs: pickle file with probability dictionary (probs, disps, site_coords)

    CAVEATS/choices:
    - need to decide on number of 100-yr simulations to run (n_samples = 1000000)
    """

    site_file = f"../{model_version_results_directory}/site_name_list.txt"

    with open(site_file, "r") as f:
        all_sites = f.read().splitlines()

    with open(f"../{model_version_results_directory}/cumu_PPE_slurm_task_array.sl", "wb") as f:
        f.write(f"#!/bin/bash -e\n".encode())
        f.write(f"#SBATCH --job-name=occ-{os.path.basename(model_version_results_directory)}\n".encode())
        f.write(f"#SBATCH --time={hours:02}:{mins:02}:00      # Walltime (HH:MM:SS)\n".encode())
        f.write(f"#SBATCH --mem={mem}GB\n".encode())
        f.write(f"#SBATCH --cpus-per-task={cpus}\n".encode())
        f.write(f"#SBATCH --account={account}\n".encode())
        f.write(f"#SBATCH --partition=large\n".encode())
        #f.write(f"#SBATCH --array=1-{len(all_sites)}\n".encode())
        f.write(f"#SBATCH --array=0-{n_tasks-1}\n".encode())

        #f.write(f"#SBATCH -o logs/{os.path.basename(model_version_results_directory)}_site%a_%j.out\n".encode())
        #f.write(f"#SBATCH -e logs/{os.path.basename(model_version_results_directory)}_site%a_%j.err\n\n".encode())
        f.write(f"#SBATCH -o logs/{os.path.basename(model_version_results_directory)}_task%a_%j.out\n".encode())
        f.write(f"#SBATCH -e logs/{os.path.basename(model_version_results_directory)}_task%a_%j.err\n\n".encode())

        f.write(f"# Activate the conda environment\n".encode())
        f.write(f"mkdir -p logs\n".encode())
        f.write(f"module purge  2>/dev/null\n".encode())
        f.write(f"module load Python/3.11.6-foss-2023a\n\n".encode())

        #f.write(f"python nesi_scripts.py --site `awk 'FNR == ENVIRON[\"SLURM_ARRAY_TASK_ID\"] {{print $1}}' {site_file}` --branchdir `awk 'FNR == ENVIRON[\"SLURM_ARRAY_TASK_ID\"] {{print $2}}' {site_file}` --time_interval {int(time_interval)} --n_samples {int(n_samples)} --sd {sd} --scaling `awk 'FNR == ENVIRON[\"SLURM_ARRAY_TASK_ID\"] {{print $3}}' {site_file}`\n\n".encode())
        f.write(f"python nesi_scripts.py --task_number $SLURM_ARRAY_TASK_ID --tasks_per_array {tasks_per_array} --site_file {site_file} --time_interval {int(time_interval)} --n_samples {int(n_samples)} --sd {sd} \n\n".encode())


def compile_site_cumu_PPE(branch_site_disp_dict, model_version_results_directory, extension1, taper_extension="", S="", return_dict=False):
    """
    Script to recompile all individual site PPE dictionaries into a single branch dictionary
    """

    sites = branch_site_disp_dict.keys()
    site_PPE_dict = {}

    if 'grid_meta' in sites:
        sites.remove('grid_meta')

    for site_of_interest in sites:
        with open(f"../{model_version_results_directory}/{extension1}/site_cumu_exceed{S}/{site_of_interest}.pkl", "rb") as f:
            single_site_dict = pkl.load(f)
        site_PPE_dict[site_of_interest] = single_site_dict
        os.remove(f"../{model_version_results_directory}/{extension1}/site_cumu_exceed{S}/{site_of_interest}.pkl")
    os.rmdir(f"../{model_version_results_directory}/{extension1}/site_cumu_exceed{S}")

    if 'grid_meta' in branch_site_disp_dict.keys():
            site_PPE_dict['grid_meta'] = branch_site_disp_dict['grid_meta']

    if not return_dict:
        with open(f"../{model_version_results_directory}/{extension1}/cumu_exceed_prob_{extension1}"
              f"{taper_extension}.pkl", "wb") as f:
            pkl.dump(site_PPE_dict, f)

    else:
        return site_PPE_dict

if __name__ == "__main__":
    # Import here to prevent circular imports
    from probabalistic_displacement_scripts import get_cumu_PPE, nesi_print

    parser = argparse.ArgumentParser(description="Script to calculate cumulative exceedance probabilities for each site")
    #parser.add_argument("--site", type=str, required=True, help="Site to calculate exceedance probabilities for")
    #parser.add_argument("--branchdir", type=str, required=True, help="Directory of the branch results")
    parser.add_argument("--task_number", type=int, required=True, help="Task number for the SLURM array")
    parser.add_argument("--tasks_per_array", type=int, required=True, help="Number of tasks per SLURM array")
    parser.add_argument("--site_file", type=str, required=True, help="File containing the site information")
    parser.add_argument("--time_interval", type=int, default=100, help="Time interval to calculate exceedance probabilities over")
    parser.add_argument("--n_samples", type=int, default=1e5, help="Number of samples to use for the poissonian simulation")
    parser.add_argument("--sd", type=float, default=0.4, help="Standard deviation of the normal distribution to use for uncertainty in displacements")
    parser.add_argument("--scaling", type=str, default="", help="Scaling factor for the displacements")
    parser.add_argument("--slip_taper", default=False, action='store_true', help="Tapered slip distribution")
    args = parser.parse_args()

    """
    start = time()
    site_of_interest = args.site
    branch_results_directory = args.branchdir
    investigation_time = args.time_interval
    n_samples = args.n_samples
    sd = args.sd
    if args.scaling == '_':
        scaling = ""
    else:
        scaling = args.scaling

    print(f"Running site {site_of_interest}....")

    rng = np.random.default_rng()

    extension1 = os.path.basename(branch_results_directory)

    with open(f"{branch_results_directory}/branch_site_disp_dict_{extension1}.pkl", "rb") as fid:
            branch_disp_dict = pkl.load(fid)

    # Additional check for if keys are integers
    if isinstance([key for key in branch_disp_dict.keys()][0], int):
        if '_' not in site_of_interest:
            site_of_interest = int(site_of_interest)

    site_dict_i = branch_disp_dict[site_of_interest]

    del branch_disp_dict

    if "scaled_rates" not in site_dict_i.keys():
        # if no scaled_rate column, assumes scaling of 1 (equal to "rates")
        scaled_rates = site_dict_i["rates"]
    else:
        scaled_rates = site_dict_i["scaled_rates"]

    # average number of events per time interval (effectively R*T from Ned's guide)
    lambdas = investigation_time * np.array(scaled_rates)

    # Generate n_samples of possible earthquake ruptures for random 100 year intervals
    # returns boolean array where 0 means "no event" and 1 means "event". rows = 100 yr window, columns = earthquake
    # rupture
    scenarios = rng.poisson(lambdas, size=(int(n_samples), lambdas.size))

    # assigns a normal distribution with a mean of 1 and a standard deviation of sd
    # effectively a multiplier for the displacement value
    disp_uncertainty = rng.normal(1., sd, size=(int(n_samples), lambdas.size))

    # for each 100 yr scenario, get displacements from EQs that happened
    disp_scenarios = scenarios * site_dict_i["disps"]
    # multiplies displacement by the uncertainty multiplier
    disp_scenarios = disp_scenarios * disp_uncertainty
    # sum all displacement values at that site in that 100 yr interval
    cumulative_disp_scenarios = disp_scenarios.sum(axis=1)

    # get displacement thresholds for calculating exceedance (hazard curve x axis)
    thresholds = np.arange(0, 3, 0.01)
    thresholds_neg = thresholds * -1
    # sum all the displacements in the 100 year window that exceed threshold
    # n_exceedances_total = np.zeros_like(thresholds)
    n_exceedances_total_abs = np.zeros_like(thresholds)
    n_exceedances_up = np.zeros_like(thresholds)
    n_exceedances_down = np.zeros_like(thresholds)
    # for threshold value:
    for threshold in thresholds:
        # replaces index in zero array with the number of times the cumulative displacement exceeded the threshold
        # across all of the 100 yr scenarios

        # sums the absolute value of the disps if the abs value is greater than threshold. e.g., -0.5 + 0.5 = 1
        n_exceedances_total_abs[thresholds == threshold] = (np.abs(cumulative_disp_scenarios) > threshold).sum()
        n_exceedances_up[thresholds == threshold] = (cumulative_disp_scenarios > threshold).sum()
    for threshold in thresholds_neg:
        n_exceedances_down[thresholds_neg == threshold] = (cumulative_disp_scenarios < threshold).sum()

    # the probability is the number of times that threshold was exceeded divided by the number of samples. so,
    # quite high for low displacements (25%). Means there's a ~25% chance an earthquake will exceed 0 m in next 100
    # years across all earthquakes in the catalogue (at that site).
    exceedance_probs_total_abs = n_exceedances_total_abs / n_samples
    exceedance_probs_up = n_exceedances_up / n_samples
    exceedance_probs_down = n_exceedances_down / n_samples

    # CAVEAT: at the moment only absolute value thresholds are stored, but for "down" the thresholds are
    # actually negative.
    single_site_dict = {"thresholds": thresholds,
                                    "exceedance_probs_total_abs": exceedance_probs_total_abs,
                                    "exceedance_probs_up": exceedance_probs_up,
                                    "exceedance_probs_down": exceedance_probs_down,
                                    "site_coords": site_dict_i["site_coords"],
                                    "standard_deviation": sd}

    with open(f"{branch_results_directory}/site_cumu_exceed{scaling}/{site_of_interest}.pkl", "wb") as f:
        pkl.dump(single_site_dict, f)
    
    print(f"Time taken: {time() - start:.2f} seconds")
    print(f"Site: {site_of_interest} complete")
    """

    """
    start = time()
    site_of_interest = args.site
    branch_results_directory = args.branchdir
    investigation_time = args.time_interval
    n_samples = args.n_samples
    sd = args.sd
    if args.scaling == '_' or args.scaling == '_\r':
        scaling = ""
    else:
        scaling = args.scaling

    print(f"Running site {site_of_interest}....")

    extension1 = os.path.basename(branch_results_directory)

    with open(f"../{branch_results_directory}/branch_site_disp_dict_{extension1}.pkl", "rb") as fid:
            branch_disp_dict = pkl.load(fid)

    # Additional check for if keys are integers
    if isinstance([key for key in branch_disp_dict.keys()][0], int):
        if '_' not in site_of_interest:
            site_of_interest = int(site_of_interest)

    get_cumu_PPE(args.slip_taper, os.path.dirname(branch_results_directory), branch_disp_dict, [site_of_interest], n_samples,
                 extension1, branch_key="nan", time_interval=100, sd=0.4, scaling=scaling)

    print(f"Time taken: {time() - start:.2f} seconds")
    print(f"Site: {site_of_interest} complete")
    """

    start = time()

    investigation_time = args.time_interval
    n_samples = args.n_samples
    sd = args.sd

    with open(args.site_file, "r") as f:
        all_sites = f.read().splitlines()

    task_sites = all_sites[args.task_number * args.tasks_per_array:(args.task_number + 1) * args.tasks_per_array]

    sites = np.array([site_info.split(" ")[0] for site_info in task_sites])
    branch_directories = np.array([site_info.split(" ")[1] for site_info in task_sites])
    scalings = np.array([site_info.split(" ")[2] for site_info in task_sites])
    site_df = pd.DataFrame(np.vstack([sites, branch_directories, scalings]).T, columns=['Site', 'Branch Directory', 'Scaling'])

    # Group all of these based on scaling and branch
    site_groups = site_df.groupby(['Branch Directory', 'Scaling'])

    for name, group in site_groups:
        begin = time()
        branch_results_directory, scaling = name

        if scaling == '_' or scaling == '_\r':
            scaling = ""
        
        extension1 = os.path.basename(branch_results_directory)
        with open(f"../{branch_results_directory}/branch_site_disp_dict_{extension1}.pkl", "rb") as fid:
            branch_disp_dict = pkl.load(fid)

        sites_of_interest = group['Site'].values

        if isinstance([key for key in branch_disp_dict.keys()][0], int):
            if '_' not in sites_of_interest[0]:
                sites_of_interest = [int(site) for site in sites_of_interest]

        if scaling == "":
            if nesi_print:
                os.system(f"echo Running {len(sites_of_interest)} sites in branch {extension1}...")
            print(f"Running {len(sites_of_interest)} sites in branch {extension1}...")
        else:
            if nesi_print:
                os.system(f"echo Running {len(sites_of_interest)} sites in branch {extension1} with scaling {scaling}...")
            print(f"Running {len(sites_of_interest)} sites in branch {extension1} with scaling {scaling}...")

        # Needs to be run one site at a time so sites can be recombined later
        for site in sites_of_interest:
            get_cumu_PPE(args.slip_taper, os.path.dirname(branch_results_directory), branch_disp_dict, [site], n_samples,
                    extension1, branch_key="nan", time_interval=investigation_time, sd=sd, scaling=scaling)

        if nesi_print:
            os.system(f"echo {extension1} complete in : {time() - begin:.2f} seconds\n")
        print(f"{extension1} complete in : {time() - begin:.2f} seconds\n")
    
    print(f"All sites complete in {time() - start:.2f} seconds (Average {(time() - start) / len(task_sites):.2f} seconds per site)")