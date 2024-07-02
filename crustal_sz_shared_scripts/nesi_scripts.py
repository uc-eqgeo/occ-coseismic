import os
import argparse
import pickle as pkl
import numpy as np
from time import time

def prep_cumu_PPE_NESI(model_version_results_directory, branch_site_disp_dict, extension1, 
                       hours : int = 0, mins: int= 3, mem: int= 40, cpus: int= 1, account: str= 'uc03610',
                       time_interval: int = 100, n_samples: int = 1000000, sd: float = 0.4, S=""):
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

    os.makedirs(f"../{model_version_results_directory}/{extension1}/site_cumu_exceed{S}/nesi_scripts", exist_ok=True)

    with open(f"../{model_version_results_directory}/{extension1}/site_cumu_exceed{S}/site_name_list.txt", "w") as f:
        for site in sites_of_interest:
            f.write(f"{site}\n")
    
    for site_of_interest in sites_of_interest:
        with open(f"../{model_version_results_directory}/{extension1}/site_cumu_exceed{S}/nesi_scripts/{site_of_interest}.sl", "wb") as f:
            f.write(f"#!/bin/bash -e\n".encode())
            f.write(f"#SBATCH --job-name=occ-{site_of_interest}\n".encode())
            f.write(f"#SBATCH --time={hours:02}:{mins:02}:00      # Walltime (HH:MM:SS)\n".encode())
            f.write(f"#SBATCH --mem-per-cpu={mem}GB\n".encode())
            f.write(f"#SBATCH --cpus-per-task={cpus}\n".encode())
            f.write(f"#SBATCH --account={account}\n".encode())

            f.write(f"#SBATCH -o logs/{site_of_interest}_%j.out\n".encode())
            f.write(f"#SBATCH -e logs/{site_of_interest}_%j.err\n\n".encode())

            f.write(f"# Activate the conda environment\n".encode())
            f.write(f"mkdir -p logs\n".encode())
            f.write(f"module purge && module load Miniconda3\n".encode())
            f.write(f"module load Python/3.11.3-gimkl-2022a\n\n".encode())

            if S:
                f.write(f"python nesi_scripts.py --site {site_of_interest} --branchdir {f'../{model_version_results_directory}/{extension1}'} --time_interval {int(time_interval)} --n_samples {int(n_samples)} --sd {sd}\n\n".encode())
            else:
                f.write(f"python nesi_scripts.py --site {site_of_interest} --branchdir {f'../{model_version_results_directory}/{extension1}'} --time_interval {int(time_interval)} --n_samples {int(n_samples)} --sd {sd} --scaling {S}\n\n".encode())

            f.write(f"# to call:\n".encode())
            f.write(f"# sbatch slurm_example.sl\n".encode())

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
    parser = argparse.ArgumentParser(description="Script to calculate cumulative exceedance probabilities for each site")
    parser.add_argument("--site", type=str, required=True, help="Site to calculate exceedance probabilities for")
    parser.add_argument("--branchdir", type=str, required=True, help="Directory of the branch results")
    parser.add_argument("--time_interval", type=int, default=100, help="Time interval to calculate exceedance probabilities over")
    parser.add_argument("--n_samples", type=int, default=1e6, help="Number of samples to use for the poissonian simulation")
    parser.add_argument("--sd", type=float, default=0.4, help="Standard deviation of the normal distribution to use for uncertainty in displacements")
    parser.add_argument("--scaling", type=str, default="", help="Scaling factor for the displacements")
    args = parser.parse_args()

    start = time()
    site_of_interest = args.site
    branch_results_directory = args.branchdir
    investigation_time = args.time_interval
    n_samples = args.n_samples
    sd = args.sd
    scaling = args.scaling

    rng = np.random.default_rng()

    extension1 = os.path.basename(branch_results_directory)

    with open(f"{branch_results_directory}/branch_site_disp_dict_{extension1}.pkl", "rb") as fid:
            branch_disp_dict = pkl.load(fid)

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