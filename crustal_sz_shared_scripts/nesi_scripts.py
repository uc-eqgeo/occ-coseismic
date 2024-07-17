import os
import argparse
import pickle as pkl
import numpy as np
import pandas as pd
from time import time, sleep


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

    os.makedirs(f"../{branchdir}/site_cumu_exceed{S}", exist_ok=True)

    site_file = f"../{model_version_results_directory}/site_name_list.txt"

    if S == "":
        S = '_'

    # Append site information for this branch to the main list
    with open(site_file, "a") as f:
        for site in sites_of_interest:
            f.write(f"{site} {branchdir} {S}\n")


def prep_SLURM_submission(model_version_results_directory, tasks_per_array, n_tasks,
                          hours: int = 0, mins: int = 3, mem: int = 45, cpus: int = 1, account: str = 'uc03610',
                          time_interval: int = 100, n_samples: int = 1000000, sd: float = 0.4, job_time=5):
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

    with open(f"../{model_version_results_directory}/cumu_PPE_slurm_task_array.sl", "wb") as f:
        f.write("#!/bin/bash -e\n".encode())
        f.write(f"#SBATCH --job-name=occ-{os.path.basename(model_version_results_directory)}\n".encode())
        f.write(f"#SBATCH --time={hours:02}:{mins:02}:00      # Walltime (HH:MM:SS)\n".encode())
        f.write(f"#SBATCH --mem={mem}GB\n".encode())
        f.write(f"#SBATCH --cpus-per-task={cpus}\n".encode())
        f.write(f"#SBATCH --account={account}\n".encode())
        f.write("#SBATCH --partition=large\n".encode())
        f.write(f"#SBATCH --array=0-{n_tasks-1}\n".encode())

        f.write(f"#SBATCH -o logs/{os.path.basename(model_version_results_directory)}_task%a_%j.out\n".encode())
        f.write(f"#SBATCH -e logs/{os.path.basename(model_version_results_directory)}_task%a_%j.err\n\n".encode())

        f.write("# Activate the conda environment\n".encode())
        f.write("mkdir -p logs\n".encode())
        f.write("module purge  2>/dev/null\n".encode())
        f.write("module load Python/3.11.6-foss-2023a\n\n".encode())

        f.write(f"python nesi_scripts.py --task_number $SLURM_ARRAY_TASK_ID --tasks_per_array {int(tasks_per_array)} --site_file {site_file} --time_interval {int(time_interval)} --n_samples {int(n_samples)} --sd {sd} \n\n".encode())


def compile_site_cumu_PPE(branch_site_disp_dict, model_version_results_directory, extension1, taper_extension="", S=""):
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
        site_PPE_dict.update(single_site_dict)
        # os.remove(f"../{model_version_results_directory}/{extension1}/site_cumu_exceed{S}/{site_of_interest}.pkl")
    # os.rmdir(f"../{model_version_results_directory}/{extension1}/site_cumu_exceed{S}")
    if S == "":
        extension2 = taper_extension
    else:
        extension2 = S

    if 'grid_meta' in branch_site_disp_dict.keys():
        site_PPE_dict['grid_meta'] = branch_site_disp_dict['grid_meta']

    with open(f"../{model_version_results_directory}/{extension1}/cumu_exceed_prob_{extension1}{extension2}.pkl", "wb") as f:
        pkl.dump(site_PPE_dict, f)

    return site_PPE_dict

def nesi_get_weighted_mean_PPE_dict(out_directory='', ppe_name='', outfile_extension='', slip_taper=False, sbatch=False,
                                    hours: int = 1, mins: int = 0, mem: int = 50, account='uc03610', cpus=1):

    if outfile_extension != "":
        optional = outfile_extension
    else:
        optional= ""
    
    if slip_taper:
        optional += " --slip_taper"

    with open(f"../{out_directory}/get_weighted_mean_PPE.sl", "wb") as f:
        f.write("#!/bin/bash -e\n".encode())
        f.write(f"#SBATCH --job-name=occ-{out_directory}\n".encode())
        f.write(f"#SBATCH --time={hours:02}:{mins:02}:00      # Walltime (HH:MM:SS)\n".encode())
        f.write(f"#SBATCH --mem={mem}GB\n".encode())
        f.write(f"#SBATCH --cpus-per-task={cpus}\n".encode())
        f.write(f"#SBATCH --account={account}\n".encode())

        f.write(f"#SBATCH -o logs/{out_directory}_task%a_%j.out\n".encode())
        f.write(f"#SBATCH -e logs/{out_directory}_task%a_%j.err\n\n".encode())

        f.write("# Activate the conda environment\n".encode())
        f.write("mkdir -p logs\n".encode())
        f.write("module purge  2>/dev/null\n".encode())
        f.write("module load Python/3.11.6-foss-2023a\n\n".encode())

        f.write(f"python nesi_scripts.py --outDir {out_directory} --PPE_name {ppe_name} {optional }--weighted_PPE \n\n".encode())

    if sbatch:
        os.system(f"sbatch ../{out_directory}/get_weighted_mean_PPE.sl")
        raise Exception("../{out_directory}/get_weighted_mean_PPE.sl submitted")
    else:
        raise Exception(f"Now run\n\tsbatch ../{out_directory}/get_weighted_mean_PPE.sl")


if __name__ == "__main__":
    # Import here to prevent circular imports
    from probabalistic_displacement_scripts import get_cumu_PPE, get_weighted_mean_PPE_dict

    parser = argparse.ArgumentParser(description="Script to calculate cumulative exceedance probabilities for each site")
    parser.add_argument("--task_number", type=int, default=0, help="Task number for the SLURM array")
    parser.add_argument("--tasks_per_array", type=int, default=10, help="Number of tasks per SLURM array")
    parser.add_argument("--site_file", type=str, default='site_name_list.txt', help="File containing the site information")
    parser.add_argument("--time_interval", type=int, default=100, help="Time interval to calculate exceedance probabilities over")
    parser.add_argument("--n_samples", type=int, default=1e5, help="Number of samples to use for the poissonian simulation")
    parser.add_argument("--sd", type=float, default=0.4, help="Standard deviation of the normal distribution to use for uncertainty in displacements")
    parser.add_argument("--scaling", type=str, default="", help="Scaling factor for the displacements")
    parser.add_argument("--slip_taper", default=False, action='store_true', help="Tapered slip distribution")
    parser.add_argument("--overwrite", default=False, action='store_true', help="Overwrite existing files")
    parser.add_argument("--weighted_PPE", dest='site_PPE', default=True, action='store_false', help="Run get_weighted_mean_PPE_dict not get_cumu_PPE")
    parser.add_argument("--outDir", type=str, help="Output directory for the results")
    parser.add_argument("--PPE_name", type=str, default="", help="Name of the Paired PPE file to load")
    parser.add_argument("--outfile_extension", type=str, default="", help="Extension for the output file")
    args = parser.parse_args()

    start = time()

    if args.slip_taper:
        taper = "_tapered"
    else:
        taper = "_uniform"


    if args.site_PPE:
        investigation_time = args.time_interval
        n_samples = args.n_samples
        sd = args.sd

        # This is a hack to get around multiple tasks trying to open the file at once, so it appearing like it doesn't exist
        find_file_count = 0
        attempt_limit = 10
        while find_file_count < attempt_limit:
            try:
                with open(args.site_file, "r") as f:
                    all_sites = f.read().splitlines()
                find_file_count = attempt_limit + 1
            except FileNotFoundError:
                sleep(1 + np.random.rand())
                find_file_count += 1
                print(f"Attempt {find_file_count} to find {args.site_file}")
        
        if find_file_count == attempt_limit:
            raise FileNotFoundError(f"File {args.site_file} not found")

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
                scaling = ''

            extension1 = os.path.basename(branch_results_directory)
            with open(f"../{branch_results_directory}/branch_site_disp_dict_{extension1}{scaling}.pkl", "rb") as fid:
                branch_disp_dict = pkl.load(fid)

            sites_of_interest = group['Site'].values

            if isinstance([key for key in branch_disp_dict.keys()][0], int):
                if '_' not in sites_of_interest[0]:
                    sites_of_interest = [int(site) for site in sites_of_interest]

            if scaling == "":
                os.system(f"echo {args.task_number}: Running {n_samples} scenarios for {len(sites_of_interest)} sites in branch {extension1}...")
                print(f"{args.task_number}: Running {len(sites_of_interest)} sites in branch {extension1}...")
            else:
                os.system(f"echo {args.task_number}: echo Running {n_samples} scenarios for {len(sites_of_interest)} sites in branch {extension1} with scaling {scaling}...")
                print(f"{args.task_number}: Running {len(sites_of_interest)} sites in branch {extension1} with scaling {scaling}...")

            # Needs to be run one site at a time so sites can be recombined later
            for ix, site in enumerate(sites_of_interest):
                lap = time()
                if os.path.exists(f"../{branch_results_directory}/site_cumu_exceed{scaling}/{site}.pkl") and not args.overwrite:
                    os.system(f"echo {ix} {extension1} {site}.pkl already exists\n")
                    continue
                get_cumu_PPE(args.slip_taper, os.path.dirname(branch_results_directory), branch_disp_dict, [site], n_samples,
                            extension1, branch_key="nan", time_interval=investigation_time, sd=sd, scaling=scaling, load_random=False,
                            plot_maximum_displacement=False, array_process=True)
                os.system(f"echo {ix} {extension1} {site} complete in {time() - lap:.2f} seconds\n")

            os.system(f"echo {extension1} complete in {time() - begin:.2f} seconds\n")
            print(f"{extension1} complete in : {time() - begin:.2f} seconds\n")

        print(f"All sites complete in {time() - start:.2f} seconds (Average {(time() - start) / len(task_sites):.2f} seconds per site)")

    else:
        os.system('echo Loading fault model PPE dictionary...')
        print('Loading fault model PPE dictionary...')
        paired_PPE_filepath = f"../{args.outDir}/{args.PPE_name}"
        with open(paired_PPE_filepath, 'rb') as f:
            PPE_dict = pkl.load(f)

        os.system('echo Calculating weighted mean PPE dictionary...')
        print('Calculating weighted mean PPE dictionary...')
        get_weighted_mean_PPE_dict(fault_model_PPE_dict=PPE_dict, out_directory=args.outDir, outfile_extension=args.outfile_extension, slip_taper=taper)