import os
import shutil
import argparse
import pickle as pkl
import numpy as np
import pandas as pd
import h5py as h5
from time import time, sleep
from helper_scripts import hdf5_to_dict

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def nesiprint(string):
    os.system(f'echo {string}')
    print(string)


def prep_nesi_site_list(model_version_results_directory, sites_of_interest, extension1, S=""):
    """
    Must first run get_site_disp_dict to get the dictionary of displacements and rates

    inputs: runs for one logic tree branch
    Time_interval is in years

    function: calculates the poissonian probability of exceedance for each site for each displacement threshold value

    outputs: pickle file with probability dictionary (probs, disps, site_coords)

    CAVEATS/choices:
    - need to decide on number of 100-yr simulations to run (n_samples = 100000)
    """

    branchdir = f"{model_version_results_directory}/{extension1}"

    os.makedirs(f"../{branchdir}/site_cumu_exceed{S}", exist_ok=True)

    site_file = f"../{model_version_results_directory}/site_name_list.txt"

    if S == "":
        S = '_'

    # Append site information for this branch to the main list
    with open(site_file, "a") as f:
        for site in sites_of_interest:
            f.write(f"{str(site).replace(' ','-/-')} {branchdir} {S}\n")


def prep_SLURM_submission(model_version_results_directory, tasks_per_array, n_tasks,
                          hours: int = 0, mins: int = 3, mem: int = 45, cpus: int = 1, account: str = '',
                          time_interval: int = 100, n_samples: int = 1000000, sd: float = 0.4, job_time = 0, NSHM_branch=True,
                          thresh_lims=[0, 3], thresh_step=0.01):
    """
    Must first run get_site_disp_dict to get the dictionary of displacements and rates

    inputs: runs for one logic tree branch
    Time_interval is in years

    function: calculates the poissonian probability of exceedance for each site for each displacement threshold value

    outputs: pickle file with probability dictionary (probs, disps, site_coords)

    CAVEATS/choices:
    - need to decide on number of 100-yr simulations to run (n_samples = 1000000)
    """

    slurm_file = f"../{model_version_results_directory}/cumu_PPE_slurm_task_array.sl"

    if os.path.exists(slurm_file):
                os.remove(slurm_file)

    site_file = f"../{model_version_results_directory}/site_name_list.txt"

    if NSHM_branch:
        NSHM = ''
    else:
        NSHM = '--paired_branch'

    with open(slurm_file, "wb") as f:
        f.write("#!/bin/bash -e\n".encode())
        f.write(f"#SBATCH --job-name=occ-{os.path.basename(model_version_results_directory)}\n".encode())
        f.write(f"#SBATCH --time={hours:02}:{mins:02}:00      # Walltime (HH:MM:SS), {job_time} secs/job\n".encode())
        f.write(f"#SBATCH --mem={mem}GB\n".encode())
        f.write(f"#SBATCH --cpus-per-task={cpus}\n".encode())
        f.write(f"#SBATCH --account={account}\n".encode())
        if mem > 25:
            f.write("#SBATCH --partition=large\n".encode())
        f.write(f"#SBATCH --array=0-{n_tasks-1}\n".encode())

        f.write(f"#SBATCH -o logs/{os.path.basename(model_version_results_directory)}_sites_%j_task%a.out\n".encode())
        f.write(f"#SBATCH -e logs/{os.path.basename(model_version_results_directory)}_sites_%j_task%a.err\n\n".encode())

        f.write("# Activate the conda environment\n".encode())
        f.write("mkdir -p logs\n".encode())
        f.write("module purge  2>/dev/null\n".encode())
        f.write("module load Python/3.11.6-foss-2023a\n\n".encode())

        f.write(f"python nesi_scripts.py --task_number $SLURM_ARRAY_TASK_ID --tasks_per_array {int(tasks_per_array)} --site_file {site_file} --time_interval {int(time_interval)} --n_samples {int(n_samples)} ".encode())
        f.write(f"--sd {sd} --nesi_job site_PPE {NSHM} --thresh_lims {thresh_lims[0]}/{thresh_lims[1]} --thresh_step {thresh_step}\n\n".encode())


def compile_site_cumu_PPE(sites, model_version_results_directory, extension1, branch_h5file="", taper_extension="", S="", weight=None, thresholds=None):
    """
    Script to recompile all individual site PPE dictionaries into a single branch dictionary.
    For the sake of saving space, the individual site dictionaries are deleted after being combined into the branch dictionary.
    """

    branch_h5 = h5.File(branch_h5file, "r+")
    if 'grid_meta' in sites:
        sites.remove('grid_meta')
    
    if S == "":
        S = taper_extension

    if not 'thresholds' in branch_h5.keys():
        branch_h5.create_dataset('thresholds', data=thresholds)

    all_good = True
    bad_sites = []
    bad_flag = ''
    start = time()
    printProgressBar(0, len(sites), prefix=f'\tAdded {0}/{len(sites)} Sites:', suffix='0 secs', length=50)
    for ix, site_of_interest in enumerate(sites):
        try:
            if not os.path.exists(f"../{model_version_results_directory}/{extension1}/site_cumu_exceed{S}/{site_of_interest}.h5"):
                continue
            with h5.File(f"../{model_version_results_directory}/{extension1}/site_cumu_exceed{S}/{site_of_interest}.h5", "r") as site_h5:
                if site_of_interest in branch_h5.keys():
                    del branch_h5[site_of_interest]
                branch_h5.create_group(site_of_interest)
                if all_good:
                    for key in site_h5[site_of_interest].keys():
                        if site_h5[site_of_interest][key][()].shape == ():   # Check for scalar datasets that cannot be compressed
                            branch_h5[site_of_interest].create_dataset(key, data=site_h5[site_of_interest][key][()])
                        else:
                            branch_h5[site_of_interest].create_dataset(key, data=site_h5[site_of_interest][key][()], compression='gzip', compression_opts=5)
            printProgressBar(ix + 1, len(sites), prefix=f'\tAdded {ix + 1}/{len(sites)} Sites:', suffix=f'{time() - start:.2f} seconds {site_of_interest}{bad_flag}', length=50)
        except:
            bad_sites.append(site_of_interest)
            all_good = False
            bad_flag = f" (Error with {len(bad_sites)} sites)"

    if weight:
        if weight > 0 and 'branch_weight' not in branch_h5.keys():
            branch_h5.create_dataset('branch_weight', data=weight)

    branch_h5.close()
    if all_good:
        #shutil.rmtree(f"../{model_version_results_directory}/{extension1}/site_cumu_exceed{S}")
        print(f'\nNot deleting ../{model_version_results_directory}/{extension1}/site_cumu_exceed{S}!')
    else:
        print(f"\nError with {len(bad_sites)} sites: ../{model_version_results_directory}/bad_sites_{os.path.basename(branch_h5file).replace('.h5', '.txt')}")
        print(f"Deleting {branch_h5file}")
        os.remove(branch_h5file)
        with open(f"../{model_version_results_directory}/bad_sites_{os.path.basename(branch_h5file).replace('.h5', '.txt')}", "w") as f:
            for site in bad_sites:
                f.write(f"{site} {model_version_results_directory}/{extension1} {S}\n")

    return


def prep_combine_branch_list(branch_site_disp_dict_file, model_version_results_directory, extension1, branch_h5file="", taper_extension="", S="", weight=0, thresholds=None):

    with open(f"../{model_version_results_directory}/combine_site_meta.pkl", "rb") as f:
        combine_dict = pkl.load(f)

    combine_dict[os.path.basename(branch_h5file)] = {'branch_site_disp_dict': branch_site_disp_dict_file,
                                                     'model_version_results_directory': model_version_results_directory,
                                                     'extension1': extension1,
                                                     'branch_h5file': branch_h5file,
                                                     'taper_extension': taper_extension,
                                                     'S': S,
                                                     'weight': weight,
                                                     'thresholds': thresholds}

    with open(f"../{model_version_results_directory}/combine_site_meta.pkl", "wb") as f:
        pkl.dump(combine_dict, f)

    with open(f"../{model_version_results_directory}/branch_combine_list.txt", "a") as f:
        f.write(f"{os.path.basename(branch_h5file)}\n")


def prep_SLURM_combine_submission(combine_dict_file, branch_combine_list, model_version_results_directory, 
                                  tasks_per_array, n_tasks, hours: int = 0, mins: int = 3, mem: int = 10,
                                  cpus: int = 1, account: str = ''):


    slurm_file = f"../{model_version_results_directory}/combine_sites.sl"

    if os.path.exists(slurm_file):
                os.remove(slurm_file)

    with open(slurm_file, "wb") as f:
        f.write("#!/bin/bash -e\n".encode())
        f.write(f"#SBATCH --job-name=occ-{os.path.basename(model_version_results_directory)}\n".encode())
        f.write(f"#SBATCH --time={hours:02}:{mins:02}:00      # Walltime (HH:MM:SS)\n".encode())
        f.write(f"#SBATCH --mem={mem}GB\n".encode())
        f.write(f"#SBATCH --cpus-per-task={cpus}\n".encode())
        f.write(f"#SBATCH --account={account}\n".encode())
        f.write(f"#SBATCH --array=0-{n_tasks-1}\n".encode())

        f.write(f"#SBATCH -o logs/{os.path.basename(model_version_results_directory)}_combine_%j_task%a.out\n".encode())
        f.write(f"#SBATCH -e logs/{os.path.basename(model_version_results_directory)}_combine_%j_task%a.err\n\n".encode())

        f.write("# Activate the conda environment\n".encode())
        f.write("mkdir -p logs\n".encode())
        f.write("module purge  2>/dev/null\n".encode())
        f.write("module load Python/3.11.6-foss-2023a\n\n".encode())

        f.write(f"python nesi_scripts.py --task_number $SLURM_ARRAY_TASK_ID --tasks_per_array {tasks_per_array} --combine_dict_file {combine_dict_file} --branch_combine_list {branch_combine_list} --nesi_job combine_sites\n\n".encode())


def prep_SLURM_weighted_sites_submission(out_directory, tasks_per_array, n_tasks, site_file,
                                         hours: int = 0, mins: int = 3, mem: int = 45, cpus: int = 1, account: str = 'uc03610',
                                         job_time = 0):
    """
    Prep the SLURM submission script to create a task array to calculate the weighted mean PPE for each site
    """

    slurm_file = f"../{out_directory}/weighted_sites_slurm_task_array.sl"

    if os.path.exists(slurm_file):
                os.remove(slurm_file)
    

    with open(slurm_file, "wb") as f:
        f.write("#!/bin/bash -e\n".encode())
        f.write(f"#SBATCH --job-name=occ-{os.path.basename(out_directory)}\n".encode())
        f.write(f"#SBATCH --time={hours:02}:{mins:02}:00      # Walltime (HH:MM:SS), {job_time} secs/site\n".encode())
        f.write(f"#SBATCH --mem={mem}GB\n".encode())
        f.write(f"#SBATCH --cpus-per-task={cpus}\n".encode())
        f.write(f"#SBATCH --account={account}\n".encode())
        if mem > 25:
            f.write("#SBATCH --partition=large\n".encode())
        f.write(f"#SBATCH --array=0-{n_tasks-1}\n".encode())

        f.write(f"#SBATCH -o logs/{os.path.basename(out_directory)}_site_weights_%j_task%a.out\n".encode())
        f.write(f"#SBATCH -e logs/{os.path.basename(out_directory)}_site_weights_%j_task%a.err\n\n".encode())

        f.write("# Activate the conda environment\n".encode())
        f.write("mkdir -p logs\n".encode())
        f.write("module purge  2>/dev/null\n".encode())
        f.write("module load Python/3.11.6-foss-2023a\n\n".encode())

        f.write(f"python nesi_scripts.py --task_number $SLURM_ARRAY_TASK_ID --tasks_per_array {int(tasks_per_array)} --site_file {site_file} --nesi_job site_weights \n\n".encode())
    print('')

    return slurm_file


def nesi_get_weighted_mean_PPE_dict(out_directory='', ppe_name='', outfile_extension='', slip_taper=False, sbatch=False,
                                    hours: int = 1, mins: int = 0, mem: int = 50, account='', cpus=1):

    if outfile_extension != "":
        optional = outfile_extension
    else:
        optional= ""
    
    if slip_taper:
        optional += " --slip_taper"

    with open(f"../{out_directory}/get_weighted_mean_PPE.sl", "wb") as f:
        f.write("#!/bin/bash -e\n".encode())
        f.write(f"#SBATCH --job-name=occ-w_{os.path.basename(out_directory)}\n".encode())
        f.write(f"#SBATCH --time={hours:02}:{mins:02}:00      # Walltime (HH:MM:SS)\n".encode())
        f.write(f"#SBATCH --mem={mem}GB\n".encode())
        f.write(f"#SBATCH --cpus-per-task={cpus}\n".encode())
        f.write(f"#SBATCH --account={account}\n".encode())

        f.write(f"#SBATCH -o logs/{os.path.basename(out_directory)}_weighted_%j.out\n".encode())
        f.write(f"#SBATCH -e logs/{os.path.basename(out_directory)}_weighted_%j.err\n\n".encode())

        f.write("# Activate the conda environment\n".encode())
        f.write("mkdir -p logs\n".encode())
        f.write("module purge  2>/dev/null\n".encode())
        f.write("module load Python/3.11.6-foss-2023a\n\n".encode())

        f.write(f"python nesi_scripts.py --outDir {out_directory} --PPE_name {ppe_name} --nesi_job weighted_mean {optional} \n\n".encode())

    raise Exception(f"Now run\n\tsbatch ../{out_directory}/get_weighted_mean_PPE.sl")


if __name__ == "__main__":
    # Import here to prevent circular imports
    from probabalistic_displacement_scripts import get_cumu_PPE, get_weighted_mean_PPE_dict, get_all_branches_site_disp_dict, create_site_weighted_mean

    parser = argparse.ArgumentParser(description="Script to calculate cumulative exceedance probabilities for each site")
    parser.add_argument("--task_number", type=int, default=0, help="Task number for the SLURM array")
    parser.add_argument("--tasks_per_array", type=int, default=10, help="Number of tasks per SLURM array")
    parser.add_argument("--site_file", type=str, default='site_name_list.txt', help="File containing the site information")
    parser.add_argument("--time_interval", type=int, default=100, help="Time interval to calculate exceedance probabilities over")
    parser.add_argument("--n_samples", type=int, default=1e5, help="Number of samples to use for the poissonian simulation")
    parser.add_argument("--sd", type=float, default=0.4, help="Standard deviation of the normal distribution to use for uncertainty in displacements")
    parser.add_argument("--thresh_lims", type=str, default="0/3", help="Threshold limits for the exceedance probabilities")
    parser.add_argument("--thresh_step", type=float, default=0.01, help="Step size for the threshold limits")
    parser.add_argument("--scaling", type=str, default="", help="Scaling factor for the displacements")
    parser.add_argument("--slip_taper", default=False, action='store_true', help="Tapered slip distribution")
    parser.add_argument("--paired_branch", dest='NSHM_branch', default=True, action='store_false', help="Run for paired branch")
    parser.add_argument("--overwrite", default=False, action='store_true', help="Overwrite existing files")
    parser.add_argument("--nesi_job", default='site_PPE', help="Select which job you need to run (site_PPE, combine_sites, weighted_mean)")
    parser.add_argument("--outDir", type=str, help="Output directory for the results")
    parser.add_argument("--PPE_name", type=str, default="", help="Name of the Paired PPE file to load")
    parser.add_argument("--outfile_extension", type=str, default="", help="Extension for the output file")
    parser.add_argument("--combine_dict_file", type=str, default="", help="File containing the dictionary of sites to combine")
    parser.add_argument("--branch_combine_list", type=str, default="", help="List of branches to combine")
    args = parser.parse_args()

    start = time()

    if args.slip_taper:
        taper = "_tapered"
    else:
        taper = "_uniform"

    if args.nesi_job == 'site_PPE':
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
                nesiprint(f"Attempt {find_file_count} to find {args.site_file}")
        
        if find_file_count == attempt_limit:
            raise FileNotFoundError(f"File {args.site_file} not found")

        # all_sites = [all_sites[ix] for ix in np.random.permutation(len(all_sites))]  # Shuffle for reduced chance of task arrays not having unprocessed sites (useful if time expired on previous attempt)
        if args.tasks_per_array == 0:
            task_sites = all_sites
        else:
            task_sites = all_sites[args.task_number * args.tasks_per_array:(args.task_number + 1) * args.tasks_per_array]
        if len(task_sites) == 0:
            raise Exception(f"Task {args.task_number} has no sites to process")

        sites = np.array([site_info.split(" ")[0].replace('-/-', ' ') for site_info in task_sites])
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

            sites_of_interest = group['Site'].values

            extension1 = os.path.basename(branch_results_directory)
            if args.NSHM_branch:
                branch_disp_dict = f"../{branch_results_directory}/branch_site_disp_dict_{extension1}{scaling}.h5"
                with h5.File(branch_disp_dict, "r") as branch_h5:
                    if isinstance([key for key in branch_h5.keys()][0], int):
                        if '_' not in sites_of_interest[0]:
                            sites_of_interest = [int(site) for site in sites_of_interest]
                branch_unique_ids = 'nan'
                crustal_model_dir = ''
                subduction_model_dir = ''
            else:
                with h5.File(f"../{branch_results_directory}/branch_site_disp_dict_{extension1}{scaling}.h5", "r") as branch_h5:
                    pair_site_disp_dict = hdf5_to_dict(branch_h5)

                out_directory = pair_site_disp_dict[[key for key in pair_site_disp_dict.keys()][0]]['out_directory']
                crustal_model_dir = pair_site_disp_dict[[key for key in pair_site_disp_dict.keys()][0]]['crustal_directory']
                subduction_model_dir = pair_site_disp_dict[[key for key in pair_site_disp_dict.keys()][0]]['subduction_directory']
                with open(f"../{out_directory}/branch_weight_dict_crustal.pkl", "rb") as f:
                    crustal_branch_weight_dict = pkl.load(f)
                all_crustal_branches_site_disp_dict = get_all_branches_site_disp_dict(crustal_branch_weight_dict, 'sites', args.slip_taper,
                                                                                      crustal_model_dir)
                all_sz_branches_site_disp_dict = {}
                ix = 0
                for fault_type in ['sz', 'py']:
                    if f'_{fault_type}_' in os.path.basename(out_directory):
                        with open(f"../{out_directory}/branch_weight_dict_{fault_type}.pkl", "rb") as f:
                            sz_branch_weight_dict = pkl.load(f)
                        all_single_sz_branches_site_disp_dict = get_all_branches_site_disp_dict(sz_branch_weight_dict, 'sites', args.slip_taper,
                                                                                                subduction_model_dir[ix])
                        ix += 1
                    all_sz_branches_site_disp_dict = all_sz_branches_site_disp_dict | all_single_sz_branches_site_disp_dict
                
                branch_disp_dict = {}

                for site in sites_of_interest:
                    crustal_unique_id = pair_site_disp_dict[site]['crustal_unique_id']
                    sz_unique_ids = pair_site_disp_dict[site]['sz_unique_ids']
                    site_coords = all_crustal_branches_site_disp_dict[crustal_unique_id]["site_disp_dict"][site]["site_coords"]

                    pair_site_disps = all_crustal_branches_site_disp_dict[crustal_unique_id]["site_disp_dict"][site]["disps"]
                    pair_scaled_rates = all_crustal_branches_site_disp_dict[crustal_unique_id]["site_disp_dict"][
                        site]["scaled_rates"]

                    for sz_unique_id in sz_unique_ids:
                        sz_site_disps = all_sz_branches_site_disp_dict[sz_unique_id]["site_disp_dict"][site]["disps"]
                        sz_site_scaled_rates = all_sz_branches_site_disp_dict[sz_unique_id]["site_disp_dict"][site]["scaled_rates"]

                        pair_site_disps += sz_site_disps
                        pair_scaled_rates += sz_site_scaled_rates
                    
                    branch_disp_dict[site] = {"disps": pair_site_disps, "scaled_rates": pair_scaled_rates, "site_coords": site_coords}

                if isinstance([key for key in branch_disp_dict.keys()][0], int):
                    if '_' not in sites_of_interest[0]:
                        sites_of_interest = [int(site) for site in sites_of_interest]

            if scaling == "":
                nesiprint(f"{args.task_number}: Running {len(sites_of_interest)} sites in branch {extension1}...")
            else:
                nesiprint(f"{args.task_number}: Running {len(sites_of_interest)} sites in branch {extension1} with scaling {scaling}...")

            # Needs to be run one site at a time so sites can be recombined later
            for ix, site in enumerate(sites_of_interest):
                if not args.NSHM_branch:
                    branch_unique_ids = pair_site_disp_dict[site]['branch_key']
                lap = time()
                if os.path.exists(f"../{branch_results_directory}/site_cumu_exceed{scaling}/{site}.pkl") and not args.overwrite:
                    print(f"{ix} {extension1} {site}.pkl already exists")
                    continue
                get_cumu_PPE(args.slip_taper, os.path.dirname(branch_results_directory), branch_disp_dict, [site], n_samples,
                            extension1, branch_key=branch_unique_ids, time_interval=investigation_time, sd=sd, scaling=scaling, load_random=False,
                            plot_maximum_displacement=False, array_process=True, NSHM_branch=args.NSHM_branch, crustal_model_dir=crustal_model_dir, subduction_model_dirs=subduction_model_dir,
                            thresh_lims=[float(val) for val in args.thresh_lims.split('/')], thresh_step=float(args.thresh_step))
                # os.system(f"echo {ix} {extension1} {site} complete in {time() - lap:.2f} seconds\n")

            nesiprint(f"{extension1} complete in : {time() - begin:.2f} seconds\n")

        print(f"All sites complete in {time() - start:.2f} seconds (Average {(time() - start) / len(task_sites):.2f} seconds per site)")

    elif args.nesi_job == 'combine_sites':
        # This is a hack to get around multiple tasks trying to open the file at once, so it appearing like it doesn't exist
        find_file_count = 0
        attempt_limit = 10
        while find_file_count < attempt_limit:
            try:
                with open(args.branch_combine_list, "r") as f:
                    all_branches = f.read().splitlines()
                find_file_count = attempt_limit + 1
            except FileNotFoundError:
                sleep(1 + np.random.rand())
                find_file_count += 1
                nesiprint(f"Attempt {find_file_count} to find {args.branch_combine_list}")
        
        if args.tasks_per_array == 0:
            task_branches = all_branches
        else:
            task_branches = all_branches[args.task_number * args.tasks_per_array:(args.task_number + 1) * args.tasks_per_array]
        if len(task_branches) == 0:
            raise Exception(f"Task {args.task_number} has no sites to process")
    
        with open(f"{args.combine_dict_file}", "rb") as f:
            combine_dict = pkl.load(f)
       
        for branch in task_branches:
            with h5.File(combine_dict[branch]['branch_site_disp_dict'], "r") as branch_h5:
                site_list = [key for key in branch_h5.keys() if key not in ['rates', 'scaled_rates']]
            nesiprint(f"\tCombining site dictionaries into {combine_dict[branch]['branch_h5file']}....")
            compile_site_cumu_PPE(site_list, combine_dict[branch]['model_version_results_directory'], combine_dict[branch]['extension1'],
                                branch_h5file=combine_dict[branch]['branch_h5file'], taper_extension=combine_dict[branch]['taper_extension'], S=combine_dict[branch]['S'], weight=combine_dict[branch]['weight'],
                                thresholds=combine_dict[branch]['thresholds'])
        print('\nAll branches combined!')

    elif args.nesi_job == 'weighted_mean':
        nesiprint('Loading fault model PPE dictionary...')
        paired_PPE_filepath = f"../{args.outDir}/{args.PPE_name}"
        with open(paired_PPE_filepath, 'rb') as f:
            PPE_dict = pkl.load(f)

        nesiprint('Calculating weighted mean PPE dictionary...')
        get_weighted_mean_PPE_dict(fault_model_PPE_dict=PPE_dict, out_directory=args.outDir, outfile_extension=args.outfile_extension, slip_taper=args.slip_taper)
    
    elif args.nesi_job == 'site_weights':
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
                nesiprint(f"Attempt {find_file_count} to find {args.site_file}")
        
        if find_file_count == attempt_limit:
            raise FileNotFoundError(f"File {args.site_file} not found")

        if args.tasks_per_array == 0:
            task_sites = all_sites
        else:
            task_sites = all_sites[args.task_number * args.tasks_per_array:(args.task_number + 1) * args.tasks_per_array]
        if len(task_sites) == 0:
            raise Exception(f"Task {args.task_number} has no sites to process")
    
        nesiprint(f"Finding weighted means for {len(task_sites)} sites...")
        for site in task_sites:
            site_name = os.path.basename(site).replace('.h5', '')
            with h5.File(site, "a") as site_h5:
                create_site_weighted_mean(site_h5, site_name, site_h5['n_samples'][()], 
                                          site_h5['crustal_model_version_results_directory'][()].decode('utf-8'), 
                                          [val.decode('utf-8') for val in site_h5['sz_model_version_results_directory_list'][()]], 
                                          site_h5['gf_name'][()].decode('utf-8'), 
                                          site_h5['thresholds'][:],
                                          [val.decode('utf-8') for val in site_h5['exceed_type_list'][()]],
                                          [val.decode('utf-8') for val in site_h5['branch_id_list'][()]], 
                                          site_h5['sigma_lims'], 
                                          site_h5['branch_weights'],
                                          compression='gzip')
            nesiprint(f"Site {site_name} complete")
        print('\nAll sites complete!')

    else:
        raise Exception(f"Job {args.nesi_job} not recognised")