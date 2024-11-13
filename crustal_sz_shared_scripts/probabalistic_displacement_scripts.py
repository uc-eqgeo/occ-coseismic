try:
    import geopandas as gpd
    import rasterio
    from rasterio.transform import Affine
    from weighted_mean_plotting_scripts import get_mean_prob_barchart_data, get_mean_disp_barchart_data
except:
    print("Running on NESI. Some functions won't work....")
from helper_scripts import get_figure_bounds, make_qualitative_colormap, tol_cset, get_probability_color, percentile, maximum_displacement_plot, dict_to_hdf5, hdf5_to_dict
import xarray as xr
import h5py as h5
import json
import os
import shutil
import random
import itertools
import numpy as np
import pandas as pd
import pickle as pkl
from time import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from nesi_scripts import prep_nesi_site_list, prep_SLURM_submission, compile_site_cumu_PPE, \
                         prep_combine_branch_list, prep_SLURM_combine_submission, prep_SLURM_weighted_sites_submission
numba_flag = True
try:
    from numba import njit
except:
    numba_flag = False

matplotlib.rcParams['pdf.fonttype'] = 42

def write_site_disp_dict(extension1, slip_taper, model_version_results_directory, site_disp_h5file):
    """
        inputs: uses extension naming scheme to load displacement dictionary created with the
        get_rupture_disp_dict function. State slip taper (True or False).

        functions: reshapes the dictionary to be organized by site name (key = site name).

        outputs: a dictionary where each key is a location/site name. contains displacements (length = number of
        rupture ids), annual rate (length = number of rupture ids), site name list (should be same length as green's
        function), and site coordinates (same length as site name list)

        CAVEATS/choices:
        - a little clunky because most of the dictionary columns are repeated across all keys.
        """
    print('Making', os.path.basename(site_disp_h5file))
    if slip_taper is True:
        taper_extension = "_tapered"
    else:
        taper_extension = "_uniform"

    # load saved displacement data
    # disp dictionary has keys for each rupture id and displacement data for each site ( disp length = # of sites)
    # for grids, the disp_dict is still in grid shape instead of list form.
    with open(f"../{model_version_results_directory}/{extension1}/all_rupture_disps_{extension1}{taper_extension}.pkl",
              "rb") as fid:
        rupture_disp_dictionary = pkl.load(fid)

    ###### reshape displacement data to be grouped by site location.
    site_coords = np.array([])
    key = 0
    while site_coords.shape[0] == 0:
        rupt = list(rupture_disp_dictionary.keys())[key]
        site_coords = rupture_disp_dictionary[rupt]["site_coords"]
        key += 1
    site_names = rupture_disp_dictionary[rupt]["site_name_list"]

    # makes a list of lists. each item is a rupture scenario that contains a list of displacements at each site.
    # could probably simplify this because it's the same shape as the dictionary, but I'm not sure how to do that yet
    # and this is easier to understand I think.
    disps_by_scenario = []
    annual_rates_by_scenario = []
    for ix, rupture_id in enumerate(rupture_disp_dictionary.keys()):
        print(f"\tPreparing disps by scenario... {ix}\{len(rupture_disp_dictionary.keys())}", end='\r')
        disps = np.zeros(len(site_names))
        non_zero_disps = np.array(rupture_disp_dictionary[rupture_id]["v_disps_m"])
        if non_zero_disps.shape[0] == 0:
            continue
        disps[non_zero_disps[:, 0].astype(int)] = non_zero_disps[:, 1]
        disps_by_scenario.append(disps)
        annual_rate = rupture_disp_dictionary[rupture_id]["annual_rate"]
        annual_rates_by_scenario.append(annual_rate)
    print('')
    # list of lists. each item is a site location that contains displacements from each scenario (disp list length =
    # number of rupture scenarios)
    disps_by_location = []
    scenarios_with_disps = []
    #annual_rates_by_location = []
    for site_num in range(len(site_names)):
        print(f"\tPreparing disps by location... {site_num}\{len(site_names)}", end='\r')
        site_disp = np.array([scenario[site_num] for scenario in disps_by_scenario])
        disp_ix = [ix for ix, disp in enumerate(site_disp) if disp != 0]
        disps_by_location.append(site_disp[disp_ix].tolist())
        scenarios_with_disps.append(disp_ix)
        # annual_rates_by_location.append(annual_rates_by_scenario)
    print('')
    # make dictionary of displacements and other data. key is the site name.
    if os.path.exists(site_disp_h5file):
        os.remove(site_disp_h5file)
    with h5.File(site_disp_h5file, "w") as site_disp_PPEh5:
        site_disp_PPEh5.create_dataset("rates", data=annual_rates_by_scenario)
        for i, site in enumerate(site_names):
            print(f"\tWriting sites to {site_disp_h5file}... {i}\{len(site_names)}", end='\r')
            site_group = site_disp_PPEh5.create_group(site)
            site_group.create_dataset("disps", data=disps_by_location[i])
            site_group.create_dataset("disps_ix", data=scenarios_with_disps[i])
            site_group.create_dataset("site_coords", data=site_coords[i])
    print('')
    return 

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

def time_elasped(current_time, start_time, decimal=False):
    elapsed_time = current_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    if not decimal:
        return "{:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds))
    else:
        seconds, rem = divmod(seconds, 1)
        return "{:0>2}:{:0>2}:{:0>2}.{:.0f}".format(int(hours), int(minutes), int(seconds), rem * 10)


def get_all_branches_site_disp_dict(branch_weight_dict, gf_name, slip_taper, model_version_results_directory):
    """
    Combine all site_disp_dicts for a each branch of a fault model into a single dictionary
    """
    all_branches_site_disp_dict = {}
    for branch_id in branch_weight_dict.keys():
        extension1 = gf_name + branch_weight_dict[branch_id]["file_suffix"]
        # get site displacement dictionary
        # this extracts the rates from the solution directory, but it is not scaled by the rate scaling factor
        # multiply the rates by the rate scaling factor
        rate_scaling_factor = branch_weight_dict[branch_id]["S"]

        branch_site_disp_dict_file = f"../{model_version_results_directory}/{extension1}/branch_site_disp_dict_{extension1}_S{str(rate_scaling_factor).replace('.', '')}.h5"
        if os.path.exists(branch_site_disp_dict_file):
            try:  # check in case h5 is corrupted
                branch_h5 = h5.File(branch_site_disp_dict_file, "a")
                branch_h5.close()
            except:
                write_site_disp_dict(extension1, slip_taper=slip_taper, model_version_results_directory=model_version_results_directory, site_disp_h5file=branch_site_disp_dict_file)
        else:
            write_site_disp_dict(extension1, slip_taper=slip_taper, model_version_results_directory=model_version_results_directory, site_disp_h5file=branch_site_disp_dict_file)

        with h5.File(branch_site_disp_dict_file, "a") as branch_h5:
            if "scaled_rates" not in branch_h5.keys():
                # multiply each value in the rates array by the rate scaling factor
                branch_h5.create_dataset("scaled_rates", data=branch_h5["rates"][:] * rate_scaling_factor)


        all_branches_site_disp_dict[branch_id] = {"site_disp_dict":branch_site_disp_dict_file,
                                                "branch_weight":branch_weight_dict[branch_id]["total_weight_RN"]}

    return all_branches_site_disp_dict

if numba_flag:
    @njit()
    def calc_thresholds(thresholds, cumulative_disp_scenarios, n_chunks=1):
        n_exceedances_total_abs = np.zeros((len(thresholds), n_chunks))
        n_exceedances_up = np.zeros((len(thresholds), n_chunks))
        n_exceedances_down = np.zeros((len(thresholds), n_chunks))

        for tix, threshold in enumerate(thresholds):
            # replaces index in zero array with the number of times the cumulative displacement exceeded the threshold
            # across all of the 100 yr scenarios
            # sums the absolute value of the disps if the abs value is greater than threshold. e.g., -0.5 + 0.5 = 1
            n_exceedances_total_abs[tix, :] = np.sum(np.abs(cumulative_disp_scenarios) > threshold)
            n_exceedances_up[tix, :] = np.sum(cumulative_disp_scenarios > threshold)
            n_exceedances_down[tix, :] = np.sum(cumulative_disp_scenarios < -threshold)

        return n_exceedances_total_abs, n_exceedances_up, n_exceedances_down
else:
    def calc_thresholds(thresholds, cumulative_disp_scenarios, n_chunks=1):
        n_exceedances_total_abs = np.zeros((len(thresholds), n_chunks))
        n_exceedances_up = np.zeros((len(thresholds), n_chunks))
        n_exceedances_down = np.zeros((len(thresholds), n_chunks))

        for tix, threshold in enumerate(thresholds):
            # replaces index in zero array with the number of times the cumulative displacement exceeded the threshold
            # across all of the 100 yr scenarios
            # sums the absolute value of the disps if the abs value is greater than threshold. e.g., -0.5 + 0.5 = 1
            n_exceedances_total_abs[tix, :] = np.sum(np.abs(cumulative_disp_scenarios) > threshold, axis=1)
            n_exceedances_up[tix, :] = np.sum(cumulative_disp_scenarios > threshold, axis=1)
            n_exceedances_down[tix, :] = np.sum(cumulative_disp_scenarios < -threshold, axis=1)

        return n_exceedances_total_abs, n_exceedances_up, n_exceedances_down


def prepare_random_arrays(branch_site_disp_dict_file, randdir, time_interval, n_samples, sd):
        print('\tPreparing random arrays...')
        branch_site_disp_dict = h5.File(branch_site_disp_dict_file, "r")
        rng = np.random.default_rng()
        if "scaled_rates" not in branch_site_disp_dict.keys():
            # if no scaled_rate column, assumes scaling of 1 (equal to "rates")
            rates = np.array(branch_site_disp_dict["rates"])
        else:
            rates = np.array(branch_site_disp_dict["scaled_rates"])
        branch_site_disp_dict.close()

        n_ruptures = rates.shape[0]
        scenarios = rng.poisson(time_interval * rates, size=(int(n_samples), n_ruptures))
        disp_uncertainty = rng.normal(1, sd, size=(int(n_samples), n_ruptures))
        with open(f"{randdir}/scenarios.pkl", "wb") as fid:
            pkl.dump(scenarios, fid)
        with open(f"{randdir}/disp_uncertainty.pkl", "wb") as fid:
            pkl.dump(disp_uncertainty, fid)


def get_cumu_PPE(slip_taper, model_version_results_directory, branch_site_disp_dict, site_ids, n_samples,
                 extension1, branch_key="nan", time_interval=100, sd=0.4, error_chunking=1000, scaling='', load_random=False,
                 thresh_lims=[0, 3], thresh_step=0.01, plot_maximum_displacement=False, array_process=False,
                 crustal_model_dir="", subduction_model_dirs="", NSHM_branch=True, pair_unique_id=None):
    """
    Must first run get_site_disp_dict to get the dictionary of displacements and rates, with 1 sigma error bars

    inputs: runs for one logic tree branch
    Time_interval is in years

    function: calculates the poissonian probability of exceedance for each site for each displacement threshold value

    outputs: pickle file with probability dictionary (probs, disps, site_coords)

    CAVEATS/choices:
    - need to decide on number of 100-yr simulations to run (n_samples = 1000000)
    """
    commence = time()

    procdir = os.path.relpath(os.path.dirname(__file__) + '/..')

    if numba_flag:
        _ = calc_thresholds(np.arange(0,1,1), np.ones((1, 100)))

    # use random number generator to initialise monte carlo sampling
    rng = np.random.default_rng()

    # Load the displacement/rate data for all sites
    if slip_taper is True:
        taper_extension = "_tapered"
    else:
        taper_extension = "_uniform"

    n_chunks = int(n_samples / error_chunking)
    if n_chunks < 10:
        error_chunking = int(n_samples / 10)
        print(f'Too few chunks for accurate error estimation. Decreasing error_chunking to {error_chunking}')

    if not NSHM_branch:  # If making a PPE for a paired branch, not a NSHM branch, load pre-made cumu_PPE dicts
        NSHM_PPEh5_list = []
        for branch in branch_key:
            model, branch_id = branch.split('_')[-2:]
            branch_scaling = branch.split('_')[3]
            if model == 'c':
                branch_PPE_h5 = f"../{crustal_model_dir}/sites_{model}_{branch_id}/{branch}_cumu_PPE.h5"
                if not os.path.exists(branch_PPE_h5):
                    NSHM_branch = True
            else:
                for subduction_model_dir in subduction_model_dirs:
                    ix = 0
                    branch_PPE_h5 = f"../{subduction_model_dir}/sites_{model}_{branch_id}/{branch}_cumu_PPE.h5"
                    while not os.path.exists(branch_PPE_h5):
                        ix += 1
                        if ix >= len(subduction_model_dirs):
                            NSHM_branch = True
                            break
            if NSHM_branch:
                print(f"Could not find *cumu_PPE.h5 for {branch}...")
            else:
                NSHM_PPEh5_list.append(branch_PPE_h5)

    if not os.path.exists(f"{procdir}/{model_version_results_directory}/{extension1}/scenarios.pkl") or not os.path.exists(f"../{model_version_results_directory}/{extension1}/disp_uncertainty.pkl"):
        load_random = False

    if load_random:
        # Load array of random samples rather than regenerating them
        lap = time()
        if array_process:
            randomdir = f"{procdir}/{model_version_results_directory}/{extension1}/site_cumu_exceed{scaling}"
        else:
            randomdir = f"{procdir}/{model_version_results_directory}/{extension1}"

        with open(f"{randomdir}/scenarios.pkl", 'rb') as f:
            all_scenarios = pkl.load(f)
        with open(f"{randomdir}/disp_uncertainty.pkl", 'rb') as f:
            all_uncertainty = pkl.load(f)

    ## loop through each site and generate a bunch of 100 yr interval scenarios
    site_PPE_dict = {}

    # get displacement thresholds for calculating exceedance (hazard curve x axis)
    thresholds = np.round(np.arange(thresh_lims[0], thresh_lims[1] + thresh_step, thresh_step), 4)

    #site_PPE_dict["thresholds"] = thresholds
    start = time()
    benchmarking = False
    if not benchmarking:
        printProgressBar(0, len(site_ids), prefix=f'\tProcessing {len(site_ids)} Sites:', suffix='Complete 00:00:00 (00:00s/site)', length=50)
    for i, site_of_interest in enumerate(site_ids):
        begin = time()
        lap = time()

        if isinstance(branch_site_disp_dict, str):
            with h5.File(branch_site_disp_dict, "r") as branch_h5:
                site_dict_i = hdf5_to_dict(branch_h5[site_of_interest])
                site_dict_i["scaled_rates"] = branch_h5["scaled_rates"][:]
        else:
            site_dict_i = branch_site_disp_dict[site_of_interest]
            site_dict_i["scaled_rates"] = branch_h5["scaled_rates"][:]

        if not NSHM_branch:
            cumulative_disp_scenarios = np.zeros(n_samples)
            for NSHM_PPE in NSHM_PPEh5_list[1:]:
                with h5.File(NSHM_PPE, "r") as PPEh5:
                    NSHM_displacements = PPEh5[site_of_interest]["scenario_displacements"][:]
                    slip_scenarios = PPEh5[site_of_interest]["slip_scenarios_ix"][:]

                cumulative_disp_scenarios[slip_scenarios] += NSHM_displacements.reshape(-1)
            if benchmarking:
                print(f"Loaded PPE: {time() - begin:.5f} s")
            lap = time()
        else:
            # Set up params for sampling
            investigation_time = time_interval

            if "scaled_rates" not in site_dict_i.keys():
                # if no scaled_rate column, assumes scaling of 1 (equal to "rates")
                scaled_rates = site_dict_i["rates"]
            else:
                scaled_rates = site_dict_i["scaled_rates"]

            # Drop ruptures that don't cause slip at this site
            drop_noslip = True
            if drop_noslip:
                disps = site_dict_i['disps']
                # Check this isn't a site that is uneffected by earthquakes
                if disps.shape[0] > 0:
                    scaled_rates = scaled_rates[site_dict_i["disps_ix"]]
                else:
                    disps = np.zeros(1)
                    scaled_rates = np.zeros(1)
            else:
                disps = np.zeros_like(scaled_rates)
                if site_dict_i["disps_ix"] > 0:
                    disps[site_dict_i["disps_ix"]] = site_dict_i['disps']

            # average number of events per time interval (effectively R*T from Ned's guide)
            lambdas = investigation_time * np.array(scaled_rates)

            if load_random:
                # As a concession to not regenerating random samples and scenarios, randomly shift the loaded uncertainty arrays
                sample_shift = np.random.randint(-all_uncertainty.shape[0], all_uncertainty.shape[0])
                rupture_shift = np.random.randint(-all_uncertainty.shape[1], all_uncertainty.shape[1])
                disp_uncertainty = np.roll(all_uncertainty, (sample_shift, rupture_shift))[:, :lambdas.size]
                # Leave scenarios alone - no point rolling sample order, and can't shift sideways as can't appy one rupture's distribution
                # to another
                if site_dict_i["disps_ix"].shape[0] > 0:
                    scenarios = all_scenarios[:, site_dict_i["disps_ix"]]
                else:
                    scenarios = np.zeros((int(n_samples), 1))
                if benchmarking:
                    print(f"Time taken to prep random samples: {time() - begin:.5f} s")
                    lap = time()
            else:
                # Generate n_samples of possible earthquake ruptures for random 100 year intervals
                # returns boolean array where 0 means "no event" and 1 means "event". rows = 100 yr window, columns = earthquake
                # rupture
                scenarios = rng.poisson(lambdas, size=(int(n_samples), lambdas.size))

                # assigns a normal distribution with a mean of 1 and a standard deviation of sd
                # effectively a multiplier for the displacement value
                disp_uncertainty = rng.normal(1., sd, size=(int(n_samples), lambdas.size))
                if benchmarking:
                    print(f"Time taken to generate random samples: {time() - begin:.5f} s")
                    lap = time()

            # for each 100 yr scenario, get displacements from EQs that happened
            disp_scenarios = scenarios * disps
            # multiplies displacement by the uncertainty multiplier
            disp_scenarios = disp_scenarios * disp_uncertainty
            # sum all displacement values at that site in that 100 yr interval
            cumulative_disp_scenarios = disp_scenarios.sum(axis=1)
            if benchmarking:
                print(f"Calculated PPE: {time() - lap:.5f} s")
            lap = time()    

        # sum all the displacements in the 100 year window that exceed threshold
        cumulative_disp_scenarios = cumulative_disp_scenarios.reshape(1, len(cumulative_disp_scenarios))   
        # Find indexes of scenarios where slip occurred
        slip_scenarios = np.where(cumulative_disp_scenarios != 0)[1]    
        lap = time()
        n_exceedances_total_abs, n_exceedances_up, n_exceedances_down = calc_thresholds(thresholds, cumulative_disp_scenarios)
        if benchmarking:
            print(f"Cumulative Displacements : {time() - lap:.15f} s")
        lap = time()

        # the probability is the number of times that threshold was exceeded divided by the number of samples. so,
        # quite high for low displacements (25%). Means there's a ~25% chance an earthquake will exceed 0 m in next 100
        # years across all earthquakes in the catalogue (at that site).
        exceedance_probs_total_abs = n_exceedances_total_abs / n_samples
        exceedance_probs_up = n_exceedances_up / n_samples
        exceedance_probs_down = n_exceedances_down / n_samples

        # Minimum data needed for weighted_mean_PPE (done to reduce required storage, and if errors can be recalculated later if needed)
        site_PPE_dict[site_of_interest] = {"exceedance_probs_total_abs": exceedance_probs_total_abs[exceedance_probs_total_abs != 0],
                                           "exceedance_probs_up": exceedance_probs_up[exceedance_probs_up != 0],
                                           "exceedance_probs_down": exceedance_probs_down[exceedance_probs_down != 0]}

        # Save the rest of the data if this is a NSHM branch
        if NSHM_branch:
            chunked_disp_scenarios = cumulative_disp_scenarios[:(n_chunks * error_chunking)].reshape(n_chunks, error_chunking)

            n_exceedances_total_abs, n_exceedances_up, n_exceedances_down = calc_thresholds(thresholds, chunked_disp_scenarios, n_chunks=n_chunks)
            if benchmarking:
                print(f"Chunked Displacements : {time() - lap:.15f} s")

            lap = time()
            # the probability is the number of times that threshold was exceeded divided by the number of samples. so,
            # quite high for low displacements (25%). Means there's a ~25% chance an earthquake will exceed 0 m in next 100
            # years across all earthquakes in the catalogue (at that site).
            exceedance_errs_total_abs = n_exceedances_total_abs / error_chunking
            exceedance_errs_up = n_exceedances_up / error_chunking
            exceedance_errs_down = n_exceedances_down / error_chunking

            # Output errors
            sigma_lims = [2.27, 15.865, 84.135, 97.725]
            error_abs = np.percentile(exceedance_errs_total_abs, sigma_lims, axis=1)
            error_up = np.percentile(exceedance_errs_up, sigma_lims, axis=1)
            error_down = np.percentile(exceedance_errs_down, sigma_lims, axis=1)

            site_PPE_dict[site_of_interest].update({"scenario_displacements": cumulative_disp_scenarios[0, slip_scenarios],
                                                    "slip_scenarios_ix": slip_scenarios,
                                                    "site_coords": site_dict_i["site_coords"],
                                                    "standard_deviation": sd,
                                                    "error_total_abs": error_abs[:, error_abs.sum(axis=0) != 0],
                                                    "error_up": error_up[:, error_up.sum(axis=0) != 0],
                                                    "error_down": error_down[:, error_down.sum(axis=0) != 0],
                                                    "sigma_lims": sigma_lims,
                                                    "n_samples": n_samples,
                                                    "thresh_para": np.hstack([thresh_lims, thresh_step])})

        elapsed = time_elasped(time(), start)
        if benchmarking:
            print(f"Site Complete: {time() - begin:.5f} s\n")
        else:
            printProgressBar(i + 1, len(site_ids), prefix=f'\tProcessing {len(site_ids)} Sites:', suffix=f'Complete {elapsed} ({(time()-start) / (i + 1):.2f}s/site)', length=50)

    if benchmarking:
        print(f"Total Time: {time() - commence:.5f} s")

    if array_process:
        os.makedirs(f"../{model_version_results_directory}/{extension1}/site_cumu_exceed{scaling}", exist_ok=True)
        site_file = f"../{model_version_results_directory}/{extension1}/site_cumu_exceed{scaling}/{site_of_interest}.h5"
        if os.path.exists(site_file):
            os.remove(site_file)
        with h5.File(site_file, "w") as site_PPEh5:
            dict_to_hdf5(site_PPEh5, site_PPE_dict, compression='gzip', compression_opts=5)
    else:
        if extension1 != "" and scaling == "":
            with h5.File(f"../{model_version_results_directory}/{extension1}/cumu_exceed_prob_{extension1}{taper_extension}.h5", "w") as site_PPEh5:
                dict_to_hdf5(site_PPEh5, site_PPE_dict, compression='gzip', compression_opts=5)
        elif scaling != "":
            with h5.File(f"../{model_version_results_directory}/site_cumu_exceed{scaling}/{site_of_interest}.h5", "w") as site_PPEh5:
                dict_to_hdf5(site_PPEh5, site_PPE_dict, compression='gzip', compression_opts=5)
        else:
            return site_PPE_dict

def make_fault_model_PPE_dict(branch_weight_dict, model_version_results_directory, slip_taper, n_samples, outfile_extension, inv_sites=[],
                              nesi=False, nesi_step = None, hours : int = 0, mins: int= 3, mem: int= 5, cpus: int= 1, account: str= '',
                              time_interval=int(100), sd=0.4, n_array_tasks=1000, min_tasks_per_array=100, job_time=3, load_random=False,
                              remake_PPE=True, sbatch=False, thresh_lims=[0, 3], thresh_step=0.01):
    """ This function takes the branch dictionary and calculates the PPEs for each branch.
    It then combines the PPEs (key = unique branch ID).

    Must run this function with crustal, subduction, or a combination of two.

    :param crustal_branch_dict: from the function make_branch_weight_dict
    :param results_version_directory: string; path to the directory with the solution files
    :return mega_branch_PPE_dictionary and saves a pickle file.
    """

    gf_name = "sites"

    if slip_taper:
        taper_extension = "_tapered"
    else:
        taper_extension = "_uniform"

    if nesi:
        if nesi_step == 'prep' and os.path.exists(f"../{model_version_results_directory}/site_name_list.txt"):
            os.remove(f"../{model_version_results_directory}/site_name_list.txt")
        if nesi_step == 'combine':
            if os.path.exists(f"../{model_version_results_directory}/branch_combine_list.txt"):
                os.remove(f"../{model_version_results_directory}/branch_combine_list.txt")
            if os.path.exists(f"../{model_version_results_directory}/combine_site_meta.pkl"):
                os.remove(f"../{model_version_results_directory}/combine_site_meta.pkl")
            with open(f"../{model_version_results_directory}/combine_site_meta.pkl", "wb") as f:
                pkl.dump({}, f)

    branch_weight_list = []
    fault_model_allbranch_PPE_dict = {}
    combine_branches = 0
    for counter, branch_id in enumerate(branch_weight_dict.keys()):
        print(f"calculating {branch_id} PPE\t({counter + 1} of {len(branch_weight_dict.keys())} branches)")

        # get site displacement dictionary and branch weights
        extension1 = gf_name + branch_weight_dict[branch_id]["file_suffix"]
        branch_weight = branch_weight_dict[branch_id]["total_weight_RN"]
        branch_weight_list.append(branch_weight)
        rate_scaling_factor = branch_weight_dict[branch_id]["S"]

        branch_site_disp_dict_file = f"../{model_version_results_directory}/{extension1}/branch_site_disp_dict_{extension1}_S{str(rate_scaling_factor).replace('.', '')}.h5"
        if os.path.exists(branch_site_disp_dict_file):
            with h5.File(branch_site_disp_dict_file, 'r') as branch_h5:
                site_list = [site for site in branch_h5.keys() if "rates" not in site]
            missing_sites = [site for site in inv_sites if site not in site_list]
            if len(missing_sites) > 0:
                write_site_disp_dict(extension1, slip_taper=slip_taper, model_version_results_directory=model_version_results_directory, site_disp_h5file=branch_site_disp_dict_file)
                with h5.File(branch_site_disp_dict_file, "a") as branch_site_disp_dict:
                    branch_site_disp_dict.create_dataset("scaled_rates", data=branch_site_disp_dict["rates"][:] * rate_scaling_factor)

        else:
            # Extract rates from the NSHM solution directory, but it is not scaled by the rate scaling factor
            write_site_disp_dict(extension1, slip_taper=slip_taper, model_version_results_directory=model_version_results_directory, site_disp_h5file=branch_site_disp_dict_file)
            with h5.File(branch_site_disp_dict_file, "a") as branch_site_disp_dict:
                # multiply each value in the rates array by the rate scaling factor
                branch_site_disp_dict.create_dataset("scaled_rates", data=branch_site_disp_dict["rates"][:] * rate_scaling_factor)
                site_list = [site for site in branch_site_disp_dict.keys() if not site in ["rates", "scaled_rates"]]

        branch_cumu_PPE_dict_file = f"../{model_version_results_directory}/{extension1}/{branch_id}_cumu_PPE.h5"
        fault_model_allbranch_PPE_dict[branch_id] = branch_cumu_PPE_dict_file

        # Reduce site list to only those that have not been processed or not processed to the required number of samples
        thresholds = np.round(np.arange(thresh_lims[0], thresh_lims[1] + thresh_step, thresh_step), 4)
        if os.path.exists(fault_model_allbranch_PPE_dict[branch_id]):
            print('\tChecking for existing PPE at each site...')
            with h5.File(fault_model_allbranch_PPE_dict[branch_id], "r") as branch_PPEh5:
                # Checks that sites have been processed
                existing_sites = [site for site in branch_PPEh5.keys() if site not in ["branch_weight", "thresholds"]]
                # Checks that previous processing had required sampling (i.e. wasn't a testing run)
                well_processed_sites = []
                for site in existing_sites:
                    if all([True if key in branch_PPEh5[site].keys() else False for key in ['n_samples', 'thresh_para']]):
                        if branch_PPEh5[site]['n_samples'][()] >= n_samples:
                            well_processed_sites.append(site)
                #             # Check previous processing had required threshold limits  # Might be redundant, unless you want to plot only the single branch
                #             proc_thresholds = np.round(np.arange(branch_PPEh5[site]['thresh_para'][0], branch_PPEh5[site]['thresh_para'][1] + branch_PPEh5[site]['thresh_para'][2], branch_PPEh5[site]['thresh_para'][2]), 4)
                #             if np.in1d(thresholds, proc_thresholds).all():
                #                 well_processed_sites.append(site)

            prep_list = [site for site in inv_sites if site not in well_processed_sites]
        else:
            branch_PPEh5 = h5.File(fault_model_allbranch_PPE_dict[branch_id], "w")
            branch_PPEh5.close()
            prep_list = inv_sites
        
        if len(prep_list) == 0:
            print(f"\tAll sites have been processed for {branch_id}. Skipping...")
            continue
        else:
            print(f"\t{len(prep_list)}/{len(inv_sites)} requested sites need processing for {branch_id}...")
            remake_PPE = True

        ### get exceedance probability dictionary
        if nesi:
            if nesi_step == 'prep':
                if load_random:
                    randdir = f"../{model_version_results_directory}/{extension1}/site_cumu_exceed_S{str(rate_scaling_factor).replace('.', '')}"
                    prepare_random_arrays(branch_site_disp_dict_file, randdir, time_interval, n_samples, sd)

                print(f"\tPrepping for NESI....")
                prep_nesi_site_list(model_version_results_directory, prep_list, extension1, S=f"_S{str(rate_scaling_factor).replace('.', '')}")
                continue
            elif nesi_step == 'combine':
                if sbatch:
                    print(f"\tPreparing NESI combination for {fault_model_allbranch_PPE_dict[branch_id]}....")
                    prep_combine_branch_list(branch_site_disp_dict_file, model_version_results_directory, extension1, branch_h5file=fault_model_allbranch_PPE_dict[branch_id],
                                        taper_extension=taper_extension, S=f"_S{str(rate_scaling_factor).replace('.', '')}", weight=branch_weight_list[-1], thresholds=thresholds)
                    combine_branches += 1
                    continue
                else:
                    print(f"\tCombining site dictionaries into {fault_model_allbranch_PPE_dict[branch_id]}....")
                    compile_site_cumu_PPE(prep_list, model_version_results_directory, extension1, branch_h5file=fault_model_allbranch_PPE_dict[branch_id],
                                        taper_extension=taper_extension, S=f"_S{str(rate_scaling_factor).replace('.', '')}", thresholds=thresholds)

        else:
            if os.path.exists(fault_model_allbranch_PPE_dict[branch_id]) and not remake_PPE:
                print(f"\tFound Pre-Prepared Branch PPE:  {fault_model_allbranch_PPE_dict[branch_id]}")
            else:
                if load_random:
                    randdir = f"../{model_version_results_directory}"
                    prepare_random_arrays(branch_site_disp_dict_file, randdir, time_interval, n_samples, sd)

                branch_cumu_PPE_dict = get_cumu_PPE(branch_key=branch_id, branch_site_disp_dict=branch_site_disp_dict_file,
                                                    site_ids=prep_list, slip_taper=slip_taper, load_random=load_random,
                                                    model_version_results_directory=model_version_results_directory,
                                                    time_interval=time_interval, n_samples=n_samples, extension1="",
                                                    thresh_lims=thresh_lims, thresh_step=thresh_step)
                with h5.File(fault_model_allbranch_PPE_dict[branch_id], "r+") as branch_PPEh5:
                    dict_to_hdf5(branch_PPEh5, branch_cumu_PPE_dict, replace_groups=True)

        if not all([nesi, nesi_step == 'combine', sbatch]):
            if os.path.exists(fault_model_allbranch_PPE_dict[branch_id]):
                with h5.File(fault_model_allbranch_PPE_dict[branch_id], "r+") as branch_PPEh5:
                    if 'branch_weight' in branch_PPEh5.keys():
                        del branch_PPEh5['branch_weight']
                    branch_PPEh5.create_dataset('branch_weight', data=branch_weight_list[-1])

    n_sites = len(prep_list)
    if nesi and nesi_step == 'prep':
        n_jobs = len(branch_weight_dict.keys()) * n_sites
        tasks_per_array = np.ceil(n_jobs / n_array_tasks)
        if tasks_per_array < min_tasks_per_array:
            tasks_per_array = min_tasks_per_array
        array_time = job_time * tasks_per_array
        hours, secs = divmod(array_time, 3600)
        mins = np.ceil(secs / 60)
        n_tasks = int(np.ceil(n_jobs / tasks_per_array))
        print('\nCreating SLURM submission script....')
        prep_SLURM_submission(model_version_results_directory, int(tasks_per_array), int(n_tasks), hours=int(hours), mins=int(mins), job_time=job_time, mem=mem, cpus=cpus,
                              account=account, time_interval=time_interval, n_samples=n_samples, sd=sd, thresh_lims=thresh_lims, thresh_step=thresh_step)
        raise Exception(f"Now run\n\tsbatch ../{model_version_results_directory}/cumu_PPE_slurm_task_array.sl")

    elif nesi and nesi_step == 'combine' and sbatch:
        tasks_per_array = np.ceil(combine_branches / n_array_tasks)
        min_branches_per_array = 1
        if tasks_per_array < min_branches_per_array:
            tasks_per_array = min_branches_per_array
        time_per_site = 0.2
        array_time = 60 + time_per_site * n_sites * tasks_per_array
        hours, secs = divmod(array_time, 3600)
        mins = np.ceil(secs / 60)
        n_tasks = int(np.ceil(combine_branches / tasks_per_array))
        print('\nCreating SLURM submission script....')
        combine_dict_file = f"../{model_version_results_directory}/combine_site_meta.pkl"
        branch_combine_list_file = f"../{model_version_results_directory}/branch_combine_list.txt"
        prep_SLURM_combine_submission(combine_dict_file, branch_combine_list_file, model_version_results_directory,
                                      int(tasks_per_array), int(n_tasks), hours=int(hours), mins=int(mins), mem=10, account=account)
        raise Exception(f"Now run\n\tsbatch ../{model_version_results_directory}/combine_sites.sl")
    else:
        print('Building all branch PPE dictionary....')
        branch_list = [branch_id for branch_id in branch_weight_dict.keys()]

        site_coords_dict = {}
        with h5.File(branch_site_disp_dict_file, "r") as branch_h5:
            for site in site_list:
                site_coords_dict[site] = branch_h5[site]['site_coords'][:]
        fault_model_allbranch_PPE_dict['meta'] = {'branch_ids': branch_list, 'site_ids': site_list, 'branch_weights': branch_weight_list, 'site_coords_dict': site_coords_dict}

        outfile_name = f"all_branch_PPE_dict{outfile_extension}{taper_extension}"
        print(f"\nSaving {model_version_results_directory}/{outfile_name}.pkl....")
        with open(f"../{model_version_results_directory}/{outfile_name}.pkl", "wb") as f:
            pkl.dump(fault_model_allbranch_PPE_dict, f)

        return fault_model_allbranch_PPE_dict

def get_weighted_mean_PPE_dict(fault_model_PPE_dict, out_directory, outfile_extension, slip_taper, site_list=[], thresh_lims=[0, 3], thresh_step=0.01, nesi=False, nesi_step='prep', n_samples=100000,
                               min_tasks_per_array=100, n_array_tasks=100, mem=10, cpus=1, account='', job_time=60):
    """takes all the branch PPEs and combines them based on the branch weights into a weighted mean PPE dictionary

    :param fault_model_PPE_dict: The dictionary has PPEs for each branch (or branch pairing).
    Each branch contains "branch_weight" and "cumu_PPE_dict".
    "cumu_PPE_dict" is organized by site. Nested in sites is "thresholds", "exceedance_probs_up",
    "exceedance_probs_down", and "exceedance_probs_total_abs"
    :return dictionary of sites, with lists of weighted mean PPEs and threshold values.
    """
    start = time()
    if slip_taper:
        taper_extension = "_tapered"
    else:
        taper_extension = "_uniform"

    unique_id_list = fault_model_PPE_dict['meta']['branch_ids']
    # site_list = fault_model_PPE_dict['meta']['site_ids']

    # weight the probabilities by NSHM branch weights to get a weighted mean
    branch_weights = np.array(fault_model_PPE_dict['meta']['branch_weights'])

    # need a more elegant solution to this I think
    thresholds = np.round(np.arange(thresh_lims[0], thresh_lims[1] + thresh_step, thresh_step), 4)

    # extract site coordinates from fault model PPE dictionary
    site_coords_dict = fault_model_PPE_dict['meta']['site_coords_dict']

    # Create variables
    sigma_lims = [2.275, 15.865, 84.135, 97.725]
    sigma_lims.sort()
    exceed_type_list = ["total_abs", "up", "down"]

    # Check if previous h5 exists. If it does, preserve it until new weighted mean file is complete
    weighted_h5_file = f"../{out_directory}/weighted_mean_PPE_dict{outfile_extension}{taper_extension}.h5"
    preserve_previous_h5 = False
    # if os.path.exists(weighted_h5_file):
    #     weighted_h5_file = weighted_h5_file.replace('.h5', '_processing.h5')
    #     preserve_previous_h5 = True
    #     if os.path.exists(weighted_h5_file):
    #         os.remove(weighted_h5_file)

    meta_dict = {'branch_weights': branch_weights,
                 'thresholds': thresholds,
                 'branch_ids': np.array(unique_id_list, dtype='S'),
                 'sigma_lims': sigma_lims}

    if not os.path.exists(weighted_h5_file):
        weighted_h5 = h5.File(weighted_h5_file, "w")
        for key in meta_dict.keys():
            weighted_h5.create_dataset(key, data=meta_dict[key])
    else:
        weighted_h5 = h5.File(weighted_h5_file, "r+")
        for key in meta_dict.keys():
            if not np.array_equal(weighted_h5[key][:], meta_dict[key]):
                if key == 'thresholds':
                    # Check to see if the previous run had the same thrshold start and step, just a lower maximum limit
                    if np.array_equal(weighted_h5[key][:], meta_dict[key][:weighted_h5[key][:].shape[0]]):
                        print(f"Previous threshold limits were different, but the same start and step. Extending the limits to match the new limits...")
                        del weighted_h5[key]
                        weighted_h5[key] = meta_dict[key]
                    else:
                        Exception(f"Meta data for cannot match newly requested thresholds to those from previous runs....")
                else:
                    print(f"Meta data for **{key}** does not match what was in the weighted.h5...")

    model_directory = '/'.join(fault_model_PPE_dict[unique_id_list[0]].split('/')[1:-2])

    n_sites = len(site_list)
    if not nesi:
        start = time()
        printProgressBar(0, n_sites, prefix=f'\t0/{n_sites} sites:', suffix='00:00:00 0 s/site', length=50)
        for ix, site in enumerate(site_list):
            if site not in weighted_h5.keys():
                site_group = weighted_h5.create_group(site)
                site_group.create_dataset("site_coords", data=site_coords_dict[site])
                create_site_weighted_mean(site_group, site, n_samples, model_directory, [model_directory], 'sites', thresholds, exceed_type_list, unique_id_list, sigma_lims, branch_weights, compression=None)
            elapsed = time_elasped(time(), start, decimal=False)
            printProgressBar(ix + 1, len(site_list), prefix=f'\tProcessing Site {site}', suffix=f'Complete {elapsed} ({(time()-start) / (ix + 1):.2f}s/site)', length=50)
        weighted_h5.close()
    else:
        if nesi_step == 'prep':
            weighted_h5.close()
            os.makedirs(f"../{out_directory}/weighted_sites", exist_ok=True)
            for ix, site in enumerate(site_list):
                print(f'Preparing site {site} for NESI task array... ({100 * ix / len(site_list):.0f}%)', end ='\r')
                site_h5_file = f"../{out_directory}/weighted_sites/{site}.h5"
                if os.path.exists(site_h5_file):
                    os.remove(site_h5_file)
                site_meta_dict = {'site_coords': site_coords_dict[site],
                                  'n_samples': n_samples,
                                  'crustal_model_version_results_directory': model_directory,
                                  'sz_model_version_results_directory_list': [model_directory],
                                  'gf_name': 'sites',
                                  'thresholds': thresholds,
                                  'exceed_type_list': exceed_type_list,
                                  'branch_id_list': unique_id_list,
                                  'sigma_lims': sigma_lims,
                                  'branch_weights': branch_weights}
                with h5.File(site_h5_file, 'w') as site_h5:
                    dict_to_hdf5(site_h5, site_meta_dict)
            
            site_file = f"../{out_directory}/weighted_sites/site_list.txt"
            with open(site_file, "w") as f:
                for site in site_list:
                    f.write(f"../{out_directory}/weighted_sites/{site}.h5\n")
    
            n_sites = len(site_list)
            tasks_per_array = np.ceil(n_sites / n_array_tasks)
            if tasks_per_array < min_tasks_per_array:
                tasks_per_array = min_tasks_per_array
            n_array_tasks = int(np.ceil(n_sites / tasks_per_array))
            array_time = 60 + job_time * tasks_per_array
            hours, rem = divmod(array_time, 3600)
            mins = np.ceil(rem / 60)

            slurm_file = prep_SLURM_weighted_sites_submission(out_directory, tasks_per_array, n_array_tasks, site_file,
                                         hours=int(hours), mins=int(mins), mem=mem, cpus=cpus, account=account, job_time=job_time)

            raise Exception(f"Now run\n\tsbatch {slurm_file}")
        
        elif nesi_step == 'combine':
            start = time()
            printProgressBar(0, len(site_list), prefix=f'\tAdding Site {site_list[0]}', suffix='Complete 00:00:00 (00:00s/site)', length=50)
            for ix, site in enumerate(site_list):
                site_h5_file = f"../{out_directory}/weighted_sites/{site}.h5"
                site_group = weighted_h5.create_group(site)
                with h5.File(site_h5_file, 'r') as site_h5:
                    site_group.create_dataset("site_coords", data=site_h5['site_coords'])
                    for exceed_type in exceed_type_list:
                        for dataset in ['weighted_exceedance_probs_*-*', '*-*_max_vals', '*-*_min_vals', 'branch_exceedance_probs_*-*']:
                            site_group.create_dataset(dataset.replace('*-*', exceed_type), data=site_h5[dataset.replace('*-*', exceed_type)], compression=None)
                elapsed = time_elasped(time(), start, decimal=False)
                printProgressBar(ix + 1, len(site_list), prefix=f'\tAdding Site {site}', suffix=f'Complete {elapsed} ({(time()-start) / (ix + 1):.2f}s/site)', length=50)
            weighted_h5.close()
            shutil.rmtree(f"../{out_directory}/weighted_sites")


    if preserve_previous_h5:
        os.remove(weighted_h5_file.replace('_processing.h5', '.h5'))
        os.rename(weighted_h5_file, weighted_h5_file.replace('_processing.h5', '.h5'))
        weighted_h5_file = weighted_h5_file.replace('_processing.h5', '.h5')

    return weighted_h5_file


def make_sz_crustal_paired_PPE_dict(crustal_branch_weight_dict, sz_branch_weight_dict_list,
                                    crustal_model_version_results_directory, sz_model_version_results_directory_list,
                                    paired_PPE_pickle_name, slip_taper, n_samples, out_directory, outfile_extension, sz_type_list,
                                    nesi=False, nesi_step='prep', hours : int = 0, mins: int= 3, mem: int= 5, cpus: int= 1, account: str= '',
                                    n_array_tasks=100, min_tasks_per_array=100, time_interval=int(100), sd=0.4, job_time=3, remake_PPE=True, load_random=False,
                                    sbatch=True, thresh_lims=[0, 3], thresh_step=0.01, site_gdf=None):
    """ This function takes the branch dictionary and calculates the PPEs for each branch.
    It then combines the PPEs (key = unique branch ID).

    Must run this function with crustal, subduction, or a combination of two.
    :param crustal_branch_dict: from the function make_branch_weight_dict
    :param results_version_directory: string; path to the directory with the solution files
    :return mega_branch_PPE_dictionary and saves a pickle file.
    """

    gf_name = "sites"

    if slip_taper:
        taper_extension = "_tapered"
    else:
        taper_extension = "_uniform"

    if nesi:
        if nesi_step == 'prep' and os.path.exists(f"../{out_directory}/site_name_list.txt"):
            os.remove(f"../{out_directory}/site_name_list.txt")
        if nesi_step == 'combine':
            if os.path.exists(f"../{out_directory}/branch_combine_list.txt"):
                os.remove(f"../{out_directory}/branch_combine_list.txt")
            if os.path.exists(f"../{out_directory}/combine_site_meta.pkl"):
                os.remove(f"../{out_directory}/combine_site_meta.pkl")
                with open(f"../{out_directory}/combine_site_meta.pkl", "wb") as f:
                    pkl.dump({}, f)

    # Make crustal_sz pair list
    all_crustal_branches_site_disp_dict = get_all_branches_site_disp_dict(crustal_branch_weight_dict, gf_name, slip_taper,
                                                                          crustal_model_version_results_directory)
    
    with h5.File(all_crustal_branches_site_disp_dict[list(crustal_branch_weight_dict.keys())[0]]["site_disp_dict"]) as crust_h5:
        processed_site_names = [site for site in crust_h5.keys() if "rates" not in site]

    if isinstance(site_gdf, list):
        site_names = site_gdf
    else:
        site_names = site_gdf['siteId'].values.tolist()
        site_gdf['tree_count'] = 0

    # make a dictionary of displacements at each site from all the crustal earthquake scenarios
    all_sz_branches_site_disp_dict = {}
    crustal_sz_branch_pairs = list(crustal_branch_weight_dict.keys())
    for ix, sz_branch_weight_dict in enumerate(sz_branch_weight_dict_list):
        all_single_sz_branches_site_disp_dict = get_all_branches_site_disp_dict(sz_branch_weight_dict, gf_name, slip_taper,
                                                                                sz_model_version_results_directory_list[ix])

        with h5.File(all_single_sz_branches_site_disp_dict[list(sz_branch_weight_dict.keys())[0]]["site_disp_dict"]) as sz_h5:
            processed_site_names = list(set(processed_site_names + [site for site in sz_h5.keys() if "rates" not in site]))
            processed_site_names.sort()

        if not isinstance(site_gdf, list):
            site_gdf['tree_count'] += [1 if site in processed_site_names else 0 for site in site_names]
        all_sz_branches_site_disp_dict = all_sz_branches_site_disp_dict | all_single_sz_branches_site_disp_dict

        # make all the combinations of crustal and subduction zone branch pairs
        crustal_sz_branch_pairs = list(itertools.product(crustal_sz_branch_pairs,
                                                            sz_branch_weight_dict.keys()))

        if isinstance(crustal_sz_branch_pairs[0][0], tuple):
            crustal_sz_branch_pairs = [t1 + tuple([t2]) for t1, t2 in crustal_sz_branch_pairs]

    if not isinstance(site_gdf, list):
        site_gdf.to_file(f"../{out_directory}/{out_directory.split('/')[-1]}_site_locations.geojson", driver="GeoJSON")
    missing_sites = [site for site in site_names if site not in processed_site_names]
    assert len(missing_sites) == 0, f"Missing sites: {missing_sites}"

    pair_weight_list = []
    pair_id_list = []
    paired_crustal_sz_PPE_dict = {}
    for pair in crustal_sz_branch_pairs:
        # get the branch unique ID for the crustal and sz combos
        crustal_unique_id, sz_unique_ids = pair[0], pair[1:]
        pair_unique_id = crustal_unique_id
        branch_unique_ids = [crustal_unique_id]
        for sz_unique_id in sz_unique_ids:
            pair_unique_id += "_-_" + sz_unique_id
            branch_unique_ids.append(sz_unique_id)
        pair_id_list.append(pair_unique_id)
        pair_cumu_PPE_dict_file = f"../{out_directory}/{pair_unique_id}/{pair_unique_id}_cumu_PPE.h5"
        paired_crustal_sz_PPE_dict[pair_unique_id] = pair_cumu_PPE_dict_file

        pair_weight = all_crustal_branches_site_disp_dict[crustal_unique_id]["branch_weight"]
        
        for sz_unique_id in sz_unique_ids:
            pair_weight = pair_weight * all_sz_branches_site_disp_dict[sz_unique_id]["branch_weight"]
        pair_weight_list.append(pair_weight)


    # Create variables
    thresholds = np.round(np.arange(thresh_lims[0], thresh_lims[1] + thresh_step, thresh_step), 4)
    sigma_lims = [2.275, 15.865, 84.135, 97.725]
    sigma_lims.sort()
    exceed_type_list = ["total_abs", "up", "down"]

    # Check if previous h5 exists. If it does, preserve it until new weighted mean file is complete
    weighted_h5_file = f"../{out_directory}/weighted_mean_PPE_dict{outfile_extension}{taper_extension}.h5"
    preserve_previous_h5 = False
    if os.path.exists(weighted_h5_file):
        weighted_h5_file = weighted_h5_file.replace('.h5', '_processing.h5')
        preserve_previous_h5 = True
        if os.path.exists(weighted_h5_file):
            os.remove(weighted_h5_file)

    # Create weighted h5 file with associated metadata
    weighted_h5 = h5.File(weighted_h5_file, "w")
    
    weighted_h5.create_dataset('branch_weights', data=pair_weight_list)
    weighted_h5.create_dataset('branch_ids', data=pair_id_list)
    weighted_h5.create_dataset("thresholds", data=thresholds)
    weighted_h5.create_dataset('sigma_lims', data=sigma_lims)
    
    if not nesi:
        start = time()
        printProgressBar(0, len(site_names), prefix=f'\tProcessing Site {site_names[0]}', suffix='Complete 00:00:00 (00:00s/site)', length=50)
        for ix, site in enumerate(site_names):
            site_group = weighted_h5.create_group(site)
            with h5.File(all_crustal_branches_site_disp_dict[crustal_unique_id]["site_disp_dict"], 'r') as branch_site_h5:
                site_group.create_dataset("site_coords", data=branch_site_h5[site]["site_coords"])
            create_site_weighted_mean(site_group, site, n_samples, crustal_model_version_results_directory, sz_model_version_results_directory_list, gf_name, thresholds,
                                      exceed_type_list, pair_id_list, sigma_lims, pair_weight_list)
            elapsed = time_elasped(time(), start, decimal=False)
            printProgressBar(ix + 1, len(site_names), prefix=f'\tProcessing Site {site}', suffix=f'Complete {elapsed} ({(time()-start) / (ix + 1):.2f}s/site)', length=50)
        weighted_h5.close()
    
    else:
        if nesi_step == 'prep':
            weighted_h5.close()
            os.makedirs(f"../{out_directory}/weighted_sites", exist_ok=True)
            for ix, site in enumerate(site_names):
                print(f'Preparing site {site} for NESI task array...', end ='\r')
                site_h5_file = f"../{out_directory}/weighted_sites/{site}.h5"
                if os.path.exists(site_h5_file):
                    os.remove(site_h5_file)
                with h5.File(all_crustal_branches_site_disp_dict[crustal_unique_id]["site_disp_dict"], 'r') as branch_site_h5:
                    site_meta_dict = {'site_coords': branch_site_h5[site]["site_coords"][:],
                                    'n_samples': n_samples,
                                    'crustal_model_version_results_directory': crustal_model_version_results_directory,
                                    'sz_model_version_results_directory_list': sz_model_version_results_directory_list,
                                    'gf_name': gf_name,
                                    'thresholds': thresholds,
                                    'exceed_type_list': exceed_type_list,
                                    'branch_id_list': pair_id_list,
                                    'sigma_lims': sigma_lims,
                                    'branch_weights': pair_weight_list}
                with h5.File(site_h5_file, 'w') as site_h5:
                    dict_to_hdf5(site_h5, site_meta_dict)
            
            site_file = f"../{out_directory}/weighted_sites/site_list.txt"
            with open(site_file, "w") as f:
                for site in site_names:
                    f.write(f"../{out_directory}/weighted_sites/{site}.h5\n")
    
            n_sites = len(site_names)
            tasks_per_array = np.ceil(n_sites / n_array_tasks)
            if tasks_per_array < min_tasks_per_array:
                tasks_per_array = min_tasks_per_array
            n_array_tasks = int(np.ceil(n_sites / tasks_per_array))
            array_time = 60 + job_time * tasks_per_array
            hours, rem = divmod(array_time, 3600)
            mins = np.ceil(rem / 60)

            slurm_file = prep_SLURM_weighted_sites_submission(out_directory, tasks_per_array, n_array_tasks, site_file,
                                         hours=int(hours), mins=int(mins), mem=mem, cpus=cpus, account=account, job_time=job_time)

            raise Exception(f"Now run\n\tsbatch {slurm_file}")
        
        elif nesi_step == 'combine':
            start = time()
            printProgressBar(0, len(site_names), prefix=f'\tAdding Site {site_names[0]}', suffix='Complete 00:00:00 (00:00s/site)', length=50)
            for ix, site in enumerate(site_names):
                site_h5_file = f"../{out_directory}/weighted_sites/{site}.h5"
                site_group = weighted_h5.create_group(site)
                with h5.File(site_h5_file, 'r') as site_h5:
                    site_group.create_dataset("site_coords", data=site_h5['site_coords'])
                    for exceed_type in exceed_type_list:
                        for dataset in ['weighted_exceedance_probs_*-*', '*-*_max_vals', '*-*_min_vals', 'branch_exceedance_probs_*-*']:
                            site_group.create_dataset(dataset.replace('*-*', exceed_type), data=site_h5[dataset.replace('*-*', exceed_type)], compression=None)
                elapsed = time_elasped(time(), start, decimal=False)
                printProgressBar(ix + 1, len(site_names), prefix=f'\tAdding Site {site}', suffix=f'Complete {elapsed} ({(time()-start) / (ix + 1):.2f}s/site)', length=50)
            weighted_h5.close()
            shutil.rmtree(f"../{out_directory}/weighted_sites")


    if preserve_previous_h5:
        os.remove(weighted_h5_file.replace('_processing.h5', '.h5'))
        os.rename(weighted_h5_file, weighted_h5_file.replace('_processing.h5', '.h5'))
        weighted_h5_file = weighted_h5_file.replace('_processing.h5', '.h5')
    
    return

def create_site_weighted_mean(site_h5, site, n_samples, crustal_directory, sz_directory_list, gf_name, thresholds, exceed_type_list, pair_id_list, sigma_lims, branch_weights, compression=None):    
        site_df_abs = {}
        site_df_up = {}
        site_df_down = {}
        for pair_id in pair_id_list:
            cumulative_disp_scenarios = np.zeros(n_samples)
            for branch in pair_id.split('_-_'):
                fault_tag = branch.split('_')[-2]
                branch_tag = branch.split('_')[-1]
                if '_sz_' in branch:
                    fault_type = 'sz'
                    sz_name = 'hikkerk'
                elif '_py_' in branch:
                    fault_type = 'py'
                    sz_name = 'puysegur'
                if fault_tag == 'fq':
                    fault_type += '_fq'
                    sz_name = 'fq_' + sz_name

                if fault_tag == 'c':
                    fault_type = fault_tag
                    fault_dir = crustal_directory
                else:
                    fault_dir = next((sz_dir for sz_dir in sz_directory_list if f'/{sz_name}' in sz_dir))
                NSHM_file = f"../{fault_dir}/{gf_name}_{fault_type}_{branch_tag}/{branch}_cumu_PPE.h5"
                with h5.File(NSHM_file, 'r') as NSHM_h5:
                    if site in NSHM_h5.keys():
                        NSHM_displacements = NSHM_h5[site]['scenario_displacements'][:]
                        slip_scenarios = NSHM_h5[site]['slip_scenarios_ix'][:]
                        cumulative_disp_scenarios[slip_scenarios] += NSHM_displacements.reshape(-1)

            n_exceedances_total_abs, n_exceedances_up, n_exceedances_down = calc_thresholds(thresholds, cumulative_disp_scenarios.reshape(1, -1))
            site_df_abs[pair_id] = (n_exceedances_total_abs / n_samples).reshape(-1)
            site_df_up[pair_id] = (n_exceedances_up / n_samples).reshape(-1)
            site_df_down[pair_id] = (n_exceedances_down / n_samples).reshape(-1)

        site_df_dict = {"total_abs": site_df_abs, "up": site_df_up, "down": site_df_down}
        for exceed_type in exceed_type_list:
            for dataset in ['weighted_exceedance_probs_*-*', '*-*_max_vals', '*-*_min_vals', 'branch_exceedance_probs_*-*', '*-*_weighted_percentile_error']:
                if dataset.replace('*-*', exceed_type) in site_h5.keys():
                    del site_h5[dataset.replace('*-*', exceed_type)]
            site_probabilities_df = pd.DataFrame(site_df_dict[exceed_type])

            # collapse each row into a weighted mean value
            branch_weighted_mean_probs = site_probabilities_df.apply(
                lambda x: np.average(x, weights=branch_weights), axis=1)
            site_max_probs = site_probabilities_df.max(axis=1)
            site_min_probs = site_probabilities_df.min(axis=1)

            site_h5.create_dataset(f"weighted_exceedance_probs_{exceed_type}", data=branch_weighted_mean_probs, compression=compression)
            site_h5.create_dataset(f"{exceed_type}_max_vals", data=site_max_probs, compression=compression)
            site_h5.create_dataset(f"{exceed_type}_min_vals", data=site_min_probs, compression=compression)
            site_h5.create_dataset(f"branch_exceedance_probs_{exceed_type}", data=site_probabilities_df.to_numpy(), compression='gzip', compression_opts=6)
            site_h5[f'branch_exceedance_probs_{exceed_type}'].attrs['branch_ids'] = pair_id_list

            # Calculate errors based on 1 and 2 sigma WEIGHTED percentiles of all of the branches for each threshold (better option)
            percentiles = percentile(site_probabilities_df, sigma_lims, axis=1, weights=branch_weights)
            site_h5.create_dataset(f"{exceed_type}_weighted_percentile_error", data=percentiles, compression=compression)

def get_exceedance_bar_chart_data(site_PPE_dictionary, probability, exceed_type, site_list, weighted=False, err_index=None):
    """returns displacements at the X% probabilities of exceedance for each site

    define exceedance type. Options are "total_abs", "up", "down"
    """

    if weighted:
        prefix = 'weighted_'
    else:
        prefix = ''

    thresholds = np.array([round(val, 4) for val in site_PPE_dictionary["thresholds"]])

    # displacement thresholds are negative for "down" exceedances
    if exceed_type == "down":
        thresholds = -thresholds

    # get disp threshold (x-value) at defined probability (y-value)
    disps = []
    if err_index:
        errs = []

    for site in site_list:
        site_PPE = site_PPE_dictionary[site][f"{prefix}exceedance_probs_{exceed_type}"]

        # get first index that is < 10% (ideally we would interpolate for exact value but don't have a function)
        exceedance_index = next((index for index, value in enumerate(site_PPE) if value <= round(probability,4)), -1)
        disp = thresholds[exceedance_index]
        disps.append(disp)

        if err_index:
            site_err_PPE = site_PPE_dictionary[site][f'{exceed_type}_{prefix}percentile_error']
            site_err = []
            for err_ind in err_index:
                exceedance_index = next((index for index, value in enumerate(site_err_PPE[err_ind, :]) if value <= round(probability,4)), -1)
                site_err.append(thresholds[exceedance_index])
            errs.append(site_err)

    if err_index:
        return disps, errs
    else:
        return disps


def get_probability_bar_chart_data(site_PPE_dictionary, exceed_type, threshold, site_list=None, weighted=False):
    """ function that finds the probability at each site for the specified displacement threshold on the hazard curve
        Inputs:
        :param: dictionary of exceedance probabilities for each site (key = site)
        :param exceedance type: string; "total_abs", "up", or "down"
        :param: list of sites to get data for. If None, will get data for all sites in site_PPE_dictionary.
                I made this option so that you could skip the sites you didn't care about (e.g., use "plot_order")

        Outputs:
        :return    probs_threshold: list of probabilities of exceeding the specified threshold (one per site)
            """

    if weighted:
        prefix = 'weighted_'
    else:
        prefix = ''	

    if site_list == None:
        site_list = list(site_PPE_dictionary.keys())

    thresholds = [round(val, 4) for val in site_PPE_dictionary["thresholds"]]
    # find index in thresholds where the value matches the parameter threshold
    index = thresholds.index(round(threshold, 4))

    # get list of probabilities at defined displacement threshold (one for each site)
    probs_threshold = []
    for site in site_list:
        site_PPE = site_PPE_dictionary[site][f"{prefix}exceedance_probs_{exceed_type}"]
        probs_threshold.append(site_PPE[index])

    return probs_threshold


def plot_branch_hazard_curve(extension1, slip_taper, model_version_results_directory, file_type_list, plot_order=[]):
    """makes hazard curves for each site. includes the probability of cumulative displacement from multiple
    earthquakes exceeding a threshold in 100 years."""

    exceed_type_list = ["total_abs", "up", "down"]

    if slip_taper is True:
        taper_extension = "_tapered"
    else:
        taper_extension = "_uniform"

    with open(f"../{model_version_results_directory}/{extension1}/cumu_exceed_prob_{extension1}"
              f"{taper_extension}.pkl",
              "rb") as fid:
        PPE_dictionary = pkl.load(fid)

    plt.close("all")

    if not plot_order:  # Take default plot order from the dictionary keys
        plot_order = [key for key in PPE_dictionary.keys()]

    n_plots = int(np.ceil(len(plot_order) / 12))
    printProgressBar(0, n_plots, prefix = '\tCompleted Plots:', suffix = 'Complete', length = 50)

    for plot_n in range(n_plots):
        sites = plot_order[plot_n*12:(plot_n+1)*12]
        if len(sites) >= 5 or len(sites) == 3:
            n_cols = 3
            n_rows = int(np.ceil(len(sites) / 3))
        elif len(sites) == 4 or len(sites) == 2:
            n_cols = 2
            n_rows = int(np.ceil(len(sites) / 2))
        else:
            n_cols = 1
            n_rows = len(sites)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.63 + 0.12, n_rows * 2.32 + 0.71))  # Replicate 8x10 inch figure if there are less than 12 subplots
        plt.subplots_adjust(hspace=0.3, wspace=0.3)

        #loop through sites and make a subplot for each one
        for i, site in enumerate(sites):
            ax = plt.subplot(n_rows, n_cols, i + 1)

            # plots all three types of exceedance (total_abs, up, down) on the same plot
            for j, exceed_type in enumerate(exceed_type_list):
                curve_color = get_probability_color(exceed_type)
                exceedance_probs = PPE_dictionary[site][f"exceedance_probs_{exceed_type}"]
                thresholds = PPE_dictionary[site]["thresholds"]

                ax.plot(thresholds, exceedance_probs, color=curve_color)
                ax.axhline(y=0.02, color="0.7", linestyle='dashed')
                ax.axhline(y=0.1, color="0.7", linestyle='dotted')

            ax.set_title(site)
            ax.set_yscale('log'), ax.set_xscale('log')
            ax.set_yticks([0.00001, 0.0001, 0.001, 0.01, 0.1, 1])
            ymin, ymax = 0.000005, 1
            ax.set_ylim([ymin, ymax])
            ax.get_xaxis().set_major_formatter(ScalarFormatter())
            ax.ticklabel_format(axis='x', style='plain')
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        fig.text(0.5, 0, 'Vertical displacement threshold (m)', ha='center')
        fig.text(0, 0.5, 'Probability of exceedance in 100 years', va='center', rotation='vertical')
        fig.suptitle(f"cumulative exceedance hazard curves \n{taper_extension}".replace("_", " "))
        plt.tight_layout()

        #save hazard curve figure
        # make directory for hazard curve if it doesn't exist
        if not os.path.exists(f"../{model_version_results_directory}/{extension1}/probability_figures"):
            os.mkdir(f"../{model_version_results_directory}/{extension1}/probability_figures")

        for file_type in file_type_list:
            plt.savefig(f"../{model_version_results_directory}/{extension1}/probability_figures/hazard_curve_{extension1}"
                        f"{taper_extension}_{plot_n + 1}.{file_type}", dpi=300)
        plt.close()
        printProgressBar(plot_n + 1, n_plots, prefix = '\tCompleted Plots:', suffix = 'Complete', length = 50)


def plot_many_hazard_curves(file_suffix_list, slip_taper, gf_name, fault_type, model_version_results_directory, model_version,
                            color_map): #, plot_order=plot_order):

    plot_order = ["Paraparaumu", "Porirua CBD north", "South Coast", "Wellington Airport", "Wellington CBD", "Petone",
                  "Seaview", "Eastbourne", "Turakirae Head", "Lake Ferry", "Cape Palliser",
                  "Flat Point"]

    exceed_type_list = ["total_abs"]

    if slip_taper is True:
        taper_extension = "_tapered"
    else:
        taper_extension = "_uniform"

    plt.close("all")
    fig, axs = plt.subplots(figsize=(8, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    fig.suptitle("Sites", fontsize=18, y=0.95)

    if color_map == "light_grey":
        plot_colors = ["0.8"]*len(file_suffix_list)
    else:
        plot_colors = plt.get_cmap(color_map)(np.linspace(0, 1, len(file_suffix_list)))
        random.shuffle(plot_colors)

    print("making hazard curves for:")
    for k, file_suffix in enumerate(file_suffix_list):
        print(file_suffix)
        with open(f"../{model_version_results_directory}{model_version}/{gf_name}{file_suffix}/"
                  f"{fault_type}_cumu_exceed_prob_{gf_name}{file_suffix}{taper_extension}.pkl", "rb") as fid:
            PPE_dictionary = pkl.load(fid)

        for i, site in enumerate(plot_order):
            ax = plt.subplot(4, 3, i + 1)

            # plots all three types of exceedance (total_abs, up, down) on the same plot
            for j, exceed_type in enumerate(exceed_type_list):

                curve_color = plot_colors[k]

                exceedance_probs = PPE_dictionary[site][f"exceedance_probs_{exceed_type}"]
                thresholds = PPE_dictionary[site]["thresholds"]

                ax.plot(thresholds, exceedance_probs, color=curve_color)
                ax.axhline(y=0.02, color="0.8", linestyle='dashed')
                ax.axhline(y=0.1, color="0.8", linestyle='dotted')

            ax.set_title(site)
            xmin, xmax = 0.01, 3
            ymin, ymax = 0.000005, 1
            ax.set_yscale('log'), ax.set_xscale('log')
            ax.set_yticks([0.00001, 0.0001, 0.001, 0.01, 0.1, 1])
            ax.get_xaxis().set_major_formatter(ScalarFormatter())
            ax.ticklabel_format(axis='x', style='plain')
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
        fig.text(0.5, 0, 'Vertical displacement threshold (m)', ha='center')
        fig.text(0, 0.5, 'Probability of exceedance in 100 years', va='center', rotation='vertical')
        fig.suptitle(f"all branch displacement hazard curves \n{fault_type} {model_version} {taper_extension}")
        plt.tight_layout()

        # save hazard curve figure
        # make directory for hazard curve if it doesn't exist
    if not os.path.exists(f"../{model_version_results_directory}/all_branches_compare"):
        os.mkdir(f"../{model_version_results_directory}/all_branches_compare")

    plt.savefig(
        f"../{model_version_results_directory}/all_branches_compare/hazard_curve_compare"
        f"{model_version}.png", dpi=300)

    return fig, axs


def plot_weighted_mean_haz_curves(weighted_mean_PPE_dictionary, exceed_type_list,
                                  model_version_title, out_directory, file_type_list, slip_taper, plot_order, sigma=2):
    """
    Plots the weighted mean hazard curve for each site, for each exceedance type (total_abs, up, down)
    :param weighted_mean_PPE_dictionary: dictionary containing the weighted mean exceedance probabilities for each site.
    :param PPE_dictionary: dictionary containing the weighted mean exceedance probabilities for each branch
    :param exceed_type_list: list of strings, either "total_abs", "up", or "down"
    :return:
    """

    if slip_taper is True:
        taper_extension = "_tapered"
    else:
        taper_extension = "_uniform"

    weighted_mean_PPE_dictionary = h5.File(weighted_mean_PPE_dictionary, 'r')

    unique_id_list = weighted_mean_PPE_dictionary['branch_ids'].asstr()
    weights = weighted_mean_PPE_dictionary['branch_weights'][:]
    thresholds = weighted_mean_PPE_dictionary["thresholds"][:]
    thresholds = thresholds[1:]
    weight_order = np.argsort(weights)
    weight_colouring = True

    if 'sigma_lims' in weighted_mean_PPE_dictionary.keys():
        sigma_lims = weighted_mean_PPE_dictionary['sigma_lims'][:]
        if sigma == 2:
            sigma_ix = [ix for ix, sig in enumerate(sigma_lims) if sig in [2.275, 97.725]]
        elif sigma == 1:
            sigma_ix = [ix for ix, sig in enumerate(sigma_lims) if sig in [15.865, 84.135]]
        else:
            print("Can't find requested sigma values in weighted_mean_PPE. Defaulting to max and min")
            sigma_ix = [0, -1]

    if weight_colouring:
        colouring = "_c"
        c_weight = weights / max(weights)
        colours = plt.get_cmap('plasma')(c_weight)
    else:
        colouring = ""

    plt.close("all")

    n_plots = int(np.ceil(len(plot_order) / 12))
    printProgressBar(0, n_plots, prefix = '\tCompleted Plots:', suffix = 'Complete', length = 50)

    for plot_n in range(n_plots):
        sites = plot_order[plot_n*12:(plot_n+1)*12]
        if len(sites) >= 5 or len(sites) == 3:
            n_cols = 3
            n_rows = int(np.ceil(len(sites) / 3))
        elif len(sites) == 4 or len(sites) == 2:
            n_cols = 2
            n_rows = int(np.ceil(len(sites) / 2))
        else:
            n_cols = 1
            n_rows = len(sites)

        for exceed_type in exceed_type_list:
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.63 + 0.12, n_rows * 2.32 + 0.71))  # Replicate 8x10 inch figure if there are less than 12 subplots
            plt.subplots_adjust(hspace=0.3, wspace=0.3)

            # shade the region between the max and min value of all the curves at each site
            for i, site in enumerate(sites):
                ax = plt.subplot(n_rows, n_cols, i + 1)

                # plots all three types of exceedance (total_abs, up, down) on the same plot
                max_probs = weighted_mean_PPE_dictionary[site][f"{exceed_type}_max_vals"][:]
                min_probs = weighted_mean_PPE_dictionary[site][f"{exceed_type}_min_vals"][:]
                max_probs = max_probs[1:]
                min_probs = min_probs[1:]

                # Shade based on max-min
                #ax.fill_between(thresholds, max_probs, min_probs, color='0.9')
                # Shade based on weighted errors
                #ax.fill_between(thresholds, weighted_mean_PPE_dictionary[site][f"weighted_exceedance_probs_{exceed_type}"][1:] + weighted_mean_PPE_dictionary[site][f"{exceed_type}_error"][1:],
                #                weighted_mean_PPE_dictionary[site][f"weighted_exceedance_probs_{exceed_type}"][1:] - weighted_mean_PPE_dictionary[site][f"{exceed_type}_error"][1:], color='0.9')
                # Shade based on weighted 2 sigma percentiles
                ax.fill_between(thresholds, weighted_mean_PPE_dictionary[site][f"{exceed_type}_weighted_percentile_error"][sigma_ix[0], 1:],
                                weighted_mean_PPE_dictionary[site][f"{exceed_type}_weighted_percentile_error"][sigma_ix[1], 1:], color='0.8')

            # plot all the branches as light grey lines
            # for each branch, plot the exceedance probabilities for each site
            for kk in weight_order:
                for i, site in enumerate(sites):
                    site_exceedance_probs = weighted_mean_PPE_dictionary[site][f'branch_exceedance_probs_{exceed_type}'][1:, kk]
                    ax = plt.subplot(n_rows, n_cols, i + 1)
                    #ax.plot(thresholds, site_exceedance_probs, color=[weights[weight_order[k]] / max_weight, 1-(weights[weight_order[k]] / max_weight), 0],
                    #        linewidth=0.1)
                    if weight_colouring:
                        ax.plot(thresholds, site_exceedance_probs, color=colours[kk], linewidth=0.2, alpha=0.5)
                    else:
                        ax.plot(thresholds, site_exceedance_probs, color='grey', linewidth=0.2, alpha=0.5)


            # loop through sites and add the weighted mean lines
            for i, site in enumerate(sites):
                ax = plt.subplot(n_rows, n_cols, i + 1)

                # plots all three types of exceedance (total_abs, up, down) on the same plot
                weighted_mean_exceedance_probs = weighted_mean_PPE_dictionary[site][f"weighted_exceedance_probs_{exceed_type}"][1:]

                line_color = get_probability_color(exceed_type)
               
                # Unweighted 1 sigma lines
                # ax.plot(thresholds, weighted_mean_PPE_dictionary[site][f"{exceed_type}_84_135_vals"], color=line_color, linewidth=0.75, linestyle='-.')
                # ax.plot(thresholds, weighted_mean_PPE_dictionary[site][f"{exceed_type}_15_865_vals"], color=line_color, linewidth=0.75, linestyle='-.')
                # Unweighted 2 sigma lines
                # ax.plot(thresholds, weighted_mean_PPE_dictionary[site][f"{exceed_type}_97_725_vals"], color=line_color, linewidth=0.75, linestyle='--')
                # ax.plot(thresholds, weighted_mean_PPE_dictionary[site][f"{exceed_type}_2_275_vals"], color=line_color, linewidth=0.75, linestyle='--')

                # Weighted 1 sigma lines
                # ax.plot(thresholds, weighted_mean_PPE_dictionary[site][f"{exceed_type}_w84_135_vals"], color=line_color, linewidth=0.75, linestyle=':')
                # ax.plot(thresholds, weighted_mean_PPE_dictionary[site][f"{exceed_type}_w15_865_vals"], color=line_color, linewidth=0.75, linestyle=':')
                # Weighted 2 sigma lines
                ax.plot(thresholds, weighted_mean_PPE_dictionary[site][f"{exceed_type}_weighted_percentile_error"][0,1:], color='black', linewidth=0.75, linestyle='-.')
                ax.plot(thresholds, weighted_mean_PPE_dictionary[site][f"{exceed_type}_weighted_percentile_error"][-1,1:], color='black', linewidth=0.75, linestyle='-.')

                ax.plot(thresholds, weighted_mean_exceedance_probs, color=line_color, linewidth=1.5)

                # Uncertainty weighted mean
                #ax.plot(thresholds, weighted_mean_PPE_dictionary[site][f"uc_weighted_exceedance_probs_{exceed_type}"], color='black', linewidth=1)

                ax.axhline(y=0.02, color="g", linestyle='dashed')
                ax.axhline(y=0.1, color="g", linestyle='dotted')

                xmin, xmax = 0.01, 3
                ymin, ymax = 0.000005, 1
                ax.set_title(site)
                ax.set_yscale('log'), ax.set_xscale('log')
                ax.set_ylim([ymin, ymax])
                ax.set_yticks([0.00001, 0.0001, 0.001, 0.01, 0.1, 1])
                ax.get_xaxis().set_major_formatter(ScalarFormatter())
                ax.ticklabel_format(axis='x', style='plain')
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.set_xlim([xmin, xmax])

            fig.text(0.5, 0, 'Vertical displacement threshold (m)', ha='center')
            fig.text(0, 0.5, 'Probability of exceedance in 100 years', va='center', rotation='vertical')
            fig.suptitle(f"weighted mean hazard curves\n{model_version_title} {taper_extension}\n{exceed_type}")
            plt.tight_layout()

            if not os.path.exists(f"../{out_directory}/weighted_mean_figures"):
                os.mkdir(f"../{out_directory}/weighted_mean_figures")

            for file_type in file_type_list:
                plt.savefig(
                    f"../{out_directory}/weighted_mean_figures/weighted_mean_hazcurve_{exceed_type}{taper_extension}_{plot_n}{colouring}.{file_type}", dpi=300)
            plt.close()
            printProgressBar(plot_n + 0.5, n_plots, prefix = '\tCompleted Plots:', suffix = 'Complete', length = 50)

        # make a second graph with just the shaded envelope and weighted mean lines
        if len(exceed_type_list) > 1:
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.63 + 0.12, n_rows * 2.32 + 0.71))
            plt.subplots_adjust(hspace=0.3, wspace=0.3)

            for i, site in enumerate(sites):
                ax = plt.subplot(n_rows, n_cols, i + 1)
                for exceed_type in exceed_type_list:
                    # Shade based on max-min
                    # ax.fill_between(thresholds, weighted_mean_max_probs, weighted_mean_min_probs, color=fill_color, alpha=0.2)
                    # Shade based on weighted errors
                    #ax.fill_between(thresholds, weighted_mean_PPE_dictionary[site][f"weighted_exceedance_probs_{exceed_type}"][1:] + weighted_mean_PPE_dictionary[site][f"{exceed_type}_error"][1:],
                    #                weighted_mean_PPE_dictionary[site][f"weighted_exceedance_probs_{exceed_type}"][1:] - weighted_mean_PPE_dictionary[site][f"{exceed_type}_error"][1:], color='0.9')
                    # Shade based on 2 sigma percentiles
                    ax.fill_between(thresholds, weighted_mean_PPE_dictionary[site][f"{exceed_type}_weighted_percentile_error"][0, 1:],
                                    weighted_mean_PPE_dictionary[site][f"{exceed_type}_weighted_percentile_error"][-1, 1:], color='0.8')

                # plot solid lines on top of the shaded regions
                for exceed_type in exceed_type_list:
                    line_color = get_probability_color(exceed_type)
                    weighted_mean_exceedance_probs = weighted_mean_PPE_dictionary[site][f"weighted_exceedance_probs_{exceed_type}"][1:]
                    ax.plot(thresholds, weighted_mean_exceedance_probs, color=line_color, linewidth=2)

                # add 10% and 2% lines
                ax.axhline(y=0.02, color="g", linestyle='dashed')
                ax.axhline(y=0.1, color="g", linestyle='dotted')

                # make axes pretty
                ax.set_title(site)
                ax.set_yscale('log'), ax.set_xscale('log')
                ax.set_ylim([0.000005, 1]), ax.set_xlim([0.01, 3])
                ax.set_yticks([0.00001, 0.0001, 0.001, 0.01, 0.1, 1])
                ax.get_xaxis().set_major_formatter(ScalarFormatter())
                ax.ticklabel_format(axis='x', style='plain')
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            fig.text(0.5, 0, 'Vertical displacement threshold (m)', ha='center')
            fig.text(0, 0.5, 'Probability of exceedance in 100 years', va='center', rotation='vertical')
            exceed_types_string = ", ".join(exceed_type_list)
            fig.suptitle(f"weighted mean hazard curves \n{model_version_title} {taper_extension} \n{exceed_types_string} ")
            plt.tight_layout()

            for file_type in file_type_list:
                plt.savefig(f"../{out_directory}/weighted_mean_figures/weighted_mean_hazcurves{taper_extension}_{plot_n}"
                            f".{file_type}", dpi=300)
            plt.close()
            printProgressBar(plot_n + 1, n_plots, prefix = '\tCompleted Plots:', suffix = 'Complete', length = 50)
    weighted_mean_PPE_dictionary.close()

def plot_weighted_mean_haz_curves_colorful(weighted_mean_PPE_dictionary, PPE_dictionary, exceed_type_list,
                                           model_version_title, out_directory, file_type_list, slip_taper, file_name,
                                           string_list, plot_order):
    """
    Plots the weighted mean hazard curve for each site, for each exceedance type (total_abs, up, down)
    :param weighted_mean_PPE_dictionary: dictionary containing the weighted mean exceedance probabilities for each site.
    :param PPE_dictionary: dictionary containing the weighted mean exceedance probabilities for each branch
    :param exceed_type_list: list of strings, either "total_abs", "up", or "down"
    :return:
    """

    if slip_taper is True:
        taper_extension = "_tapered"
    else:
        taper_extension = "_uniform"

    unique_id_list = list(PPE_dictionary.keys())

    plt.close("all")

    n_plots = int(np.ceil(len(plot_order) / 12))
    printProgressBar(0, n_plots, prefix = '\tCompleted Plots:', suffix = 'Complete', length = 50)

    for plot_n in range(n_plots):
        sites = plot_order[plot_n*12:(plot_n+1)*12]
        if len(sites) >= 5 or len(sites) == 3:
            n_cols = 3
            n_rows = int(np.ceil(len(sites) / 3))
        elif len(sites) == 4 or len(sites) == 2:
            n_cols = 2
            n_rows = int(np.ceil(len(sites) / 2))
        else:
            n_cols = 1
            n_rows = len(sites)

        for exceed_type in exceed_type_list:
            fig, axs = plt.subplots(n_rows, n_cols + 1, sharex=True, sharey=True, figsize=((n_cols + 1) * 2.63 + 0.12, n_rows * 2.32 + 0.71))  # Replicate 8x10 inch figure if there are less than 12 subplots
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
            subplot_indices = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]

            # shade the region between the max and min value of all the curves at each site
            for i, site in enumerate(sites):
                ax = plt.subplot(n_rows, n_cols + 1, subplot_indices[i])

                # plots all three types of exceedance (total_abs, up, down) on the same plot
                max_probs = weighted_mean_PPE_dictionary[site][f"{exceed_type}_max_vals"]
                min_probs = weighted_mean_PPE_dictionary[site][f"{exceed_type}_min_vals"]
                thresholds = weighted_mean_PPE_dictionary[site]["thresholds"]

                thresholds = thresholds[1:]
                max_probs = max_probs[1:]
                min_probs = min_probs[1:]

                # Shade based on max-min
                #ax.fill_between(thresholds, max_probs, min_probs, color='0.9', label="_nolegend_")
                # Shade based on weighted errors
                #ax.fill_between(thresholds, weighted_mean_PPE_dictionary[site][f"weighted_exceedance_probs_{exceed_type}"][1:] + weighted_mean_PPE_dictionary[site][f"{exceed_type}_error"][1:],
                #                weighted_mean_PPE_dictionary[site][f"weighted_exceedance_probs_{exceed_type}"][1:] - weighted_mean_PPE_dictionary[site][f"{exceed_type}_error"][1:], color='0.9', label="_nolegend_")
                # Shade based on 2 sigma percentiles
                ax.fill_between(thresholds, weighted_mean_PPE_dictionary[site][f"{exceed_type}_w97_725_vals"][1:],
                                weighted_mean_PPE_dictionary[site][f"{exceed_type}_w2_275_vals"][1:], color='0.8', label="_nolegend_")

            # plot all the branches as light grey lines
            # for each branch, plot the exceedance probabilities for each site
            # make a list of random colors in the length on unique_ids
            # random_colors = [
            #     (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)) for i in range(len(unique_id_list))
            # ]
            special_colors = [(235 / 255, 97 / 255, 35 / 255), (64 / 255, 176 / 255, 166 / 255), (116 / 255, 43 / 255,
                                                                                            140 / 255)]

            for k, unique_id in enumerate(unique_id_list):
                # this loop isn't really needed, but it's useful if you calculate Green's functions
                # at more sites than you want to plot
                if string_list[0] in unique_id:
                    line_color = special_colors[0]
                    linewidth = 1
                elif string_list[1] in unique_id:
                    line_color = special_colors[1]
                    linewidth = 1
                else:
                    line_color = special_colors[2]
                    linewidth = 1

                for i, site in enumerate(sites):

                    thresholds = PPE_dictionary[unique_id]["cumu_PPE_dict"][site]["thresholds"]
                    site_exceedance_probs = PPE_dictionary[unique_id]["cumu_PPE_dict"][site][f"exceedance_probs_{exceed_type}"]

                    # skip the 0 value in the list
                    thresholds = thresholds[1:]
                    site_exceedance_probs = site_exceedance_probs[1:]

                    # ax = plt.subplot(4, 3, i + 1)
                    ax = plt.subplot(n_rows, n_cols + 1, subplot_indices[i])

                    #ax.plot(thresholds, site_exceedance_probs, color='0.7')
                    ax.plot(thresholds, site_exceedance_probs, color=line_color, linewidth=linewidth)

            # loop through sites and add the weighted mean lines
            for i, site in enumerate(sites):
                # ax = plt.subplot(4, 3, i + 1)
                ax = plt.subplot(n_rows, n_cols + 1, subplot_indices[i])

                # plots all three types of exceedance (total_abs, up, down) on the same plot
                weighted_mean_exceedance_probs = weighted_mean_PPE_dictionary[site][f"weighted_exceedance_probs_{exceed_type}"]
                thresholds = weighted_mean_PPE_dictionary[site]["thresholds"]

                thresholds = thresholds[1:]
                weighted_mean_exceedance_probs = weighted_mean_exceedance_probs[1:]

                line_color = get_probability_color(exceed_type)
                # Weighted 2 sigma lines
                ax.plot(thresholds, weighted_mean_PPE_dictionary[site][f"{exceed_type}_w97_725_vals"][1:], color='black', linewidth=0.75, linestyle='-.')
                ax.plot(thresholds, weighted_mean_PPE_dictionary[site][f"{exceed_type}_w2_275_vals"][1:], color='black', linewidth=0.75, linestyle='-.')
                ax.plot(thresholds, weighted_mean_exceedance_probs, color=line_color, linewidth=2)

                ax.axhline(y=0.02, color="0.3", linestyle='dashed')
                ax.axhline(y=0.1, color="0.3", linestyle='dotted')

                xmin, xmax = 0.01, 3
                ymin, ymax = 0.000005, 1
                ax.set_title(site)
                ax.set_yscale('log'), ax.set_xscale('log')
                ax.set_ylim([ymin, ymax])
                ax.set_yticks([0.00001, 0.0001, 0.001, 0.01, 0.1, 1])
                ax.get_xaxis().set_major_formatter(ScalarFormatter())
                ax.ticklabel_format(axis='x', style='plain')
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.set_xlim([xmin, xmax])

            # fig.show()
            fig.legend(unique_id_list, loc='center right', fontsize="x-small")
            fig.subplots_adjust(left=0.0)

            fig.text(0.5, 0, 'Vertical displacement threshold (m)', ha='center')
            fig.text(0, 0.5, 'Probability of exceedance in 100 years', va='center', rotation='vertical')
            fig.suptitle(f"weighted mean hazard curves\n{model_version_title} {taper_extension}\n{exceed_type}")
            plt.tight_layout()

            if not os.path.exists(f"../{out_directory}/weighted_mean_figures"):
                os.mkdir(f"../{out_directory}/weighted_mean_figures")

            for file_type in file_type_list:
                plt.savefig(
                    f"../{out_directory}/weighted_mean_figures/"
                    f"{file_name}weighted_mean_hazcurve_{exceed_type}{taper_extension}_{plot_n}.{file_type}", dpi=300)
            plt.close()
            printProgressBar(plot_n + 1, n_plots, prefix = '\tCompleted Plots:', suffix = 'Complete', length = 50)

# What is the displacement at 10% and 2% probability?
def make_10_2_disp_plot(extension1, slip_taper, model_version_results_directory, file_type_list=["png", "pdf"], probability_list=[0.1, 0.02],
                        plot_order=[], max_sites=12):
    """ makes bar charts of the displacement value at the 10% and 2% probability of exceence thresholds for each site
        extension1 = "sites_c_MDEz" or whatever
        fault_type = "crustal" or "sz"
        slip_taper = True or False
        model_version_results_directory = "{results_directory}/{fault_type}{fault_model}"
    """
    if slip_taper is True:
        taper_extension = "_tapered"
    else:
        taper_extension = "_uniform"


    with open(f"../{model_version_results_directory}/{extension1}/cumu_exceed_prob_{extension1}"
              f"{taper_extension}.pkl", "rb") as fid:
        site_PPE_dictionary = pkl.load(fid)

    plt.close("all")
    if not plot_order:  # Take default plot order from the dictionary keys
        plot_order = list(site_PPE_dictionary.keys())
    
    n_plots = int(np.ceil(len(plot_order) / max_sites))

    for plot_n in range(n_plots):
        if (plot_n + 1) % 10 == 0:
            print(f"\t\t{plot_n + 1}/{n_plots}")
        sites = plot_order[plot_n * max_sites:(plot_n + 1) * max_sites]
        main_plot_labels = [site.replace("CBD ", "CBD").replace(" ", "\n").replace("CBD", "CBD ") if isinstance(site, str) else site for site in sites]

        fig, axs = plt.subplots(1, 2, figsize=(7, 3.4))
        x = np.arange(len(sites))  # the site label locations
        width = 0.4  # the width of the bars
        # find maximum value in all the "up" columns in PPE dictionary

        max_min_y_vals = []
        for i, probability in enumerate(probability_list):
            disps_up = \
                get_exceedance_bar_chart_data(site_PPE_dictionary=site_PPE_dictionary, exceed_type="up",
                                        site_list=sites, probability=probability)
            disps_down= \
                get_exceedance_bar_chart_data(site_PPE_dictionary=site_PPE_dictionary, exceed_type="down",
                                        site_list=sites, probability=probability)

            max_min_y_vals.append(max(disps_up))
            max_min_y_vals.append(min(disps_down))

            color_up = (189/255, 0, 0)
            color_down = (15/255, 72/255, 186/255)
            label_size = 6
            label_offset = label_size / 60

            # add bars to plot, add black horizontal line at zero.
            bars_up = axs[i].bar(x, disps_up, width, color=color_up, linewidth=0.5)
            bars_down = axs[i].bar(x, disps_down, width, color=color_down, linewidth=0.5)
            axs[i].axhline(y=0, color="k", linewidth=0.5)

            # add value labels to bars
            for bar in bars_up:
                bar_color = bar.get_facecolor()
                axs[i].text(x=bar.get_x(), y=bar.get_height() + label_offset, s=round(bar.get_height(), 1), ha='left',
                            va='center', color=bar_color, fontsize=label_size, fontweight='bold')

            for bar in bars_down:
                bar_color = bar.get_facecolor()
                axs[i].text(x=bar.get_x(), y=bar.get_height() - label_offset, s=round(bar.get_height(), 1), ha='left',
                            va='center', color=bar_color, fontsize=label_size, fontweight='bold')

        for i in range(len(probability_list)):
            axs[i].set_ylim(min(max_min_y_vals) - 0.25, max(max_min_y_vals) + 0.25)
            axs[i].tick_params(axis='x', labelrotation=90, labelsize=label_size)
            axs[i].set_xticks(x, main_plot_labels)
            axs[i].tick_params(axis='y', labelsize=8)
            axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            # set tick labels to be every 0.2
            if max(max_min_y_vals) + 0.25 < 1.0:
                axs[i].yaxis.set_major_locator(mticker.MultipleLocator(0.25))
            else:
                axs[i].yaxis.set_major_locator(mticker.MultipleLocator(0.5))

        # set indidual subplot stuff
        axs[0].set_ylabel("Displacement (m)", fontsize=8)
        axs[1].tick_params(axis='y', labelleft=False)

        axs[0].set_title(f"10% probability of exceedance", fontsize=8)
        axs[1].set_title(f"2% probability of exceedance", fontsize=8)

        # manually make legend with rectangles and text
        max_min_y_vals.append(0.2)  # Incase all values are 0
        swatch_width, swatch_height = width, max(max_min_y_vals) * 0.08
        swatch_minx, swatch_miny = -1 * (len(sites) / 30), max(max_min_y_vals)
        axs[0].add_patch(Rectangle((swatch_minx, swatch_miny), swatch_width, swatch_height,
                                facecolor=color_up, edgecolor=None))
        axs[0].add_patch(Rectangle((swatch_minx, swatch_miny - 2 * swatch_height), swatch_width, swatch_height,
                                facecolor=color_down, edgecolor=None))


        axs[0].text(swatch_minx + 2 * swatch_width, swatch_miny, "uplift", fontsize=8)
        axs[0].text(swatch_minx + 2 * swatch_width, swatch_miny - 2 * swatch_height, "subsidence", fontsize=8)

        fig.suptitle(f"100 yr exceedance displacements\n{extension1}{taper_extension}", fontsize=10)
        fig.tight_layout()

        # make a directory for the figures if it doesn't already exist
        outfile_directory = f"../{model_version_results_directory}/{extension1}/probability_figures"
        if not os.path.exists(f"{outfile_directory}"):
            os.makedirs(f"{outfile_directory}")
        for file_type in file_type_list:
            fig.savefig(f"{outfile_directory}/10_2_disps_{extension1}{taper_extension}_{plot_n + 1}.{file_type}", dpi=300)
        
        plt.close("all")


def save_10_2_disp(extension1, slip_taper, model_version_results_directory):
    """ save displacement value at the 10% and 2% probability of exceence thresholds for each site
        extension1 = "sites_c_MDEz" or whatever
        fault_type = "crustal" or "sz"
        slip_taper = True or False
        model_version_results_directory = "{results_directory}/{fault_type}{fault_model}"
    """
    probability_list = [0.1, 0.02]
    displacement_list = [0.5, 1.0, 2.0, 2.5]

    if slip_taper is True:
        taper_extension = "_tapered"
    else:
        taper_extension = "_uniform"


    with open(f"../{model_version_results_directory}/{extension1}/cumu_exceed_prob_{extension1}"
              f"{taper_extension}.pkl", "rb") as fid:
        site_PPE_dictionary = pkl.load(fid)

    outfile_directory = f"../{model_version_results_directory}/{extension1}/displacement_arrays"
    if not os.path.exists(f"{outfile_directory}"):
        os.makedirs(f"{outfile_directory}")

    site_list = [site for site in site_PPE_dictionary.keys()]
    xy_array = np.zeros((len(site_list), 2))

    for ix, site in enumerate(site_list):
        xy_array[ix, :] = site_PPE_dictionary[site]['site_coords'][:2]

    for i, probability in enumerate(probability_list):
        disps_up = \
            get_exceedance_bar_chart_data(site_PPE_dictionary=site_PPE_dictionary, exceed_type="up",
                                    site_list=site_list, probability=probability)
        disps_down= \
            get_exceedance_bar_chart_data(site_PPE_dictionary=site_PPE_dictionary, exceed_type="down",
                                     site_list=site_list, probability=probability)
        disps_abs= \
            get_exceedance_bar_chart_data(site_PPE_dictionary=site_PPE_dictionary, exceed_type="total_abs",
                                     site_list=site_list, probability=probability)
        
        data = {'sites': site_list, 'uplift': disps_up, 'subsidence': disps_down, 'total_abs': disps_abs}

        disp_gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(xy_array[:, 0], xy_array[:, 1]), crs="EPSG:2193")
        disp_gdf.to_file(f"{outfile_directory}/{int(probability * 100)}perc_disps_{extension1}{taper_extension}.geojson", driver='GeoJSON')

    for disp in displacement_list:
        perc_up = \
            get_probability_bar_chart_data(site_PPE_dictionary=site_PPE_dictionary, exceed_type="up",
                                      threshold=disp, site_list=site_list)
        perc_down= \
            get_probability_bar_chart_data(site_PPE_dictionary=site_PPE_dictionary, exceed_type="down",
                                      threshold=disp, site_list=site_list)
        perc_abs= \
            get_probability_bar_chart_data(site_PPE_dictionary=site_PPE_dictionary, exceed_type="total_abs",
                                      threshold=disp, site_list=site_list)
        
        data = {'sites': site_list, 'uplift': perc_up, 'subsidence': perc_down, 'total_abs': perc_abs}

        perc_gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(xy_array[:, 0], xy_array[:, 1]), crs="EPSG:2193")
        perc_gdf.to_file(f"{outfile_directory}/{disp}mdisp_perc_{extension1}{taper_extension}.geojson", driver='GeoJSON')

# What is the probability of exceeding 0.2 m subsidence, 0.2 m uplift at each site?
def make_prob_bar_chart(extension1,  slip_taper, model_version, model_version_results_directory,
                        threshold=0.2):
    # What is the probability of exceeding 0.2 m subsidence, 0.2 m uplift at each site?
    """ determines the probability of exceeding a defined displacement threshold at each site and plots as a bar chart
        two-part plot, one for up and one for down. y axis is probability, x axis is site name
        :param extension1: string, name of the NSHM branch suffix etc.
        :param slip_taper: boolean, True if slip tapers, False if uniform slip
        :param fault_type: string, "crustal" or sz"
        :param threshold: float, displacement threshold to determine exceedance probability
        :param results_directory: string, name of directory where results are stored
    """

    plot_order = ["Paraparaumu", "Porirua CBD north", "South Coast", "Wellington Airport", "Wellington CBD", "Petone",
                  "Seaview", "Eastbourne", "Turakirae Head", "Lake Ferry", "Cape Palliser",
                  "Flat Point"]

    exceed_type_list = ["up", "down"]

    if slip_taper is True:
        taper_extension = "_tapered"
    else:
        taper_extension = "_uniform"

    with open(f"../{model_version_results_directory}/{extension1}/cumu_exceed_prob_{extension1}"
              f"{taper_extension}.pkl",
              "rb") as fid:
        site_PPE_dictionary = pkl.load(fid)

    # set up custom color scheme
    colors = make_qualitative_colormap("custom", len(plot_order))

    # set up figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(7, 5))
    x = np.arange(len(plot_order))  # the site label locations
    width = 0.6  # the width of the bars

    for i, exceed_type in enumerate(exceed_type_list):
        probs_threshold_exceed_type = \
            get_probability_bar_chart_data(site_PPE_dictionary=site_PPE_dictionary, exceed_type=exceed_type,
                                      threshold=threshold, site_list=plot_order)

        # add bars to plot
        bars_10cm = axs[i].bar(x, probs_threshold_exceed_type, width, color=colors)

        for bar in bars_10cm:
            bar_color = bar.get_facecolor()
            # add value label to each bar
            axs[i].text(x=bar.get_x() + bar.get_width() / 2, y=bar.get_height() + 0.03,
                        s=f"{int(100 * round(bar.get_height(), 2))}%", horizontalalignment='center', color=bar_color,
                        fontsize=6, fontweight='bold')

        axs[i].set_ylim(0.0, 0.5)
        axs[i].tick_params(axis='x', labelrotation=90, labelsize=6)
        axs[i].tick_params(axis='y', labelsize=8)
        axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # set tick labels to be every 0.2
        axs[i].yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        axs[i].set_xticks(x, plot_order)

    # set indidual subplot stuff
    axs[0].set_ylabel("Probabilty", fontsize=8)
    axs[1].tick_params(axis='y', labelleft=False)

    axs[0].set_title(f"Probability of exceeding {threshold} m uplift", fontsize=8)
    axs[1].set_title(f"Probability of exceeding {threshold} m subsidence", fontsize=8)

    fig.suptitle(f"{model_version} faults (100 yrs)")
    fig.tight_layout()

    # make directory for hazard curve if it doesn't exist
    if not os.path.exists(f"../{model_version_results_directory}/{extension1}/probability_figures"):
        os.mkdir(f"../{model_version_results_directory}/{extension1}/probability_figures")
    fig.savefig(f"../{model_version_results_directory}/{extension1}/probability_figures/prob_bar_chart_{extension1}"
                f"{taper_extension}.pdf", dpi=300)
    fig.savefig(f"../{model_version_results_directory}/{extension1}/probability_figures/prob_bar_chart_{extension1}"
                f"{taper_extension}.png", dpi=300)


def make_branch_prob_plot(extension1,  slip_taper, model_version, model_version_results_directory,
                      file_type_list=["png", "pdf"], threshold=0.2, plot_order=None, max_sites=12):
    """ """

    exceed_type_list = ["up", "down"]

    if slip_taper is True:
        taper_extension = "_tapered"
    else:
        taper_extension = "_uniform"

    with open(f"../{model_version_results_directory}/{extension1}/cumu_exceed_prob_{extension1}"
              f"{taper_extension}.pkl",
              "rb") as fid:
        PPE_dict = pkl.load(fid)
    
    if not plot_order:  # Take default plot order from the dictionary keys
        plot_order = [key for key in PPE_dict.keys()]
    
    n_plots = int(np.ceil(len(plot_order) / max_sites))
    printProgressBar(0, n_plots, prefix = '\tCompleted Plots:', suffix = 'Complete', length = 50)
    for plot_n in range(n_plots):
        printProgressBar(plot_n, n_plots, prefix = '\tCompleted Plots:', suffix = 'Complete', length = 50)
        sites = plot_order[plot_n * max_sites:(plot_n + 1) * max_sites]
        main_plot_labels = [site.replace("CBD ", "CBD").replace(" ", "\n").replace("CBD", "CBD ") if isinstance(site, str) else site for site in sites]

        # set up custom color scheme
        # set up custom color scheme
        colors = make_qualitative_colormap("custom", len(sites))
        point_size = [35]

        # set up figure and subplots
        fig, axs = plt.subplots(1, 2, figsize=(7, 3.5))
        x = np.arange(len(sites))  # the site label locations

        for i, exceed_type in enumerate(exceed_type_list):
            probs = \
                get_probability_bar_chart_data(site_PPE_dictionary=PPE_dict, exceed_type=exceed_type,
                                            threshold=threshold, site_list=sites)

            # add point and error bars to plot
            axs[i].scatter(x, probs, s=point_size, color=colors, zorder=3, edgecolors='k', linewidths=0.5)

            labels = [f"{int(100 * round(prob, 2))}%" for prob in probs]
            label_y_vals = [prob + 0.03 for prob in probs]
            for site, q in enumerate(x):
                axs[i].text(x=x[q], y=label_y_vals[q], s=labels[q],
                            horizontalalignment='center', fontsize=6, fontweight='bold')

            ymin, ymax  = 0.0, 0.3
            if ymax < max(probs) + 0.05:
                ymax = np.ceil((max(probs) + 0.05) * 20) / 20
            axs[i].set_ylim([ymin, ymax])
            axs[i].tick_params(axis='x', labelrotation=45, labelsize=6)
            axs[i].tick_params(axis='y', labelsize=8)
            axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            # set tick labels to be every 0.2
            axs[i].yaxis.set_major_locator(mticker.MultipleLocator(0.1))
            axs[i].set_xticks(x, main_plot_labels, va='top', ha='center')

        # set indidual subplot stuff
        fontsize = 8
        # I'm doing it this way instead of just using "names" because I want to make sure the legend is in the correct
        # order.

        axs[0].set_ylabel("Probabilty", fontsize=8)
        axs[1].tick_params(axis='y', labelleft=False)

        axs[0].set_title(f"Probability of exceeding {threshold} m uplift", fontsize=fontsize)
        axs[1].set_title(f"Probability of exceeding {threshold} m subsidence", fontsize=fontsize)

        fig.suptitle(f"{model_version} {extension1} (100 yrs)")
        fig.tight_layout()

        outfile_directory = f"../{model_version_results_directory}/{extension1}/probability_figures"
        if not os.path.exists(f"{outfile_directory}"):
            os.mkdir(f"{outfile_directory}")

        for file_type in file_type_list:
            fig.savefig(f"{outfile_directory}/probs_chart_{extension1}{taper_extension}_{plot_n + 1}.{file_type}", dpi=300)

        plt.close()


def save_disp_prob_tifs(extension1, slip_taper, model_version_results_directory, thresh_lims=[0, 3], thresh_step=0.1, thresholds=None,
                        probs_lims=[0.01, 0.2], probs_step=0.01, probabilites=None, output_thresh=True, output_probs=True, weighted=False, grid=False):
    """
    Create multiband geotiffs of the probability of exceeding given displacements across all sites.
    This assumes that sites are derived from a regularly spaced grid
    """

    # Define File Paths
    exceed_type_list = ["total_abs", "up", "down"]

    if slip_taper is True:
        taper_extension = "_tapered"
    else:
        taper_extension = "_uniform"

    if weighted:
        dict_file = f"../{model_version_results_directory}/weighted_mean_PPE_dict_{extension1}{taper_extension}.pkl"
        outfile_directory = f"../{model_version_results_directory}/weighted_mean_tifs"
        
    else:
        dict_file = f"../{model_version_results_directory}/{extension1}/cumu_exceed_prob_{extension1}{taper_extension}.pkl"
        outfile_directory = f"../{model_version_results_directory}/{extension1}/probability_grids"

    with open(dict_file, "rb") as fid:
        PPE_dict = pkl.load(fid)

    sites = [*PPE_dict]
    if 'branch_weights' in sites:
        sites.remove('branch_weights')

    # Calculate XY extents and resolution for tifs
    if grid:
        with open(f"../{model_version_results_directory}/{extension1}/grid_limits.pkl", "rb") as fid:
            grid_meta = pkl.load(fid)

        x, y, buffer_size, x_res, y_res = grid_meta['x'], grid_meta['y'], grid_meta['buffer_size'], grid_meta['cell_size'], grid_meta['cell_size']

        x_data = np.arange(round(x - buffer_size, -3), round(x + buffer_size, -3), x_res)
        y_data = np.arange(round(y - buffer_size, -3), round(y + buffer_size, -3), y_res)

    else:
        site_x = [pixel['site_coords'][0] for key, pixel in PPE_dict.items() if key != 'branch_weights']
        site_y = [pixel['site_coords'][1] for key, pixel in PPE_dict.items() if key != 'branch_weights']

        x_data = np.unique(site_x)
        y_data = np.unique(site_y)

        xmin, xmax = min(x_data), max(x_data)
        ymin, ymax = min(y_data), max(y_data)
        x_res, y_res = min(np.diff(x_data)), min(np.diff(y_data))

        x_data = np.arange(xmin, xmax + x_res, x_res)
        y_data = np.arange(ymin, ymax + y_res, y_res)

        if not all(np.isin(site_x, x_data)) or not all(np.isin(site_y, y_data)):
            print("Site coordinates cant all be aligned to grid. Check sites are evenly spaced. Skipping step...")
            return

        site_x = (np.array(site_x) - x_data[0]) / x_res
        site_y = (np.array(site_y) - y_data[0]) / y_res

    transform = Affine.translation(x_data[0] - x_res / 2, y_data[0] - y_res / 2) * Affine.scale(x_res, y_res)

    if not os.path.exists(f"{outfile_directory}"):
        os.mkdir(f"{outfile_directory}")

    # Create GeoTifs
    if output_thresh:
        print(f"\tCreating displacement probability geoTifs....")
        if thresholds is None:
            thresholds = np.arange(thresh_lims[0], thresh_lims[1] + thresh_step, thresh_step)

        if not all(np.isin(thresholds, PPE_dict[sites[0]]["thresholds"])):
            thresholds = thresholds[np.isin(thresholds, PPE_dict[sites[0]]["thresholds"])]
        if len(thresholds) == 0:
            print('No requested thresholds were in the PPE dictionary. Change requested thresholds')
        else:
            print('Not all requested thresholds were in PPE dictionary. Running available thresholds...')
            print('Available thresholds are:', thresholds)

        for exceed_type in exceed_type_list:
            thresh_grd = np.zeros([len(thresholds), len(y_data), len(x_data)]) * np.nan
            probs = np.zeros([len(sites), len(thresholds)])
            for ii, threshold in enumerate(thresholds):
                probs[:, ii] = get_probability_bar_chart_data(site_PPE_dictionary=PPE_dict, exceed_type=exceed_type,
                                                              threshold=threshold, site_list=sites, weighted=weighted)
            if grid:
                thresh_grd[ii, :, :] = np.reshape(probs, (len(y_data), len(x_data)))
            else:
                for jj in range(len(sites)):
                    thresh_grd[:, int(site_y[jj]), int(site_x[jj])] = probs[jj, :]

            file_name = f"{extension1}{taper_extension}_{exceed_type}_disp_prob_sites.tif".strip('_')
            with rasterio.open(f"{outfile_directory}/{file_name}", 'w',
                               driver='GTiff', count=thresh_grd.shape[0], height=thresh_grd.shape[1], width=thresh_grd.shape[2],
                               dtype=thresh_grd.dtype, crs='EPSG:2193', transform=transform) as dst:
                dst.write(thresh_grd)
                dst.descriptions = [f"{threshold:.2f}" for threshold in thresholds]


    if output_probs:
        print(f"\tCreating probability exceedence geoTifs....")
        if probabilites is None:
            probabilites = np.arange(probs_lims[0], probs_lims[1] + probs_step, probs_step)

        for exceed_type in exceed_type_list:
            thresh_grd = np.zeros([len(probabilites), len(y_data), len(x_data)]) * np.nan
            disps = np.zeros([len(sites), len(probabilites)])
            for ii, probability in enumerate(probabilites):
                disps[:, ii] = get_exceedance_bar_chart_data(site_PPE_dictionary=PPE_dict, exceed_type=exceed_type,
                                                            site_list=sites, probability=probability, weighted=weighted)
                if exceed_type == 'down':
                    disps[:, ii] = -1 * disps[:, ii]
            if grid:
                thresh_grd[ii, :, :] = np.reshape(disps, (len(y_data), len(x_data)))
            else:
                for jj in range(len(sites)):
                    thresh_grd[:, int(site_y[jj]), int(site_x[jj])] = disps[jj, :]
            
            file_name=f"{extension1}{taper_extension}_{exceed_type}_prob_disp_sites.tif".strip('_')
            with rasterio.open(f"{outfile_directory}/{file_name}", 'w', \
                            driver='GTiff', count=thresh_grd.shape[0], height=thresh_grd.shape[1], width=thresh_grd.shape[2], \
                            dtype=thresh_grd.dtype, crs='EPSG:2193', transform=transform) as dst:
                dst.write(thresh_grd)
                dst.descriptions = [str(probability) for probability in probabilites]


def save_disp_prob_xarrays(extension1, slip_taper, model_version_results_directory, thresh_lims=[0, 3], thresh_step=0,
                           probs_lims=[0.01, 0.2], probs_step=0, output_thresh=True, output_probs=True, weighted=False,
                           output_grids=True, thresholds=None, probabilities=None, sites=[], out_tag=''):
    """
    Add all results to x_array datasets, and save as netcdf files
    """

    # Define File Paths
    exceed_type_list = ["total_abs", "up", "down"]

    if slip_taper is True:
        taper_extension = "_tapered"
    else:
        taper_extension = "_uniform"

    if weighted:
        h5_file = f"../{model_version_results_directory}/weighted_mean_PPE_dict{extension1}{taper_extension}.h5"
        outfile_directory = f"../{model_version_results_directory}/weighted_mean_xarray"

    else:
        h5_file = f"../{model_version_results_directory}/{extension1}/cumu_exceed_prob{extension1}{taper_extension}.h5"
        outfile_directory = f"../{model_version_results_directory}/{extension1}/probability_grids"

    PPEh5 = h5.File(h5_file, 'r')

    metadata_keys = ['branch_weights', 'branch_ids', 'thresholds', 'threshold_vals', 'sigma_lims']

    if sites == []:
        sites = [*PPEh5.keys()]

    for meta in metadata_keys:
        if meta in sites:
            sites.remove(meta)
    
    if thresholds is None:
        if thresh_step != 0:
            thresholds = np.arange(thresh_lims[0], thresh_lims[1] + thresh_step, thresh_step)
        else:
            thresholds = PPEh5["thresholds"]
    
    thresholds = [round(val, 4) for val in thresholds]   # Rounding to try and deal with the floating point errors

    if not os.path.exists(f"{outfile_directory}"):
        os.mkdir(f"{outfile_directory}")

    if output_grids:
        site_x = [PPEh5[site]['site_coords'][:][0] for site in sites]
        site_y = [PPEh5[site]['site_coords'][:][1] for site in sites]

        x_data = np.unique(site_x)
        y_data = np.unique(site_y)

        xmin, xmax = min(x_data), max(x_data)
        ymin, ymax = min(y_data), max(y_data)
        x_res, y_res = min(np.diff(x_data)), min(np.diff(y_data))

        x_data = np.arange(xmin, xmax + x_res, x_res)
        y_data = np.arange(ymin, ymax + y_res, y_res)

        if not all(np.isin(site_x, x_data)) or not all(np.isin(site_y, y_data)):
            print("Site coordinates cant all be aligned to grid. Check sites are evenly spaced. Saving as site geojson instead...")
            save_disp_prob_geojson(extension1, slip_taper, model_version_results_directory, thresh_lims=thresh_lims, thresh_step=thresh_step, thresholds=thresholds,
                                       probs_lims=probs_lims, probs_step=probs_step, probabilities=probabilities, weighted=weighted)
            return

        site_x = (np.array(site_x) - x_data[0]) / x_res
        site_y = (np.array(site_y) - y_data[0]) / y_res

        # Create Datasets
        da = {}
        ds = xr.Dataset()
        if extension1 == "":
            out_name = ''
            branch_name = os.path.basename(model_version_results_directory)
        else:
            out_name = f"{extension1}_"
            branch_name = extension1

        if output_thresh:
            print(f"\tAdding Displacement Probability DataArrays....")

            if not all(np.isin(thresholds, np.round(PPEh5["thresholds"][:], 4))):
                dropped_thresholds = thresholds[np.isin(thresholds, PPEh5["thresholds"][:], invert=True)]
                thresholds = thresholds[np.isin(thresholds, PPEh5["thresholds"][:])]
                if len(thresholds) == 0:
                    print('No requested thresholds were in the PPE dictionary. Change requested thresholds')
                    pass
                else:
                    print('Not all requested thresholds were in PPE dictionary.\nMissing thresholds:\n', dropped_thresholds)
                    print('Running available thresholds:\n', thresholds)

            for exceed_type in exceed_type_list:
                thresh_grd = np.zeros([len(thresholds), len(y_data), len(x_data)]) * np.nan
                probs = np.zeros([len(sites), len(thresholds)])
                printProgressBar(0, len(thresholds), prefix=f'\tProcessing 0.00 m', suffix=f'{exceed_type}', length=50)
                for ii, threshold in enumerate(thresholds):
                    probs[:, ii] = get_probability_bar_chart_data(site_PPE_dictionary=PPEh5, exceed_type=exceed_type,
                                                                  threshold=threshold, site_list=sites, weighted=weighted)
                    printProgressBar(ii + 1, len(thresholds), prefix=f'\tProcessing {threshold:.2f} m', suffix=f'{exceed_type}', length=50)
                for jj in range(len(sites)):
                    thresh_grd[:, int(site_y[jj]), int(site_x[jj])] = probs[jj, :]

                da[exceed_type] = xr.DataArray(thresh_grd, dims=['threshold', 'lat', 'lon'], coords={'threshold': thresholds, 'lat': y_data, 'lon': x_data})
                da[exceed_type].attrs['exceed_type'] = exceed_type
                da[exceed_type].attrs['threshold'] = 'Displacement (m)'
                da[exceed_type].attrs['crs'] = 'EPSG:2193'

                ds['disp_' + exceed_type] = da[exceed_type]
            out_name += 'disp_'

        if output_probs:
            print(f"\tAdding Probability Exceedence DataArrays....")
            if probabilities is None:
                if probs_step != 0:
                    probabilities = np.round(np.arange(probs_lims[0], probs_lims[1] + probs_step, probs_step), 4)
                else:
                    probabilities = np.array([round(val, 4) for val in probabilities])

            for exceed_type in exceed_type_list:
                thresh_grd = np.zeros([len(probabilities), len(y_data), len(x_data)]) * np.nan
                disps = np.zeros([len(sites), len(probabilities)])
                printProgressBar(0, len(probabilities), prefix=f'\tProcessing 00 %', suffix=f'{exceed_type}', length=50)
                for ii, probability in enumerate(probabilities):
                    disps[:, ii] = get_exceedance_bar_chart_data(site_PPE_dictionary=PPEh5, exceed_type=exceed_type,
                                                                 site_list=sites, probability=probability, weighted=weighted)
                    printProgressBar(ii + 1, len(probabilities), prefix=f'\tProcessing {int(100 * probability):0>2} %', suffix=f'{exceed_type}', length=50)
                    if exceed_type == 'down':
                        disps[:, ii] = -1 * disps[:, ii]
                for jj in range(len(sites)):
                    thresh_grd[:, int(site_y[jj]), int(site_x[jj])] = disps[jj, :]

                da[exceed_type] = xr.DataArray(thresh_grd, dims=['probability', 'lat', 'lon'], coords={'probability': (probabilities * 100).astype(int), 'lat': y_data, 'lon': x_data})
                da[exceed_type].attrs['exceed_type'] = exceed_type
                da[exceed_type].attrs['threshold'] = 'Exceedance Probability (%)'
                da[exceed_type].attrs['crs'] = 'EPSG:2193'

                ds['prob_' + exceed_type] = da[exceed_type]
            out_name += 'prob_'

        ds.attrs['branch'] = branch_name
        ds.to_netcdf(f"{outfile_directory}/{model_version_results_directory.split('/')[-1]}_{out_name}{out_tag}_grids.nc".replace('__', '_'))

    return ds


def save_disp_prob_geojson(extension1, slip_taper, model_version_results_directory, thresh_lims=[0, 3], thresh_step=0.1, thresholds=None,
                           probs_lims=[0.01, 0.2], probs_step=0.01, probabilities=None, weighted=False, epsg=2193):
    """
    Write site data out as geojson
    """
    # Define File Paths
    exceed_type_list = ["total_abs", "up", "down"]

    if slip_taper is True:
        taper_extension = "_tapered"
    else:
        taper_extension = "_uniform"

    if weighted:
        h5_file = f"../{model_version_results_directory}/weighted_mean_PPE_dict{extension1}{taper_extension}.h5"
        outfile_directory = f"../{model_version_results_directory}/weighted_mean_xarray"
        
    else:
        h5_file = f"../{model_version_results_directory}/{extension1}/cumu_exceed_prob{extension1}{taper_extension}.h5"
        outfile_directory = f"../{model_version_results_directory}/{extension1}/probability_grids"

    PPEh5 = h5.File(h5_file)

    sites = [*PPEh5.keys()]
    metadata_keys = ['branch_weights', 'branch_ids', 'thresholds', 'threshold_vals', 'sigma_lims']
    for meta in metadata_keys:
        if meta in sites:
            sites.remove(meta)

    if thresholds is None:
        if thresh_step != 0:
            thresholds = np.arange(thresh_lims[0], thresh_lims[1] + thresh_step, thresh_step)
        else:
            thresholds = PPEh5["thresholds"]

    thresholds = [round(val, 4) for val in thresholds]   # Rounding to try and deal with the floating point errors

    if not os.path.exists(f"{outfile_directory}"):
        os.mkdir(f"{outfile_directory}")

    if probabilities is None:
        if probs_step != 0:
            probabilities = list(np.round(np.arange(probs_lims[0], probs_lims[1] + probs_step, probs_step), 4))
        else:
            probabilities = np.array([round(val, 4) for val in probabilities])
        
    probs = np.zeros([len(sites), len(thresholds), 3])
    for ii, exceed_type in enumerate(exceed_type_list):
        for jj, threshold in enumerate(thresholds):
            print(f"\tRetreiving {exceed_type} probability at {threshold} m...", end='\r')
            probs[:, jj, ii] = get_probability_bar_chart_data(site_PPE_dictionary=PPEh5, exceed_type=exceed_type,
                                                              threshold=round(threshold, 4), site_list=sites, weighted=weighted)
        print('')
    
    geojson = {
        "type": "FeatureCollection",
        "features": [],
        "crs": {
        "type": "name",
        "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}  # NZTM CRS (EPSG:2193)
        }}
    
    for ix, site in enumerate(sites):
        print(f'\tWriting site data to geojson... {ix}/{len(sites)}', end='\r')
        df = pd.DataFrame(probs[ix, :, :], columns=exceed_type_list, index=thresholds)
        for index, row in df.iterrows():
            properties = {"Threshold (m)": round(index, 4)}
            properties = properties | row.astype(float).to_dict()
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(coord) for coord in PPEh5[site]['site_coords'][:][:2]]
                    },
                "properties": properties,
                }
            geojson["features"].append(feature)
    print('')

    # Convert the GeoJSON object to a JSON string
    geojson_str = json.dumps(geojson, indent=2)
    
    # Write the GeoJSON string to a file
    with open(f"{outfile_directory}/displacements.geojson", 'w') as f:
        f.write(geojson_str)
    
    disps = np.zeros([len(sites), len(probabilities), 3])
    for ii, exceed_type in enumerate(exceed_type_list):
        for jj, probability in enumerate(probabilities):
                print(f"\tRetreiving {exceed_type} displacement at {probability} %...", end='\r')
                disps[:, jj, ii] = get_exceedance_bar_chart_data(site_PPE_dictionary=PPEh5, exceed_type=exceed_type,
                                                                 site_list=sites, probability=probability, weighted=weighted)
        print('')

    geojson = {
        "type": "FeatureCollection",
        "features": [],
        "crs": {
        "type": "name",
        "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}  # NZTM CRS (EPSG:2193)
        }}
    
    for ix, site in enumerate(sites):
        print(f'\tWriting site data to geojson... {ix}/{len(sites)}', end='\r')
        df = pd.DataFrame(disps[ix, :, :], columns=exceed_type_list, index=probabilities)
        for index, row in df.iterrows():
            properties = {"Probability (%)": round(index, 4)}
            properties = properties | row.astype(float).to_dict()
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(coord) for coord in PPEh5[site]['site_coords']]
                    },
                "properties": properties,
                }
            geojson["features"].append(feature)
    print('')

    # Convert the GeoJSON object to a JSON string
    geojson_str = json.dumps(geojson, indent=2)
    
    # Write the GeoJSON string to a file
    with open(f"{outfile_directory}/probabilities.geojson", 'w') as f:
        f.write(geojson_str)