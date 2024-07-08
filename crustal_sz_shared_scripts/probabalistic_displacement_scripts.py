import random
import geopandas as gpd
import pandas as pd
import os
import itertools
import pickle as pkl
import matplotlib.ticker as mticker
import rasterio
from rasterio.transform import Affine
from time import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from helper_scripts import get_figure_bounds, make_qualitative_colormap, tol_cset, get_probability_color, percentile
from matplotlib.patches import Rectangle
from weighted_mean_plotting_scripts import get_mean_prob_barchart_data, get_mean_disp_barchart_data

matplotlib.rcParams['pdf.fonttype'] = 42

def get_site_disp_dict(extension1, slip_taper, model_version_results_directory):
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
    first_key = list(rupture_disp_dictionary.keys())[0]
    site_names = rupture_disp_dictionary[first_key]["site_name_list"]
    site_coords = rupture_disp_dictionary[first_key]["site_coords"]

    # makes a list of lists. each item is a rupture scenario that contains a list of displacements at each site.
    # could probably simplify this because it's the same shape as the dictionary, but I'm not sure how to do that yet
    # and this is easier to understand I think.
    disps_by_scenario = []
    annual_rates_by_scenario = []
    for rupture_id in rupture_disp_dictionary.keys():
        disps = rupture_disp_dictionary[rupture_id]["v_disps_m"]
        disps_by_scenario.append(disps)
        annual_rate = rupture_disp_dictionary[rupture_id]["annual_rate"]
        annual_rates_by_scenario.append(annual_rate)

    # list of lists. each item is a site location that contains displacements from each scenario (disp list length =
    # number of rupture scenarios)
    disps_by_location = []
    annual_rates_by_location = []
    for site_num in range(len(site_names)):
        site_disp = [scenario[site_num] for scenario in disps_by_scenario]
        disps_by_location.append(site_disp)
        annual_rates_by_location.append(annual_rates_by_scenario)

    # make dictionary of displacements and other data. key is the site name.
    site_disp_dictionary = {}
    for i, site in enumerate(site_names):
        site_disp_dictionary[site] = {"disps": disps_by_location[i], "rates": annual_rates_by_location[i],
                                           "site_coords": site_coords[i]}

    if slip_taper is True:
        extension3 = "_tapered"
    else:
        extension3 = "_uniform"
    
    if 'grid_meta' in rupture_disp_dictionary.keys():
        site_disp_dictionary['grid_meta'] = rupture_disp_dictionary['grid_meta']

    # with open(f"../{results_version_directory}/{extension1}/site_disp_dict_{extension1}{extension3}.pkl",
    #           "wb") as f:
    #     pkl.dump(site_disp_dictionary, f)
    return site_disp_dictionary

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

def get_cumu_PPE(slip_taper, model_version_results_directory, branch_site_disp_dict, n_samples,
                 extension1, branch_key="nan", time_interval=100, sd=0.4, error_chunking=1000):
    """
    Must first run get_site_disp_dict to get the dictionary of displacements and rates, with 1 sigma error bars

    inputs: runs for one logic tree branch
    Time_interval is in years

    function: calculates the poissonian probability of exceedance for each site for each displacement threshold value

    outputs: pickle file with probability dictionary (probs, disps, site_coords)

    CAVEATS/choices:
    - need to decide on number of 100-yr simulations to run (n_samples = 1000000)
    """

    # use random number generator to initialise monte carlo sampling
    rng = np.random.default_rng()

    # Load the displacement/rate data for all sites
    if slip_taper is True:
        taper_extension = "_tapered"
    else:
        taper_extension = "_uniform"

    ## loop through each site and generate a bunch of 100 yr interval scenarios
    site_PPE_dict = {}
    printProgressBar(0, len(branch_site_disp_dict.keys()), prefix = f'\tProcessing {len(branch_site_disp_dict.keys())} Sites:', suffix = 'Complete', length = 50)
    for i, site_of_interest in enumerate(branch_site_disp_dict.keys()):
        printProgressBar(i + 1, len(branch_site_disp_dict.keys()), prefix = f'\tProcessing {len(branch_site_disp_dict.keys())} Sites:', suffix = 'Complete', length = 50)
        # print('\t\tSite:', site_of_interest, '(', i, 'of', len(branch_site_disp_dict.keys()), ')')
        # if i == 0:
        #     if branch_key not in ["nan", ""]:
        #         print(f"calculating {branch_key} PPE for site {i} of {len(branch_site_disp_dict.keys())}")
        #     if extension1 not in ["nan", ""]:
        #         print(f"calculating {extension1} PPE for site {i} of {len(branch_site_disp_dict.keys())}")
        # print(f"calculating {branch_key} PPE ({i} of {len(branch_site_disp_dict.keys())} branches)")
        site_dict_i = branch_site_disp_dict[site_of_interest]

        ## Set up params for sampling
        investigation_time = time_interval

        if "scaled_rates" not in site_dict_i.keys():
            # if no scaled_rate column, assumes scaling of 1 (equal to "rates")
            scaled_rates = site_dict_i["rates"]
        else:
            scaled_rates = site_dict_i["scaled_rates"]

        # Drop ruptures that don't cause slip at this site
        drop_noslip = True
        if drop_noslip:
            no_slip = [ix for ix, slip in enumerate(site_dict_i["disps"]) if slip == 0]
            disps = [slip for ix, slip in enumerate(site_dict_i['disps']) if ix not in no_slip]
            scaled_rates = [rate for ix, rate in enumerate(scaled_rates) if ix not in no_slip]
        else:
            disps = site_dict_i['disps']

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
        disp_scenarios = scenarios * disps
        # multiplies displacement by the uncertainty multiplier
        disp_scenarios = disp_scenarios * disp_uncertainty
        # sum all displacement values at that site in that 100 yr interval
        cumulative_disp_scenarios = disp_scenarios.sum(axis=1)

        # get displacement thresholds for calculating exceedance (hazard curve x axis)
        thresholds = np.arange(0, 3, 0.01)
        thresholds_neg = thresholds * -1
        # sum all the displacements in the 100 year window that exceed threshold
        n_exceedances_total_abs = np.zeros_like(thresholds)
        n_exceedances_up = np.zeros_like(thresholds)
        n_exceedances_down = np.zeros_like(thresholds)
        # Initially use all samples to come up with a best estimate of exceedence probability
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

        # Now chunk the scenarios to get a better estimate of the exceedance probability
        n_chunks = int(n_samples / error_chunking)
        if n_chunks < 10:
            error_chunking = int(n_samples / 10)
            print(f'\nToo few chunks for accurate error estimation. Decreasing error_chunking to {error_chunking}\n')

        n_exceedances_total_abs = np.zeros((len(thresholds), n_chunks))
        n_exceedances_up = np.zeros((len(thresholds), n_chunks))
        n_exceedances_down = np.zeros((len(thresholds), n_chunks))

        cumulative_disp_scenarios = cumulative_disp_scenarios[:(n_chunks * error_chunking)].reshape(n_chunks, error_chunking)
        for tix, threshold in enumerate(thresholds):
            # replaces index in zero array with the number of times the cumulative displacement exceeded the threshold
            # across all of the 100 yr scenarios
            # sums the absolute value of the disps if the abs value is greater than threshold. e.g., -0.5 + 0.5 = 1
            n_exceedances_total_abs[tix, :] = (np.abs(cumulative_disp_scenarios) > threshold).sum(axis=1)
            n_exceedances_up[tix, :] = (cumulative_disp_scenarios > threshold).sum(axis=1)
            n_exceedances_down[tix, :] = (cumulative_disp_scenarios < -threshold).sum(axis=1)

        # the probability is the number of times that threshold was exceeded divided by the number of samples. so,
        # quite high for low displacements (25%). Means there's a ~25% chance an earthquake will exceed 0 m in next 100
        # years across all earthquakes in the catalogue (at that site).
        exceedance_errs_total_abs = n_exceedances_total_abs / error_chunking
        exceedance_errs_up = n_exceedances_up / error_chunking
        exceedance_errs_down = n_exceedances_down / error_chunking

        # Output 1 sigma error limits
        error_abs = np.std(exceedance_errs_total_abs, axis=1)
        error_up = np.std(exceedance_errs_up, axis=1)
        error_down = np.std(exceedance_errs_down, axis=1)

        # CAVEAT: at the moment only absolute value thresholds are stored, but for "down" the thresholds are
        # actually negative.
        site_PPE_dict[site_of_interest] = {"thresholds": thresholds,
                                           "exceedance_probs_total_abs": exceedance_probs_total_abs,
                                           "exceedance_probs_up": exceedance_probs_up,
                                           "exceedance_probs_down": exceedance_probs_down,
                                           "site_coords": site_dict_i["site_coords"],
                                           "standard_deviation": sd,
                                           "error_total_abs": error_abs,
                                           "error_up": error_up,
                                           "error_down": error_down}

    if 'grid_meta' in branch_site_disp_dict.keys():
            site_PPE_dict['grid_meta'] = branch_site_disp_dict['grid_meta']
    
    if extension1 != "":
        with open(f"../{model_version_results_directory}/{extension1}/cumu_exceed_prob_{extension1}"
              f"{taper_extension}.pkl", "wb") as f:
            pkl.dump(site_PPE_dict, f)

    else:
        return site_PPE_dict

def make_fault_model_PPE_dict(branch_weight_dict, model_version_results_directory, slip_taper, n_samples,
                              outfile_extension):
    """ This function takes the branch dictionary and calculates the PPEs for each branch.
    It then combines the PPEs (key = unique branch ID).

    Must run this function with crustal, subduction, or a combination of two.

    :param crustal_branch_dict: from the function make_branch_weight_dict
    :param results_version_directory: string; path to the directory with the solution files
    :return mega_branch_PPE_dictionary and saves a pickle file.
    """

    gf_name = "sites"
    counter = 0

    if slip_taper:
        taper_extension = "_tapered"
    else:
        taper_extension = "_uniform"


    fault_model_allbranch_PPE_dict = {}
    for branch_id in branch_weight_dict.keys():

        print(f"calculating {branch_id} PPE\t({counter} of {len(branch_weight_dict.keys())} branches)")
        counter += 1

        # get site displacement dictionary and branch weights
        extension1 = gf_name + branch_weight_dict[branch_id]["file_suffix"]
        branch_weight = branch_weight_dict[branch_id]["total_weight_RN"]

        # Extract rates from the NSHM solution directory, but it is not scaled by the rate scaling factor
        branch_site_disp_dict = get_site_disp_dict(extension1, slip_taper=slip_taper,
                           model_version_results_directory=model_version_results_directory)
        # multiply the rates by the rate scaling factor
        rate_scaling_factor = branch_weight_dict[branch_id]["S"]
        for site in branch_site_disp_dict.keys():
            # multiply each value in the rates array by the rate scaling factor
            branch_site_disp_dict[site]["scaled_rates"] = [rate * rate_scaling_factor for rate in branch_site_disp_dict[
                site]["rates"]]

        ### get exceedance probability dictionary
        branch_cumu_PPE_dict = get_cumu_PPE(branch_key=branch_id, branch_site_disp_dict=branch_site_disp_dict,
                    model_version_results_directory=model_version_results_directory, slip_taper=slip_taper,
                    time_interval=100, n_samples=n_samples, extension1="")

        fault_model_allbranch_PPE_dict[branch_id] = {"cumu_PPE_dict": branch_cumu_PPE_dict, "branch_weight":
            branch_weight}

    outfile_name = f"allbranch_PPE_dict_{outfile_extension}{taper_extension}"

    with open(f"../{model_version_results_directory}/{outfile_name}.pkl", "wb") as f:
        pkl.dump(fault_model_allbranch_PPE_dict, f)
    return fault_model_allbranch_PPE_dict

def get_weighted_mean_PPE_dict(fault_model_PPE_dict, out_directory, outfile_extension, slip_taper):
    """takes all the branch PPEs and combines them based on the branch weights into a weighted mean PPE dictionary

    :param fault_model_PPE_dict: The dictionary has PPEs for each branch (or branch pairing).
    Each branch contains "branch_weight" and "cumu_PPE_dict".
    "cumu_PPE_dict" is organized by site. Nested in sites is "thresholds", "exceedance_probs_up",
    "exceedance_probs_down", and "exceedance_probs_total_abs"
    :return dictionary of sites, with lists of weighted mean PPEs and threshold values.
    """

    if slip_taper:
        taper_extension = "_tapered"
    else:
        taper_extension = "_uniform"

    unique_id_list = list(fault_model_PPE_dict.keys())
    n_branches = len(unique_id_list)
    site_list = fault_model_PPE_dict[unique_id_list[0]]["cumu_PPE_dict"].keys()

    # weight the probabilities by NSHM branch weights to get a weighted mean
    branch_weights = [fault_model_PPE_dict[unique_id]["branch_weight"] for unique_id in
                      fault_model_PPE_dict.keys()]

    # need a more elegant solution to this I think
    threshold_vals = np.arange(0, 3, 0.01)

    # extract site coordinates from fault model PPE dictionary
    site_coords_dict = {}
    for site in site_list:
        site_coords = fault_model_PPE_dict[unique_id_list[0]]["cumu_PPE_dict"][site]["site_coords"]
        site_coords_dict[site] = site_coords

    weighted_mean_site_probs_dictionary = {}
    weighted_mean_site_probs_dictionary['branch_weights'] = branch_weights
    for site in site_list:
        weighted_mean_site_probs_dictionary[site] = {}

    for exceed_type in ["total_abs", "up", "down"]:
        for i, site in enumerate(site_list):
            site_df = {}
            errors_df = {}
            for unique_id in unique_id_list:
                probabilities_i_site = fault_model_PPE_dict[unique_id]["cumu_PPE_dict"][site][
                    f"exceedance_probs_{exceed_type}"]
                site_df[unique_id] = probabilities_i_site
                errors_df[unique_id] = fault_model_PPE_dict[unique_id]["cumu_PPE_dict"][site][
                    f"error_{exceed_type}"]
            site_probabilities_df = pd.DataFrame(site_df)

            # collapse each row into a weighted mean value
            branch_weighted_mean_probs = site_probabilities_df.apply(
                lambda x: np.average(x, weights=branch_weights), axis=1)
            site_max_probs = site_probabilities_df.max(axis=1)
            site_min_probs = site_probabilities_df.min(axis=1)

            weighted_mean_site_probs_dictionary[site]["threshold_vals"] = threshold_vals
            weighted_mean_site_probs_dictionary[site][f"weighted_exceedance_probs_{exceed_type}"] = branch_weighted_mean_probs
            weighted_mean_site_probs_dictionary[site][f"{exceed_type}_max_vals"] = site_max_probs
            weighted_mean_site_probs_dictionary[site][f"{exceed_type}_min_vals"] = site_min_probs
            weighted_mean_site_probs_dictionary[site]["site_coords"] = site_coords_dict[site]

            # Calculate errors based on 1 and 2 sigma percentiles of all of the branches for each threshold
            weighted_mean_site_probs_dictionary[site][f"{exceed_type}_97_725_vals"] = np.percentile(site_probabilities_df, 97.725, axis=1)
            weighted_mean_site_probs_dictionary[site][f"{exceed_type}_84_135_vals"] = np.percentile(site_probabilities_df, 84.135, axis=1)
            weighted_mean_site_probs_dictionary[site][f"{exceed_type}_15_865_vals"] = np.percentile(site_probabilities_df, 15.865, axis=1)
            weighted_mean_site_probs_dictionary[site][f"{exceed_type}_2_275_vals"] = np.percentile(site_probabilities_df, 2.275, axis=1)

            # Calculate errors based on 1 and 2 sigma WEIGHTED percentiles of all of the branches for each threshold (better option)
            weighted_mean_site_probs_dictionary[site][f"{exceed_type}_w97_725_vals"] = percentile(site_probabilities_df, 97.725, axis=1, weights=branch_weights)
            weighted_mean_site_probs_dictionary[site][f"{exceed_type}_w84_135_vals"] = percentile(site_probabilities_df, 84.135, axis=1, weights=branch_weights)
            weighted_mean_site_probs_dictionary[site][f"{exceed_type}_w15_865_vals"] = percentile(site_probabilities_df, 15.865, axis=1, weights=branch_weights)
            weighted_mean_site_probs_dictionary[site][f"{exceed_type}_w2_275_vals"] = percentile(site_probabilities_df, 2.275, axis=1, weights=branch_weights)

            calc_uc_weighting = True
            # This method uses the uncertainty calculated for each branch, as well as the branch weights, to calculate the weighted mean and error.
            # However, it's not great, and using the branch weighting seems to work better for calculating the exceedence probabilities.
            # Additionally, when combining all the errors, the error is seemingly so small it only surrounds the weighted mean value branch, and doesn't
            # really reflect the variation in branches. Keeping the calculation anyway though, just so you can plot it if you want to.
            if calc_uc_weighting:
                site_errors_df = pd.DataFrame(errors_df)
                full_weights = branch_weights/((site_errors_df) ** 2)
                full_weights[full_weights == np.inf] = 0
                zero_weights = np.where(np.sum(full_weights, axis=1) == 0)[0]
                full_weights.loc[zero_weights] = 1
                site_probabilities_df = pd.concat([site_probabilities_df, full_weights], axis='columns')
                site_weighted_mean_probs = site_probabilities_df.apply(lambda x: np.average(x[:n_branches], weights=x[n_branches:]), axis=1)
                site_weighted_error = np.sqrt(1 / full_weights.sum(axis=1))
                site_weighted_error[zero_weights] = 0
                weighted_mean_site_probs_dictionary[site][f"uc_weighted_exceedance_probs_{exceed_type}"] = site_weighted_mean_probs
                weighted_mean_site_probs_dictionary[site][f"{exceed_type}_error"] = site_weighted_error

    with open(f"../{out_directory}/weighted_mean_PPE_dict_{outfile_extension}{taper_extension}.pkl", "wb") as f:
        pkl.dump(weighted_mean_site_probs_dictionary, f)

    return weighted_mean_site_probs_dictionary

def make_sz_crustal_paired_PPE_dict(crustal_branch_weight_dict, sz_branch_weight_dict,
                                    crustal_model_version_results_directory, sz_model_version_results_directory,
                                    slip_taper, n_samples, out_directory, outfile_extension, sz_type="sz"):
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
    paired_PPE_pickle_name = f"{sz_type}_crustal_paired_PPE_dict_{outfile_extension}{taper_extension}.pkl"

    # make a dictionary of displacements at each site from all the crustal earthquake scenarios

    all_crustal_branches_site_disp_dict = {}
    for branch_id in crustal_branch_weight_dict.keys():

        extension1 = gf_name + crustal_branch_weight_dict[branch_id]["file_suffix"]
        # get site displacement dictionary
        # this extracts the rates from the solution directory, but it is not scaled by the rate scaling factor
        crustal_branch_site_disp_dict = get_site_disp_dict(
            extension1, slip_taper=slip_taper, model_version_results_directory=crustal_model_version_results_directory)

        # multiply the rates by the rate scaling factor
        rate_scaling_factor = crustal_branch_weight_dict[branch_id]["S"]
        for site in crustal_branch_site_disp_dict.keys():
            # multiply each value in the rates array by the rate scaling factor
            crustal_branch_site_disp_dict[site]["scaled_rates"] = \
                [rate * rate_scaling_factor for rate in crustal_branch_site_disp_dict[site]["rates"]]

        all_crustal_branches_site_disp_dict[branch_id] = {"site_disp_dict":crustal_branch_site_disp_dict,
                                                   "branch_weight":crustal_branch_weight_dict[branch_id][
                                                       "total_weight_RN"]}

    # make a dictionary of displacements at each site from all the crustal earthquake scenarios
    all_sz_branches_site_disp_dict = {}
    for branch_id in sz_branch_weight_dict.keys():
        sz_slip_taper = False

        extension1 = gf_name + sz_branch_weight_dict[branch_id]["file_suffix"]
        # get displacement dictionary
        # this extracts the rates from the solution directory, but it is not scaled by the rate scaling factor
        sz_branch_site_disp_dict = get_site_disp_dict(
            extension1, slip_taper=sz_slip_taper, model_version_results_directory=sz_model_version_results_directory)

        # multiply the rates by the rate scaling factor
        rate_scaling_factor = sz_branch_weight_dict[branch_id]["S"]
        for site in sz_branch_site_disp_dict.keys():
            # multiply each value in the rates array by the rate scaling factor
            sz_branch_site_disp_dict[site]["scaled_rates"] = \
                [rate * rate_scaling_factor for rate in sz_branch_site_disp_dict[site]["rates"]]

        all_sz_branches_site_disp_dict[branch_id] = {"site_disp_dict":sz_branch_site_disp_dict,
                                                   "branch_weight":sz_branch_weight_dict[branch_id]["total_weight_RN"]}

    # make all the combinations of crustal and subduction zone branch pairs
    crustal_sz_branch_pairs = list(itertools.product(crustal_branch_weight_dict.keys(),
                                                      sz_branch_weight_dict.keys()))

    counter = 0
    paired_crustal_sz_PPE_dict = {}
    for pair in crustal_sz_branch_pairs:
        # get the branch unique ID for the crustal and sz combos
        crustal_unique_id, sz_unique_id = pair[0], pair[1]
        pair_unique_id = crustal_unique_id + "_" + sz_unique_id

        print(f"calculating {pair_unique_id} PPE\t({counter} of {len(crustal_sz_branch_pairs)} branches)")
        counter += 1

        site_names = list(all_crustal_branches_site_disp_dict[crustal_unique_id]["site_disp_dict"].keys())

        pair_weight = all_crustal_branches_site_disp_dict[crustal_unique_id]["branch_weight"] * \
                       all_sz_branches_site_disp_dict[sz_unique_id]["branch_weight"]

        # loop over all the sites for the crustal and sz branches of interest
        # make one long list of displacements and corresponding scaled rates per site
        pair_site_disp_dict = {}
        for j, site in enumerate(site_names):
            site_coords = all_crustal_branches_site_disp_dict[crustal_unique_id]["site_disp_dict"][site]["site_coords"]

            crustal_site_disps = all_crustal_branches_site_disp_dict[crustal_unique_id]["site_disp_dict"][site]["disps"]
            sz_site_disps = all_sz_branches_site_disp_dict[sz_unique_id]["site_disp_dict"][site]["disps"]

            crustal_site_scaled_rates = all_crustal_branches_site_disp_dict[crustal_unique_id]["site_disp_dict"][
                site]["scaled_rates"]
            sz_site_scaled_rates = all_sz_branches_site_disp_dict[sz_unique_id]["site_disp_dict"][site]["scaled_rates"]

            pair_site_disps = crustal_site_disps + sz_site_disps
            pair_scaled_rates = crustal_site_scaled_rates + sz_site_scaled_rates

            pair_site_disp_dict[site] = {"disps": pair_site_disps, "scaled_rates": pair_scaled_rates,
                                                 "site_coords": site_coords}


        # get exceedence probabilities for each crustal/sz pair

        if not os.path.exists(f"../{out_directory}"):
            os.mkdir(f"../{out_directory}")
        pair_cumu_PPE_dict = get_cumu_PPE(branch_key=pair_unique_id, branch_site_disp_dict=pair_site_disp_dict,
                                            model_version_results_directory=out_directory,
                                            slip_taper=slip_taper, time_interval=100,
                                            n_samples=n_samples, extension1="")

        paired_crustal_sz_PPE_dict[pair_unique_id] = {"cumu_PPE_dict": pair_cumu_PPE_dict, "branch_weight": pair_weight}

    with open(f"../{out_directory}/{paired_PPE_pickle_name}", "wb") as f:
        pkl.dump(paired_crustal_sz_PPE_dict, f)
    return paired_crustal_sz_PPE_dict

def get_exceedance_bar_chart_data(site_PPE_dictionary, probability, exceed_type, site_list, weighted=False):
    """returns displacements at the X% probabilities of exceedance for each site

    define exceedance type. Options are "total_abs", "up", "down"
    """

    if weighted:
        prefix = 'weighted_'
        thresh = 'threshold_vals'
    else:
        prefix, thresh = '', 'thresholds'

    # get disp threshold (x-value) at defined probability (y-value)
    disps = []
    for site in site_list:
        threshold_vals = site_PPE_dictionary[site][thresh]

        # displacement thresholds are negative for "down" exceedances
        if exceed_type == "down":
            threshold_vals = -threshold_vals

        site_PPE = site_PPE_dictionary[site][f"{prefix}exceedance_probs_{exceed_type}"]

        # get first index that is < 10% (ideally we would interpolate for exact value but don't have a function)
        exceedance_index = next((index for index, value in enumerate(site_PPE) if value <= probability), -1)
        disp = threshold_vals[exceedance_index]
        disps.append(disp)

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
        thresh = 'threshold_vals'
    else:
        prefix, thresh = '', 'thresholds'	

    if site_list == None:
        site_list = list(site_PPE_dictionary.keys())

    # get list of probabilities at defined displacement threshold (one for each site)
    probs_threshold = []
    for site in site_list:
        site_PPE = site_PPE_dictionary[site][f"{prefix}exceedance_probs_{exceed_type}"]
        threshold_vals = list(site_PPE_dictionary[site][thresh])

        # find index in threshold_vals where the value matches the parameter threshold
        index = threshold_vals.index(threshold)
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
                threshold_vals = PPE_dictionary[site]["thresholds"]

                ax.plot(threshold_vals, exceedance_probs, color=curve_color)
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
                threshold_vals = PPE_dictionary[site]["thresholds"]

                ax.plot(threshold_vals, exceedance_probs, color=curve_color)
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

def plot_weighted_mean_haz_curves(weighted_mean_PPE_dictionary, PPE_dictionary, exceed_type_list,
                                  model_version_title, out_directory, file_type_list, slip_taper, plot_order):
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
    weights = weighted_mean_PPE_dictionary['branch_weights']
    weight_order = np.argsort(weights)
    weight_colouring = False
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
                max_probs = weighted_mean_PPE_dictionary[site][f"{exceed_type}_max_vals"]
                min_probs = weighted_mean_PPE_dictionary[site][f"{exceed_type}_min_vals"]
                threshold_vals = weighted_mean_PPE_dictionary[site]["threshold_vals"]
                threshold_vals = threshold_vals[1:]
                max_probs = max_probs[1:]
                min_probs = min_probs[1:]

                # Shade based on max-min
                #ax.fill_between(threshold_vals, max_probs, min_probs, color='0.9')
                # Shade based on weighted errors
                #ax.fill_between(threshold_vals, weighted_mean_PPE_dictionary[site][f"weighted_exceedance_probs_{exceed_type}"][1:] + weighted_mean_PPE_dictionary[site][f"{exceed_type}_error"][1:],
                #                weighted_mean_PPE_dictionary[site][f"weighted_exceedance_probs_{exceed_type}"][1:] - weighted_mean_PPE_dictionary[site][f"{exceed_type}_error"][1:], color='0.9')
                # Shade based on weighted 2 sigma percentiles
                ax.fill_between(threshold_vals, weighted_mean_PPE_dictionary[site][f"{exceed_type}_w97_725_vals"][1:],
                                weighted_mean_PPE_dictionary[site][f"{exceed_type}_w2_275_vals"][1:], color='0.8')

            # plot all the branches as light grey lines
            # for each branch, plot the exceedance probabilities for each site
            for k, unique_id in enumerate([unique_id_list[id] for id in weight_order]):
                # this loop isn't really needed, but it's useful if you calculate Green's functions
                # at more sites than you want to plot
                for i, site in enumerate(sites):
                    threshold_vals = PPE_dictionary[unique_id]["cumu_PPE_dict"][site]["thresholds"]
                    site_exceedance_probs = PPE_dictionary[unique_id]["cumu_PPE_dict"][site][f"exceedance_probs_{exceed_type}"]
                    ax = plt.subplot(n_rows, n_cols, i + 1)
                    #ax.plot(threshold_vals, site_exceedance_probs, color=[weights[weight_order[k]] / max_weight, 1-(weights[weight_order[k]] / max_weight), 0],
                    #        linewidth=0.1)
                    if weight_colouring:
                        ax.plot(threshold_vals, site_exceedance_probs, color=colours[weight_order[k]], linewidth=0.2, alpha=0.5)
                    else:
                        ax.plot(threshold_vals, site_exceedance_probs, color='grey', linewidth=0.2, alpha=0.5)


            # loop through sites and add the weighted mean lines
            for i, site in enumerate(sites):
                ax = plt.subplot(n_rows, n_cols, i + 1)

                # plots all three types of exceedance (total_abs, up, down) on the same plot
                weighted_mean_exceedance_probs = weighted_mean_PPE_dictionary[site][f"weighted_exceedance_probs_{exceed_type}"]
                threshold_vals = weighted_mean_PPE_dictionary[site]["threshold_vals"]

                line_color = get_probability_color(exceed_type)
               
                # Unweighted 1 sigma lines
                # ax.plot(threshold_vals, weighted_mean_PPE_dictionary[site][f"{exceed_type}_84_135_vals"], color=line_color, linewidth=0.75, linestyle='-.')
                # ax.plot(threshold_vals, weighted_mean_PPE_dictionary[site][f"{exceed_type}_15_865_vals"], color=line_color, linewidth=0.75, linestyle='-.')
                # Unweighted 2 sigma lines
                # ax.plot(threshold_vals, weighted_mean_PPE_dictionary[site][f"{exceed_type}_97_725_vals"], color=line_color, linewidth=0.75, linestyle='--')
                # ax.plot(threshold_vals, weighted_mean_PPE_dictionary[site][f"{exceed_type}_2_275_vals"], color=line_color, linewidth=0.75, linestyle='--')

                # Weighted 1 sigma lines
                # ax.plot(threshold_vals, weighted_mean_PPE_dictionary[site][f"{exceed_type}_w84_135_vals"], color=line_color, linewidth=0.75, linestyle=':')
                # ax.plot(threshold_vals, weighted_mean_PPE_dictionary[site][f"{exceed_type}_w15_865_vals"], color=line_color, linewidth=0.75, linestyle=':')
                # Weighted 2 sigma lines
                ax.plot(threshold_vals, weighted_mean_PPE_dictionary[site][f"{exceed_type}_w97_725_vals"], color='black', linewidth=0.75, linestyle='-.')
                ax.plot(threshold_vals, weighted_mean_PPE_dictionary[site][f"{exceed_type}_w2_275_vals"], color='black', linewidth=0.75, linestyle='-.')

                ax.plot(threshold_vals, weighted_mean_exceedance_probs, color=line_color, linewidth=1.5)

                # Uncertainty weighted mean
                #ax.plot(threshold_vals, weighted_mean_PPE_dictionary[site][f"uc_weighted_exceedance_probs_{exceed_type}"], color='black', linewidth=1)

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
                    fill_color = get_probability_color(exceed_type)

                    weighted_mean_max_probs = weighted_mean_PPE_dictionary[site][f"{exceed_type}_max_vals"]
                    weighted_mean_min_probs = weighted_mean_PPE_dictionary[site][f"{exceed_type}_min_vals"]
                    threshold_vals = weighted_mean_PPE_dictionary[site]["threshold_vals"]
                    # Shade based on max-min
                    # ax.fill_between(threshold_vals, weighted_mean_max_probs, weighted_mean_min_probs, color=fill_color, alpha=0.2)
                    # Shade based on weighted errors
                    #ax.fill_between(threshold_vals, weighted_mean_PPE_dictionary[site][f"weighted_exceedance_probs_{exceed_type}"][1:] + weighted_mean_PPE_dictionary[site][f"{exceed_type}_error"][1:],
                    #                weighted_mean_PPE_dictionary[site][f"weighted_exceedance_probs_{exceed_type}"][1:] - weighted_mean_PPE_dictionary[site][f"{exceed_type}_error"][1:], color='0.9')
                    # Shade based on 2 sigma percentiles
                    ax.fill_between(threshold_vals, weighted_mean_PPE_dictionary[site][f"{exceed_type}_97_725_vals"],
                                    weighted_mean_PPE_dictionary[site][f"{exceed_type}_2_275_vals"], color='0.8')

                # plot solid lines on top of the shaded regions
                for exceed_type in exceed_type_list:
                    line_color = get_probability_color(exceed_type)
                    weighted_mean_exceedance_probs = weighted_mean_PPE_dictionary[site][
                        f"weighted_exceedance_probs_{exceed_type}"]
                    ax.plot(threshold_vals, weighted_mean_exceedance_probs, color=line_color, linewidth=2)

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
                threshold_vals = weighted_mean_PPE_dictionary[site]["threshold_vals"]

                threshold_vals = threshold_vals[1:]
                max_probs = max_probs[1:]
                min_probs = min_probs[1:]

                # Shade based on max-min
                #ax.fill_between(threshold_vals, max_probs, min_probs, color='0.9', label="_nolegend_")
                # Shade based on weighted errors
                #ax.fill_between(threshold_vals, weighted_mean_PPE_dictionary[site][f"weighted_exceedance_probs_{exceed_type}"][1:] + weighted_mean_PPE_dictionary[site][f"{exceed_type}_error"][1:],
                #                weighted_mean_PPE_dictionary[site][f"weighted_exceedance_probs_{exceed_type}"][1:] - weighted_mean_PPE_dictionary[site][f"{exceed_type}_error"][1:], color='0.9', label="_nolegend_")
                # Shade based on 2 sigma percentiles
                ax.fill_between(threshold_vals, weighted_mean_PPE_dictionary[site][f"{exceed_type}_w97_725_vals"][1:],
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

                    threshold_vals = PPE_dictionary[unique_id]["cumu_PPE_dict"][site]["thresholds"]
                    site_exceedance_probs = PPE_dictionary[unique_id]["cumu_PPE_dict"][site][f"exceedance_probs_{exceed_type}"]

                    # skip the 0 value in the list
                    threshold_vals = threshold_vals[1:]
                    site_exceedance_probs = site_exceedance_probs[1:]

                    # ax = plt.subplot(4, 3, i + 1)
                    ax = plt.subplot(n_rows, n_cols + 1, subplot_indices[i])

                    #ax.plot(threshold_vals, site_exceedance_probs, color='0.7')
                    ax.plot(threshold_vals, site_exceedance_probs, color=line_color, linewidth=linewidth)

            # loop through sites and add the weighted mean lines
            for i, site in enumerate(sites):
                # ax = plt.subplot(4, 3, i + 1)
                ax = plt.subplot(n_rows, n_cols + 1, subplot_indices[i])

                # plots all three types of exceedance (total_abs, up, down) on the same plot
                weighted_mean_exceedance_probs = weighted_mean_PPE_dictionary[site][f"weighted_exceedance_probs_{exceed_type}"]
                threshold_vals = weighted_mean_PPE_dictionary[site]["threshold_vals"]

                threshold_vals = threshold_vals[1:]
                weighted_mean_exceedance_probs = weighted_mean_exceedance_probs[1:]

                line_color = get_probability_color(exceed_type)
                # Weighted 2 sigma lines
                ax.plot(threshold_vals, weighted_mean_PPE_dictionary[site][f"{exceed_type}_w97_725_vals"][1:], color='black', linewidth=0.75, linestyle='-.')
                ax.plot(threshold_vals, weighted_mean_PPE_dictionary[site][f"{exceed_type}_w2_275_vals"][1:], color='black', linewidth=0.75, linestyle='-.')
                ax.plot(threshold_vals, weighted_mean_exceedance_probs, color=line_color, linewidth=2)

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

def save_disp_prob_tifs(extension1,  slip_taper, model_version_results_directory, thresh_lims=[0, 3], thresh_step=0.1, thresholds=None, \
                        probs_lims = [0.02, 0.5], probs_step=0.02, probabilites=None, output_thresh=True, output_probs=True, weighted=False, grid=False):
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
        threshold_key = 'threshold_vals'
    else:
        dict_file = f"../{model_version_results_directory}/{extension1}/cumu_exceed_prob_{extension1}{taper_extension}.pkl"
        outfile_directory = f"../{model_version_results_directory}/{extension1}/probability_grids"
        threshold_key = 'thresholds'
    
    with open(dict_file, "rb") as fid:
        PPE_dict = pkl.load(fid)

    sites = [*PPE_dict]

    # Calculate XY extents and resolution for tifs
    if grid:
        with open(f"../{model_version_results_directory}/{extension1}/grid_limits.pkl", "rb") as fid:
            grid_meta = pkl.load(fid)  

        x, y, buffer_size, x_res, y_res = grid_meta['x'], grid_meta['y'], grid_meta['buffer_size'], grid_meta['cell_size'], grid_meta['cell_size'] 

        x_data = np.arange(round(x - buffer_size, -3), round(x + buffer_size, -3), x_res)
        y_data = np.arange(round(y - buffer_size, -3), round(y + buffer_size, -3), y_res)

    else:
        site_x = [pixel['site_coords'][0] for _, pixel in PPE_dict.items()]
        site_y = [pixel['site_coords'][1] for _, pixel in PPE_dict.items()]

        x_data = np.unique(site_x)
        y_data = np.unique(site_y)

        xmin, xmax = min(x_data), max(x_data)
        ymin, ymax = min(y_data), max(y_data)
        x_res, y_res = min(np.diff(x_data)), min(np.diff(y_data))

        x_data = np.arange(xmin, xmax + x_res, x_res)
        y_data = np.arange(ymin, ymax + y_res, y_res)

        if not all(np.in1d(site_x, x_data)) or not all(np.in1d(site_y, y_data)):
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
    
        if not all(np.in1d(thresholds, PPE_dict[sites[0]][threshold_key])):
            thresholds = thresholds[np.in1d(thresholds, PPE_dict[sites[0]][threshold_key])]
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

            file_name=f"{extension1}{taper_extension}_{exceed_type}_disp_prob_sites.tif".strip('_')
            with rasterio.open(f"{outfile_directory}/{file_name}", 'w', \
                            driver='GTiff', count=thresh_grd.shape[0], height=thresh_grd.shape[1], width=thresh_grd.shape[2], \
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
