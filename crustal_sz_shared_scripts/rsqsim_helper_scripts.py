import numpy as np

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
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
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def time_elasped(current_time, start_time):
    elapsed_time = current_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds))

def get_probability_bar_chart_data(site_PPE_dictionary, exceed_type, threshold, sigmas, site_list=None):
    """ function that finds the probability at each site for the specified displacement threshold on the hazard curve
        Inputs:
        :param: dictionary of exceedance probabilities for each site (key = site)
        :param exceedance type: string; "total_abs", "up", or "down"
        :param: list of sites to get data for. If None, will get data for all sites in site_PPE_dictionary.
                I made this option so that you could skip the sites you didn't care about (e.g., use "plot_order")

        Outputs:
        :return    probs_threshold: list of probabilities of exceeding the specified threshold (one per site)
            """

    if site_list == None:
        site_list = list(site_PPE_dictionary.keys())

    thresholds = [round(val, 4) for val in site_PPE_dictionary["thresholds"]]
    # find index in thresholds where the value matches the parameter threshold
    index = thresholds.index(round(threshold, 4))

    # get list of probabilities at defined displacement threshold (one for each site)
    probs_threshold = []
    disps_err_low = []
    disps_err_high = []
    sigma_low = np.where(site_PPE_dictionary["sigma_lims"][:] == sigmas[0])[0][0]
    sigma_high = np.where(site_PPE_dictionary["sigma_lims"][:] == sigmas[1])[0][0]
    for site in site_list:
        site_PPE = site_PPE_dictionary[site][f"exceedance_probs_{exceed_type}"]
        probs_threshold.append(site_PPE[index])
        err_low = site_PPE_dictionary[site][f"error_exceedance_{exceed_type}"][:][index, sigma_low]
        err_high = site_PPE_dictionary[site][f"error_exceedance_{exceed_type}"][:][index, sigma_high]

        disps_err_low.append(err_low)
        disps_err_high.append(err_high)

    return probs_threshold, disps_err_low, disps_err_high

def get_exceedance_bar_chart_data(site_PPE_dictionary, probability, exceed_type, site_list, sigmas):
    """returns displacements at the X% probabilities of exceedance for each site

    define exceedance type. Options are "total_abs", "up", "down"
    """

    thresholds = np.array([round(val, 4) for val in site_PPE_dictionary["thresholds"]])

    # displacement thresholds are negative for "down" exceedances
    if exceed_type == "down":
        thresholds = -thresholds

    # get disp threshold (x-value) at defined probability (y-value)
    disps = []
    disps_err_low = []
    disps_err_high = []
    sigma_low = np.where(site_PPE_dictionary["sigma_lims"][:] == sigmas[0])[0][0]
    sigma_high = np.where(site_PPE_dictionary["sigma_lims"][:] == sigmas[1])[0][0]

    for site in site_list:
        site_PPE = site_PPE_dictionary[site][f"exceedance_probs_{exceed_type}"]
        err_low_PPE = site_PPE_dictionary[site][f"error_exceedance_{exceed_type}"][:, sigma_low]
        err_high_PPE = site_PPE_dictionary[site][f"error_exceedance_{exceed_type}"][:, sigma_high]

        # get first index that is < 10% (ideally we would interpolate for exact value but don't have a function)
        exceedance_index = next((index for index, value in enumerate(site_PPE) if value <= round(probability,4)), -1)
        err_low_ix = next((index for index, value in enumerate(err_low_PPE) if value <= round(probability,4)), -1)
        err_high_ix = next((index for index, value in enumerate(err_high_PPE) if value <= round(probability,4)), -1)

        if any([err_high_ix != -1, err_high_ix != 10]):
            disps.append(thresholds[exceedance_index])
            disps_err_low.append(thresholds[err_low_ix])
            disps_err_high.append(thresholds[err_high_ix])

    return disps, disps_err_low, disps_err_high