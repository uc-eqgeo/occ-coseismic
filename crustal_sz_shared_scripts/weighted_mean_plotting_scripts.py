# from probabalistic_displacement_scripts import make_10_2_disp_bar_chart, get_exceedance_bar_chart_data, \
#     get_probability_bar_chart_data
import os
import pickle as pkl
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from helper_scripts import get_figure_bounds
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import matplotlib.ticker as mticker
from helper_scripts import make_qualitative_colormap

# define parameters
slip_taper = False                           # True or False, only matters if crustal. Defaults to False for sz.
fault_type = "crustal"                      # "crustal or "sz"; only matters for single fault model
crustal_model_version = "_CFM"           # "_camillas_meshes", "_Model1", "_Model2", or "_CFM"
sz_model_version = "_v1"                    # must match suffix in the subduction directory with gfs
outfile_extension = "testing"               # Optional; something to tack on to the end so you don't overwrite
# files

# Do you want to calculate the PPEs for a single fault model or a paired crustal/subduction model?
single_fault_model = True                   # True or False
paired_crustal_sz = False

file_type_list = ["png"]
results_directory = "results_jde"
##############

plot_order = ["Paraparaumu", "Porirua CBD north", "South Coast", "Wellington Airport", "Wellington CBD", "Petone",
                  "Seaview", "Eastbourne", "Turakirae Head", "Lake Ferry", "Cape Palliser", "Flat Point"]
gf_name = "sites"
extension1 = ""

if single_fault_model == paired_crustal_sz:
    raise ValueError("you must choose either single fault model or paired crustal/subduction model")

# these directories should already be made from calculating displacements in a previous script
crustal_model_version_results_directory = f"{results_directory}/crustal{crustal_model_version}"
sz_model_version_results_directory = f"{results_directory}/sz{sz_model_version}"
paired_model_version_results_directory = f"{results_directory}/paired_c{crustal_model_version}_sz{sz_model_version}"

if paired_crustal_sz:
    model_version_results_directory = paired_model_version_results_directory
    model_version = f"c{crustal_model_version}_sz{sz_model_version}"
elif single_fault_model:
    if fault_type == "crustal":
        model_version_results_directory = crustal_model_version_results_directory
        model_version = f"c{crustal_model_version}"
    elif fault_type == "sz":
        model_version_results_directory = sz_model_version_results_directory
        model_version = f"sz{sz_model_version}"


def get_mean_disp_barchart_data(site_PPE_dictionary, probability, exceed_type, site_list):
    """returns displacements at the X% probabilities of exceedance for each site
    This is effectively " find the X value for the desired Y"
    :param exceed_type: Options are "total_abs", "up", "down"
    """

    # get disp at 10% exceedance
    disps = []
    errs_plus = []
    errs_minus = []
    for site in site_list:
        threshold_vals = site_PPE_dictionary[site]["threshold_vals"]

        # displacement thresholds are negative for "down" exceedances
        if exceed_type == "down":
            threshold_vals = -threshold_vals

        # probability values at each threshold
        site_mean_probs = site_PPE_dictionary[site][f"weighted_exceedance_probs_{exceed_type}"]
        max_probs = site_PPE_dictionary[site][f"{exceed_type}_max_vals"]
        min_probs = site_PPE_dictionary[site][f"{exceed_type}_min_vals"]

        # get first index (x-value) for the probability (y-value) that is *just* < prob% (ideally we would interpolate
        # for exact value but don't have a function for that)
        mean_exceedance_index = next((index for index, value in enumerate(site_mean_probs) if value <= probability), -1)
        max_exceedance_index = next((index for index, value in enumerate(max_probs) if value <= probability), -1)
        min_exceedance_index = next((index for index, value in enumerate(min_probs) if value <= probability), -1)

        # displacement value at the X% probability
        disp = threshold_vals[mean_exceedance_index]
        disps.append(disp)

        #minimum and maximum values at the same index
        max_disp = threshold_vals[max_exceedance_index]
        min_disp = threshold_vals[min_exceedance_index]
        if exceed_type == "down":
            err_plus = abs(min_disp - disp)
            err_minus = abs(disp - max_disp)
        if exceed_type == "up" or exceed_type == "total_abs":
            err_plus = abs(max_disp - disp)
            err_minus = abs(disp - min_disp)

        errs_plus.append(err_plus)
        errs_minus.append(err_minus)

    return disps, errs_plus, errs_minus

def get_mean_prob_barchart_data(site_PPE_dictionary, threshold, exceed_type, site_list):
    """ function that finds the probability at each site for the specified displacement threshold on the hazard curve
        Inputs:
        :param: dictionary of exceedance probabilities for each site (key = site)
        :param exceedance type: string; "total_abs", "up", or "down"
        :param: list of sites to get data for. If None, will get data for all sites in site_PPE_dictionary.
                I made this option so that you could skip the sites you didn't care about (e.g., use "plot_order")

        Outputs:
        :return    probs: list of probabilities of exceeding the specified threshold (one per site)
        :return    errs_plus: list of (+) errors for each probability (one per site)
        :return    errs_minus: list of (-) errors for each probability (one per site)
            """

    # get disp at 10% exceedance
    probs = []
    errs_plus = []
    errs_minus = []

    # get list of probabilities at defined displacement threshold (one for each site)
    for site in site_list:
        site_PPE = site_PPE_dictionary[site][f"weighted_exceedance_probs_{exceed_type}"]
        threshold_vals = list(site_PPE_dictionary[site]["threshold_vals"])

        # find index in threshold_vals where the value matches the parameter threshold
        probs_index = threshold_vals.index(threshold)
        probs.append(site_PPE[probs_index])

        mean_prob = site_PPE[probs_index]
        max_prob = site_PPE_dictionary[site][f"{exceed_type}_max_vals"][probs_index]
        min_prob = site_PPE_dictionary[site][f"{exceed_type}_min_vals"][probs_index]

        err_plus = max_prob - mean_prob
        err_minus = mean_prob - min_prob
        errs_plus.append(err_plus)
        errs_minus.append(err_minus)

    return probs, errs_plus, errs_minus


def make_mean_10_2_disp_bar_chart(slip_taper, model_version, model_version_results_directory,
                                  outfile_extension, file_type_list):
    """ makes bar charts of the displacement value at the 10% and 2% probability of exceence thresholds for each site
        fault_type = "crustal" or "sz"
        slip_taper = True or False
    """

    probability_list = [0.1, 0.02]

    if slip_taper is True:
        taper_extension = "_tapered"
    else:
        taper_extension = "_uniform"

    with open(f"../{model_version_results_directory}/weighted_mean_PPE_dict_{outfile_extension}{taper_extension}.pkl",
              "rb") as fid:
        site_PPE_dictionary = pkl.load(fid)

    fig, axs = plt.subplots(1, 2, figsize=(7, 5))
    x = np.arange(len(plot_order))  # the site label locations
    width = 0.4  # the width of the bars
    # find maximum value in all the "up" columns in PPE dictionary

    max_min_y_vals = []
    max_min_errs_y_val = []
    for i, probability in enumerate(probability_list):
        disps_up, errs_up_plus, errs_up_minus = \
            get_mean_disp_barchart_data(site_PPE_dictionary=site_PPE_dictionary, exceed_type="up",
                                    site_list=plot_order, probability=probability)
        disps_down, errs_down_plus, errs_down_minus = \
            get_mean_disp_barchart_data(site_PPE_dictionary=site_PPE_dictionary, exceed_type="down",
                                     site_list=plot_order, probability=probability)

        max_min_y_vals.append(max(disps_up))
        max_min_y_vals.append(min(disps_down))
        errs_y_pos = max([disps_up[j] + errs_up_plus[j] for j in range(len(disps_up))])
        errs_y_neg = min([disps_down[j] - errs_down_minus[j] for j in range(len(disps_down))])
        max_min_errs_y_val.append(errs_y_pos)
        max_min_errs_y_val.append(errs_y_neg)

        color_up, color_down = (189/255, 0, 0), (15/255, 72/255, 186/255)
        label_size = 6
        #label_offset = label_size / 100

        # add bars to plot, add black horizontal line at zero.
        bars_up = axs[i].bar(x, disps_up, width, color=color_up, linewidth=0.5)
        bars_down = axs[i].bar(x, disps_down, width, color=color_down, linewidth=0.5)

        # add error bars
        axs[i].errorbar(x, disps_up, yerr=[errs_up_minus, errs_up_plus], fmt='none', ecolor='0.6', capsize=0,
                            linewidth=1, markeredgewidth=0)
        axs[i].errorbar(x, disps_down, yerr=[errs_down_minus, errs_down_plus], fmt='none', ecolor='0.6', capsize=0,
                            linewidth=1, markeredgewidth=0)

        # add zero line
        axs[i].axhline(y=0, color="k", linewidth=0.5)

        # add value labels to bars
        label_offset = 3 * 0.05
        # add value labels to bars
        for bar in bars_up:
            bar_color = bar.get_facecolor()
            axs[i].text(x=(bar.get_x() + bar.get_width() * 0.6), y=bar.get_height() + label_offset,
                        s=round(bar.get_height(), 1), ha='left',
                        va='center', color=bar_color, fontsize=label_size, fontweight='regular')
        for bar in bars_down:
            bar_color = bar.get_facecolor()
            axs[i].text(x=(bar.get_x() + bar.get_width() * 0.6), y=bar.get_height() - label_offset,
                        s=round(bar.get_height(), 1), ha='left',
                        va='center', color=bar_color, fontsize=label_size, fontweight='regular')

    for i in range(len(probability_list)):
        #axs[i].set_ylim(min(max_min_y_vals) - 0.2, max(max_min_y_vals) + 0.2)

        if max(max_min_errs_y_val) < 0.3:
            plot_ymax, plot_ymin = 0.3, -0.3
        else:
            plot_ymax = max(max_min_errs_y_val) + 1.5*label_offset + label_size / 100
            plot_ymin = min(max_min_errs_y_val) - 1.5*label_offset - label_size / 100
        axs[i].set_ylim(plot_ymin, plot_ymax)
        axs[i].tick_params(axis='x', labelrotation=90, labelsize=label_size)
        axs[i].tick_params(axis='y', labelsize=8)
        axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # set tick labels to be every 0.2
        axs[i].yaxis.set_major_locator(mticker.MultipleLocator(0.5))
        axs[i].set_xticks(x, plot_order)

    # set indidual subplot stuff

    axs[0].set_ylabel("Displacement (m)", fontsize=8)
    axs[1].tick_params(axis='y', labelleft=False)

    axs[0].set_title(f"10% probability of exceedance", fontsize=8)
    axs[1].set_title(f"2% probability of exceedance", fontsize=8)

    # manually make legend with rectangles and text
    swatch_width, swatch_height = width, plot_ymax / 20
    swatch_minx, swatch_miny = -1 * (len(plot_order) / 30), plot_ymax - (swatch_height * 2)
    axs[0].add_patch(Rectangle((swatch_minx, swatch_miny), swatch_width, swatch_height,
                               facecolor=color_up, edgecolor=None))
    axs[0].add_patch(Rectangle((swatch_minx, swatch_miny - 2 * swatch_height), swatch_width, swatch_height,
                               facecolor=color_down, edgecolor=None))


    axs[0].text(swatch_minx + 2 * swatch_width, swatch_miny, "uplift", fontsize=8)
    axs[0].text(swatch_minx + 2 * swatch_width, swatch_miny - 2 * swatch_height, "subsidence", fontsize=8)

    fig.suptitle(f"100 yr exceedance displacements \nweighted mean {model_version}", fontsize=10)
    fig.tight_layout()

    # make a directory for the figures if it doesn't already exist
    if not os.path.exists(f"../{model_version_results_directory}/weighted_mean_figures"):
        os.makedirs(f"../{model_version_results_directory}/weighted_mean_figures")

    for file_type in file_type_list:
        fig.savefig(f"../{model_version_results_directory}/weighted_mean_figures/"
                    f"10_2_disps{extension1}{taper_extension}.{file_type}", dpi=300)


def make_site_prob_barchart(slip_taper, fault_type, model_version_results_directory, model_version, outfile_extension,
                        file_type_list, threshold=0.2, optional_extension=""):
    # What is the probability of exceeding 0.2 m subsidence, 0.2 m uplift at each site?
    """ determines the probability of exceeding a defined displacement threshold at each site and plots as a bar chart
        two-part plot, one for up and one for down. y axis is probability, x axis is site name
        :param extension1: string, name of the NSHM branch suffix etc.
        :param slip_taper: boolean, True if slip tapers, False if uniform slip
        :param fault_type: string, "crustal" or sz"
        :param threshold: float, displacement threshold to determine exceedance probability
        :param results_directory: string, name of directory where results are stored
    """

    exceed_type_list = ["up", "down"]

    if slip_taper is True:
        taper_extension = "_tapered"
    else:
        taper_extension = "_uniform"

    with open(f"../{model_version_results_directory}/weighted_mean_PPE_dict_{outfile_extension}.pkl",
              "rb") as fid:
        site_PPE_dictionary = pkl.load(fid)

    # set up custom color scheme
    colors = make_qualitative_colormap("custom", len(plot_order))

    # set up figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(7, 5))
    x = np.arange(len(plot_order))  # the site label locations
    width = 0.6  # the width of the bars

    for i, exceed_type in enumerate(exceed_type_list):
        mean_prob, errs_plus, errs_minus = \
            get_mean_prob_barchart_data(site_PPE_dictionary=site_PPE_dictionary, exceed_type=exceed_type,
                                        threshold=threshold, site_list=plot_order)

        # add bars to plot
        bars = axs[i].bar(x, mean_prob, width, color=colors)

        max_errs_y = max([mean_prob[j] + errs_plus[j] for j in range(len(mean_prob))])

        for bar in bars:
            bar_color = bar.get_facecolor()
            # add error bars
            axs[i].errorbar(x, mean_prob, yerr=[errs_plus, errs_minus], fmt='none', ecolor='0.5', capsize=0,
                            linewidth=0.5, markeredgewidth=0)

            # add value label to each bar
            label_offset = 0.05 * max_errs_y
            axs[i].text(x=(bar.get_x() + bar.get_width() * 0.6), y=bar.get_height() + label_offset,
                        s=f"{int(100 * round(bar.get_height(), 2))}%", horizontalalignment='left', color=bar_color,
                        fontsize=6, fontweight='demibold')

        if max_errs_y < 0.3:
            axs[i].set_ylim(0.0, 0.3)
        else:
            axs[i].set_ylim(0.0, max_errs_y)
        axs[i].tick_params(axis='x', labelrotation=90, labelsize=6)
        axs[i].tick_params(axis='y', labelsize=8)
        axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # set tick labels to be every 0.2
        axs[i].yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        axs[i].set_xticks(x, plot_order)

    # set indidual subplot stuff
    axs[0].set_ylabel("Probabilty", fontsize=8)
    axs[1].tick_params(axis='y', labelleft=False)

    axs[0].set_title(f"Probability of exceeding {threshold} m uplift", fontsize=8)
    axs[1].set_title(f"Probability of exceeding {threshold} m subsidence", fontsize=8)

    fig.suptitle(f"{model_version} {fault_type} faults (100 yrs)")
    fig.tight_layout()

    # make directory for hazard curve if it doesn't exist
    if not os.path.exists(f"../{model_version_results_directory}/weighted_mean_figures"):
        os.makedirs(f"../{model_version_results_directory}/weighted_mean_figures")
    for file_type in file_type_list:
        fig.savefig(f"../{model_version_results_directory}/weighted_mean_figures/prob_bar_chart{extension1}"
                    f"{taper_extension}.{file_type}", dpi=300)

def make_site_prob_plot(slip_taper, fault_type, model_version_results_directory, model_version, outfile_extension,
                        file_type_list, threshold=0.2, optional_extension=""):
    # What is the probability of exceeding 0.2 m subsidence, 0.2 m uplift at each site?
    """ determines the probability of exceeding a defined displacement threshold at each site and plots as a bar chart
        two-part plot, one for up and one for down. y axis is probability, x axis is site name
        :param extension1: string, name of the NSHM branch suffix etc.
        :param slip_taper: boolean, True if slip tapers, False if uniform slip
        :param fault_type: string, "crustal" or sz"
        :param threshold: float, displacement threshold to determine exceedance probability
        :param results_directory: string, name of directory where results are stored
    """

    exceed_type_list = ["up", "down"]

    if slip_taper is True:
        taper_extension = "_tapered"
    else:
        taper_extension = "_uniform"

    with open(f"../{model_version_results_directory}/weighted_mean_PPE_dict_{outfile_extension}{taper_extension}.pkl",
              "rb") as fid:
        site_PPE_dictionary = pkl.load(fid)

    # set up custom color scheme
    colors = make_qualitative_colormap("custom", len(plot_order))

    # set up figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(7, 5))
    x = np.arange(len(plot_order))  # the site label locations
    width = 0.6  # the width of the bars

    for i, exceed_type in enumerate(exceed_type_list):
        mean_prob, errs_plus, errs_minus = \
            get_mean_prob_barchart_data(site_PPE_dictionary=site_PPE_dictionary, exceed_type=exceed_type,
                                        threshold=threshold, site_list=plot_order)

        # add point and error bars to plot
        axs[i].errorbar(x, mean_prob, yerr=[errs_plus, errs_minus], fmt='none', ecolor='0.5', capsize=0,
                        linewidth=0.5, markeredgewidth=0)
        axs[i].scatter(x, mean_prob, s=24, color=colors, zorder=3, edgecolors='k', linewidths=0.5)

        max_errs_y = max([mean_prob[j] + errs_plus[j] for j in range(len(mean_prob))])

        for j, value in enumerate(mean_prob):
            point_color = colors[j]

            # add value label to each bar
            label_offset = 0.05 * max_errs_y
            axs[i].text(x=x[j] + label_offset, y=mean_prob[j] + label_offset,
                        s=f"{int(100 * round(mean_prob[j], 2))}%",
                        horizontalalignment='left', color=point_color,
                        fontsize=6, fontweight='demibold')

        if max_errs_y < 0.3:
            axs[i].set_ylim(0.0, 0.3)
        else:
            axs[i].set_ylim(0.0, max_errs_y)
        axs[i].tick_params(axis='x', labelrotation=90, labelsize=6)
        axs[i].tick_params(axis='y', labelsize=8)
        axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # set tick labels to be every 0.2
        axs[i].yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        axs[i].set_xticks(x, plot_order)

    # set indidual subplot stuff
    axs[0].set_ylabel("Probabilty", fontsize=8)
    axs[1].tick_params(axis='y', labelleft=False)

    axs[0].set_title(f"Probability of exceeding {threshold} m uplift", fontsize=8)
    axs[1].set_title(f"Probability of exceeding {threshold} m subsidence", fontsize=8)

    fig.suptitle(f"{model_version} {fault_type} faults (100 yrs)")
    fig.tight_layout()

    # make directory for hazard curve if it doesn't exist
    if not os.path.exists(f"../{model_version_results_directory}/weighted_mean_figures"):
        os.makedirs(f"../{model_version_results_directory}/weighted_mean_figures")
    for file_type in file_type_list:
        fig.savefig(f"../{model_version_results_directory}/weighted_mean_figures/prob_plot{extension1}"
                    f"{taper_extension}{optional_extension}.{file_type}", dpi=300)

def map_mean_disp(weighted_mean_PPE_dictionary, exceed_type, model_version_title,
                                           out_directory, file_type_list,  slip_taper, file_name,
                  probability_list = [0.1, 0.02], grid=False):

    """ plots the displacement threshold value (m) for x% probability of exceedance in 100 years
    CAVEATS/choices:
    - currently set up with two subplots, a 10% and 2% probability of exceedance
    - exeed_type is a list of ["total_abs", "up", "down"]
    - fault_type can be "crustal" or "sz"

    use grid=True if you want to plot the gridded data as an image rather than a set of scatter points"""


    plot_order = ["Paraparaumu", "Porirua CBD north", "South Coast", "Wellington Airport", "Wellington CBD", "Petone",
                  "Seaview", "Eastbourne", "Turakirae Head", "Lake Ferry", "Cape Palliser",
                  "Flat Point"]
    site_coords_dict = {"Wellington CBD": np.array([1749376, 5427530, 0]),
                        "South Coast": np.array([1736455, 5427195, 0]),
                        "Wellington Airport": np.array([1751064, 5423128, 0]),
                        "Porirua CBD north": np.array([1754357, 5445716, 0]),
                        "Porirua CBD south": np.array([1754557, 5445119, 0]),
                        "Petone": np.array([1757199, 5434207, 0]),
                        "Seaview": np.array([1759240, 5432111, 0]),
                        "Paraparaumu": np.array([1766726, 5471342, 0]),
                        "Eastbourne": np.array([1758789, 5427418, 0]),
                        "Turakirae Head": np.array([1760183, 5410911, 0]),
                        "Lake Ferry": np.array([1779348, 5415831, 0]),
                        "Cape Palliser": np.array([1789451, 5391086, 0]),
                        "Flat Point": np.array([1848038, 5429751, 0])}

    if slip_taper is True:
        taper_extension = "_tapered"
    else:
        taper_extension = "_uniform"

    # load data

    site_coords = [site_coords_dict[site] for site in plot_order]
    x_data = [coords[0] for coords in site_coords]
    y_data = [coords[1] for coords in site_coords]


    coastline = gpd.read_file("../data/coastline/coastline_jde1.geojson")
    wellington_boundary = gpd.read_file("../data/wellington_region_boundary.geojson")
    plate_boundary = gpd.read_file("../data/coastline/plate_boundary.geojson")

    # get plot bounds (set for Wellington Region at the moment)
    plot_xmin, plot_ymin, plot_xmax, plot_ymax, \
        xmin_tick, xmax_tick, ymin_tick, ymax_tick, tick_separation \
        = get_figure_bounds(extent="Wellington close")

    plt.close("all")
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 5))

    if exceed_type == "total_abs":
        color_ramp = "magma"
    elif exceed_type == "up" or exceed_type == "down":
        color_ramp = "seismic"


    for i, probability in enumerate(probability_list):

        disps, errs_plus, errs_minus = \
            get_mean_disp_barchart_data(site_PPE_dictionary=weighted_mean_PPE_dictionary, exceed_type=exceed_type,
                                        site_list=plot_order, probability=probability)
        max_disp = max(disps)
        #max_disp = 0.5
        if exceed_type == "total_abs":
            min_disp = 0
        else:
            min_disp = -1 * max_disp

        # Format subplots
        coastline.plot(ax=axs[i], color="k", linewidth=0.5)
        plate_boundary.plot(ax=axs[i], color="0.75", linewidth=1.0)
        axs[i].set_xticks(np.arange(xmin_tick, xmax_tick, tick_separation))
        axs[i].xaxis.set_major_formatter(mticker.FormatStrFormatter('%.f mE'))
        axs[i].set_yticks(np.arange(ymin_tick, ymax_tick, tick_separation))
        axs[i].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.f mN'))
        plt.setp(axs[i].get_yticklabels(), rotation=90, ha="center", rotation_mode="anchor")
        axs[i].tick_params(axis="both", which='major', labelsize=6)
        axs[i].set_xlim(plot_xmin, plot_xmax)
        axs[i].set_ylim(plot_ymin, plot_ymax)
        axs[i].set_aspect("equal")


        if grid is True:        # reshape point data into a grid, plot grid point as a raster/image
            # get x and y dimensions
            length_unique_x, length_unique_y = len(np.unique(x_data)), len(np.unique(y_data))
            #reshape list back into a grid (for plotting)
            disps_grid = np.reshape(disps, (length_unique_y, length_unique_x))

            disp_hazard_map = axs[0].imshow(disps_grid[-1::-1], cmap=color_ramp, vmin=min_disp, vmax=max_disp,
                                            zorder=1,
                                               extent=[x_data.min(), x_data.max(), y_data.min(), y_data.max()])

        else:
            # plot as points, (10% and 2% disp value at each point)
            if len(x_data) < 20:   # if there are less than 20 points, plot with black edges
                disp_hazard_map = axs[i].scatter(x_data, y_data, s=20, c=disps, cmap=color_ramp, edgecolors='black',
                                                    linewidth=0.5, zorder=2, vmin=min_disp, vmax=max_disp)

            else:   # otherwise plot without black edges
                disp_hazard_map = axs[i].scatter(x_data, y_data, s=20, c=disps, cmap=color_ramp, edgecolors=None,
                                                    linewidth=0.5, zorder=2, vmin=min_disp, vmax=max_disp)


        # make colorbars and set ticks, etc.
        divider = make_axes_locatable(axs[i])
        cax1 = divider.append_axes('top', size='6%', pad=0.05)
        cbar1 = fig.colorbar(disp_hazard_map, cax=cax1, orientation='horizontal')
        probability_string = str(int(100 * probability))
        cbar1.set_label(f"Vertical displacement (m); {probability_string}%", fontsize=8)
        cbar1.ax.tick_params(labelsize=6)
        cax1.xaxis.set_ticks_position("top")
        cbar1.ax.xaxis.set_label_position('top')

    fig.suptitle(f"Cumulative displacement hazard {model_version_title}\n{exceed_type} (100 yrs)")
    fig.tight_layout()

    # make directory for hazard map if it doesn't exist
    if not os.path.exists(f"../{out_directory}/weighted_mean_figures"):
        os.mkdir(f"../{out_directory}/weighted_mean_figures")

    for type in file_type_list:
        fig.savefig(f"../{out_directory}/weighted_mean_figures/hazard_map_{file_name}_{taper_extension}."
                    f"{type}", dpi=300)

def map_plot_mean_disp(weighted_mean_PPE_dictionary, exceed_type, model_version_title, out_directory, file_type_list,
                       slip_taper, file_name, probability=0.1, grid=False):

    """ plots the displacement threshold value (m) for x% probability of exceedance in 100 years
    CAVEATS/choices:
    - currently set up with two subplots, a 10% and 2% probability of exceedance
    - exeed_type is a list of ["total_abs", "up", "down"]
    - fault_type can be "crustal" or "sz"

    use grid=True if you want to plot the gridded data as an image rather than a set of scatter points"""


    plot_order = ["Paraparaumu", "Porirua CBD north", "South Coast", "Wellington Airport", "Wellington CBD", "Petone",
                  "Seaview", "Eastbourne", "Turakirae Head", "Lake Ferry", "Cape Palliser",
                  "Flat Point"]


    if slip_taper is True:
        taper_extension = "_tapered"
    else:
        taper_extension = "_uniform"

    # load data
    site_coords = [weighted_mean_PPE_dictionary[site]["site_coords"] for site in plot_order]
    x_data = [coords[0] for coords in site_coords]
    y_data = [coords[1] for coords in site_coords]


    coastline = gpd.read_file("../data/coastline/coastline_jde1.geojson")
    wellington_boundary = gpd.read_file("../data/wellington_region_boundary.geojson")
    plate_boundary = gpd.read_file("../data/coastline/plate_boundary.geojson")

    # get plot bounds (set for Wellington Region at the moment)
    plot_xmin, plot_ymin, plot_xmax, plot_ymax, \
        xmin_tick, xmax_tick, ymin_tick, ymax_tick, tick_separation \
        = get_figure_bounds(extent="Wellington close")

    plt.close("all")
    # two part figure, plot on the left and map on the right
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.5))

    colors = make_qualitative_colormap("custom", len(plot_order))


    ##### plot disps on the left
    disps, errs_plus, errs_minus = \
        get_mean_disp_barchart_data(site_PPE_dictionary=weighted_mean_PPE_dictionary, exceed_type=exceed_type,
                                    site_list=plot_order, probability=probability)
    bar_width = 0.6

    # plot bars and error bars
    x = np.arange(len(plot_order))
    bars = axs[0].bar(x, disps, bar_width, color=colors, linewidth=0.5)
    axs[0].errorbar(x, disps, yerr=[errs_minus, errs_plus], fmt='none', ecolor='0.6', capsize=3,
                    linewidth=1, markeredgewidth=1)

    # add zero line
    axs[0].axhline(y=0, color="k", linewidth=0.5)

    # add value labels to bars
    label_offset = 3 * 0.05
    label_size = 6
    # add value labels to bars
    for bar in bars:
        bar_color = bar.get_facecolor()
        axs[0].text(x=(bar.get_x() + bar.get_width() * 0.6), y=bar.get_height() + label_offset,
                    s=round(bar.get_height(), 1), ha='left',
                    va='center', color=bar_color, fontsize=label_size, fontweight='bold')
    axs[0].set_ylim(0.0, 3.0)
    axs[0].set_ylabel("Displacement (m)", fontsize=8)
    axs[0].tick_params(axis='y', labelrotation=90, labelsize=6)
    axs[0].tick_params(axis='x', labelrotation=90, labelsize=6)
    axs[0].set_xticks(x, plot_order)

    #### Format map subplot
    coastline.plot(ax=axs[1], color="k", linewidth=0.5)
    plate_boundary.plot(ax=axs[1], color="0.75", linewidth=1.0)
    axs[1].set_xticks(np.arange(xmin_tick, xmax_tick, tick_separation))
    axs[1].xaxis.set_major_formatter(mticker.FormatStrFormatter('%.f mE'))
    axs[1].set_yticks(np.arange(ymin_tick, ymax_tick, tick_separation))
    axs[1].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.f mN'))
    plt.setp(axs[1].get_yticklabels(), rotation=90, ha="center", rotation_mode="anchor")
    axs[1].tick_params(axis="both", which='major', labelsize=6)
    axs[1].set_xlim(plot_xmin, plot_xmax)
    axs[1].set_ylim(plot_ymin, plot_ymax)
    axs[1].set_aspect("equal")

    # add site points
    axs[1].scatter(x_data, y_data, s=20, c=colors, edgecolors='black', linewidth=0.5, zorder=2)

    # set titles and stuff
    probability_string = str(int(100 * probability))
    fig.suptitle(f"Displacement at {probability_string}%\n{model_version_title} {exceed_type} (100 yrs)")
    fig.tight_layout()

    # make directory for hazard map if it doesn't exist
    if not os.path.exists(f"../{out_directory}/weighted_mean_figures"):
        os.mkdir(f"../{out_directory}/weighted_mean_figures")

    for type in file_type_list:
        fig.savefig(f"../{out_directory}/weighted_mean_figures/disp{probability_string}_plot_hazard_map_{file_name}."
                    f"{type}", dpi=300)



# make_mean_10_2_disp_bar_chart(slip_taper=slip_taper, model_version=model_version,
#                               model_version_results_directory=model_version_results_directory,
#                               file_type_list=file_type_list,
#                               outfile_extension=outfile_extension)
#
# make_site_prob_plot(slip_taper, fault_type, model_version_results_directory, model_version=model_version,
#                     outfile_extension=outfile_extension, file_type_list=file_type_list, threshold=0.2,
#                     optional_extension="_3oct")