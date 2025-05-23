import geopandas as gpd
import os
import pickle as pkl
import matplotlib.ticker as mticker
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
from helper_scripts import make_qualitative_colormap
from weighted_mean_plotting_scripts import get_mean_prob_barchart_data, get_mean_disp_barchart_data, \
    map_plot_mean_disp
#import seaborn as sns


probability_plot = True         # plots the probability of exceedance at the 0.2 m uplift and subsidence thresholds
displacement_chart = True       # plots the displacement at the 10% and 2% probability of exceedance thresholds
compare_hazcurves = False       # plots the different hazard curves on the same plot
make_map = True
disps_net = False
labels_on = True

main_plot_order = ["Paraparaumu", "Porirua CBD north", "South Coast", "Wellington Airport", "Wellington CBD", "Petone",
                   "Seaview", "Eastbourne", "Turakirae Head", "Lake Ferry", "Cape Palliser", "Flat Point"]
main_plot_labels = ["Paraparaumu", "Porirua\nCBD north", "South\nCoast", "Wellington\nAirport", "Wellington\nCBD",
                 "Petone", "Seaview", "Eastbourne", "Turakirae\nHead", "Lake\nFerry", "Cape\nPalliser", "Flat\nPoint"]

pared_plot_order = ["South Coast", "Porirua CBD north",  "Petone", "Eastbourne", "Turakirae Head", "Lake Ferry",
                    "Cape Palliser", "Flat Point"]
pared_plot_labels = ["South\nCoast", "Porirua\nCBD north", "Petone", "Eastbourne", "Turakirae\nHead", "Lake\nFerry",
                     "Cape\nPalliser", "Flat\nPoint"]

porirua_plot_order = ["Porirua CBD north","Porirua CBD south"]
porirua_plot_labels = ["Porirua\nCBD north", "Porirua\nCBD south"]

# names = ["CFM mean uniform", "CFM mean tapered"]
# mean_PPE_paths = ["../results_jde/crustal_CFM/weighted_mean_PPE_dict_testing_uniform.pkl",
#                     "../results_jde/crustal_CFM/weighted_mean_PPE_dict_testing_tapered.pkl"]
# plot_name = "CFM_mean_uniform_tapered"
# title = "Crustal slip distribution sensitivity test\n CFM uniform vs CFM tapered"

names = ["CFM mean uniform", "SZ uniform", "CFM SZ combined"]
mean_PPE_paths = ["../results_jde/crustal_CFM/weighted_mean_PPE_dict_testing_uniform.pkl",
                  "../results_jde/sz_v1/weighted_mean_PPE_dict_testing.pkl",
                  "../results_jde/paired_c_CFM_sz_v1/weighted_mean_PPE_dict_testing_uniform.pkl"]
plot_name = "CFM_SZ_combined_uniform"
title = "CFM vs SZ vs combined uniform"

# names = ["Alt. Fault 2 mean uniform", "Alt. Fault 2 mean tapered"]
# mean_PPE_paths = ["../results_jde/crustal_Model2/weighted_mean_PPE_dict_testing_uniform.pkl",
#                    "../results_jde/crustal_Model2/weighted_mean_PPE_dict_testing_tapered.pkl"]
# plot_name = "alt2_mean_uniform_tapered"
# title = "Alt. Fault 2 uniform vs tapered"

# names = ["CFM mean uniform", "Alt. Fault 1 mean uniform", "Alt. Fault 2 mean uniform"]
# mean_PPE_paths = ["../results_jde/crustal_CFM/weighted_mean_PPE_dict_testing_uniform.pkl",
#                    "../results_jde/crustal_Model1/weighted_mean_PPE_dict_testing_uniform.pkl",
#                   "../results_jde/crustal_Model2/weighted_mean_PPE_dict_testing_uniform.pkl"]
# plot_name = "CFM_alt1_alt2_uniform_porirua"
# title = "Crustal model uniform slip comparison "

# names = ["CFM uniform", "Alt. Fault 2 uniform"]
# mean_PPE_paths = ["../results_jde/crustal_CFM/weighted_mean_PPE_dict_testing_uniform.pkl",
#                   "../results_jde/crustal_Model2/weighted_mean_PPE_dict_testing_uniform.pkl"]
# plot_name = "CFM_alt2_uniform"
# title = "CFM vs Alt. Fault 2 uniform"

# names = ["CFM uniform"]
# mean_PPE_paths = ["../results_jde/crustal_CFM/weighted_mean_PPE_dict_testing_uniform.pkl"]
# plot_name = "CFM_uniform"
# title = "CFM uniform"
# outfile_directory = "results_jde/crustal_CFM/"

# names = ["CFM and SZ uniform"]
# mean_PPE_paths = ["../results_jde/paired_c_CFM_sz_v1/weighted_mean_PPE_dict_testing_uniform.pkl"]
# plot_name = "paired_CFM_SZ_uniform"
# title = "paired CFM SZ uniform"
# outfile_directory = "results_jde/paired_c_CFM_sz_v1/"

# names = ["Alt. Fault 2 and SZ uniform"]
# mean_PPE_paths = ["../results_jde/paired_c_Model2_sz_v1/weighted_mean_PPE_dict_testing_uniform.pkl"]
# plot_name = "paired_Model2_SZ_uniform"
# title = "paired Alt. Fualt 2 SZ uniform"
# outfile_directory = "results_jde/paired_c_Model2_sz_v1/"

# names = ["Alt. Fault 2"]
# mean_PPE_paths = ["../results_jde/crustal_Model2/weighted_mean_PPE_dict_testing_uniform.pkl"]
# plot_name = "Model2_"
# title = "Alt. Fault 2"

# names = ["CFM SZ combined"]
# mean_PPE_paths = ["../results_jde/paired_c_CFM_sz_v1/weighted_mean_PPE_dict_testing_uniform.pkl"]
# plot_name = "CFM_SZ_and_combined"
# title = "CFM SZ combined"

# names = ["SZ mid", "SZ steeper dip", "SZ gentler dip"]
# mean_PPE_paths = ["../results_jde/sz_v1/weighted_mean_PPE_dict_testing.pkl",
#                    "../results_jde/sz_v1_steeperdip/weighted_mean_PPE_dict_testing_uniform.pkl",
#                   "../results_jde/sz_v1_gentlerdip/weighted_mean_PPE_dict_testing_uniform.pkl"]
# plot_name = "SZ_mid_steeperdip_gentlerdip"
# title = "Subduction interface dip sensitivity test"

# names = ["Model2 uniform", "SZ", "Model1 SZ combined"]
# mean_PPE_paths = ["../results_jde/crustal_Model2/weighted_mean_PPE_dict_testing_uniform.pkl",
#                    "../results_jde/sz_v1/weighted_mean_PPE_dict_testing_uniform.pkl",
#                   "../results_jde/paired_c_Model2_sz_v1/weighted_mean_PPE_dict_testing_uniform.pkl"]
# plot_name = "Model2_SZ_and_combined"
# title = "Model2, SZ, and combined"

# plot_order = main_plot_order
# plot_labels = main_plot_labels
plot_order = main_plot_order
plot_labels = main_plot_labels
outfile_directory = "../results_jde/compare_fault_models"
file_type_list = ["png", "pdf"]
exceed_type = "down"
#######################
matplotlib.rcParams['pdf.fonttype'] = 42

def compare_faultmodel_prob_plot(PPE_paths, plot_name, title, names, outfile_directory, file_type_list=["png"],
                                      threshold=0.2):
    """ """

    exceed_type_list = ["up", "down"]

    PPE_dicts = []
    for i, PPE_path in enumerate(PPE_paths):
        with open(f"{PPE_path}",  "rb") as fid:
            PPE_dict = pkl.load(fid)
        PPE_dicts.append(PPE_dict)

    # set up custom color scheme
    # orange, teal, yellow
    #colors = [(235/255, 97/255, 35/255), (64/255, 176/255, 166/255),  (255/255, 190/255, 106/255)]
    #orange, teal, purple
    colors = [(235 / 255, 97 / 255, 35 / 255), (64 / 255, 176 / 255, 166 / 255), (116 / 255, 43 / 255, 140 / 255)]
    point_sizes = [35, 30, 30]
    point_shapes = ["^", "o", "s"]
    #point_shapes = ["o"]

    # set up figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.5))
    x = np.arange(len(plot_order))  # the site label locations

    for p, PPE_dict in enumerate(PPE_dicts):
        all_max_errs_y = []
        for i, exceed_type in enumerate(exceed_type_list):
            mean_prob, errs_plus, errs_minus = \
                get_mean_prob_barchart_data(site_PPE_dictionary=PPE_dict, exceed_type=exceed_type,
                                            threshold=threshold, site_list=plot_order)

            # add point and error bars to plot
            axs[i].errorbar(x, mean_prob, yerr=[errs_minus, errs_plus], fmt='none', ecolor=colors[p], capsize=4,
                            linewidth=1, markeredgewidth=1)
            axs[i].scatter(x, mean_prob, s=point_sizes[p], marker=point_shapes[p], color=colors[p], zorder=3,
                           edgecolors='k', linewidths=0.5)

            max_errs_y = max([mean_prob[j] + errs_plus[j] for j in range(len(mean_prob))])
            all_max_errs_y.append(max_errs_y)

            # add labels
            labels = [f"{int(100 * round(prob, 2))}%" for prob in mean_prob]
            label_y_vals = [prob + 0.03 for prob in mean_prob]
            for site, q in enumerate(x):
                axs[i].text(x=x[q], y=label_y_vals[q], s=labels[q],
                            horizontalalignment='center', fontsize=6, fontweight='bold')


            if max(all_max_errs_y) < 0.25:
                axs[i].set_ylim(0.0, 0.25)
            else:
                axs[i].set_ylim(0.0, max(all_max_errs_y))
            axs[i].tick_params(axis='x', labelrotation=90, labelsize=6)
            axs[i].tick_params(axis='y', labelsize=8)
            axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            # set tick labels to be every 0.2
            axs[i].yaxis.set_major_locator(mticker.MultipleLocator(0.1))

            axs[i].set_xticks(x, plot_labels, va='top', ha='center')

    # set indidual subplot stuff
    fontsize = 8
    # I'm doing it this way instead of just using "names" because I want to make sure the legend is in the correct
    # order.

    axs[0].legend(names, loc="upper left", title="Fault models", title_fontsize=fontsize, fontsize=fontsize)

    axs[0].set_ylabel("Probabilty", fontsize=8)
    axs[1].tick_params(axis='y', labelleft=False)

    axs[0].set_title(f"Probability of exceeding {threshold} m uplift", fontsize=fontsize)
    axs[1].set_title(f"Probability of exceeding {threshold} m subsidence", fontsize=fontsize)

    fig.suptitle(f"{title} (100 yrs)")
    fig.tight_layout()

    if not os.path.exists(outfile_directory):
        os.makedirs(outfile_directory)

    for file_type in file_type_list:
        fig.savefig(f"{outfile_directory}/compare_probs_{plot_name}.{file_type}", dpi=300)

def compare_disps_chart(PPE_paths, plot_name, title, names, outfile_directory, file_type_list=["png"],
                        disp_type_list=["up", "down"]):
    """ makes bar charts of the displacement value at the 10% and 2% probability of exceence thresholds for each site

    """

    probability_list = [0.1, 0.02]
    color_up, color_down = (189 / 255, 0, 0), (15 / 255, 72 / 255, 186 / 255)
    color_total_abs = (235 / 255, 97 / 255, 35 / 255)
    # orange, teal, yellow
    # errbar_colors = [(235 / 255, 97 / 255, 35 / 255), (64 / 255, 176 / 255, 166 / 255),
    #                  (255 / 255, 190 / 255, 106 / 255)]
    # orange, teal, purple
    errbar_colors = [(235 / 255, 97 / 255, 35 / 255), (64 / 255, 176 / 255, 166 / 255), (116 / 255, 43 / 255, 140 / 255)]

    alpha_list = [1, 0.5, 0.3]

    PPE_dicts = []
    for PPE_path in PPE_paths:
        with open(f"{PPE_path}", "rb") as fid:
            PPE_dict = pkl.load(fid)
        PPE_dicts.append(PPE_dict)

    plt.close("all")
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.4))
    x = np.arange(len(plot_order))  # the site label locations

    if len(PPE_dicts) == 1:
        bar_width = 0.5
    else:
        bar_width = 0.2


    max_min_y_vals = []
    max_min_errs_y_val = []
    for p, PPE_dict in enumerate(PPE_dicts):
        for i, probability in enumerate(probability_list):
            disps_up, errs_up_plus, errs_up_minus = \
                get_mean_disp_barchart_data(site_PPE_dictionary=PPE_dict, exceed_type="up",
                                        site_list=plot_order, probability=probability)
            disps_down, errs_down_plus, errs_down_minus = \
                get_mean_disp_barchart_data(site_PPE_dictionary=PPE_dict, exceed_type="down",
                                         site_list=plot_order, probability=probability)
            disps_total_abs, errs_total_abs_plus, errs_total_abs_minus = \
                get_mean_disp_barchart_data(site_PPE_dictionary=PPE_dict, exceed_type="total_abs",
                                            site_list=plot_order, probability=probability)

            max_min_y_vals.append(max(disps_up))
            max_min_y_vals.append(min(disps_down))
            errs_y_pos = max([disps_up[j] + errs_up_plus[j] for j in range(len(disps_up))])
            errs_y_neg = min([disps_down[j] - errs_down_minus[j] for j in range(len(disps_down))])
            max_min_errs_y_val.append(errs_y_pos)
            max_min_errs_y_val.append(errs_y_neg)


            label_size = 6
            #label_offset = label_size / 100

            # add bars to plot, add black horizontal line at zero.
            bars_up = axs[i].bar(x + bar_width * p, disps_up, bar_width, color=color_up, alpha=alpha_list[p],
                                 linewidth=0.5)
            bars_down = axs[i].bar(x + bar_width * p, disps_down, bar_width, color=color_down, alpha=alpha_list[p],
                                   linewidth=0.5)

            #add error bars
            axs[i].errorbar(x + bar_width * p, disps_up, yerr=[errs_up_minus, errs_up_plus], fmt='none',
                            ecolor=errbar_colors[p], capsize=2, linewidth=0.6, markeredgewidth=0.5)
            axs[i].errorbar(x + bar_width * p, disps_down, yerr=[errs_down_minus, errs_down_plus], fmt='none',
                            ecolor=errbar_colors[p], capsize=2, linewidth=0.6, markeredgewidth=0.5)

            # add zero line
            axs[i].axhline(y=0, color="k", linewidth=0.5)

            # add value labels to bars
            label_offset = 3 * 0.05
            if labels_on:
                for bar in bars_up:
                    bar_color = bar.get_facecolor()
                    axs[i].text(x=bar.get_x(), y=bar.get_height() + label_offset, s=round(bar.get_height(), 1), ha='left',
                                va='center', color=bar_color, fontsize=label_size, fontweight='bold')

                for bar in bars_down:
                    bar_color = bar.get_facecolor()
                    axs[i].text(x=bar.get_x(), y=bar.get_height() - label_offset, s=round(bar.get_height(), 1), ha='left',
                                va='center', color=bar_color, fontsize=label_size, fontweight='bold')


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
            axs[i].set_xticks(x, plot_labels)

    # set indidual subplot stuff

    axs[0].set_ylabel("Displacement (m)", fontsize=8)
    axs[1].tick_params(axis='y', labelleft=False)

    axs[0].set_title(f"10% probability of exceedance", fontsize=8)
    axs[1].set_title(f"2% probability of exceedance", fontsize=8)

    # manually make legend with rectangles and text
    swatch_width, swatch_height = bar_width, plot_ymax / 20
    swatch_minx, swatch_miny = -1 * (len(plot_order) / 30), plot_ymax - (swatch_height * 2)
    for q in range(len(PPE_dicts)):
        axs[0].add_patch(Rectangle((swatch_minx, swatch_miny - q * swatch_height * 2),
                                   swatch_width, swatch_height,
                                   facecolor=color_up, edgecolor=None, alpha=alpha_list[q]))
        axs[0].add_patch(Rectangle((swatch_minx + swatch_width, swatch_miny - q * swatch_height * 2),
                                   swatch_width, swatch_height,
                                   facecolor=color_down, edgecolor=None, alpha=alpha_list[q]))


        axs[0].text(swatch_minx + len(PPE_dicts) * swatch_width+ 0.5 * swatch_width, swatch_miny - q * swatch_height *
                    2,  f"{names[q]}", fontsize=8)
        #axs[0].text(swatch_minx + 2 * swatch_width, swatch_miny - 2 * swatch_height, f"{names[q]}", fontsize=8)

    fig.suptitle(f"{title} (100 years)", fontsize=10)
    fig.tight_layout()

    # make a directory for the figures if it doesn't already exist
    if not os.path.exists(f"{outfile_directory}"):
        os.makedirs(f"{outfile_directory}")

    for file_type in file_type_list:
        fig.savefig(f"{outfile_directory}/10_2_disps_{plot_name}.{file_type}", dpi=300)

def compare_disps_with_net(PPE_paths, plot_name, title, names, outfile_directory, file_type_list=["png"],):
    """ makes bar charts of the displacement value at the 10% and 2% probability of exceence thresholds for each site
    adds a lines that is the "net displacement" (i.e., the sum of the up and down displacements)

    """

    sites = ["South Coast", "Petone", "Eastbourne", "Turakirae Head", "Lake Ferry", "Cape Palliser"]
    site_dists = [15, 24.3, 32.5, 43.6, 54.9, 78.9]

    probability_list = [0.1, 0.02]
    color_up, color_down = (189 / 255, 0, 0), (15 / 255, 72 / 255, 186 / 255)
    # orange, teal, purple
    errbar_colors = [(235 / 255, 97 / 255, 35 / 255), (64 / 255, 176 / 255, 166 / 255), (116 / 255, 43 / 255, 140 / 255)]
    alpha_list = [1, 0.5, 0.3]

    PPE_dicts = []
    for PPE_path in PPE_paths:
        with open(f"{PPE_path}", "rb") as fid:
            PPE_dict = pkl.load(fid)
        PPE_dicts.append(PPE_dict)

    plt.close("all")
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.4))
    # x = np.arange(len(sites))  # the site label locations
    bar_width = 2
    x2 = [site_dist + bar_width for site_dist in site_dists]
    exes = [site_dists, x2]

    max_min_y_vals = []
    max_min_errs_y_val = []
    for p, PPE_dict in enumerate(PPE_dicts):
        for i, probability in enumerate(probability_list):
            disps_up, errs_up_plus, errs_up_minus = \
                get_mean_disp_barchart_data(site_PPE_dictionary=PPE_dict, exceed_type="up",
                                        site_list=sites, probability=probability)
            disps_down, errs_down_plus, errs_down_minus = \
                get_mean_disp_barchart_data(site_PPE_dictionary=PPE_dict, exceed_type="down",
                                         site_list=sites, probability=probability)
            disps_net = [disps_up[j] + disps_down[j] for j in range(len(disps_up))]

            max_min_y_vals.append(max(disps_up))
            max_min_y_vals.append(min(disps_down))
            errs_y_pos = max([disps_up[j] + errs_up_plus[j] for j in range(len(disps_up))])
            errs_y_neg = min([disps_down[j] - errs_down_minus[j] for j in range(len(disps_down))])
            max_min_errs_y_val.append(errs_y_pos)
            max_min_errs_y_val.append(errs_y_neg)

            label_size = 6
            # add bars to plot, add black horizontal line at zero.
            axs[i].bar(exes[p], disps_up, bar_width, color=color_up, alpha=alpha_list[p], linewidth=0.5)
            axs[i].bar(exes[p], disps_down, bar_width, color=color_down, alpha=alpha_list[p], linewidth=0.5)

            #add error bars
            axs[i].errorbar(exes[p], disps_up, yerr=[errs_up_minus, errs_up_plus], fmt='none',
                            ecolor=errbar_colors[p], capsize=2, linewidth=0.6, markeredgewidth=0.5)
            axs[i].errorbar(exes[p], disps_down, yerr=[errs_down_minus, errs_down_plus], fmt='none',
                            ecolor=errbar_colors[p], capsize=2, linewidth=0.6, markeredgewidth=0.5)

            # add net disp line
            line_exes = [site_dist + bar_width / 2 for site_dist in site_dists]
            axs[i].plot(line_exes, disps_net, color="k", linewidth=1, linestyle="dashed", alpha=alpha_list[p])
            # add zero line
            axs[i].axhline(y=0, color="k", linewidth=0.5)

            # add value labels to bars
            label_offset = 3 * 0.05

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
            axs[i].set_xticks(site_dists, sites)

    # set indidual subplot stuff
    axs[0].set_ylabel("Displacement (m)", fontsize=8)
    axs[1].tick_params(axis='y', labelleft=False)

    axs[0].set_title(f"10% probability of exceedance", fontsize=8)
    axs[1].set_title(f"2% probability of exceedance", fontsize=8)

    # manually make legend with rectangles and text
    swatch_width, swatch_height = bar_width, plot_ymax / 20
    swatch_minx, swatch_miny = -1 * (len(plot_order) / 30), plot_ymax - (swatch_height * 2)
    for q in range(len(PPE_dicts)):
        axs[0].add_patch(Rectangle((swatch_minx, swatch_miny - q * swatch_height * 2),
                                   swatch_width, swatch_height,
                                   facecolor=color_up, edgecolor=None, alpha=alpha_list[q]))
        axs[0].add_patch(Rectangle((swatch_minx + swatch_width, swatch_miny - q * swatch_height * 2),
                                   swatch_width, swatch_height,
                                   facecolor=color_down, edgecolor=None, alpha=alpha_list[q]))

        axs[0].text(swatch_minx + len(PPE_dicts) * swatch_width+ 0.5 * swatch_width, swatch_miny - q * swatch_height *
                    2,  f"{names[q]}", fontsize=8)

    fig.suptitle(f"{title} (100 years)", fontsize=10)
    fig.tight_layout()

    # make a directory for the figures if it doesn't already exist
    if not os.path.exists(f"{outfile_directory}"):
        os.makedirs(f"{outfile_directory}")

    for file_type in file_type_list:
        fig.savefig(f"{outfile_directory}/10_2_disps_net_{plot_name}.{file_type}", dpi=300)

def compare_mean_hazcurves(PPE_paths, plot_name, outfile_directory, title, names, exceed_type, file_type_list):

    PPE_dict_list = []
    for PPE_path in PPE_paths:
        PPE_dict = pkl.load(open(PPE_path, "rb"))
        PPE_dict_list.append(PPE_dict)

    plt.close("all")
    ncols = 4
    nrows = math.ceil(len(plot_order) / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.7*ncols, 2.5*nrows), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # orange, teal, yellow
    #colors = [(235/255, 97/255, 35/255), (64/255, 176/255, 166/255), (255/255, 190/255, 106/255)]
    #orange, teal, purple
    colors = [(235/255, 97/255, 35/255), (64/255, 176/255, 166/255), (116/255, 43/255, 140/255)]

    for p, PPE_dict in enumerate(PPE_dict_list):
        color = colors[p]

        # shade the region between the max and min value of all the curves at each site
        for i, site in enumerate(plot_order):
            # ax = plt.subplot(nrows, ncols, i + 1)
            ax = axs.flatten()[i]

            # plots all three types of exceedance (total_abs, up, down) on the same plot
            max_probs = PPE_dict[site][f"{exceed_type}_max_vals"]
            min_probs = PPE_dict[site][f"{exceed_type}_min_vals"]
            threshold_vals = PPE_dict[site]["threshold_vals"]

            threshold_vals = threshold_vals[1:]
            max_probs = max_probs[1:]
            min_probs = min_probs[1:]

            ax.fill_between(threshold_vals, max_probs, min_probs, color=color, alpha=0.2)

            # loop through sites and add the weighted mean lines
            weighted_mean_exceedance_probs = PPE_dict[site][f"weighted_exceedance_probs_{exceed_type}"]
            weighted_mean_exceedance_probs = weighted_mean_exceedance_probs[1:]

            if i == 0:
                ax.plot(threshold_vals, weighted_mean_exceedance_probs, color=color, linewidth=2, label=names[p])
            else:
                ax.plot(threshold_vals, weighted_mean_exceedance_probs, color=color, linewidth=2, label='_nolegend_')
            if i == 0 and p == (len(PPE_dict_list) - 1):
                ax.axhline(y=0.02, color="0.4", linestyle='dashed', label="2% probability")
                ax.axhline(y=0.1, color="0.4", linestyle='dotted', label="10% probability")
            if i!= 0 and p == 0:
                ax.axhline(y=0.02, color="0.4", linestyle='dashed', label='_nolegend_')
                ax.axhline(y=0.1, color="0.4", linestyle='dotted', label='_nolegend_')

            ax.set_title(site)
            ax.set_yscale('log'), ax.set_xscale('log')
            ax.set_ylim([0.000005, 1])
            # ax.get_xaxis().set_major_formatter(ScalarFormatter())
            # ax.ticklabel_format(axis='x', style='plain')
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.set_xlim([0.01, 3])
            ax.set_yticks([0.00001, 0.0001, 0.001, 0.01, 0.1, 1])
            ax.tick_params(axis='x', length=6, which='major')
            ax.tick_params(axis='x', length=4, which='minor')


    #axs.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig.text(0.5, 0, 'Vertical displacement threshold (m)', ha='center')
    fig.text(0, 0.5, 'Probability of exceedance in 100 years', va='center', rotation='vertical')
    fig.suptitle(f"{title}\n100-year hazard curve, {exceed_type} displacements", fontsize=10)

    # handles, labels = plt.gca().get_legend_handles_labels()
    #
    # # specify order of items in legend
    # order = [0, 2,3, 1]
    #
    # # add legend to plot
    # plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    fig.legend(loc='lower right', fontsize=6)
    plt.tight_layout()

    if not os.path.exists(f"{outfile_directory}"):
        os.makedirs(f"{outfile_directory}")

    for file_type in file_type_list:
        plt.savefig(f"{outfile_directory}/compare_hazcurves_{plot_name}.{file_type}", dpi=300)



# def compare_hazcurves():
#     pass

if probability_plot:
    compare_faultmodel_prob_plot(PPE_paths=mean_PPE_paths, plot_name=plot_name,
                                 outfile_directory=outfile_directory, title=title, names=names,
                                 file_type_list=file_type_list,
                                 threshold=0.2)

if displacement_chart:
    compare_disps_chart(PPE_paths=mean_PPE_paths, plot_name=plot_name, outfile_directory=outfile_directory,
                            title=title, names=names, file_type_list=file_type_list)

if compare_hazcurves:
    compare_mean_hazcurves(PPE_paths=mean_PPE_paths, plot_name=plot_name, outfile_directory=outfile_directory,
                           title=title, names=names, exceed_type=exceed_type, file_type_list=file_type_list)

if disps_net:
    compare_disps_with_net(PPE_paths=mean_PPE_paths, plot_name=plot_name, outfile_directory=outfile_directory,
                           title=title, names=names, file_type_list=file_type_list)
if make_map:
    PPE_dicts = []
    for PPE_path in mean_PPE_paths:
        with open(f"{PPE_path}", "rb") as fid:
            PPE_dict = pkl.load(fid)
        PPE_dicts.append(PPE_dict)
    for prob in [0.1, 0.02]:
        map_plot_mean_disp(weighted_mean_PPE_dictionary=PPE_dicts[0],
                      exceed_type="total_abs", model_version_title=title, out_directory=outfile_directory,
                      file_type_list=file_type_list, slip_taper=False, probability=prob, file_name=f"{plot_name}")