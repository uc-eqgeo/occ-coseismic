import os
import h5py as h5
import matplotlib.ticker as mticker
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle
from single_branch_plotting import get_mean_prob_plot_data, get_mean_disp_barchart_data, get_sigma_lims

def compare_faultmodel_prob_plot(PPE_paths, plot_name, title, pretty_names, outfile_directory, plot_order,
                                 labels_on=True, file_type_list=["png"], threshold=0.2, transect=False, site_names=None, sigma='minmax'):
    """ """

    exceed_type_list = ["up", "down"]

    if site_names is None:
        site_names = plot_order[:, 0]

    PPE_dicts = []
    for PPE_path in PPE_paths:
        PPE_dicts.append(h5.File(PPE_path, "r"))

    # set up custom color scheme
    #orange, teal, purple
    colors = [(235 / 255, 97 / 255, 35 / 255), (64 / 255, 176 / 255, 166 / 255), (116 / 255, 43 / 255, 140 / 255), 'cyan', 'forestgreen', 'fuchsia']
    point_sizes = [35, 30, 30]
    point_shapes = ["^", "o", "s"] * np.ceil(len(PPE_paths) / 3).astype(int)
    colors = colors * np.ceil(len(PPE_paths) / len(colors)).astype(int)
    point_sizes = [col for col in point_sizes for _ in np.arange(np.ceil(len(PPE_paths) / len(point_sizes)).astype(int))]
    point_shapes = [col for col in point_shapes for _ in np.arange(np.ceil(len(PPE_paths) / len(point_shapes)).astype(int))]

    # set up figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.5))
    x = plot_order[:, 1].astype(float)  # the site label locations

    # find the maximum y value for error bars so that the plot can be scaled correctly
    all_max_errs_y = []
    for p, PPE_dict in enumerate(PPE_dicts):
        for i, exceed_type in enumerate(exceed_type_list):
            mean_probs, errs_plus, errs_minus = \
                get_mean_prob_plot_data(site_PPE_dictionary=PPE_dict, exceed_type=exceed_type,
                                            threshold=threshold, site_list=plot_order[:, 0], sig_lim=sigma)

            # add point and error bars to plot
            axs[i].errorbar(x, mean_probs, yerr=[errs_minus, errs_plus], fmt='none', ecolor=colors[p], capsize=4,
                            linewidth=1, markeredgewidth=1)
            axs[i].scatter(x, mean_probs, s=point_sizes[p], marker=point_shapes[p], color=colors[p], zorder=3,
                           edgecolors='k', linewidths=0.5)

            max_errs_y = max([mean_probs[j] + errs_plus[j] for j in range(len(mean_probs))])
            all_max_errs_y.append(max_errs_y)

            # add labels
            labels = [f"{round(100 * prob)}%" for prob in mean_probs]
            label_y_vals = [0.01 + prob + err for prob, err in zip(mean_probs, errs_plus)]
            if labels_on:
                for site, q in enumerate(x):
                    axs[i].text(x=x[site], y=label_y_vals[site], s=labels[site],
                                horizontalalignment='center', fontsize=6, fontweight='bold')

            if max(all_max_errs_y) < 0.25:
                axs[i].set_ylim(0.0, 0.25)
            else:
                axs[i].set_ylim(0.0, np.ceil(max(all_max_errs_y) * 10) / 10)
            axs[i].tick_params(axis='x', labelrotation=90, labelsize=6)
            axs[i].tick_params(axis='y', labelsize=8)
            axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            axs[i].yaxis.set_major_locator(mticker.MultipleLocator(0.1))
            if not transect:
                axs[i].set_xticks(x, site_names, va='top', ha='center')

    # set indidual subplot stuff
    fontsize = 7
    # I'm doing it this way instead of just using "pretty_names" because I want to make sure the legend is in the correct
    # order.

    axs[0].legend(pretty_names, title_fontsize=fontsize, fontsize=fontsize)

    axs[0].set_ylabel("Probabilty", fontsize=8)
    axs[1].tick_params(axis='y', labelleft=False)

    axs[0].set_title(f"Probability of exceeding {threshold} m uplift", fontsize=fontsize)
    axs[1].set_title(f"Probability of exceeding {threshold} m subsidence", fontsize=fontsize)

    fig.suptitle(f"{title} (100 yrs)")
    fig.tight_layout()

    if not os.path.exists(f"../{outfile_directory}"):
        os.makedirs(f"../{outfile_directory}")

    for file_type in file_type_list:
        fig.savefig(f"../{outfile_directory}/compare_probs_{plot_name}.{file_type}", dpi=300)

def compare_disps_chart(PPE_paths, plot_name, title, pretty_names, outfile_directory, plot_order,
                        labels_on=True, file_type_list=["png"],
                        transect=False, site_names=None, plot_bars=True, sigma='minmax'):
    """ makes bar charts of the displacement value at the 10% and 2% probability of exceence thresholds for each site

    """

    if site_names is None:
        site_names = plot_order[:, 0]

    probability_list = [0.1, 0.02]
    color_up, color_down = (189 / 255, 0, 0), (15 / 255, 72 / 255, 186 / 255)
    # orange, teal, yellow
    # errbar_colors = [(235 / 255, 97 / 255, 35 / 255), (64 / 255, 176 / 255, 166 / 255),
    #                  (255 / 255, 190 / 255, 106 / 255)]
    # orange, teal, purple
    errbar_colors = [(235 / 255, 97 / 255, 35 / 255), (64 / 255, 176 / 255, 166 / 255), (116 / 255, 43 / 255, 140 / 255), 'cyan', 'forestgreen', 'fuchsia']
    point_sizes = [7, 6, 6] * np.ceil(len(PPE_paths) / 3).astype(int)
    point_shapes = ["^", "o", "s"] * np.ceil(len(PPE_paths) / 3).astype(int)
    errbar_colors = errbar_colors * np.ceil(len(PPE_paths) / len(errbar_colors)).astype(int)
    point_sizes = [col for col in point_sizes for _ in np.arange(np.ceil(len(PPE_paths) / len(point_sizes)).astype(int))]
    point_shapes = [col for col in point_shapes for _ in np.arange(np.ceil(len(PPE_paths) / len(point_shapes)).astype(int))]
    alpha_list = list(np.linspace(0.3, 1, len(PPE_paths)))[::-1]

    PPE_dicts = []
    for PPE_path in PPE_paths:
        PPE_dicts.append(h5.File(PPE_path, "r"))

    plt.close("all")
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.4))
    x = plot_order[:, 1].astype(float)  # the site label locations

    if len(PPE_dicts) == 1:
        bar_width = 0.5
    elif transect:
        bar_width = float(plot_order[-1, 1]) / plot_order.shape[0] / len(PPE_dicts)
    else:
        bar_width = 0.2

    legend_width = bar_width
    if not plot_bars:
        bar_width = 0

    max_min_y_vals = []
    max_min_errs_y_val = []
    for p, PPE_dict in enumerate(PPE_dicts):
        for i, probability in enumerate(probability_list):
            disps_up, errs_up_plus, errs_up_minus = \
                get_mean_disp_barchart_data(site_PPE_dictionary=PPE_dict, exceed_type="up",
                                        site_list=plot_order[:, 0], probability=probability, sig_lim=sigma)
            disps_down, errs_down_plus, errs_down_minus = \
                get_mean_disp_barchart_data(site_PPE_dictionary=PPE_dict, exceed_type="down",
                                         site_list=plot_order[:, 0], probability=probability, sig_lim=sigma)

            max_min_y_vals.append(max(disps_up))
            max_min_y_vals.append(min(disps_down))
            errs_y_pos = max([disps_up[j] + errs_up_plus[j] for j in range(len(disps_up))])
            errs_y_neg = min([disps_down[j] - errs_down_minus[j] for j in range(len(disps_down))])
            max_min_errs_y_val.append(errs_y_pos)
            max_min_errs_y_val.append(errs_y_neg)


            label_size = 6
            # add bars to plot, add black horizontal line at zero.
            if plot_bars:
                bars_up = axs[i].bar(x + bar_width * p, disps_up, bar_width, color=color_up, alpha=alpha_list[p],
                                    linewidth=0.5)
                bars_down = axs[i].bar(x + bar_width * p, disps_down, bar_width, color=color_down, alpha=alpha_list[p],
                                    linewidth=0.5)

            #add error bars
            axs[i].errorbar(x + bar_width * p, disps_up, yerr=[errs_up_minus, errs_up_plus], fmt='none',
                            ecolor=errbar_colors[p], capsize=2, linewidth=0.6, markeredgewidth=0.5)
            axs[i].errorbar(x + bar_width * p, disps_down, yerr=[errs_down_minus, errs_down_plus], fmt='none',
                            ecolor=errbar_colors[p], capsize=2, linewidth=0.6, markeredgewidth=0.5)
            axs[i].scatter(x + bar_width * p, disps_up, s=point_sizes[p], marker=point_shapes[p], color=errbar_colors[p], zorder=3,
                edgecolors='k', linewidths=0.5)
            axs[i].scatter(x + bar_width * p, disps_down, s=point_sizes[p], marker=point_shapes[p], color=errbar_colors[p], zorder=3,
                edgecolors='k', linewidths=0.5)

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
            if not transect:
                axs[i].set_xticks(x, site_names)

    # set indidual subplot stuff

    axs[0].set_ylabel("Minimum dsisplacement (m)", fontsize=8)
    axs[1].tick_params(axis='y', labelleft=False)

    axs[0].set_title(f"10% probability of exceedance", fontsize=8)
    axs[1].set_title(f"2% probability of exceedance", fontsize=8)

    # manually make legend with rectangles and text
    swatch_width, swatch_height = legend_width, plot_ymax / 20
    swatch_minx, swatch_miny = -1 * (len(plot_order[:, 0]) / 30), plot_ymax - (swatch_height * 2)
    if transect:
        swatch_width = float(plot_order[-1, 1]) / 50
    for q in range(len(PPE_dicts)):
        if plot_bars:
            axs[0].add_patch(Rectangle((swatch_minx - swatch_width, swatch_miny - q * swatch_height * 2),
                                    swatch_width, swatch_height,
                                    facecolor=errbar_colors[q], edgecolor=None, alpha=alpha_list[0]))
            axs[0].add_patch(Rectangle((swatch_minx, swatch_miny - q * swatch_height * 2),
                                    swatch_width, swatch_height,
                                    facecolor=color_up, edgecolor=None, alpha=alpha_list[q]))
            axs[0].add_patch(Rectangle((swatch_minx + swatch_width, swatch_miny - q * swatch_height * 2),
                                    swatch_width, swatch_height,
                                    facecolor=color_down, edgecolor=None, alpha=alpha_list[q]))
            axs[0].text(swatch_minx + len(PPE_dicts) * swatch_width + 0.5 * swatch_width, swatch_miny - q * swatch_height * 2,
                        f"{pretty_names[q]}", fontsize=8)
        else:
            axs[0].scatter(swatch_minx - swatch_width, swatch_miny - q * swatch_height * 2, s=point_sizes[q], marker=point_shapes[q],
                           color=errbar_colors[q], zorder=3, edgecolors='k', linewidths=0.5)
            axs[0].text(swatch_minx, swatch_miny - q * swatch_height * 2,
                        f"{pretty_names[q]}", fontsize=6, verticalalignment='center')

    fig.suptitle(f"{title} (100 years)", fontsize=10)
    fig.tight_layout()

    # make a directory for the figures if it doesn't already exist
    if not os.path.exists(f"../{outfile_directory}"):
        os.makedirs(f"../{outfile_directory}")

    for file_type in file_type_list:
        fig.savefig(f"../{outfile_directory}/10_2_disps_{plot_name}.{file_type}", dpi=300)

def compare_disps_with_net(PPE_paths, plot_name, title, pretty_names, outfile_directory, sites, site_dists = [],
                           file_type_list=["png"], transect=False, site_names=None):
    """ makes bar charts of the displacement value at the 10% and 2% probability of exceence thresholds for each site
    adds a lines that is the "net displacement" (i.e., the sum of the up and down displacements)

    """

    #sites = ["South Coast", "Petone", "Eastbourne", "Turakirae Head", "Lake Ferry", "Cape Palliser"]
    #site_dists = [15, 24.3, 32.5, 43.6, 54.9, 78.9]

    if site_names is None:
        site_names = sites[:, 0]

    if len(site_dists) == 0:
        # Calculates distance assuming all sites are in a straight line. May cause issues if sites are scattered, e.g. on coasts
        site_coords = np.array([(float(x), float(y)) for site in sites[:, 0] for x, y in [site.split('_')]])
        site_coords -= site_coords[0, :]
        site_dists = np.sqrt(np.sum(site_coords ** 2, axis=1))
        sites_ordered = np.argsort(site_dists)
        sites = sites[sites_ordered, :]
        site_dists = site_dists[sites_ordered]
        site_names = [site_names[ix] for ix in sites_ordered]


    probability_list = [0.1, 0.02]
    color_up, color_down = (189 / 255, 0, 0), (15 / 255, 72 / 255, 186 / 255)
    # orange, teal, purple
    errbar_colors = [(235 / 255, 97 / 255, 35 / 255), (64 / 255, 176 / 255, 166 / 255), (116 / 255, 43 / 255, 140 / 255), 'cyan', 'forestgreen', 'fuchsia']
    errbar_colors = errbar_colors * np.ceil(len(PPE_paths) / len(errbar_colors)).astype(int)
    alpha_list = list(np.linspace(0.3, 1, len(PPE_paths)))[::-1]

    PPE_dicts = []
    for PPE_path in PPE_paths:
        PPE_dicts.append(h5.File(PPE_path, "r"))

    plt.close("all")
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.4))
    # x = np.arange(len(sites))  # the site label locations
    bar_width = site_dists.max() / site_dists.shape[0] / len(PPE_dicts)
    exes = [site_dists]
    for ix, site_dist in enumerate(site_dists[:-1]):
        exes.append(site_dist + [site_dist + (ix + 1) * bar_width for site_dist in site_dists])

    max_min_y_vals = []
    max_min_errs_y_val = []
    for p, PPE_dict in enumerate(PPE_dicts):
        for i, probability in enumerate(probability_list):
            disps_up, errs_up_plus, errs_up_minus = \
                get_mean_disp_barchart_data(site_PPE_dictionary=PPE_dict, exceed_type="up",
                                        site_list=sites[:, 0], probability=probability)
            disps_down, errs_down_plus, errs_down_minus = \
                get_mean_disp_barchart_data(site_PPE_dictionary=PPE_dict, exceed_type="down",
                                         site_list=sites[:, 0], probability=probability)
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
            if not transect:
                axs[i].set_xticks(site_dists, site_names)

    # set indidual subplot stuff
    axs[0].set_ylabel("Minimum displacement (m)", fontsize=8)
    axs[1].tick_params(axis='y', labelleft=False)

    axs[0].set_title(f"10% probability of exceedance", fontsize=8)
    axs[1].set_title(f"2% probability of exceedance", fontsize=8)


    swatch_width, swatch_height = bar_width, plot_ymax / 20
    swatch_minx, swatch_miny = site_dists[0], plot_ymax - (swatch_height * 2)
    if transect:
        swatch_width = site_dists.max() / 50
    for q in range(len(PPE_dicts)):
        axs[0].add_patch(Rectangle((swatch_minx - swatch_width, swatch_miny - q * swatch_height * 2),
                                   swatch_width, swatch_height,
                                   facecolor=errbar_colors[q], edgecolor=None, alpha=alpha_list[0]))
        axs[0].add_patch(Rectangle((swatch_minx, swatch_miny - q * swatch_height * 2),
                                   swatch_width, swatch_height,
                                   facecolor=color_up, edgecolor=None, alpha=alpha_list[q]))
        axs[0].add_patch(Rectangle((swatch_minx + swatch_width, swatch_miny - q * swatch_height * 2),
                                   swatch_width, swatch_height,
                                   facecolor=color_down, edgecolor=None, alpha=alpha_list[q]))

        axs[0].text(swatch_minx + len(PPE_dicts) * swatch_width + 0.5 * swatch_width, swatch_miny - q * swatch_height *
                    2, f"{pretty_names[q]}", fontsize=8)

    fig.suptitle(f"{title} (100 years)", fontsize=10)
    fig.tight_layout()

    # make a directory for the figures if it doesn't already exist
    if not os.path.exists(f"../{outfile_directory}"):
        os.makedirs(f"../{outfile_directory}")

    for file_type in file_type_list:
        fig.savefig(f"../{outfile_directory}/10_2_disps_net_{plot_name}.{file_type}", dpi=300)

def compare_mean_hazcurves(PPE_paths, plot_name, outfile_directory, title, pretty_names, plot_order,
                           file_type_list, site_names=None, sigma='minmax'):

    if site_names is None:
        site_names = plot_order[:, 0]

    PPE_dicts = []
    for PPE_path in PPE_paths:
        PPE_dicts.append(h5.File(PPE_path, "r"))

    ncols = 4
    # orange, teal, yellow
    #colors = [(235/255, 97/255, 35/255), (64/255, 176/255, 166/255), (255/255, 190/255, 106/255)]
    #orange, teal, purple
    colors = [(235 / 255, 97 / 255, 35 / 255), (64 / 255, 176 / 255, 166 / 255), (116 / 255, 43 / 255, 140 / 255), 'cyan', 'forestgreen', 'fuchsia', 'black', 'yellow']
    colors = colors * np.ceil(len(PPE_paths) / len(colors)).astype(int)

    if not os.path.exists(f"../{outfile_directory}"):
        os.makedirs(f"../{outfile_directory}")

    for exceed_type in ['up', 'down']:
        plt.close("all")
        nrows = math.ceil(len(plot_order) / ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(2.7*ncols, 2.5*nrows), sharex=True, sharey=True)
        plt.subplots_adjust(hspace=0.3, wspace=0.3)

        for p, PPE_dict in enumerate(PPE_dicts):
            color = colors[p]

            # shade the region between the max and min value of all the curves at each site
            for i, site in enumerate(plot_order[:, 0]):
                # ax = plt.subplot(nrows, ncols, i + 1)
                ax = axs.flatten()[i]

                # plots all three types of exceedance (total_abs, up, down) on the same plot
                smin, smax, s50 = get_sigma_lims(PPE_dict[site]['sigma_lims'][:], sig_lim=sigma)
                max_probs = PPE_dict[site][f"error_{exceed_type}"][smax, :]
                min_probs = PPE_dict[site][f"error_{exceed_type}"][smin, :]
                probs_50 = PPE_dict[site][f"error_{exceed_type}"][s50, :]
                t_min, t_max, t_step = PPE_dict[site]['thresh_para'][:]
                threshold_vals = np.round(np.arange(t_min, t_max + t_step, t_step), 4)

                threshold_vals = threshold_vals[1:]
                max_probs_zeros = np.zeros_like(threshold_vals)
                min_probs_zeros = np.zeros_like(threshold_vals)
                probs_50_zeros = np.zeros_like(threshold_vals)
                max_probs_zeros[:len(max_probs) - 1] = max_probs[1:]
                min_probs_zeros[:len(min_probs) - 1] = min_probs[1:]
                probs_50_zeros[:len(probs_50) - 1] = probs_50[1:]

                ax.fill_between(threshold_vals, max_probs_zeros, min_probs_zeros, color=color, alpha=0.2)

                # loop through sites and add the weighted mean lines
                mean_exceedance_probs = np.zeros_like(threshold_vals)
                mean_exceedance_probs[:len(PPE_dict[site][f"exceedance_probs_{exceed_type}"]) - 1] = PPE_dict[site][f"exceedance_probs_{exceed_type}"][1:]

                if i == 0:
                    ax.plot(threshold_vals, mean_exceedance_probs, color=color, linewidth=2, label=pretty_names[p])
                    ax.plot(threshold_vals, probs_50_zeros, color=color, linewidth=2, linestyle='dashed', label='_nolegend_')
                else:
                    ax.plot(threshold_vals, mean_exceedance_probs, color=color, linewidth=2, label='_nolegend_')
                    ax.plot(threshold_vals, probs_50_zeros, color=color, linewidth=1, linestyle='dotted', label='_nolegend_')
                if i == 0 and p == (len(PPE_dicts) - 1):
                    ax.axhline(y=0.02, color="0.4", linestyle='dashed', label="2% probability")
                    ax.axhline(y=0.1, color="0.4", linestyle='dotted', label="10% probability")
                if i!= 0 and p == 0:
                    ax.axhline(y=0.02, color="0.4", linestyle='dashed', label='_nolegend_')
                    ax.axhline(y=0.1, color="0.4", linestyle='dotted', label='_nolegend_')

                ax.set_title(site_names[i])
                ax.set_yscale('log'), ax.set_xscale('log')
                ax.set_ylim([0.000005, 1])
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.set_xlim([0.01, 5])
                ax.set_yticks([0.00001, 0.0001, 0.001, 0.01, 0.1, 1])
                ax.tick_params(axis='x', length=6, which='major')
                ax.tick_params(axis='x', length=4, which='minor')

        #axs.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        fig.text(0.5, 0, 'Vertical displacement threshold (m)', ha='center')
        fig.text(0, 0.5, 'Probability of exceedance in 100 years', va='center', rotation='vertical')
        fig.suptitle(f"{title}\n100-year hazard curve, {exceed_type} displacements", fontsize=10)


        fig.legend(loc='lower right', fontsize=6)
        plt.tight_layout()

        for file_type in file_type_list:
            plt.savefig(f"../{outfile_directory}/compare_hazcurves_{exceed_type}_{plot_name}.{file_type}", dpi=300)


