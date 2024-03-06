
import geopandas as gpd
import os
import pickle as pkl
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from sz_helper_scripts import get_figure_bounds, get_rupture_disp_dict


def get_site_disp_dict(extension1, extension2):
    """
        inputs: uses extension naming scheme to load displacement dictionary created with the
        get_rupture_disp_dict function. Slip taper is always set to False for the subduction ruptures

        functions: reshapes the dictionary to be organized by site name (key = site name).

        outputs: a dictionary where each key is a location/site name. contains displacements (length = number of
        rupture ids), annual rate (length = number of rupture ids), site name list (should be same length as green's
        function), and site coordinates (same length as site name list)

        CAVEATS/choices:
        - a little clunky because most of the dictionary columns are repeated across all keys.
        """

    slip_taper = False

    if slip_taper is True:
        extension3 = "_tapered"
    else:
        extension3 = "_uniform"

    # load saved displacement data
    # disp dictionary has keys for each rupture id and displacement data for each site ( disp length = # of sites)
    # for grids, the disp_dict is still in grid shape instead of list form.
    with open(f"out_files/{extension1}{extension2}/all_sz_rupture_disps_{extension1}{extension2}{extension3}.pkl",
              "rb") as fid:
        rupture_disp_dictionary = pkl.load(fid)

    ###### reshape displacement data to be grouped by site location.
    first_key = list(rupture_disp_dictionary.keys())[0]
    site_names = rupture_disp_dictionary[first_key]["site_name_list"]
    site_coords = rupture_disp_dictionary[first_key]["site_coords"]

    # if disps are in grid form, need to reshape them into a list to match the other types of data.

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

    with open(f"out_files/{extension1}{extension2}/sz_site_disp_dict_{extension1}{extension2}{extension3}.pkl",
              "wb") as f:
        pkl.dump(site_disp_dictionary, f)


def get_cumu_PPE(extension1, extension2, time_interval=100, n_samples=1000000):
    """
    Must first run get_site_disp_dict to get the dictionary of displacements and rates

    inputs: using extension naming scheme to load necessary data. Time_interval is in years

    function: calculates the poissonian probability of exceedance for each site for each displacement threshold value

    outputs: pickle file with probability dictionary (probs, disps, site_coords)

    CAVEATS/choices:
    - need to decide on number of 100-yr simulations to ru (n_samples = 1000000)
    """

    # use random number generator to initial monte carlo sampling
    rng = np.random.default_rng()

    # Load the displacement/rate data for all sites
    slip_taper = False
    if slip_taper is True:
        extension3 = "_tapered"
    else:
        extension3 = "_uniform"

    with open(f"out_files/{extension1}{extension2}/sz_site_disp_dict_{extension1}{extension2}{extension3}.pkl",
              'rb') as f:
        site_disp_dict = pkl.load(f)

    ## loop through each site and generate a bunch of 100 yr interval scenarios
    site_PPE_dict = {}
    for i, site_of_interest in enumerate(site_disp_dict.keys()):
        if i % 10 == 0:
            print(f"calculating PPE for site {i} of {len(site_disp_dict.keys())}")

        site_dict_i = site_disp_dict[site_of_interest]

        ## Set up params for sampling
        investigation_time = time_interval
        # average number of events per time interval (effectively R*T from Ned's guide)
        lambdas = investigation_time * np.array(site_dict_i["rates"])

        # Generate n_samples of possible earthquake ruptures for random 100 year intervals
        # returns boolean array where 0 means "no event" and 1 means "event". rows = 100 yr window, columns = earthquake
        # rupture
        scenarios = rng.poisson(lambdas, size=(n_samples, lambdas.size))

        # assigns a normal distribution with a mean of 1 and a standard deviation of 0.2
        # efectively a multiplier for the displacement value
        disp_uncertainty = rng.normal(1., 0.2, size=(n_samples, lambdas.size))

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
        # quite high for low displacements (25%). Means there's a ~25% change an earthquake will exceed 0 m in next 100
        # years across all earthquakes in the catalogue (at that site).
        exceedance_probs_total_abs = n_exceedances_total_abs / n_samples
        exceedance_probs_up = n_exceedances_up / n_samples
        exceedance_probs_down = n_exceedances_down / n_samples

        # CAVEAT: at the moment only positive thresholds are used, but for "down" the thresholds are actually negative.
        site_PPE_dict[site_of_interest] = {"thresholds": thresholds,
                                           "exceedance_probs_total_abs": exceedance_probs_total_abs,
                                           "exceedance_probs_up": exceedance_probs_up,
                                           "exceedance_probs_down": exceedance_probs_down,
                                           "site_coords": site_dict_i["site_coords"]}

    with open(f"out_files/{extension1}{extension2}/sz_cumu_exceed_prob_{extension1}{extension2}{extension3}.pkl",
              "wb") as f:
        pkl.dump(site_PPE_dict, f)


def get_exceedance_plot_data(site_PPE_dictionary, site_list=None, exceed_type="total_abs"):
    """returns displacements at the 10% and 2% probabilities of exceedance for each site

    define exceedance type. Options are "total_abs", "up", "down"
    """

    if site_list == None:
        site_list = list(site_PPE_dictionary.keys())

    # get disp at 10% exceedance
    disps_10 = []
    disps_2 = []
    for site in site_list:
        threshold_vals = site_PPE_dictionary[site]["thresholds"]

        # dispalcement threasholds are negative for "down" exceedances
        if exceed_type == "down":
         threshold_vals = -threshold_vals

        site_PPE = site_PPE_dictionary[site][f"exceedance_probs_{exceed_type}"]

        # get first index that is < 10% (ideally we would interpolate for exaxt value but don't have a function)
        exceedance_index_10 = next((index for index, value in enumerate(site_PPE) if value <= 0.10), -1)
        disp_10 = threshold_vals[exceedance_index_10]
        disps_10.append(disp_10)

        # get first index that is < 2%
        exceedance_index_2 = next((index for index, value in enumerate(site_PPE) if value <= 0.02), -1)
        disp_2 = threshold_vals[exceedance_index_2]
        disps_2.append(disp_2)

    return disps_10, disps_2

def get_probability_plot_data(site_PPE_dictionary, exceed_type, threshold, site_list=None):
    """ function that finds the probability at each site for the specified displacement threshold on the hazard curve
            Inputs:
                site_PPE_dictionary: dictionary of exceedance probabilities for each site (key = site)
                exceedance type: "total_abs", "up", or "down"
                site_list: list of sites to get data for. If None, will get data for all sites in site_PPE_dictionary.
                    I made this option so that you could skip the sites you didn't care about (e.g., use "plot_order")

            Outputs:
                probs_threshold: list of probabilities of exceeding the specified threshold (one per site)
                """

    if site_list == None:
        site_list = list(site_PPE_dictionary.keys())

    # get probability at defined displacement threshold
    probs_threshold = []

    for site in site_list:
        site_PPE = site_PPE_dictionary[site][f"exceedance_probs_{exceed_type}"]
        threshold_vals = list(site_PPE_dictionary[site]["thresholds"])

        # find index in threshold_vals where the value matches the parameter threshold
        index = threshold_vals.index(threshold)

        probs_threshold.append(site_PPE[index])

    return probs_threshold


def plot_cumu_disp_hazard_curve(extension1, extension2, exceed_type_list=["total_abs", "up", "down"]):
    """makes hazard curves for each site. includes the probability of cumulative displacement from multiple
    earthquakes exceeding a threshold in 100 years."""
    plot_order = ["Porirua CBD north", "South Coast", "Wellington Airport", "Wellington CBD", "Petone",
                  "Seaview", "Paraparaumu", "Eastbourne", "Turakirae Head"]

    slip_taper = False
    if slip_taper is True:
        extension3 = "_tapered"
    else:
        extension3 = "_uniform"

    with open(f"out_files/{extension1}{extension2}/sz_cumu_exceed_prob_{extension1}{extension2}{extension3}.pkl",
              "rb") as fid:
        PPE_dictionary = pkl.load(fid)

    # make directory for hazard curve if it doesn't exist
    if not os.path.exists(f"out_files/{extension1}{extension2}/figures"):
        os.mkdir(f"out_files/{extension1}{extension2}/figures")


    plt.close("all")
    fig, axs = plt.subplots(figsize=(15, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    fig.suptitle("Sites", fontsize=18, y=0.95)

    #loop through sites and make a subplot for each one
    for i, site in enumerate(plot_order):
        ax = plt.subplot(3, 3, i + 1)

        # plots all three types of exceedance (total_abs, up, down) on the same plot
        for j, exceed_type in enumerate(exceed_type_list):
            if exceed_type =="total_abs":
                curve_color = "k"
            elif exceed_type == "up":
                curve_color = "r"
            elif exceed_type == "down":
                curve_color = "b"
            else:
                curve_color = "g"

            exceedance_probs = PPE_dictionary[site][f"exceedance_probs_{exceed_type}"]
            threshold_vals = PPE_dictionary[site]["thresholds"]

            # if exceed_type == "down":
            #     threshold_vals = -threshold_vals

            ax.scatter(threshold_vals, exceedance_probs, color=curve_color)
            ax.axhline(y=0.02, color="0.7", linestyle='dashed')
            ax.axhline(y=0.1, color="0.7", linestyle='dotted')

            # show the figure
            #plt.show

        ax.set_title(site)
        ax.set_yscale('log')
        ax.ticklabel_format(axis='x', style='plain')
        ax.set_xscale('log')
        #ax.set_xscale('symlog')
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(axis='x', style='plain')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig.text(0.5, 0, 'Vertical displacement threshold (m)', ha='center')
    fig.text(0, 0.5, 'Probability of exceedance in 100 years', va='center', rotation='vertical')
    fig.suptitle(f"cumulative exceedance hazard curves \n subduction, {extension3}")
    plt.tight_layout()

    #save figure
    plt.savefig(f"out_files/{extension1}{extension2}/figures/sz_hazard_curve_{extension1}{extension2}{extension3}.png",
                dpi=300)


def plot_cumu_disp_hazard_map(extension1, extension2, grid, exceed_type_list):
    """ plots the displacement threshold value (m) for x% probability of exceedance in 100 years
    CAVEATS/choices:
    - currently set up with two subplots, a 10% and 2% probability of exceedance
    - exeed_type is a list of ["total_abs", "up", "down"]

    use grid=True if you want to plot the gridded data as an image rather than a set of scatter points"""

    slip_taper = False
    if slip_taper is True:
        extension3 = "_tapered"
    else:
        extension3 = "_uniform"

    # load data
    with open(f"out_files/{extension1}{extension2}/sz_cumu_exceed_prob_{extension1}{extension2}{extension3}.pkl",
              "rb") as fid:
        site_PPE_dictionary = pkl.load(fid)

    # make directory for hazard curve if it doesn't exist
    if not os.path.exists(f"out_files/{extension1}{extension2}/figures"):
        os.mkdir(f"out_files/{extension1}{extension2}/figures")

    site_coords = np.array([list(site_PPE_dictionary[key]["site_coords"]) for key in site_PPE_dictionary.keys()])
    x_data, y_data = site_coords[:, 0], site_coords[:, 1]

    coastline = gpd.read_file("../data/coastline/coastline_jde1.geojson")
    wellington_boundary = gpd.read_file("../data/wellington_region_boundary.geojson")
    plate_boundary = gpd.read_file("../data/coastline/plate_boundary.geojson")
    discretized_polygons_gdf = gpd.read_file(f"out_files/{extension1}{extension2}/sz_discretized_polygons"
                                             f"_{extension1}{extension2}.geojson")
    discretized_polygons_gdf['fault_id'] = discretized_polygons_gdf['fault_id'].astype('int64')


    # get plot bounds (set for Wellington Region at the moment)
    plot_xmin, plot_ymin, plot_xmax, plot_ymax, \
        xmin_tick, xmax_tick, ymin_tick, ymax_tick, tick_separation \
        = get_figure_bounds(extent="Wellington", polygon_gdf=discretized_polygons_gdf)

    for exceed_type in exceed_type_list:
        disps_10, disps_2 = get_exceedance_plot_data(site_PPE_dictionary, exceed_type=exceed_type)

        extension4 = "_" + exceed_type

        # make figure object with subplots
        plt.close("all")
        fig, axs = plt.subplots(1, 2, figsize=(6.5, 5))

        # Format subplots
        for ax in axs:
            #discretized_polygons_gdf.boundary.plot(ax=ax, linewidth=0.5, color="0.6")
            coastline.plot(ax=ax, color="k", linewidth=0.5)
            plate_boundary.plot(ax=ax, color="0.75", linewidth=1.0)
            ax.set_xticks(np.arange(xmin_tick, xmax_tick, tick_separation))
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.f mE'))
            ax.set_yticks(np.arange(ymin_tick, ymax_tick, tick_separation))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.f mN'))
            plt.setp(ax.get_yticklabels(), rotation=90, ha="center", rotation_mode="anchor")
            ax.tick_params(axis="both", which='major', labelsize=6)
            ax.set_xlim(plot_xmin, plot_xmax)
            ax.set_ylim(plot_ymin, plot_ymax)
            ax.set_aspect("equal")

        # find max displacement
        max_z = max(max(disps_10), max(disps_2))
        # reshape point data into a grid, plot grid point as a raster/image
        if grid is True:
            # get x and y dimensions
            length_unique_x = len(np.unique(x_data))
            length_unique_y = len(np.unique(y_data))
            #reshape list back into a grid (for plotting)
            disps_10_grid = np.reshape(disps_10, (length_unique_y, length_unique_x))
            disps_2_grid = np.reshape(disps_2, (length_unique_y, length_unique_x))
            # plot as an image
            disp_hazard_map_10 = axs[0].imshow(disps_10_grid[-1::-1], cmap="seismic", vmin=-max_z, vmax=max_z, zorder=1,
                                               extent=[x_data.min(), x_data.max(), y_data.min(), y_data.max()])
            disp_hazard_map_2 = axs[1].imshow(disps_2_grid[-1::-1], cmap="seismic", vmin=-max_z, vmax=max_z, zorder=1,
                                              extent=[x_data.min(), x_data.max(), y_data.min(), y_data.max()])

        # plot point data using scatter
        else:
            # plot displacement hazard map (10% and 2% disp value at each point)
            if len(x_data) < 20:   # if there are less than 20 points, plot with black edges
                disp_hazard_map_10 = axs[0].scatter(x_data, y_data, s=15, c=disps_10, cmap="seismic", edgecolors='black',
                                                    linewidth=0.5, zorder=2, vmin=-max_z, vmax=max_z)
                disp_hazard_map_2 = axs[1].scatter(x_data, y_data, s=15, c=disps_2, cmap="seismic", edgecolors='black', linewidth=0.5,
                                                   zorder=2, vmin=-max_z, vmax=max_z)
            else:   # otherwise plot without black edges
                disp_hazard_map_10 = axs[0].scatter(x_data, y_data, s=15, c=disps_10, cmap="seismic", edgecolors=None,
                                                    linewidth=0.5, zorder=2, vmin=-max_z, vmax=max_z)
                disp_hazard_map_2 = axs[1].scatter(x_data, y_data, s=15, c=disps_2, cmap="seismic", edgecolors=None,
                                                   linewidth=0.5, zorder=2, vmin=-max_z, vmax=max_z)

        # make colorbars and set ticks, etc.
        divider = make_axes_locatable(axs[0])
        cax1 = divider.append_axes('top', size='6%', pad=0.05)
        cbar1 = fig.colorbar(disp_hazard_map_10, cax=cax1, orientation='horizontal')
        cbar1.set_label("Vertical displacement (m); 10%", fontsize=8)
        cbar1.ax.tick_params(labelsize=6)
        cax1.xaxis.set_ticks_position("top")
        cbar1.ax.xaxis.set_label_position('top')

        divider2 = make_axes_locatable(axs[1])
        cax2 = divider2.append_axes('top', size='6%', pad=0.05)
        cbar2 = fig.colorbar(disp_hazard_map_2, cax=cax2, orientation='horizontal')
        cbar2.set_label("Vertical displacement (m); 2%", fontsize=8)
        cbar2.ax.tick_params(labelsize=6)
        cax2.xaxis.set_ticks_position("top")
        cbar2.ax.xaxis.set_label_position('top')

        fig.suptitle(f"Cumulative displacement hazard (100 yrs) \n {exceed_type}")
        fig.tight_layout()
        fig.savefig(f"out_files/{extension1}{extension2}/figures/hazard_map_"
                    f"{extension1}{extension2}{extension3}{extension4}.png", dpi=300)


# What is the displacement at 10% and 2% probability?
def make_site_PPE_bar_charts(extension1, extension2, exceed_type_list):
    """ makes bar charts of the displacement value at the 10% and 2% probability of exceence thresholds for each site

    """

    plot_order = ["Porirua CBD north", "South Coast", "Wellington Airport", "Wellington CBD", "Petone",
                  "Seaview", "Paraparaumu", "Eastbourne", "Turakirae Head"]

    slip_taper = False
    if slip_taper is True:
        extension3 = "_tapered"
    else:
        extension3 = "_uniform"

    with open(f"out_files/{extension1}{extension2}/sz_cumu_exceed_prob_{extension1}{extension2}{extension3}.pkl",
              "rb") as fid:
        site_PPE_dictionary = pkl.load(fid)

    for exceed_type in exceed_type_list:
        plt.close()
        disps_10, disps_2 = get_exceedance_plot_data(site_PPE_dictionary, site_list=plot_order, exceed_type=exceed_type)

        extension4 = "_" + exceed_type

        all_colors = plt.get_cmap('tab20b')(np.linspace(0, 1, 20))
        color_indices = list(range(0, 20, 2))
        colors = all_colors[color_indices[0:len(plot_order)]]

        fig, axs = plt.subplots(1, 2, figsize=(7, 4))
        bars1 = axs[0].bar(plot_order, disps_10, color=colors)
        bars2 = axs[1].bar(plot_order, disps_2, color=colors)
        subplots_list = [bars1, bars2]

        for i, subplot in enumerate(subplots_list):
            if exceed_type == "up" or exceed_type == "total_abs":
                max_y = max(max(disps_10), max(disps_2))
                axs[i].set_ylim(0, max_y + 0.2)

                #set exceedance label in the plot
                axs[i].text(0, max_y, exceed_type, fontsize=8, fontweight='bold')

                # label the bar heights
                for bar in subplot:
                    bar_color = bar.get_facecolor()
                    # add value label to each bar
                    axs[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, round(bar.get_height(), 1),
                                horizontalalignment='center', color=bar_color, fontsize=8, fontweight='bold')

            if exceed_type == "down":
                min_y = min(min(disps_10), min(disps_2))
                axs[i].set_ylim(min_y - 0.2, 0)

                # set exceedance label in the plot
                axs[i].text(0, min_y, exceed_type, fontsize=8, fontweight='bold')

                # label the bar heights
                for bar in subplot:
                    bar_color = bar.get_facecolor()
                    # add value label to each bar
                    axs[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, round(bar.get_height(), 1),
                                horizontalalignment='center', color=bar_color, fontsize=8, fontweight='bold')

            axs[i].tick_params(axis='x', labelrotation=90, labelsize=6)
            axs[i].tick_params(axis='y', labelsize=8)
            axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            # set tick labels to be every 0.5 m
            axs[i].yaxis.set_major_locator(mticker.MultipleLocator(0.5))

        # set indidual subplot stuff
        axs[0].set_ylabel("Displacement (m)", fontsize=8)
        axs[1].tick_params(axis='y', labelleft=False)

        axs[0].set_title("10% probability of exceedance")
        axs[1].set_title("2% probability of exceedance")

        fig.tight_layout()
        fig.savefig(f"out_files/{extension1}{extension2}/figures/bar_chart_{extension1}{extension2}"
                    f"{extension3}{extension4}.png",
                        dpi=300)

def make_combo_site_PPE_bar_charts(extension1, extension2):
    """ makes bar charts of the displacement value at the 10% and 2% probability of exceence thresholds for each site

    """
    plot_order = ["Porirua CBD north", "South Coast", "Wellington Airport", "Wellington CBD", "Petone",
                  "Seaview", "Paraparaumu", "Eastbourne", "Turakirae Head"]
    exceed_type_list = ["up", "down"]

    slip_taper = False
    if slip_taper is True:
        extension3 = "_tapered"
    else:
        extension3 = "_uniform"

    with open(f"out_files/{extension1}{extension2}/sz_cumu_exceed_prob_{extension1}{extension2}{extension3}.pkl",
              "rb") as fid:
        site_PPE_dictionary = pkl.load(fid)

    extension4 = "_combo"

    fig, axs = plt.subplots(1, 2, figsize=(7, 5))
    x = np.arange(len(plot_order))  # the site label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    for exceed_type in exceed_type_list:
        #plt.close()
        offset = width * multiplier
        disps_10, disps_2 = get_exceedance_plot_data(site_PPE_dictionary, site_list=plot_order, exceed_type=exceed_type)

        all_colors = plt.get_cmap('tab20b')(np.linspace(0, 1, 20))
        color_indices = list(range(0, 20, 2))
        colors = all_colors[color_indices[0:len(plot_order)]]

        #fig, axs = plt.subplots(1, 2, figsize=(7, 4))
        bars10 = axs[0].bar(x + offset, disps_10, width, label=exceed_type, color=colors)
        bars2 = axs[1].bar(x + offset, disps_2, width, color=colors)
        subplots_list = [bars10, bars2]

        # label bars
        # axs[0].bar_label(bars10, padding=3)
        # axs[1].bar_label(bars2, padding=3)

        multiplier += 1
        for i, subplot in enumerate(subplots_list):
            max_y = 1.5
            min_y = -0.8
            axs[i].axhline(y=0, color="k", linewidth=0.5)

            axs[i].set_ylim(min_y - 0.2, max_y + 0.2)

            #label the bar heights in the color of the bar
            if exceed_type == "up" or exceed_type == "total_abs":
                for bar in subplot:
                    bar_color = bar.get_facecolor()
                    # add value label to each bar
                    axs[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, round(bar.get_height(), 1),
                                horizontalalignment='center', color=bar_color, fontsize=8, fontweight='bold')

            if exceed_type == "down":
                for bar in subplot:
                    bar_color = bar.get_facecolor()
                    # add value label to each bar
                    axs[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.1, round(bar.get_height(), 1),
                                horizontalalignment='center', color=bar_color, fontsize=8, fontweight='bold')

            axs[i].tick_params(axis='x', labelrotation=90, labelsize=6)
            axs[i].tick_params(axis='y', labelsize=8)
            axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            # set tick labels to be every 0.5 m
            axs[i].yaxis.set_major_locator(mticker.MultipleLocator(0.5))
            axs[i].set_xticks(x + width, plot_order)


        # set indidual subplot stuff
        axs[0].set_ylabel("Displacement (m)", fontsize=8)
        axs[1].tick_params(axis='y', labelleft=False)

        axs[0].set_title("10% probability of exceedance")
        axs[1].set_title("2% probability of exceedance")

        fig.tight_layout()
        fig.savefig(f"out_files/{extension1}{extension2}/figures/sz_combo_bar_chart_{extension1}{extension2}"
                    f"{extension3}{extension4}.png",
                        dpi=300)

def make_up_down_PPE_bar_charts(extension1, extension2):
    """ makes bar charts of the displacement value at the 10% and 2% probability of exceence thresholds for each site

    """
    plot_order = ["Porirua CBD north", "South Coast", "Wellington Airport", "Wellington CBD", "Petone",
                  "Seaview", "Paraparaumu", "Eastbourne", "Turakirae Head"]
    exceed_type_list = ["up", "down"]

    slip_taper = False
    if slip_taper is True:
        extension3 = "_tapered"
    else:
        extension3 = "_uniform"

    with open(f"out_files/{extension1}{extension2}/sz_cumu_exceed_prob_{extension1}{extension2}{extension3}.pkl",
              "rb") as fid:
        site_PPE_dictionary = pkl.load(fid)

    fig, axs = plt.subplots(1, 2, figsize=(7, 5))
    x = np.arange(len(plot_order))  # the site label locations
    width = 0.4  # the width of the bars
    multiplier = 0

    for exceed_type in exceed_type_list:
        #plt.close()
        offset = width * multiplier
        disps_10, disps_2 = get_exceedance_plot_data(site_PPE_dictionary, site_list=plot_order, exceed_type=exceed_type)

        if exceed_type == "up":
            color = (189/255, 0, 0)
        elif exceed_type == "down":
            color = (15/255, 72/255, 186/255)

        # add bars to plot
        bars10 = axs[0].bar(x, disps_10, width, color=color)
        bars2 = axs[1].bar(x, disps_2, width, color=color)
        subplots_list = [bars10, bars2]

        multiplier += 1
        for i, subplot in enumerate(subplots_list):
            max_y = 0.5
            min_y = -1.2
            axs[i].axhline(y=0, color="k", linewidth=0.5)

            axs[i].set_ylim(min_y - 0.2, max_y + 0.2)

            #label the bar heights in the color of the bar
            if exceed_type == "up" or exceed_type == "total_abs":
                for bar in subplot:
                    bar_color = bar.get_facecolor()
                    # add value label to each bar
                    axs[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, round(bar.get_height(), 1),
                                horizontalalignment='center', color=bar_color, fontsize=7, fontweight='bold')

            if exceed_type == "down":
                for bar in subplot:
                    bar_color = bar.get_facecolor()
                    # add value label to each bar
                    axs[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.1, round(bar.get_height(), 1),
                                horizontalalignment='center', color=bar_color, fontsize=7, fontweight='bold')

            axs[i].tick_params(axis='x', labelrotation=90, labelsize=6)
            axs[i].tick_params(axis='y', labelsize=8)
            axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            # set tick labels to be every 0.5 m
            axs[i].yaxis.set_major_locator(mticker.MultipleLocator(0.5))
            axs[i].set_xticks(x + width, plot_order)


        # set indidual subplot stuff
        axs[0].set_ylabel("Displacement (m)", fontsize=8)
        axs[1].tick_params(axis='y', labelleft=False)

        axs[0].set_title("10% prob. of exceedance", fontsize=10)
        axs[1].set_title("2% prob. of exceedance", fontsize=10)

        fig.suptitle(f"Subduction zone (100 yrs)")
        fig.tight_layout()

        fig.savefig(f"out_files/{extension1}{extension2}/figures/sz_up_down_bar_chart_{extension1}{extension2}"
                    f"{extension3}.png",
                        dpi=300)


# What is the probability of exceeding 0.2 m subsidence, 0.2 m uplift at each site?
def make_prob_bar_chart(extension1, extension2, threshold=0.2):
    # What is the probability of exceeding 0.2 m subsidence, 0.2 m uplift at each site?
    """ determines the probability of exceeding a defined displacement threshold at each site and plots as a bar chart
        two-part plot, one for up and one for down. y axis is probability, x axis is site name
    """

    plot_order = ["Porirua CBD north", "South Coast", "Wellington Airport", "Wellington CBD", "Petone",
                  "Seaview", "Paraparaumu", "Eastbourne", "Turakirae Head"]
    exceed_type_list = ["up", "down"]

    slip_taper = False
    if slip_taper is True:
        extension3 = "_tapered"
    else:
        extension3 = "_uniform"

    with open(f"out_files/{extension1}{extension2}/sz_cumu_exceed_prob_{extension1}{extension2}{extension3}.pkl",
              "rb") as fid:
        site_PPE_dictionary = pkl.load(fid)

    all_colors = plt.get_cmap('tab20b')(np.linspace(0, 1, 20))
    color_indices = list(range(0, 20, 2))
    colors = all_colors[color_indices[0:len(plot_order)]]

    fig, axs = plt.subplots(1, 2, figsize=(7, 5))
    x = np.arange(len(plot_order))  # the site label locations
    width = 0.5  # the width of the bars

    for i, exceed_type in enumerate(exceed_type_list):
        # plt.close()
        probs_threshold_exceed_type = get_probability_plot_data(site_PPE_dictionary=site_PPE_dictionary,
                                                                exceed_type=exceed_type, threshold=threshold,
                                                                site_list=plot_order)

        # add bars to plot
        bars_10cm = axs[i].bar(x, probs_threshold_exceed_type, width, color=colors)

        axs[i].set_ylim(0.0, 1.0)

        for bar in bars_10cm:
            bar_color = bar.get_facecolor()
            # add value label to each bar
            axs[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, round(bar.get_height(), 1),
                        horizontalalignment='center', color=bar_color, fontsize=8, fontweight='bold')


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

    fig.suptitle(f"Subduction zone (100 yrs)")
    fig.tight_layout()

    fig.savefig(f"out_files/{extension1}{extension2}/figures/sz_probs_up_down_bar_chart_{extension1}{extension2}"
                f"{extension3}.png",
                dpi=300)


NSHM_directory="NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MTUy"
extension1 = "sites_v5_Uy"
extension2 = ""
grid = False
#exceed_type_list=["total_abs", "up", "down"]

#Start with rupture displacement dictionary
# print("Getting rupture displacement dictionary...")
# all_ruptures_disp_dict = get_rupture_disp_dict(NSHM_directory=NSHM_directory, extension1=extension1,
#                                             extension2=extension2)
# step 1: get site displacement dictionary
print("Calculating displacement probabilities...")
get_site_disp_dict(extension1, extension2)

# step 2: get exceedance probability dictionary
get_cumu_PPE(extension1, extension2, time_interval=100, n_samples=1000000)

# step 3 (optional): plot hazard curves. Only do it for "site" variations
#plot_cumu_disp_hazard_curve(extension1=extension1, extension2=extension2)

# step 4: plot hazard maps
# plot_cumu_disp_hazard_map(extension1=extension1, extension2=extension2,
#                           exceed_type_list=exceed_type_list, grid=False)

# step 5: plot bar charts
#make_site_PPE_bar_charts(extension1=extension1, extension2=extension2, exceed_type_list=exceed_type_list)

#make_combo_site_PPE_bar_charts(extension1=extension1, extension2=extension2)

#make_up_down_PPE_bar_charts(extension1=extension1, extension2=extension2)
#make_prob_bar_chart(extension1, extension2, threshold=0.2)