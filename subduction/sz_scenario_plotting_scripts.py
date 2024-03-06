
import geopandas as gpd
import os
import shutil
import numpy as np
import pickle as pkl
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle
from sz_helper_scripts import read_rupture_csv, read_average_slip, make_total_slip_dictionary, get_figure_bounds, get_rupture_disp_dict


# not using right now, will have to adjust if I want to include it.
def plot_vert_difference(NSHM_directory, rupture_slip_dict, target_rupture_ids, patch_polygons, gf_dict_pkl_coast1,
                         gf_dict_pkl_coast2,
                         out_directory="v_diff", out_folder="coastal_def_diff"):
    """calculates the vertical difference between two greens functions for specific ruptures.
    use for comparing the same rupture scanrio with different geometry or other parameter"""

    # Read in rupture data and map elements
    all_ruptures = read_rupture_csv(f"{NSHM_directory}/ruptures/indices.csv")
    rupture_slip_dict = read_average_slip(f"{NSHM_directory}/ruptures/average_slips.csv")
    coastline = gpd.read_file("../data/coastline/coastline_jde1.geojson")
    patch_polygons_gdf = gpd.read_file(patch_polygons)

    # Make outfile directory, or delete existing one with same name
    if os.path.exists(f"figures/{out_directory}"):
        shutil.rmtree(f"figures/{out_directory}")
    os.mkdir(f"figures/{out_directory}")

    # make subfolder for outfiles
    if os.path.exists(f"figures/{out_directory}/{out_folder}"):
        shutil.rmtree(f"figures/{out_directory}/{out_folder}")
    os.mkdir(f"figures/{out_directory}/{out_folder}")

    # Make slip dictionaries for the two coast greens functions (e.g., steeperdip and gentler dip)
    gf_total_slip_dict1 = make_total_slip_dictionary(gf_dict_pkl_coast1)
    gf_total_slip_dict2 = make_total_slip_dictionary(gf_dict_pkl_coast2)

    # Set plot bounds
    x, y, buffer = 1749150, 5428092, 7.e4
    plot_xmin, plot_ymin, plot_xmax, plot_ymax = x - buffer, y - buffer, x + buffer, y + buffer

    # load coastaline xy data
    x_data = np.load(f"out_files/{extension1}{extension2}/xpoints_coast.npy")
    y_data = np.load(f"out_files/{extension1}{extension2}/ypoints_coast.npy")

    # Make plot
    for rupture_id in target_rupture_ids:
        plt.close("all")
        ruptured_patches = all_ruptures[rupture_id]
        # sum greens function for all patches to get scenario gf
        gfs1_i = np.sum([gf_total_slip_dict1[j] for j in ruptured_patches], axis=0)
        gfs2_i = np.sum([gf_total_slip_dict2[j] for j in ruptured_patches], axis=0)
        # calculate disps by multiplying scenario avg slip by scenario greens function [X disp, Y disp, VS]
        disps_scenario1 = rupture_slip_dict[rupture_id] * gfs1_i
        disps_scenario2 = rupture_slip_dict[rupture_id] * gfs2_i
        # storing zeros is more efficient than nearly zeros. Makes v small disps = 0
        disps_scenario1[np.abs(disps_scenario1) < 5.e-3] = 0.
        disps_scenario2[np.abs(disps_scenario2) < 5.e-3] = 0.
        # calculate difference between two variations
        disps_scenario_diff = disps_scenario2 - disps_scenario1

        max_vdisp = np.max(np.abs(disps_scenario_diff[:, -1]))
        if max_vdisp < 0.1:
            max_vdisp_cbar = 0.1
        else: max_vdisp_cbar = max_vdisp

        fig, axs = plt.subplots(1, 2, figsize=(6.5, 5))

        # plot ruptured patches in grey
        patch_polygons_gdf[patch_polygons_gdf.index.isin(ruptured_patches)].plot(ax=axs[0], color="0.5")

        # plot vertical diff (last column in disps array)
        disps = axs[1].scatter(x_data, y_data, s=4, c=disps_scenario_diff[:, -1],
                              cmap="seismic", vmin=-max_vdisp_cbar, vmax=max_vdisp_cbar)

        for ax in axs:
            coastline.plot(ax=ax, color="k", linewidth=0.5)
            ax.set_xticks(np.arange(1600000., 1900000., 50000.))
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.f mE'))
            ax.set_yticks(np.arange(5300000., 5600000., 50000.))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.f mN'))
            plt.setp(ax.get_yticklabels(), rotation=90, ha="center", rotation_mode="anchor")
            ax.tick_params(axis="both", which='major', labelsize=6)
            ax.set_xlim(plot_xmin, plot_xmax)
            ax.set_ylim(plot_ymin, plot_ymax)
            ax.set_aspect("equal")

        divider = make_axes_locatable(axs[0])
        cax1 = divider.append_axes('left', size='5%', pad=0.05)
        cax1.set_visible(False)

        divider2 = make_axes_locatable(axs[1])
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)
        cbar2 = fig.colorbar(disps, cax=cax2, orientation='vertical')
        cbar2.set_label("Vertical deformation difference (m)", fontsize=8)
        cbar2.ax.tick_params(labelsize=6)

        fig.suptitle(f"Rupture {rupture_id}{extension2}")
        fig.tight_layout()
        fig.savefig(f"figures/{out_directory}/{out_folder}/coast_rupture_{rupture_id}.png", dpi=300)
        print(f"Coastal def difference {rupture_id}")


def vertical_disp_figure(all_ruptures_disp_dict, NSHM_directory, target_rupture_ids, extension1, extension2,
                         grid, extent=""):
    """" makes two-part figure with ruptures patches on left and vertical deformation on right
    input:      dictionary of displacements (at all green's function sites), indexed/keyed by rupture id
                NSHM solutions directory name (str),
                list of rupture indicies (list)
                extension1 (str): trial name with site type, version, and NSHM directory suffix (e.g., sites_v1_Ez)
                extension2 (str): used to denote changes in geometry/dip, not using at the moment.
                slip_taper (bool): True if slip tapering is used, False if not
                grid (bool): True if using grid, False if not
                extent (str): feeds into "get figure bounds" script. "ruptured_rectangles", "North Island",
                "Wellington", or "" (for just discretized polygons extent).

    outputs:   saves figure to
                        out_files/figures/displacement_figure/rupture_{rupture_id}{extension3}.png
               saves discretized polygons with slip data to
                        out_files/{extension1}{extension2}/geojson/scenario_slip_{rupture_id}{extension3}.geojson

                """

    # set slip taper to always be false for SZ. Can change later if needed.
    slip_taper = False
    if slip_taper is True:
        extension3 = "_tapered"
    else:
        extension3 = "_uniform"

    # load data
    discretized_polygons_gdf = gpd.read_file(f"out_files/{extension1}{extension2}/sz_discretized_polygons"
                                             f"_{extension1}{extension2}.geojson")
    rectangle_outlines_gdf = gpd.read_file(f"out_files/{extension1}{extension2}/all_rectangle_outlines"
                                           f"_{extension1}{extension2}.geojson")
    # for some reason it defaults values to string. Convert to integer.
    discretized_polygons_gdf['fault_id'] = discretized_polygons_gdf['fault_id'].astype('int64')
    rectangle_outlines_gdf['fault_id'] = rectangle_outlines_gdf['fault_id'].astype('int64')

    coastline = gpd.read_file("../data/coastline/coastline_jde1.geojson")
    plate_boundary = gpd.read_file("../data/coastline/plate_boundary.geojson")

    # load in rupture indices to get fault names/IDs for each rupture ID
    all_ruptures = read_rupture_csv(f"{NSHM_directory}/ruptures/indices.csv")

    # make folder for outputs
    if not os.path.exists(f"out_files/{extension1}{extension2}/figures"):
        os.mkdir(f"out_files/{extension1}{extension2}/figures")
    if not os.path.exists(f"out_files/{extension1}{extension2}/figures/displacement_figure"):
        os.mkdir(f"out_files/{extension1}{extension2}/figures/displacement_figure")

    # get displacement figure bounds
    plot_b_xmin, plot_b_ymin, plot_b_xmax, plot_b_ymax, \
        xmin_b_tick, xmax_b_tick, ymin_b_tick, ymax_b_tick, tick_b_separation\
        = get_figure_bounds(extent="Wellington", polygon_gdf=discretized_polygons_gdf)

    ### make figure
    for rupture_id in target_rupture_ids:
        plt.close("all")
        ruptured_fault_ids = all_ruptures[rupture_id]
        # sum greens function for all patches to get scenario gf
        ruptured_discretized_polygons_gdf = discretized_polygons_gdf[
            discretized_polygons_gdf.fault_id.isin(ruptured_fault_ids)]
        ruptured_discretized_polygons_gdf = gpd.GeoDataFrame(ruptured_discretized_polygons_gdf, geometry='geometry')
        ruptured_rectangle_outlines_gdf = rectangle_outlines_gdf[
            rectangle_outlines_gdf.fault_id.isin(ruptured_fault_ids)]

        # set plot bounds and ticks for slip figure (part a)
        plot_a_xmin, plot_a_ymin, plot_a_xmax, plot_a_ymax, xmin_a_tick, xmax_a_tick, ymin_a_tick, ymax_a_tick, \
        tick_a_separation = get_figure_bounds(rectangle_outlines_gdf, extent="sz_margin")

        # extract needed data from displacement dictionary
        disps_scenario = all_ruptures_disp_dict[rupture_id]["v_disps_m"]
        patch_slips = all_ruptures_disp_dict[rupture_id]["polygon_slips_m"]
        plot_x_data = all_ruptures_disp_dict[rupture_id]["x_data"]
        plot_y_data = all_ruptures_disp_dict[rupture_id]["y_data"]

        max_vert_disp = np.max(np.abs(disps_scenario))
        ruptured_discretized_polygons_gdf["slip"] = patch_slips

        # this just prevents the slip-colored polgons from maxing out on the colorbar scale if slip is uniform
        if slip_taper is False:
            max_slip_color_val = np.max(patch_slips) / 0.76276
        else:
            max_slip_color_val = np.max(patch_slips)

        # format fig
        fig, axs = plt.subplots(1, 2, figsize=(6.5, 5))

        # ## plot ruptured patches
        ruptured_discretized_polygons_gdf.plot(ax=axs[0], column=ruptured_discretized_polygons_gdf["slip"],
                                               cmap="viridis", vmin=0, vmax = max_slip_color_val)
        ruptured_rectangle_outlines_gdf.boundary.plot(ax=axs[0], linewidth= 0.5, color="0.5")
        ruptured_discretized_polygons_gdf.boundary.plot(ax=axs[0], linewidth= 0.5, color="0.2")

        if grid is True:
            # plot displacement as a grid
            # get x and y dimensions
            length_unique_x = len(np.unique(plot_x_data))
            length_unique_y = len(np.unique(plot_y_data))
            # reshape list back into a grid (for plotting)
            disps_scenario = np.reshape(disps_scenario, (length_unique_y, length_unique_x))
            plot_x_data = np.reshape(plot_x_data, (length_unique_y, length_unique_x))
            plot_y_data = np.reshape(plot_y_data, (length_unique_y, length_unique_x))

            disp_map = axs[1].imshow(disps_scenario[-1::-1], cmap="seismic", vmin=-max_vert_disp, vmax=max_vert_disp,
                                  extent=[plot_x_data.min(), plot_x_data.max(), plot_y_data.min(), plot_y_data.max()])

            # add def contours, levels can be int or list/array
            ##############UPDATE THIS SECTION IN THE CRUSTAL SCRIPT
            neg_levels = np.arange(np.floor(disps_scenario.min()), -0.5, 0.5)
            pos_levels = np.arange(1, np.ceil(disps_scenario.max()), 1)
            neg_contour = axs[1].contour(plot_x_data, plot_y_data, disps_scenario, levels=neg_levels,
                                         colors="steelblue", linewidths=0.5)
            pos_contour = axs[1].contour(plot_x_data, plot_y_data, disps_scenario, levels=pos_levels,
                                         colors="maroon", linewidths=0.5)
            axs[1].contour(plot_x_data, plot_y_data, disps_scenario, levels=[0],
                           colors="0.5", linewidths=1)

        # plot point data using scatter
        else:
        # plot displacement hazard map (10% and 2% disp value at each point)
            if len(plot_x_data) < 20:  # if there are less than 20 points, plot with black edges
                disp_map = axs[1].scatter(plot_x_data, plot_y_data, s=15, c=disps_scenario, cmap="seismic",
                                        edgecolors='black', linewidth=0.5, zorder=3,
                                        vmin=-max_vert_disp, vmax=max_vert_disp)
            else:  # otherwise plot without black edges
                disp_map = axs[1].scatter(plot_x_data, plot_y_data, s=15, c=disps_scenario, cmap="seismic",
                                            edgecolors=None, linewidth=0.5, zorder=2,
                                            vmin=-max_vert_disp, vmax=max_vert_disp)

        # Format subplots
        for ax in axs:
            coastline.plot(ax=ax, color="k", linewidth=0.5)
            plate_boundary.plot(ax=ax, color="0.75", linewidth=1.5)

            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.f mE'))

            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.f mN'))
            plt.setp(ax.get_yticklabels(), rotation=90, ha="center", rotation_mode="anchor")
            ax.tick_params(axis="both", which='major', labelsize=6)
            ax.set_aspect("equal")

        # set left plot limits (plot a, slip plot)
        axs[0].set_xlim(plot_a_xmin, plot_a_xmax)
        axs[0].set_ylim(plot_a_ymin, plot_a_ymax)
        axs[0].set_xticks(np.arange(xmin_a_tick, xmax_a_tick, tick_a_separation))
        axs[0].set_yticks(np.arange(ymin_a_tick, ymax_a_tick, tick_a_separation))
        axs[0].set_aspect("equal")

        # add inside rectangle to plot a
        inset_width = plot_b_xmax - plot_b_xmin
        inset_height = plot_b_ymax - plot_b_ymin
        axs[0].add_patch(Rectangle((plot_b_xmin, plot_b_ymin), inset_width, inset_height, edgecolor='0.3',
                               fill=False, lw=1))

        #set right plot limits (plot b, displacement)
        axs[1].set_xlim(plot_b_xmin, plot_b_xmax)
        axs[1].set_ylim(plot_b_ymin, plot_b_ymax)
        axs[1].set_xticks(np.arange(xmin_b_tick, xmax_b_tick, tick_b_separation))
        axs[1].set_yticks(np.arange(ymin_b_tick, ymax_b_tick, tick_b_separation))
        axs[1].set_aspect("equal")


        # color bars, labels, and stuff
        divider = make_axes_locatable(axs[0])
        cax1 = divider.append_axes('top', size='6%', pad=0.05)
        cbar1 = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=max_slip_color_val)),
                                                   cax=cax1, orientation='horizontal')
        cbar1.set_label("slip (m)", fontsize=8)
        cbar1.ax.tick_params(labelsize=6)
        cbar1.ax.xaxis.set_ticks_position('top')
        cbar1.ax.xaxis.set_label_position('top')

        divider2 = make_axes_locatable(axs[1])
        cax2 = divider2.append_axes('top', size='5%', pad=0.05)
        cbar2 = fig.colorbar(disp_map, cax=cax2, orientation='horizontal')
        cbar2.set_label("Vertical deformation (m)", fontsize=8)
        cbar2.ax.tick_params(labelsize=6)
        cbar2.ax.xaxis.set_ticks_position('top')
        cbar2.ax.xaxis.set_label_position('top')

        fig.suptitle(f"Rupture {rupture_id}{extension2}{extension3}")
        fig.tight_layout()
        fig.savefig(f"out_files/{extension1}{extension2}/"
                    f"figures/displacement_figure/sz_rupture_{rupture_id}{extension3}.png", dpi=300)

        if not os.path.exists(f"out_files/{extension1}{extension2}/geojson"):
            os.mkdir(f"out_files/{extension1}{extension2}/geojson")

        ruptured_discretized_polygons_gdf.to_file(
            f"out_files/{extension1}{extension2}/geojson/sz_scenario_slip_{rupture_id}{extension3}.geojson",
            driver="GeoJSON")
        print(f"Figure {rupture_id}{extension3}")



# target_ruptures = [5403, 6483, 6513, 1638, 4475, 5105]
#target_ruptures = [956]
NSHM_directory="NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MTUy"
extension1 = "sites_v5_Uy"
extension2 = ""
grid = False
slip_taper = False

####
## CONSIDERATION: no current way to taper slip on the subduction zone that makes sense. probably fine, but ya know.
# it's written in there and it need to be modified or removed.
# run_coast_workflow(NSHM_directory="NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MTUy",
#                    extension1=extension1, extension2=extension2, grid=grid, slip_taper=slip_taper)
####

all_ruptures_disp_dict = get_rupture_disp_dict(NSHM_directory=NSHM_directory, extension1=extension1,
                                            extension2=extension2)

# with open(f"out_files/{extension1}{extension2}/all_sz_rupture_disps_{extension1}{extension2}_uniform.pkl",
#           "rb") as fid:
#     all_ruptures_disp_dict = pkl.load(fid)
#
# all_keys = list(all_ruptures_disp_dict.keys())
# target_rupture_ids = all_keys[0:10]
#
# vertical_disp_figure(NSHM_directory=NSHM_directory, all_ruptures_disp_dict=all_ruptures_disp_dict,
#                      target_rupture_ids=target_rupture_ids[0:10], extension1=extension1, extension2=extension2, grid=grid)
#
