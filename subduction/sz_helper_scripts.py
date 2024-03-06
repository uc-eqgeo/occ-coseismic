import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
from functools import reduce
import numpy as np
import pickle as pkl
from scipy.interpolate import griddata


def read_rupture_csv(csv_file: str):
    rupture_dict = {}
    with open(csv_file, "r") as fid:
        index_data = fid.readlines()
    for line in index_data[1:]:
        numbers = [int(num) for num in line.strip().split(",")]
        rupture_dict[numbers[0]] = np.array(numbers[2:])
    return rupture_dict


def read_average_slip(csv_file: str):
    df = pd.read_csv(csv_file)
    slip_dic = {}
    for i, row in df.iterrows():
        slip_dic[i] = row["Average Slip (m)"]
    return slip_dic


def make_total_slip_dictionary(gf_dict_pkl):
    """ calculates total greens function displacement using strike slip gf, dip slip gf, and rake value
    need to run the and crustal_discretized_gfs script first"""

    with open(gf_dict_pkl, "rb") as fid:
        gf_dict = pkl.load(fid)

    # Makes a new total gf displacement dictionary using rake
    gf_adjusted_dict = {}
    for i in gf_dict.keys():
        ss_gf = gf_dict[i]["ss"]
        ds_gf = gf_dict[i]["ds"]
        rake = gf_dict[i]["rake"]
        site_name_list = gf_dict[i]["site_name_list"]
        site_coords = gf_dict[i]["site_coords"]

        combined_gf = np.sin(np.radians(rake)) * ds_gf + np.cos(np.radians(rake)) * ss_gf
        gf_adjusted_dict[i] = {"combined_gf": combined_gf, "site_name_list": site_name_list, "site_coords": site_coords}

    return gf_adjusted_dict


def merge_rupture_attributes(directory, trimmed=True):
    """merge rupture csv files from NZSHM22 into single dataframe or csv
    default to trimmed = True, which trims out all rupture scenarios where Annual Rate = 0"""

    # load individual files
    avg_slip_df = pd.read_csv(f"../data/{directory}/ruptures/average_slips.csv")
    properties_df = pd.read_csv(f"../data/{directory}/ruptures/properties.csv")
    rates_df = pd.read_csv(f"../data/{directory}/solution/rates.csv")
    # only keep rupture scenarios with Annual Rates > 0
    trimmed_rates_df = rates_df[rates_df["Annual Rate"] > 0]

    # combine the data frames based on rupture index, using reduce and merge
    if trimmed == True:
        # make a list of data frames. Rates needs to be last based on merge function below
        dfs_list = [avg_slip_df, properties_df, trimmed_rates_df]
        # merge only works with 2 data frames, hence the use of reduce
        merged_df = reduce(lambda left, right: pd.merge(left, right, on='Rupture Index', how='right'), dfs_list)
    else:
        # use below if want to keep all rupture scenarios, even with rate = 0
        dfs_list = [avg_slip_df, properties_df, rates_df]
        merged_df = reduce(lambda left, right: pd.merge(left, right, on='Rupture Index', how='outer'), dfs_list)

    # save merged file to csv
    # merged_df.to_csv('merged_rupture_attributes.csv', sep=',')
    return merged_df


def filter_ruptures_by_rate(directory):
    """returns rupture indices that have annual rate >0"""

    # load individual files
    rates_df = pd.read_csv(f"../data/{directory}/solution/rates.csv")
    # only keep rupture scenarios with Annual Rates > 0
    trimmed_rates_df = rates_df[rates_df["Annual Rate"] > 0]

    filtered_ruptures = trimmed_rates_df.index.values.tolist()
    print(f"initial scenarios: {len(rates_df)}")
    print(f"rate filtered scenarios: {len(filtered_ruptures)}")
    return filtered_ruptures


# This runs a bit slowly
def filter_ruptures_by_location(NSHM_directory, target_rupture_ids, extension1, extension2, search_radius=2.5e5):
    """ filters the initial rupture scenarios by which patches are involved
        set a distance from interest area and cut out scenarios that don't intersect

        for now, rupture_df can be any list with the targeted rupture indices. For example, a list of ruptures
        that have been filtered by annual rate in the filter_ruptures_by_rate script above
        """

    # input: interest location
    Wellington = Point(1749150, 5428092)

    fault_rectangle_centroids_gdf = gpd.read_file(f"out_files/{extension1}{extension2}/all_rectangle_centroids"
                                                  f"_{extension1}{extension2}.geojson")
    all_ruptures_patch_indices = read_rupture_csv(f"{NSHM_directory}/ruptures/indices.csv")
    # find rupture scenarios that match input target ruptures (e.g., from filter by rate)
    trimmed_rupture_patch_indices = {i: all_ruptures_patch_indices[i] for i in all_ruptures_patch_indices.keys() if i in
                                     target_rupture_ids}

    # find faults patches that are within search radius
    filtered_fault_ids = []
    for i in range(len(fault_rectangle_centroids_gdf.centroid)):
        centroid = fault_rectangle_centroids_gdf.centroid[i]
        if centroid.distance(Wellington) < search_radius:
            filtered_fault_ids.append(fault_rectangle_centroids_gdf.index[i])   # FIX THIS LATER TO BE CONSISTENT
            # WITH CRUSTAL
            #filtered_fault_ids.append(int(fault_rectangle_centroids_gdf.fault_id[i]))

    # this can probably be simplified
    # include scenarios that have those patches
    filtered_scenarios = []
    for rupture_index in target_rupture_ids:
        # uses scenarios that include any patch within that search radius
        if any(fault_id in filtered_fault_ids for fault_id in trimmed_rupture_patch_indices[rupture_index]):
            filtered_scenarios.append(rupture_index)

    print(f"location filtered scenarios: {len(filtered_scenarios)}")
    return filtered_scenarios

def calculate_vertical_disps(rupture_id, ruptured_fault_ids, rupture_slip_dict, gf_total_slip_dict):
    """ calculates displacements for given rupture scenario at individual sites/coordinates based on green's function

    CAVETS/choices:
    - for SZ, no slip taper. Uniform slip.
    - this version changes all very small dispalcements to zero, or if no meshes are used, returns zero displacement
    """

    # find which patches have a mesh and which don't, to use greens functions later just with meshed patches
    ruptured_fault_ids_with_mesh = np.intersect1d(ruptured_fault_ids, list(gf_total_slip_dict.keys()))

    # calculate slip on each discretized polygon by multiplying scenario slip by scenario greens function
    # scenario gf sums displacements from all ruptured
    gfs_i = np.sum([gf_total_slip_dict[j]["combined_gf"] for j in ruptured_fault_ids_with_mesh], axis=0)
    disps_scenario = rupture_slip_dict[rupture_id] * gfs_i
    polygon_slips = rupture_slip_dict[rupture_id] * np.ones(len(ruptured_fault_ids_with_mesh))

    # storing zeros is more efficient than nearly zeros. Makes v small displacements = 0
    if len(ruptured_fault_ids_with_mesh) != 0:
        disps_scenario[np.abs(disps_scenario) < 5.e-3] = 0.
    elif len(ruptured_fault_ids_with_mesh) == 0:
        disps_scenario = None

    return disps_scenario, polygon_slips

def get_rupture_disp_dict(NSHM_directory, extension1, extension2):
    """
    inputs: uses extension naming scheme to load NSHM rate/slip data and fault geometry. Slip taper is always false,
    but keeping it in for extension naming scheme consistency.

    extracts site and rupture data, pares down the ruptures based on location (within a buffer) and annual rate >0,
    and passes that to the get_displacements function

    outputs: a dictionary where each key is the rupture id in a pickle file. contains displacements (length = same as
    greens function), annual rate (single value for the rupture), site name list (should be same length as green's
    function), and site coordinates (same length as site name list)

    CAVEATS:
    - current version omits scenarios from the output list if all locations have zero displacement
    - a little clunky because most of the dictionary columns are repeated across all keys.
    """

    # set slip taper to false for SZ ruptures. Can change this later if needed.
    slip_taper=False

    # load saved data
    print(f"loading data for {extension1}{extension2}")
    rupture_slip_dict = read_average_slip(f"{NSHM_directory}/ruptures/average_slips.csv")
    rates_df = pd.read_csv(f"../data/{NSHM_directory}/solution/rates.csv")
    discretized_polygons_gdf = gpd.read_file(f"out_files/{extension1}{extension2}/"
                                             f"sz_discretized_polygons_{extension1}{extension2}.geojson")
    gf_dict_pkl = f"out_files/{extension1}{extension2}/sz_gf_dict_{extension1}{extension2}.pkl"

    # this line takes ages, only do it once
    all_ruptures = read_rupture_csv(f"{NSHM_directory}/ruptures/indices.csv")  # change this later to clean up code
    rectangle_outlines_gdf = gpd.read_file(f"out_files/{extension1}{extension2}/all_rectangle_outlines"
                                           f"_{extension1}{extension2}.geojson")
    # for some reason it defaults values to string. Convert to integer.
    discretized_polygons_gdf['fault_id'] = discretized_polygons_gdf['fault_id'].astype('int64')
    rectangle_outlines_gdf['fault_id'] = rectangle_outlines_gdf['fault_id'].astype('int64')

    # filter ruptures by annual rate and location
    filtered_ruptures_annual_rate = filter_ruptures_by_rate(NSHM_directory)
    filtered_ruptures_location = filter_ruptures_by_location(NSHM_directory=NSHM_directory,
                                                             target_rupture_ids=filtered_ruptures_annual_rate,
                                                             extension1=extension1,
                                                             extension2=extension2)

    # make slip dictionary for all ruptures
    # Makes a new total gf displacement dictionary using rake. If points don't have a name (e.g., for whole coastline
    # calculations), the site name list is just a list of numbers
    gf_total_slip_dict = make_total_slip_dictionary(gf_dict_pkl)
    first_key = list(gf_total_slip_dict.keys())[0]
    site_name_list = gf_total_slip_dict[first_key]["site_name_list"]
    site_coords = gf_total_slip_dict[first_key]["site_coords"]


    # calculate displacements at all th sites by rupture. Output dictionary keys are by rupture ID.
    disp_dictionary = {}
    for rupture_id in filtered_ruptures_location:
        ruptured_fault_ids = all_ruptures[rupture_id]
        ruptured_discretized_polygons_gdf = discretized_polygons_gdf[
            discretized_polygons_gdf.fault_id.isin(ruptured_fault_ids)]
        ruptured_discretized_polygons_gdf = gpd.GeoDataFrame(ruptured_discretized_polygons_gdf, geometry='geometry')
        ruptured_rectangle_outlines_gdf = rectangle_outlines_gdf[
            rectangle_outlines_gdf.fault_id.isin(ruptured_fault_ids)]

        #calculate displacements
        print(f"calculating displacements for {rupture_id}")
        # output is a list of displacements for each site
        disps_scenario, polygon_slips = \
            calculate_vertical_disps(rupture_id=rupture_id, ruptured_fault_ids=ruptured_fault_ids,
                                     rupture_slip_dict=rupture_slip_dict, gf_total_slip_dict=gf_total_slip_dict)

        # extract annual rate and save data to dictionary. Key is the rupture ID. Ignores scenarios with zero
        # displacement at all sites. Sites can be a grid cell or a specific (named) site.
        if disps_scenario is not None:
            annual_rate = rates_df[rates_df.index == rupture_id]["Annual Rate"].values[0]
            # displacement dictionary for a single rupture scenario at all sites. Key is rupture id.
            rupture_disp_dict = {"rupture_id": rupture_id, "v_disps_m": disps_scenario, "annual_rate": annual_rate,
                                 "site_name_list": site_name_list, "site_coords": site_coords,
                                 "x_data": site_coords[:, 0], "y_data": site_coords[:, 1],
                                 "polygon_slips_m": polygon_slips}
            disp_dictionary[rupture_id] = rupture_disp_dict

    # print statement about how many scenarios have displacement > 0 at each site
    print(f"scenarios with displacement > 0: {len(disp_dictionary)}")

    if slip_taper is True:
        extension3 = "_tapered"
    else:
        extension3 = "_uniform"

    # save displacements
    with open(f"out_files/{extension1}{extension2}/all_sz_rupture_disps_{extension1}{extension2}{extension3}.pkl",
              "wb") as f:
        pkl.dump(disp_dictionary, f)

    return disp_dictionary



def get_rect_geojson(NSHM_directory, target_rupture_ids, extension1, extension2, slip_taper):
    """makes geojson of rectangles for specified rupture scenario indicies
    adds slip rate from sect_slip_rates.csv in NSHM solution folder

    directory: NSHM scenario folder name (str format)
    target_rupture_indices: list of rupture indices. Can generate using merge_rupture attributes (trims by rate>0)
        or filter_ruptures_by_location which cuts out scenarios outside a geographic buffer
    patch_polygons: .csv file name, (str)
    out_folder: str for output folder within figures directory. defaults to v_test_geojson"""

    # load patch ploygons and patch slip rates as geopandas dataframe
    all_rectangles_gdf = gpd.read_file(
        f"out_files/{extension1}{extension2}/all_rectangle_outlines_{extension1}{extension2}.geojson")
    sect_sliprates_df = pd.read_csv(f"../data/{NSHM_directory}/ruptures/sect_slip_rates.csv")

    # makes dictionary of rupture scenarios and corresponding ruptured patches
    all_ruptures = read_rupture_csv(f"{NSHM_directory}/ruptures/indices.csv")

    if slip_taper is True:
        extension3 = "_tapered"
    # # makes dictionary of patches used in each target rupture scenario
    # target_rupture_patch_dictionary = {i: all_ruptures[i] for i in all_ruptures.keys() if i in target_rupture_ids}

    # make directory for outputs
    if not os.path.exists(f"out_files/{extension1}{extension2}/geojson"):
        os.mkdir(f"out_files/{extension1}{extension2}/geojson")

    ## loop through target ruptures
    for rupture_id in target_rupture_ids:
        # get patch indices that ruptured in scenario
        ruptured_patches = all_ruptures[rupture_id]
        ruptured_patches.sort()

        # subset slip rates, patch polygons, and make dataframe/geodataframe
        ruptured_patch_sliprates = sect_sliprates_df[sect_sliprates_df.index.isin(ruptured_patches)]
        scenario_rectangles_gdf = all_rectangles_gdf[all_rectangles_gdf.index.isin(ruptured_patches)]
        df = pd.DataFrame({'section_index':ruptured_patches,
                           "slp_rt_m_yr": ruptured_patch_sliprates["Slip Rate (m/yr)"]})
        out_rectangles_gdf = gpd.GeoDataFrame(df, geometry=scenario_rectangles_gdf.geometry, crs=2193)

        # write patches to file
        out_rectangles_gdf.to_file\
            (f"out_files/{extension1}{extension2}/geojson/rectangles_{rupture_id}{extension3}.geojson",
                                driver="GeoJSON")

    print("Rupture patches written to geojson")



def save_target_rates(NSHM_directory, target_rupture_ids, extension1, extension2):
    """get the annual rates from NSHM solution for target ruptures, output a csv file

    NSHM_directory = name of NSHM folder
    target_rupture_ids = list of rupture ids/indices
    out_directory = directory for all the other outputfiles (figures, etc.) """

    # load annual rate file
    rates_df = pd.read_csv(f"../data/{NSHM_directory}/solution/rates.csv")
    # only keep ruptures and rates of interest
    trimmed_rates_df = rates_df[rates_df.index.isin(target_rupture_ids)]

    trimmed_rates_df.to_csv(f"out_files/{extension1}{extension2}/coastal_def_ruptures_rates.csv", sep=',')
    print("Filtered annual rates written to .csv")

def get_figure_bounds(polygon_gdf, extent=""):
    """sets figure bounds based on key words
    polygon_gdf: either discretized polygon gdf (for displacement figure) or ruptured rectangles gdf (slip figure)
    extent: can specify the extent of figure for interest area """

    if extent == "North Island":    # bounds of whole north island
        plot_xmin, plot_ymin, plot_xmax, plot_ymax = 1525000, 5247000, 2300000, 6176000
        xmin_tick, xmax_tick  = round(plot_xmin, -4), round(plot_xmax, -4)
        ymin_tick, ymax_tick = round(plot_ymin, -4), round(plot_ymax, -4)
        tick_separation = 100000.
    elif extent == "Wellington":    # bounds around wellington, with a 10 km buffer
        x, y, buffer = 1749150, 5428092, 10.e4
        plot_xmin, plot_ymin, plot_xmax, plot_ymax = x - buffer, y - buffer, x + buffer, y + buffer
        tick_separation = 100000.
        xmin_tick, xmax_tick = 1700000, 1800000 + tick_separation / 4
        ymin_tick, ymax_tick = 5380000, 5480000 + tick_separation / 4
    elif extent == "ruptured_rectangles":   # intended for ruptured rectangles gdf (slip plot)
        buffer = 20000
        plot_xmin = polygon_gdf.total_bounds[0] - buffer
        plot_ymin = polygon_gdf.total_bounds[1] - buffer
        plot_xmax = polygon_gdf.total_bounds[2] + buffer
        plot_ymax = polygon_gdf.total_bounds[3] + buffer
        xmin_tick, xmax_tick = round(plot_xmin, -5) - 100000, round(plot_xmax, -5) + 100000
        ymin_tick, ymax_tick = round(plot_ymin, -5) - 100000, round(plot_ymax, -5) + 100000
        tick_separation = round((plot_ymax - plot_ymin) / 4, -5)
    elif extent == "sz_margin":   # intended for entire subduction margin, stops at top of N. Island
        buffer = 20000
        plot_xmin = polygon_gdf.total_bounds[0] - buffer * 2
        plot_ymin = polygon_gdf.total_bounds[1] - buffer
        plot_xmax = 2280000
        plot_ymax = 6100000
        tick_separation = round((plot_ymax - plot_ymin) / 3, -5)
        xmin_tick, xmax_tick = round(plot_xmin, -5) + tick_separation / 2, round(plot_xmax, -5) + 10000
        ymin_tick, ymax_tick = round(plot_ymin, -5) + tick_separation / 2, round(plot_ymax, -5) + 10000
    elif extent == "discretized_polygons":   # intended for discretized polygons gdf (displacement plot)
        plot_xmin, plot_ymin, plot_xmax, plot_ymax = polygon_gdf.total_bounds
        xmin_tick, xmax_tick = round(plot_xmin, -5) - 100000, round(plot_xmax, -5) + 100000
        ymin_tick, ymax_tick = round(plot_ymin, -5) - 100000, round(plot_ymax, -5) + 100000
        tick_separation = 400000.
    else:   # same as discretizes polygons
        plot_xmin, plot_ymin, plot_xmax, plot_ymax = polygon_gdf.total_bounds
        xmin_tick, xmax_tick = round(plot_xmin, -5) - 100000, round(plot_xmax, -5) + 100000
        ymin_tick, ymax_tick = round(plot_ymin, -5) - 100000, round(plot_ymax, -5) + 100000
        tick_separation = 400000.
    return plot_xmin, plot_ymin, plot_xmax, plot_ymax, xmin_tick, xmax_tick, ymin_tick, ymax_tick, tick_separation


#target_rupture_ids = [209026]
#rates_df = pd.read_csv(f"../data/{NSHM_directory}/solution/rates.csv")