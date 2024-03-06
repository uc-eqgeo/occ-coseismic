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
    slip_dict = {}
    for i, row in df.iterrows():
        slip_dict[i] = row["Average Slip (m)"]
    return slip_dict

def read_rake_csv(csv_file: str):
    df = pd.read_csv(csv_file)
    rake_dict = {}
    for i, row in df.iterrows():
        fault_id = int(row["fault_id"])
        rake_dict[fault_id] = {"cfm_rake": row["cfm_rake"], "model1_rake": row["model1_rake"], "model2_rake": row["model2_rake"]}
    return rake_dict

def make_total_slip_dictionary(gf_dict_pkl):
    """ calculates total greens function displacement using strike slip gf, dip slip gf, and rake value

    need to run the and crustal_discretized_gfs script first"""

    with open(gf_dict_pkl, "rb") as fid:
        gf_dict = pkl.load(fid)

    # Makes a new total gf displacement dictionary using rake
    gf_adjusted_dict = {}
    for i in gf_dict.keys():
        # greens functions are just for the vertical component
        ss_gf = gf_dict[i]["ss"]
        ds_gf = gf_dict[i]["ds"]
        rake = gf_dict[i]["rake"]

        site_name_list = gf_dict[i]["site_name_list"]
        site_coords = gf_dict[i]["site_coords"]

        # calculate combined vertical from strike slip and dip slip using rake
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

    fault_rectangle_centroids_gdf = gpd.read_file(f"out_files/{extension1}{extension2}/named_rectangle_centroids"
                                                  f"_{extension1}{extension2}.geojson")
    all_ruptures_patch_indices = read_rupture_csv(f"../data/{NSHM_directory}/ruptures/indices.csv")
    # find rupture scenarios that match input target ruptures (e.g., from filter by rate)
    trimmed_rupture_patch_indices = {i: all_ruptures_patch_indices[i] for i in all_ruptures_patch_indices.keys() if i in
                                     target_rupture_ids}

    # find faults patches that are within search radius
    filtered_fault_ids = []
    for i in range(len(fault_rectangle_centroids_gdf.centroid)):
        centroid = fault_rectangle_centroids_gdf.centroid[i]
        if centroid.distance(Wellington) < search_radius:
            #filtered_fault_ids.append(patch_centroids_gdf.index[i])
            filtered_fault_ids.append(int(fault_rectangle_centroids_gdf.fault_id[i]))

    # this can probably be simplified
    # include scenarios that have those patches
    filtered_scenarios = []
    for rupture_index in target_rupture_ids:
        # uses scenarios that include any patch within that search radius
        if any(fault_id in filtered_fault_ids for fault_id in trimmed_rupture_patch_indices[rupture_index]):
            filtered_scenarios.append(rupture_index)

    print(f"location filtered scenarios: {len(filtered_scenarios)}")
    return filtered_scenarios

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

def calculate_vertical_disps(ruptured_discretized_polygons_gdf, ruptured_rectangle_outlines_gdf, rupture_id,
                             ruptured_fault_ids, slip_taper, rupture_slip_dict, gf_total_slip_dict):
    """ calcualtes displacements for given rupture scenario at a single site
    not yet sure if I should set it up to allow more than one site at a time

    CAVETS/choices:
    - tapered slip assigns one slip value to each discretized polygon (e.g., one fault id). the slip
    values are tapered according to total rupture length of all rectangles.
    - this version changes all very small dispalcements to zero, or if no meshes are used, returns zero displacement
    """

    # find which patches have a mesh and which don't, to use greens functions later just with meshed patches
    ruptured_fault_ids_with_mesh = np.intersect1d(ruptured_fault_ids, list(gf_total_slip_dict.keys()))

    # calculate slip on each discretized polygon
    if slip_taper is False:
        # calculate displacements by multiplying scenario slip by scenario greens function
        # scenario gf sums displacements from all ruptured
        gfs_i = np.sum([gf_total_slip_dict[j]["combined_gf"] for j in ruptured_fault_ids_with_mesh], axis=0)
        disps_scenario = rupture_slip_dict[rupture_id] * gfs_i
        polygon_slips = rupture_slip_dict[rupture_id] * np.ones(len(ruptured_fault_ids_with_mesh))

        # storing zeros is more efficient than nearly zeros. Makes v small displacements = 0
        if len(ruptured_fault_ids_with_mesh) != 0:
            disps_scenario[np.abs(disps_scenario) < 5.e-3] = 0.
        elif len(ruptured_fault_ids_with_mesh) == 0:
            disps_scenario = None

    elif slip_taper is True:
        # get centroid coords of faults discretized polygons with a mesh
        ruptured_polygon_centroid_points = ruptured_discretized_polygons_gdf.centroid
        ruptured_polygon_centroids_x = [point.x for point in ruptured_polygon_centroid_points]
        ruptured_polygon_centroids_y = [point.y for point in ruptured_polygon_centroid_points]
        ruptured_polygon_centroid_coords = np.array([ruptured_polygon_centroids_x, ruptured_polygon_centroids_y]).T

        # get bounds of fault patches, makes np array with 4 coords (minx, miny, maxx, maxy)
        rupture_bounds = ruptured_rectangle_outlines_gdf.total_bounds

        # makes 1000 points along a line between endpoints (bounds of fault rectangles).
        along_rupture_line_x = np.linspace(rupture_bounds[0], rupture_bounds[2], 1000)
        along_rupture_line_y = np.linspace(rupture_bounds[1], rupture_bounds[3], 1000)
        # stack into one column of xy pairs
        along_rupture_line_xy = np.column_stack((along_rupture_line_x, along_rupture_line_y))

        # calculate distance along line for each xy point
        start_point = Point(along_rupture_line_xy[0])
        line_distances = []
        for coord in along_rupture_line_xy:
            next_point = Point(coord)
            distance = start_point.distance(next_point)
            line_distances.append(distance)
        line_length = np.max(line_distances)

        # calculate slip at each interpolated point based on distance
        # this constant is based on the integral of the sin function from 0 to 1 (see NSHM taper)
        max_slip = rupture_slip_dict[rupture_id] / 0.76276
        # apply slip taper function to max slip. slip = sqrt(sin(pi * distance/line_length))
        # making a multiplier list is verbose but helps me keep track of things
        tapered_slip_multipliers = []
        tapered_slip_values = []
        for distance in line_distances:
            if np.sin(np.pi * distance / line_length) < 5.e-5:      # this is to fix error below of sqrt(0)
                tapered_slip_multiplier = 0.
            else:
                tapered_slip_multiplier = np.sqrt(np.sin(np.pi * distance / line_length))
            tapered_slip_multipliers.append(tapered_slip_multiplier)
            tapered_slip_values.append(max_slip * tapered_slip_multiplier)

        # interpolate slip at each discretized polygon (i.e., patch) centroid and corresponding displacement
        polygon_slips = griddata(along_rupture_line_xy, tapered_slip_values, ruptured_polygon_centroid_coords,
                               method="nearest")

        # calculate displacements by multiplying the polygon green's function by slip on each fault
        # this will be a list of lists
        disps_i_list = []
        for i, fault_id in enumerate(ruptured_discretized_polygons_gdf.fault_id):
            disp_i = gf_total_slip_dict[fault_id]["combined_gf"] * polygon_slips[i]
            disps_i_list.append(disp_i)
        #sum displacements from each patch
        disps_scenario = np.sum(disps_i_list, axis=0)
        if len(ruptured_fault_ids_with_mesh) != 0:
            disps_scenario[np.abs(disps_scenario) < 5.e-3] = 0.
        elif len(ruptured_fault_ids_with_mesh) == 0:
            disps_scenario = None


    return disps_scenario, polygon_slips

def get_rupture_disp_dict(NSHM_directory, extension1, extension2, slip_taper):
    """
    inputs: uses extension naming scheme to load NSHM rate/slip data and fault geometry, state slip taper

    extracts site and rupture data, pares down the ruptures based on location (within a buffer) and annual rate >0,
    and passes that to the get_displacements function

    outputs: a dictionary where each key is the rupture id in a pickle file. contains displacements (length = same as
    greens function), annual rate (single value for the rupture), site name list (should be same length as green's
    function), and site coordinates (same length as site name list)

    CAVEATS:
    - current version omits scenarios from the output list if all locations have zero displacement
    - a little clunky because most of the dictionary columns are repeated across all keys.
    """

    # load saved data
    print(f"loading data for {extension1}{extension2}")
    rupture_slip_dict = read_average_slip(f"../data/{NSHM_directory}/ruptures/average_slips.csv")
    rates_df = pd.read_csv(f"../data/{NSHM_directory}/solution/rates.csv")
    discretized_polygons_gdf = gpd.read_file(f"out_files/{extension1}{extension2}/"
                                             f"crustal_discretized_polygons_{extension1}{extension2}.geojson")
    gf_dict_pkl = f"out_files/{extension1}{extension2}/crustal_gf_dict_{extension1}{extension2}.pkl"

    # this line takes ages, only do it once
    all_ruptures = read_rupture_csv(f"../data/{NSHM_directory}/ruptures/indices.csv")  # change this later to clean up
    # code
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
    print(f"calculating displacements for {extension1}")
    for rupture_id in filtered_ruptures_location:
        ruptured_fault_ids = all_ruptures[rupture_id]
        ruptured_discretized_polygons_gdf = discretized_polygons_gdf[
            discretized_polygons_gdf.fault_id.isin(ruptured_fault_ids)]
        ruptured_discretized_polygons_gdf = gpd.GeoDataFrame(ruptured_discretized_polygons_gdf, geometry='geometry')
        ruptured_rectangle_outlines_gdf = rectangle_outlines_gdf[
            rectangle_outlines_gdf.fault_id.isin(ruptured_fault_ids)]

        #calculate displacements, output is a list of displacements for each site
        disps_scenario, patch_slips = \
            calculate_vertical_disps(ruptured_discretized_polygons_gdf=ruptured_discretized_polygons_gdf,
                                     ruptured_rectangle_outlines_gdf=ruptured_rectangle_outlines_gdf,
                                     rupture_id=rupture_id, ruptured_fault_ids=ruptured_fault_ids,
                                     slip_taper=slip_taper, rupture_slip_dict=rupture_slip_dict,
                                     gf_total_slip_dict=gf_total_slip_dict)

        # extract annual rate and save data to dictionary. Key is the rupture ID. Ignores scenarios with zero
        # displacement at all sites. Sites can be a grid cell or a specific (named) site.
        if disps_scenario is not None:
            annual_rate = rates_df[rates_df.index == rupture_id]["Annual Rate"].values[0]
            # displacement dictionary for a single rupture scenario at all sites. Key is rupture id.
            rupture_disp_dict = {"rupture_id": rupture_id, "v_disps_m": disps_scenario, "annual_rate": annual_rate,
                                 "site_name_list": site_name_list, "site_coords": site_coords,
                                 "x_data": site_coords[:, 0], "y_data": site_coords[:, 1],
                                 "polygon_slips_m": patch_slips}
            disp_dictionary[rupture_id] = rupture_disp_dict

    # print statement about how many scenarios have displacement > 0 at each site
    print(f"scenarios with displacement > 0: {len(disp_dictionary)}")

    if slip_taper is True:
        extension3 = "_tapered"
    else:
        extension3 = "_uniform"

    # save displacements
    with open(f"out_files/{extension1}{extension2}/all_rupture_disps_{extension1}{extension2}{extension3}.pkl",
              "wb") as f:
        pkl.dump(disp_dictionary, f)

    return disp_dictionary

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
        xmin_tick, xmax_tick = round(plot_xmin + buffer, -4), plot_xmax
        ymin_tick, ymax_tick = round(plot_ymin + buffer, -4), plot_ymax
        tick_separation = round((plot_ymax - plot_ymin) / 3, -4)
    else:   # bounds of all polyons with a 100 km buffer (intented for discretized polygons, displacement plot)
        plot_xmin, plot_ymin, plot_xmax, plot_ymax = polygon_gdf.total_bounds
        xmin_tick, xmax_tick = round(plot_xmin, -5) - 100000, round(plot_xmax, -5) + 100000
        ymin_tick, ymax_tick = round(plot_ymin, -5) - 100000, round(plot_ymax, -5) + 100000
        tick_separation = 400000.
    return plot_xmin, plot_ymin, plot_xmax, plot_ymax, xmin_tick, xmax_tick, ymin_tick, ymax_tick, tick_separation

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

#target_rupture_ids = [209026]
#rates_df = pd.read_csv(f"../data/{NSHM_directory}/solution/rates.csv")