try:
    import geopandas as gpd
    from shapely.geometry import Point
except:
    print("Running on NESI. Some functions won't work....")
finally:
    import pandas as pd
    import os
    import h5py as h5
    from functools import reduce
    import numpy as np
    from numpy.lib.function_base import _check_interpolation_as_method, _quantile_is_valid, _QuantileMethods, \
                                        _get_indexes, _get_gamma, _lerp
    from numpy.core.numeric import normalize_axis_tuple
    from numpy import minimum, take, concatenate
    import pickle as pkl
    from scipy.interpolate import griddata
    import matplotlib.pyplot as plt
    from time import time
    import h5py as h5
    from scipy.sparse import csr_matrix

def dict_to_hdf5(hdf5_group, dictionary, compression=None, compression_opts=None, replace_groups=False):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            # Recursively create sub-groups
            if replace_groups and key in hdf5_group:
                del hdf5_group[key]
            sub_group = hdf5_group.create_group(str(key))
            dict_to_hdf5(sub_group, value)
        else:
            # Create datasets for other data types
            if compression and not np.isscalar(value):
                hdf5_group.create_dataset(key, data=value, compression=compression, compression_opts=compression_opts)
            else:
                hdf5_group.create_dataset(key, data=value)


def hdf5_to_dict(group):
    result = {}
    for key, item in group.items():
        if isinstance(item, h5.Dataset):
            result[key] = item[()]
        elif isinstance(item, h5.Group):
            result[key] = hdf5_to_dict(item)
    return result


def get_probability_color(exceed_type):
    """ exceed type can be "total_abs", "up", or "down
    """
    if exceed_type == "total_abs":
        color = "k"
    elif exceed_type == "up":
        color = (189/255, 0, 0)
    elif exceed_type == "down":
        color = (15/255, 72/255, 186/255)

    return color


def make_qualitative_colormap(name, length):
    from collections import namedtuple
    # set up custom color scheme. Uses tab20 as a base, moves the greens to the start (to break up the
    # green/brown/red requence), makes the brown darker, and the red redder.
    if name == "tab20_subbed":
        #up to 20 colors
        all_colors = plt.get_cmap('tab20b')(np.linspace(0, 1, 20))
        color_indices = list(range(0, 20, 2))
        colors = all_colors[color_indices[0:length]]
        blue1, blue2 = colors[0], colors[1]
        green1, green2 = colors[2], colors[3]
        new_brown1 = [140 / 255, 84 / 255, 0 / 255, 1]
        new_brown2 = [112 / 255, 74 / 255, 1 / 255, 1]
        new_dark_red = [133 / 255, 5 / 255, 0 / 255, 1]
        #colors[0], colors[1] = green1, green2
        #colors[2], colors[3] = blue1, blue2
        colors[4], colors[6] = new_brown2, new_dark_red

    elif name == "tab20_reordered":
        # up to 20 colors
        all_colors = plt.get_cmap('tab20b')(np.linspace(0, 1, 20))
        color_indices = list(range(0, 20, 2))
        colors = all_colors[color_indices[0:length]]
        blue1, blue2 = colors[0], colors[1]
        green1, green2 = colors[2], colors[3]
        #new_brown1 = [140 / 255, 84 / 255, 0 / 255, 1]
        new_brown2 = [112 / 255, 74 / 255, 1 / 255, 1]
        new_dark_red = [133 / 255, 5 / 255, 0 / 255, 1]
        colors[0], colors[1] = green1, green2
        colors[2], colors[3] = blue1, blue2
        colors[4], colors[6] = new_brown2, new_dark_red

    elif name == "custom":
        tab20b_all_colors = plt.get_cmap('tab20b')(np.linspace(0, 1, 20))
        tab20c_all_colors = plt.get_cmap('tab20c')(np.linspace(0, 1, 20))
        tab20_all_colors = plt.get_cmap('tab20')(np.linspace(0, 1, 20))
        colors = [tab20b_all_colors[0], tab20b_all_colors[2],
                  tab20c_all_colors[0], tab20c_all_colors[2],
                  tab20b_all_colors[4], tab20b_all_colors[6],
                  tab20_all_colors[10], tab20b_all_colors[10],
                  tab20c_all_colors[4], tab20c_all_colors[6],
                  [133 / 255, 5 / 255, 0 / 255, 1], tab20b_all_colors[14],
                  tab20b_all_colors[16], tab20b_all_colors[18]]
        colors = colors[0:length]

    elif name == "tol_muted_ordered":
        # up to about 8 colors, plus light grey and black
        cset = namedtuple('Mcset',
                          'indigo cyan teal green olive sand rose wine purple pale_grey black')
        colors = cset('#332288','#88CCEE', '#44AA99','#117733', '#999933', '#DDCC77','#CC6677',
                    '#882255', '#AA4499', '#DDDDDD', '#000000')

    return colors


def read_rupture_csv(csv_file: str, fakequakes=False):
    rupture_dict = {}
    with open(csv_file, "r") as fid:
        index_data = fid.readlines()
    for ix, line in enumerate(index_data[1:]):
        numbers = [int(num) for num in line.strip().split(",")]
        if fakequakes:
            rupture_dict[ix] = np.array(numbers[2:])
        else:
            rupture_dict[numbers[0]] = np.array(numbers[2:])
    return rupture_dict


def read_average_slip(csv_file: str):
    df = pd.read_csv(csv_file)
    slip_dic = {}
    for i, row in df.iterrows():
        slip_dic[i] = row["Average Slip (m)"]
    return slip_dic


def read_fakequakes_slip_rates(NSHM_directory):
    proc_dir = os.path.relpath(os.path.dirname(__file__))
    csv_file = os.path.join(proc_dir, f"../data/{NSHM_directory}/ruptures/average_slips.csv")
    df = pd.read_csv(csv_file, index_col=0)
    patches = list(df.columns)
    patches.remove('Average Slip (m)')
    rupture_slip_dict = {}
    for ix, (_, row) in enumerate(df.iterrows()):
        print(f"Writing rupture slip dictionary... {ix}/{len(df)}", end="\r")
        rupture_slip_dict[ix] = row[patches].values.reshape(-1, 1)
    print("")
    all_ruptures = read_rupture_csv(os.path.join(proc_dir, f"../data/{NSHM_directory}/ruptures/indices.csv"), fakequakes=True)
    rates_df = pd.read_csv(os.path.join(proc_dir, f"../data/{NSHM_directory}/solution/rates.csv"))

    return rupture_slip_dict, rates_df, all_ruptures


def make_total_slip_dictionary(gf_dict_h5):
    """ calculates total greens function displacement using strike slip gf, dip slip gf, and rake value
    need to run the subduction and crustal_discretised_gfs script first"""

    gf_dict = h5.File(gf_dict_h5, "r")

    # Makes a new total gf displacement dictionary using rake
    grid_meta = None
    all_site_names = gf_dict["site_name_list"].asstr()[:]
    n_sites = len(all_site_names)
    n_ruptures = np.sum([1 for key in gf_dict.keys() if key not in ["site_coords", "site_name_list", "grid_meta"]])
    gf_adjusted_array = np.zeros([n_ruptures, n_sites])
    all_site_coords = gf_dict["site_coords"][:]
    key_list = []
    for ix, key in enumerate([key for key in gf_dict.keys() if key not in ["site_coords", "site_name_list"]]):
        print('Writing total slip dictionary: {}/{} rupture patches'.format(ix, n_ruptures), end="\r")
        if key == 'grid_meta':
            grid_meta = gf_dict[key]
        else:
            # greens functions are just for the vertical component
            gf_ix = gf_dict[key]["site_name_ix"]
#            site_name_list = all_site_names[gf_ix]
#            site_coords = all_site_coords[gf_ix, :]

            non_zero_ix = gf_ix[gf_dict[key]['non_zero_sites']]

            # calculate combined vertical from strike slip and dip slip using rake
            combined_gf = np.sin(np.radians(gf_dict[key]["rake"])) * gf_dict[key]["ds"] + np.cos(np.radians(gf_dict[key]["rake"])) * gf_dict[key]["ss"]
            gf_adjusted_array[len(key_list), non_zero_ix] = combined_gf
            key_list.append(key)

    gf_dict.close()
    print('')

    return csr_matrix(gf_adjusted_array), all_site_names.tolist(), all_site_coords, key_list, grid_meta


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


def tol_cset(colorset=None):
    """
    Discrete color sets for qualitative data.

    Define a namedtuple instance with the colors.
    Examples for: cset = tol_cset(<scheme>)
      - cset.red and cset[1] give the same color (in default 'bright' colorset)
      - cset._fields gives a tuple with all color names
      - list(cset) gives a list with all colors
    """
    from collections import namedtuple

    namelist = ('bright', 'high-contrast', 'vibrant', 'muted', 'medium-contrast', 'light')
    if colorset is None:
        return namelist
    if colorset not in namelist:
        colorset = 'bright'
        print('*** Warning: requested colorset not defined,',
              'known colorsets are {}.'.format(namelist),
              'Using {}.'.format(colorset))

    if colorset == 'bright':
        cset = namedtuple('Bcset',
                          'blue red green yellow cyan purple grey black')
        return cset('#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE',
                    '#AA3377', '#BBBBBB', '#000000')

    if colorset == 'high-contrast':
        cset = namedtuple('Hcset',
                          'blue yellow red black')
        return cset('#004488', '#DDAA33', '#BB5566', '#000000')

    if colorset == 'vibrant':
        cset = namedtuple('Vcset',
                          'orange blue cyan magenta red teal grey black')
        return cset('#EE7733', '#0077BB', '#33BBEE', '#EE3377', '#CC3311',
                    '#009988', '#BBBBBB', '#000000')

    if colorset == 'muted':
        cset = namedtuple('Mcset',
                          'rose indigo sand green cyan wine teal olive purple pale_grey black')
        return cset('#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE',
                    '#882255', '#44AA99', '#999933', '#AA4499', '#DDDDDD',
                    '#000000')

    if colorset == 'medium-contrast':
        cset = namedtuple('Mcset',
                          'light_blue dark_blue light_yellow dark_red dark_yellow light_red black')
        return cset('#6699CC', '#004488', '#EECC66', '#994455', '#997700',
                    '#EE99AA', '#000000')

    if colorset == 'light':
        cset = namedtuple('Lcset',
                          'light_blue orange light_yellow pink light_cyan mint pear olive pale_grey black')
        return cset('#77AADD', '#EE8866', '#EEDD88', '#FFAABB', '#99DDFF',
                    '#44BB99', '#BBCC33', '#AAAA00', '#DDDDDD', '#000000')

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
def filter_ruptures_by_location(NSHM_directory, target_rupture_ids, fault_type, model_version,
                                crustal_directory="crustal_files", sz_directory="subduction_files",
                                location=[1749150, 5428092], search_radius=2.5e5, fakequakes=False):
    """ filters the initial rupture scenarios by which patches are involved
        set a distance from interest area and cut out scenarios that don't intersect

        for now, rupture_df can be any list with the targeted rupture indices. For example, a list of ruptures
        that have been filtered by annual rate in the filter_ruptures_by_rate script above

        fault_type = "crustal" or "sz"
        """
    location = Point(location)

    if fault_type == "crustal":
        fault_rectangle_centroids_gdf = gpd.read_file(f"../{crustal_directory}/discretised_{model_version}"
                                                      f"/named_rectangle_centroids.geojson")
    if fault_type == "sz" or fault_type == "py":
        fault_rectangle_centroids_gdf = gpd.read_file(
            f"../{sz_directory}/discretised_{model_version}/{fault_type}_all_rectangle_centroids.geojson")

    all_ruptures_patch_indices = read_rupture_csv(f"../data/{NSHM_directory}/ruptures/indices.csv", fakequakes=fakequakes)

    # find rupture scenarios that match input target ruptures (e.g., from filter by rate)
    trimmed_rupture_patch_indices = {i: all_ruptures_patch_indices[i] for i in all_ruptures_patch_indices.keys() if i in
                                     target_rupture_ids}

    # find faults patches that are within search radius
    filtered_fault_ids = list(np.nonzero(fault_rectangle_centroids_gdf.distance(location) < search_radius)[0])
    # for i in range(len(fault_rectangle_centroids_gdf.centroid)):
    #     centroid = fault_rectangle_centroids_gdf.centroid[i]
    #     if centroid.distance(location) < search_radius:
    #         #filtered_fault_ids.append(patch_centroids_gdf.index[i])
    #         filtered_fault_ids.append(int(fault_rectangle_centroids_gdf.fault_id[i]))


    # this can probably be simplified
    # include scenarios that have those patches
    filtered_scenarios = []
    for rupture_index in target_rupture_ids:
        # uses scenarios that include any patch within that search radius
        if np.isin(trimmed_rupture_patch_indices[rupture_index], filtered_fault_ids).any():
            filtered_scenarios.append(rupture_index)
    print(f"location filtered scenarios: {len(filtered_scenarios)}")
    return filtered_scenarios



def calculate_vertical_disps(ruptured_discretised_polygons_gdf, ruptured_rectangle_outlines_gdf, rupture_id,
                             ruptured_fault_ids, slip_taper, rupture_slip_dict, gf_total_slip_array, rupture_order, fakequakes=False):
    """ calculates displacements for given rupture scenario at a single site
    not yet sure if I should set it up to allow more than one site at a time

    CAVETS/choices:
    - tapered slip assigns one slip value to each discretised polygon (e.g., one fault id). the slip
    values are tapered according to total rupture length of all rectangles.
    - this version changes all very small displacements to zero, or if no meshes are used, returns zero displacement
    """

    # find which patches have a mesh and which don't, to use greens functions later just with meshed patches
    ruptured_fault_ids_with_mesh, _, mesh_indices = np.intersect1d(ruptured_fault_ids, rupture_order, assume_unique=True, return_indices=True)
    ruptured_fault_ids_with_mesh = ruptured_fault_ids_with_mesh.astype(int)
    # calculate slip on each discretised polygon
    if slip_taper is False:
        # calculate displacements by multiplying scenario slip by scenario greens function
        # scenario gf sums displacements from all ruptured
        if fakequakes:
            sparse_rupt_slip = csr_matrix(rupture_slip_dict[rupture_id][ruptured_fault_ids_with_mesh])
            disps_scenario = np.array(gf_total_slip_array[mesh_indices, :].multiply(sparse_rupt_slip).sum(0)).reshape(-1)
            polygon_slips = rupture_slip_dict[rupture_id][ruptured_fault_ids_with_mesh]

        else:
            gf_array = gf_total_slip_array[mesh_indices, :].toarray()
            disps_scenario = rupture_slip_dict[rupture_id] * gf_array.sum(axis=0)
            polygon_slips = rupture_slip_dict[rupture_id] * np.ones(len(ruptured_fault_ids_with_mesh))

        # storing zeros is more efficient than nearly zeros. Makes v small displacements = 0
        if len(ruptured_fault_ids_with_mesh) != 0:
            disps_scenario[np.abs(disps_scenario) < 5.e-3] = 0.
        elif len(ruptured_fault_ids_with_mesh) == 0:
            disps_scenario = None

    elif slip_taper is True:
        # get centroid coords of faults discretised polygons with a mesh
        ruptured_polygon_centroid_points = ruptured_discretised_polygons_gdf.centroid
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

        # interpolate slip at each discretised polygon (i.e., patch) centroid and corresponding displacement
        polygon_slips = griddata(along_rupture_line_xy, tapered_slip_values, ruptured_polygon_centroid_coords,
                               method="nearest")

        # calculate displacements by multiplying the polygon green's function by slip on each fault
        # this will be a list of lists
        disps_i_list = []
        for i, fault_id in enumerate(ruptured_discretised_polygons_gdf.fault_id):
            # This section has never been tested following change from gf_total_slip_dict to gf_arrays
            fault_ix = rupture_order.index(fault_id)
            combined_gf = gf_total_slip_array[fault_ix, :].toarray()
            disp_i = combined_gf * polygon_slips[i]
            # disp_i = gf_total_slip_dict[fault_id]["combined_gf"] * polygon_slips[i]
            disps_i_list.append(disp_i)
        #sum displacements from each patch
        disps_scenario = np.sum(disps_i_list, axis=0)
        if len(ruptured_fault_ids_with_mesh) != 0:
            disps_scenario[np.abs(disps_scenario) < 5.e-3] = 0.
        elif len(ruptured_fault_ids_with_mesh) == 0:
            disps_scenario = None

    # Abandon ruptures that don't cause any displacement
    if disps_scenario is not None and sum(np.abs(disps_scenario)) == 0:
        disps_scenario = None

    return disps_scenario, polygon_slips

def get_rupture_disp_dict(NSHM_directory, fault_type, extension1, slip_taper, gf_name,
                          results_version_directory, disc_version_directory, crustal_directory="crustal_files", sz_directory="subduction_files",
                          location=[1749150, 5428092], search_radius=2.5e5, fakequakes=False):
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
    print(f"\nloading data for {extension1}")
    procdir = os.path.relpath(os.path.dirname(__file__)) + '/..'
    disc_version = disc_version_directory.split('/')[-1]

    if fakequakes:
        rupture_slip_dict, rates_df, all_ruptures = read_fakequakes_slip_rates(NSHM_directory)
    else:
        rupture_slip_dict = read_average_slip(f"{procdir}/data/{NSHM_directory}/ruptures/average_slips.csv")
        rates_df = pd.read_csv(f"{procdir}/data/{NSHM_directory}/solution/rates.csv")
        # this line takes ages, only do it once
        all_ruptures = read_rupture_csv(f"{procdir}/data/{NSHM_directory}/ruptures/indices.csv")

    if fault_type == "crustal":
        discretised_polygons_gdf = gpd.read_file(f"{procdir}/{crustal_directory}/discretised_{disc_version}/"
                                                 f"crustal_discretised_polygons.geojson")
        gf_dict_pkl = f"{procdir}/{crustal_directory}/discretised_{disc_version}/crustal_gf_dict_{gf_name}.h5"
        rectangle_outlines_gdf = gpd.read_file(f"{procdir}/{crustal_directory}/discretised_{disc_version}"
                                               f"/all_rectangle_outlines.geojson")
    elif fault_type == "sz" or fault_type == "py":
        discretised_polygons_gdf = gpd.read_file(f"{procdir}/{sz_directory}/discretised_{disc_version}/{fault_type}_discretised_polygons.geojson")
        gf_dict_pkl = f"{procdir}/{sz_directory}/discretised_{disc_version}/{fault_type}_gf_dict_{gf_name}.h5"
        rectangle_outlines_gdf = gpd.read_file(f"{procdir}/{sz_directory}/discretised_{disc_version}/{fault_type}_all_rectangle_outlines.geojson")

    # for some reason it defaults values to string. Convert to integer.
    discretised_polygons_gdf['fault_id'] = discretised_polygons_gdf['fault_id'].astype('int64')
    rectangle_outlines_gdf['fault_id'] = rectangle_outlines_gdf['fault_id'].astype('int64')

    # filter ruptures by annual rate and location
    filtered_ruptures_annual_rate = filter_ruptures_by_rate(NSHM_directory)
    filtered_ruptures_location = filter_ruptures_by_location(NSHM_directory=NSHM_directory,
                                                             target_rupture_ids=filtered_ruptures_annual_rate,
                                                             fault_type=fault_type, crustal_directory=crustal_directory,
                                                             sz_directory=sz_directory, model_version=disc_version,
                                                             location=location, search_radius=search_radius, fakequakes=fakequakes)

    # Makes a new total gf displacement dictionary using rake. If points don't have a name (e.g., for whole coastline
    # calculations), the site name list is just a list of numbers
    gf_total_slip_array, site_name_list, site_coords, key_order, grid_meta = make_total_slip_dictionary(gf_dict_pkl)

    # calculate displacements at all the sites by rupture. Output dictionary keys are by rupture ID.
    disp_dictionary = {}
    for ix, rupture_id in enumerate(filtered_ruptures_location):
        print(f"calculating displacements for {extension1} ({ix}/{len(filtered_ruptures_location)})", end="\r")
        ruptured_fault_ids = all_ruptures[rupture_id]
        ruptured_discretised_polygons_gdf = discretised_polygons_gdf[
            discretised_polygons_gdf.fault_id.isin(ruptured_fault_ids)]
        ruptured_discretised_polygons_gdf = gpd.GeoDataFrame(ruptured_discretised_polygons_gdf, geometry='geometry')
        ruptured_rectangle_outlines_gdf = rectangle_outlines_gdf[
            rectangle_outlines_gdf.fault_id.isin(ruptured_fault_ids)]

        #calculate displacements, output is a list of displacements for each site
        disps_scenario, patch_slips = \
            calculate_vertical_disps(ruptured_discretised_polygons_gdf=ruptured_discretised_polygons_gdf,
                                     ruptured_rectangle_outlines_gdf=ruptured_rectangle_outlines_gdf,
                                     rupture_id=rupture_id, ruptured_fault_ids=ruptured_fault_ids,
                                     slip_taper=slip_taper, rupture_slip_dict=rupture_slip_dict,
                                     gf_total_slip_array=gf_total_slip_array, rupture_order=key_order, fakequakes=fakequakes)

        # extract annual rate and save data to dictionary. Key is the rupture ID. Ignores scenarios with zero
        # displacement at all sites. Sites can be a grid cell or a specific (named) site.
        annual_rate = rates_df[rates_df.index == rupture_id]["Annual Rate"].values[0]
        if disps_scenario is not None:
            # displacement dictionary for a single rupture scenario at all sites. Key is rupture id.
            disps_scenario = [(ix, disp) for ix, disp in enumerate(disps_scenario) if disp != 0]
            rupture_disp_dict = {"rupture_id": rupture_id, "v_disps_m": disps_scenario, "annual_rate": annual_rate,
                                 "site_name_list": site_name_list, "site_coords": site_coords,
                                 "polygon_slips_m": patch_slips}
        else:
            rupture_disp_dict = {"rupture_id": rupture_id, "v_disps_m": np.array([]), "annual_rate": annual_rate,
                        "site_name_list": site_name_list, "site_coords": np.array([]),
                        "polygon_slips_m": patch_slips}
        disp_dictionary[rupture_id] = rupture_disp_dict

    # print statement about how many scenarios have displacement > 0 at each site
    print(f"\nscenarios with displacement > 0: {len(disp_dictionary)}")

    if slip_taper is True:
        extension3 = "_tapered"
    else:
        extension3 = "_uniform"

    # save displacements
    os.makedirs(f"{procdir}/results/{disc_version}/{extension1}", exist_ok=True)

    with open(f"{procdir}/results/{disc_version}/{extension1}/all_rupture_disps_{extension1}{extension3}.pkl",
              "wb") as f:
        pkl.dump(disp_dictionary, f)
    
    # Save the site names to a seperate pickle file, as it is quicker to load and check on later runs
    site_name_dict = {'site_name_list': site_name_list}
    with open(f"{procdir}/results/{disc_version}/{extension1}/all_rupture_disps_{extension1}{extension3}_sites.pkl",
              "wb") as f:
        pkl.dump(site_name_dict, f)

    if grid_meta:
        with open(f"../{results_version_directory}/{extension1}/grid_limits.pkl", "wb") as f:
            pkl.dump(grid_meta, f)

    return disp_dictionary

def get_figure_bounds(polygon_gdf="", extent=""):
    """sets figure bounds based on key words
    polygon_gdf: either discretised polygon gdf (for displacement figure) or ruptured rectangles gdf (slip figure)
    extent: can specify the extent of figure for interest area """

    if extent == "North Island":    # bounds of whole north island
        buffer = 100000
        plot_xmin, plot_ymin, plot_xmax, plot_ymax = 1525000, 5270000, 2300000, 6176000
        xmin_tick, xmax_tick  = 1600000, plot_xmax
        ymin_tick, ymax_tick = 5400000, 6176000
        tick_separation = 300000
    if extent == "South Island":    # bounds of whole south island
        buffer = 100000
        plot_xmin, plot_ymin, plot_xmax, plot_ymax = 1000000, 4737000, 1720000, 5520000
        xmin_tick, xmax_tick  = 1000000, plot_xmax
        ymin_tick, ymax_tick = 4740000, 5520000
        tick_separation = 300000
    elif extent == "Wellington":    # bounds around the wellington region, with a 10 km buffer
        x, y, buffer = 1771150, 5428092, 10.e4
        plot_xmin, plot_ymin, plot_xmax, plot_ymax = x - buffer, y - buffer, x + buffer, y + buffer
        tick_separation = 100000.
        xmin_tick, xmax_tick = 1700000, 1800000 + tick_separation / 4
        ymin_tick, ymax_tick = 5380000, 5480000 + tick_separation / 4
    elif extent == "Wellington close":  # bounds around the wellington region, with a 10 km buffer
        x, y, buffer = 1771150, 5428092, 5.e4
        plot_xmin, plot_ymin, plot_xmax, plot_ymax = x - buffer*0.9, y - buffer, x + buffer*1.7, y + buffer
        tick_separation = 50000.
        xmin_tick, xmax_tick = 1750000, 1850000 + tick_separation / 4
        ymin_tick, ymax_tick = 5390000, 5480000 + tick_separation / 4
    elif extent == "ruptured_rectangles":   # intended for ruptured rectangles gdf (slip plot)
        buffer = 20000
        plot_xmin = polygon_gdf.total_bounds[0] - buffer
        plot_ymin = polygon_gdf.total_bounds[1] - buffer
        plot_xmax = polygon_gdf.total_bounds[2] + buffer
        plot_ymax = polygon_gdf.total_bounds[3] + buffer
        xmin_tick, xmax_tick = round(plot_xmin + buffer, -4), plot_xmax
        ymin_tick, ymax_tick = round(plot_ymin + buffer, -4), plot_ymax
        tick_separation = round((plot_ymax - plot_ymin) / 3, -4)
    else:   # bounds of all polyons with a 100 km buffer (intented for discretised polygons, displacement plot)
        plot_xmin, plot_ymin, plot_xmax, plot_ymax = polygon_gdf.total_bounds
        xmin_tick, xmax_tick = round(plot_xmin, -5) - 100000, round(plot_xmax, -5) + 100000
        ymin_tick, ymax_tick = round(plot_ymin, -5) - 100000, round(plot_ymax, -5) + 100000
        tick_separation = 400000.
    return plot_xmin, plot_ymin, plot_xmax, plot_ymax, xmin_tick, xmax_tick, ymin_tick, ymax_tick, tick_separation

def save_target_rates(NSHM_directory, target_rupture_ids, extension1, results_version_directory):
    """get the annual rates from NSHM solution for target ruptures, output a csv file

    NSHM_directory = name of NSHM folder
    target_rupture_ids = list of rupture ids/indices
    out_directory = directory for all the other outputfiles (figures, etc.) """

    # load annual rate file
    rates_df = pd.read_csv(f"../data/{NSHM_directory}/solution/rates.csv")
    # only keep ruptures and rates of interest
    trimmed_rates_df = rates_df[rates_df.index.isin(target_rupture_ids)]

    if not os.path.exists(f"../{results_version_directory}/{extension1}"):
        os.mkdir(f"../{results_version_directory}/{extension1}")

    trimmed_rates_df.to_csv(f"../{results_version_directory}/{extension1}/ruptures_rates.csv", sep=',')
    print(f"{extension1} annual rates written to .csv")

#target_rupture_ids = [209026]
#rates_df = pd.read_csv(f"../data/{NSHM_directory}/solution/rates.csv")

def maximum_displacement_plot(site_ids, branch_site_disp_dict, model_dir, branch_name, threshold=0.01):
    coords = []
    up = []
    
    for site in site_ids:
        sdict = branch_site_disp_dict[site]
        coords.append(sdict['site_coords'])
        disps = np.array(sdict['disps'])
        max_disp = 0
        for disp in disps:
            if np.abs(disp) > np.abs(max_disp):
                max_disp = disp
        up.append(max_disp)
    
    coords = np.array(coords)
    up = np.array(up)

    fig, axs = plt.subplots(1, 2, figsize=(7, 3.4))
    lim = int(np.ceil(np.percentile(np.abs(up), 95)))
    im1 = axs[0].scatter(coords[:,0], coords[:,1], c=up, s=1, vmin=-lim, vmax=lim, cmap='RdYlBu')
    fig.colorbar(im1, ax=axs[0])
    axs[0].set_title("Maximum Displacement (m)")
    im2 = axs[1].scatter(coords[:,0], coords[:,1], c=np.abs(up) > threshold, s=1, vmin=0, vmax=1)
    fig.colorbar(im2, ax=axs[1])
    axs[1].set_title(f"Exceeds {threshold}m displacement")
    if branch_name == "":
        fig.savefig(f"../{model_dir}/weighted_max_disp.png", dpi=300)
    else:
        fig.savefig(f"../{model_dir}/{branch_name}/{branch_name}_max_disp.png", dpi=300)
    plt.close(fig)

def get_NSHM_directories(fault_type_list, deformation_model='geologic and geodetic', time_independent=True,
                         time_dependent=True, single_branch=False, fakequakes=False):
    # Set up which branches you want to calculate displacements and probabilities for
    # File suffixes and NSHM directories for each branch MUST be in the same order or you'll cause problems
    file_suffix_list = []
    NSHM_directory_list = []
    n_branches = []
    for fault_type in fault_type_list:
        fault_branches = 0
        if fault_type == "crustal":
            if time_independent and not single_branch:
                if "geologic" in deformation_model:
                    file_suffix_list_i = ["_c_MDA2", "_c_MDEz", "_c_MDE1"]
                    NSHM_directory_list_i = ["crustal_solutions/NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDA2",
                                             "crustal_solutions/NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDEz",
                                             "crustal_solutions/NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDE1"
                                            ]
                    fault_branches += len(file_suffix_list_i)
                    file_suffix_list.extend(file_suffix_list_i)
                    NSHM_directory_list.extend(NSHM_directory_list_i)
                if "geodetic" in deformation_model:
                    file_suffix_list_i = ["_c_MDE2", "_c_MDE5", "_c_MDI0"]
                    NSHM_directory_list_i = ["crustal_solutions/NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDE2",
                                             "crustal_solutions/NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDE5",
                                             "crustal_solutions/NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDI0"]
                    fault_branches += len(file_suffix_list_i)
                    file_suffix_list.extend(file_suffix_list_i)
                    NSHM_directory_list.extend(NSHM_directory_list_i)
            if time_dependent and not single_branch:
                if "geologic" in deformation_model:
                    file_suffix_list_i = ["_c_NjE5", "_c_NjI2", "_c_NjI3"]
                    NSHM_directory_list_i = ["crustal_solutions/NZSHM22_TimeDependentInversionSolution-QXV0b21hdGlvblRhc2s6MTExNjE5",
                                             "crustal_solutions/NZSHM22_TimeDependentInversionSolution-QXV0b21hdGlvblRhc2s6MTExNjI2",
                                             "crustal_solutions/NZSHM22_TimeDependentInversionSolution-QXV0b21hdGlvblRhc2s6MTExNjI3"]
                    fault_branches += len(file_suffix_list_i)
                    file_suffix_list.extend(file_suffix_list_i)
                    NSHM_directory_list.extend(NSHM_directory_list_i)
                if "geodetic" in deformation_model:
                    file_suffix_list_i = ["_c_NjI5", "_c_NjMy", "_c_NjM3"]
                    NSHM_directory_list_i = ["crustal_solutions/NZSHM22_TimeDependentInversionSolution-QXV0b21hdGlvblRhc2s6MTExNjI5",
                                             "crustal_solutions/NZSHM22_TimeDependentInversionSolution-QXV0b21hdGlvblRhc2s6MTExNjMy",
                                             "crustal_solutions/NZSHM22_TimeDependentInversionSolution-QXV0b21hdGlvblRhc2s6MTExNjM3"]
                    fault_branches += len(file_suffix_list_i)
                    file_suffix_list.extend(file_suffix_list_i)
                    NSHM_directory_list.extend(NSHM_directory_list_i)
            if single_branch:
                print("\n\n********\nCAUTION: SINGLE BRANCH HARD CODED FOR CRUSTAL FAULTS. MANUALLY CHANGE IN HELPER SCRIPTS UNTIL I GET ROUND TO FIXING\n********\n\n")
                file_suffix_list_i = ["_c_MDEw"]
                NSHM_directory_list_i = ["crustal_solutions/NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDEw"]
                file_suffix_list.extend(file_suffix_list_i)
                NSHM_directory_list.extend(NSHM_directory_list_i)

        elif fault_type == "sz":
            if fakequakes:
                file_suffix_list_i = ["_sz_fq_3nub110", "_sz_fq_pnub110", "_sz_fq_3nhb110", "_sz_fq_pnhb110", "_sz_fq_3lhb110", "_sz_fq_plhb110",
                                      "_sz_fq_3lhb110C1", "_sz_fq_3lhb110C100", "_sz_fq_3lhb110C1000", "_sz_fq_3nhb110C1", "_sz_fq_3nhb110C100"]
                NSHM_directory_list_i = ["sz_solutions/FakeQuakes_hk_3e10_nolocking_uniformSlip_n5000_S10_N1_GR500_b1-1_N21-5_nIt500000_narchi10",
                                         "sz_solutions/FakeQuakes_hk_prem_nolocking_uniformSlip_n5000_S10_N1_GR500_b1-1_N21-5_nIt500000_narchi10",
                                         "sz_solutions/FakeQuakes_hk_3e10_nolocking_n5000_S10_N1_GR500_b1-1_N21-5_nIt500000_narchi10",
                                         "sz_solutions/FakeQuakes_hk_prem_nolocking_n5000_S10_N1_GR500_b1-1_N21-5_nIt500000_narchi10",
                                         "sz_solutions/FakeQuakes_hk_3e10_locking_n5000_S10_N1_GR500_b1-1_N21-5_nIt500000_narchi10",
                                         "sz_solutions/FakeQuakes_hk_prem_locking_n5000_S10_N1_GR500_b1-1_N21-5_nIt500000_narchi10",
                                         "sz_solutions/FakeQuakes_hk_3e10_locking_n5000_S1_N1_GR500_b1-1_N21-5_nIt500000_narchi10",
                                         "sz_solutions/FakeQuakes_hk_3e10_locking_n5000_S100_N1_GR500_b1-1_N21-5_nIt500000_narchi10",
                                         "sz_solutions/FakeQuakes_hk_3e10_locking_n5000_S1000_N1_GR500_b1-1_N21-5_nIt500000_narchi10",
                                         "sz_solutions/FakeQuakes_hk_3e10_nolocking_n5000_S1_N1_GR500_b1-1_N21-5_nIt500000_narchi10",
                                         "sz_solutions/FakeQuakes_hk_3e10_nolocking_n5000_S100_N1_GR500_b1-1_N21-5_nIt500000_narchi10"]
            else:
                file_suffix_list_i = ["_sz_NJk2", "_sz_NzEx", "_sz_NzE0"]
                NSHM_directory_list_i = ["sz_solutions/NZSHM22_ScaledInversionSolution-QXV0b21hdGlvblRhc2s6MTA3Njk2",
                                         "sz_solutions/NZSHM22_ScaledInversionSolution-QXV0b21hdGlvblRhc2s6MTA3NzEx",
                                         "sz_solutions/NZSHM22_ScaledInversionSolution-QXV0b21hdGlvblRhc2s6MTA3NzE0"]

            if single_branch:
                branch_index = [file_suffix_list_i.index(branch) for branch in single_branch]
                file_suffix_list_i = [file_suffix_list_i[index] for index in branch_index]
                NSHM_directory_list_i = [NSHM_directory_list_i[index] for index in branch_index]
            else:
                fault_branches += len(file_suffix_list_i) * 3  # For scaling

            file_suffix_list.extend(file_suffix_list_i)
            NSHM_directory_list.extend(NSHM_directory_list_i)

        elif fault_type == "py":
            file_suffix_list_i = ["_py_M5NQ"]
            NSHM_directory_list_i = ["sz_solutions/NZSHM22_ScaledInversionSolution-QXV0b21hdGlvblRhc2s6MTMyNzM5NQ=="]

            if single_branch:
                branch_index = file_suffix_list_i.index(single_branch)
                file_suffix_list_i = [file_suffix_list_i[branch_index]]
                NSHM_directory_list_i = [NSHM_directory_list_i[branch_index]]
            else:
                fault_branches += len(file_suffix_list_i) * 3  # For scaling

            file_suffix_list.extend(file_suffix_list_i)
            NSHM_directory_list.extend(NSHM_directory_list_i)
        n_branches.append(fault_branches)

    return NSHM_directory_list, file_suffix_list, n_branches


## These scripts are those required by numpy v2.0.0 to run the weighted percentiles.
# As numpy v2.0 does not have backwards compatibility (and I couldn't install it), I have ripped
# the required functions and added them here, so that there is no need to update beyond numpy v1.*
# Useage should otherwise be the same as numpy.percentile(a, q, weights=weights).
# Sourced from: https://github.com/numpy/numpy/blob/v2.0.0/numpy/lib/_function_base_impl.py#L3942-L4288

def percentile(a,
               q,
               axis=None,
               out=None,
               overwrite_input=False,
               method="inverted_cdf",
               keepdims=False,
               *,
               weights=None,
               interpolation=None):
    """
    Compute the q-th percentile of the data along the specified axis.

    Returns the q-th percentile(s) of the array elements.

    See `numpy.percentile` man pages for the rest of the documation
    ----------
    """
    if interpolation is not None:
        method = _check_interpolation_as_method(
            method, interpolation, "percentile")

    a = np.asanyarray(a)
    if a.dtype.kind == "c":
        raise TypeError("a must be an array of real numbers")

    # Use dtype of array if possible (e.g., if q is a python int or float)
    # by making the divisor have the dtype of the data array.
    q = np.true_divide(q, a.dtype.type(100) if a.dtype.kind == "f" else 100)
    q = np.asanyarray(q)  # undo any decay that the ufunc performed (see gh-13105)
    if not _quantile_is_valid(q):
        raise ValueError("Percentiles must be in the range [0, 100]")

    if weights is not None:
        if method != "inverted_cdf":
            msg = ("Only method 'inverted_cdf' supports weights. "
                   f"Got: {method}.")
            raise ValueError(msg)
        if axis is not None:
            axis = normalize_axis_tuple(axis, a.ndim, argname="axis")
        weights = _weights_are_valid(weights=weights, a=a, axis=axis)
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")

    return _ureduce(
        a, func=_quantile_ureduce_func, q=q, weights=weights, keepdims=keepdims, axis=axis,
        out=out, overwrite_input=overwrite_input, method=method)

def _weights_are_valid(weights, a, axis):
    """Validate weights array.
    
    We assume, weights is not None.
    """
    wgt = np.asanyarray(weights)

    # Sanity checks
    if a.shape != wgt.shape:
        if axis is None:
            raise TypeError(
                "Axis must be specified when shapes of a and weights "
                "differ.")
        if wgt.shape != tuple(a.shape[ax] for ax in axis):
            raise ValueError(
                "Shape of weights must be consistent with "
                "shape of a along specified axis.")

        # setup wgt to broadcast along axis
        wgt = wgt.transpose(np.argsort(axis))
        wgt = wgt.reshape(tuple((s if ax in axis else 1)
                                for ax, s in enumerate(a.shape)))
    return wgt

def _ureduce(a, func, keepdims=False, **kwargs):
    """
    Internal Function.
    Call `func` with `a` as first argument swapping the axes to use extended
    axis on functions that don't support it natively.

    Returns result and a.shape with axis dims set to 1.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    func : callable
        Reduction function capable of receiving a single axis argument.
        It is called with `a` as first argument followed by `kwargs`.
    kwargs : keyword arguments
        additional keyword arguments to pass to `func`.

    Returns
    -------
    result : tuple
        Result of func(a, **kwargs) and a.shape with axis dims set to 1
        which can be used to reshape the result to the same shape a ufunc with
        keepdims=True would produce.

    """
    a = np.asanyarray(a)
    axis = kwargs.get('axis', None)
    out = kwargs.get('out', None)

    if keepdims is np._NoValue:
        keepdims = False

    nd = a.ndim
    if axis is not None:
        axis = normalize_axis_tuple(axis, nd)

        if keepdims:
            if out is not None:
                index_out = tuple(
                    0 if i in axis else slice(None) for i in range(nd))
                kwargs['out'] = out[(Ellipsis, ) + index_out]

        if len(axis) == 1:
            kwargs['axis'] = axis[0]
        else:
            keep = set(range(nd)) - set(axis)
            nkeep = len(keep)
            # swap axis that should not be reduced to front
            for i, s in enumerate(sorted(keep)):
                a = a.swapaxes(i, s)
            # merge reduced axis
            a = a.reshape(a.shape[:nkeep] + (-1,))
            kwargs['axis'] = -1
    else:
        if keepdims:
            if out is not None:
                index_out = (0, ) * nd
                kwargs['out'] = out[(Ellipsis, ) + index_out]

    r = func(a, **kwargs)

    if out is not None:
        return out

    if keepdims:
        if axis is None:
            index_r = (np.newaxis, ) * nd
        else:
            index_r = tuple(
                np.newaxis if i in axis else slice(None)
                for i in range(nd))
        r = r[(Ellipsis, ) + index_r]

    return r

def _quantile_ureduce_func(
        a: np.array,
        q: np.array,
        weights: np.array,
        axis: int = None,
        out=None,
        overwrite_input: bool = False,
        method="linear",
) -> np.array:
    if q.ndim > 2:
        # The code below works fine for nd, but it might not have useful
        # semantics. For now, keep the supported dimensions the same as it was
        # before.
        raise ValueError("q must be a scalar or 1d")
    if overwrite_input:
        if axis is None:
            axis = 0
            arr = a.ravel()
            wgt = None if weights is None else weights.ravel()
        else:
            arr = a
            wgt = weights
    else:
        if axis is None:
            axis = 0
            arr = a.flatten()
            wgt = None if weights is None else weights.flatten()
        else:
            arr = a.copy()
            wgt = weights
    result = _quantile(arr,
                       quantiles=q,
                       axis=axis,
                       method=method,
                       out=out,
                       weights=wgt)
    return result

def _quantile(
        arr: np.array,
        quantiles: np.array,
        axis: int = -1,
        method="linear",
        out=None,
        weights=None,
):
    """
    Private function that doesn't support extended axis or keepdims.
    These methods are extended to this function using _ureduce
    See nanpercentile for parameter usage
    It computes the quantiles of the array for the given axis.
    A linear interpolation is performed based on the `interpolation`.

    By default, the method is "linear" where alpha == beta == 1 which
    performs the 7th method of Hyndman&Fan.
    With "median_unbiased" we get alpha == beta == 1/3
    thus the 8th method of Hyndman&Fan.
    """
    # --- Setup
    arr = np.asanyarray(arr)
    values_count = arr.shape[axis]
    # The dimensions of `q` are prepended to the output shape, so we need the
    # axis being sampled from `arr` to be last.
    if axis != 0:  # But moveaxis is slow, so only call it if necessary.
        arr = np.moveaxis(arr, axis, destination=0)
    supports_nans = (
        np.issubdtype(arr.dtype, np.inexact) or arr.dtype.kind in 'Mm'
    )

    if weights is None:
        # --- Computation of indexes
        # Index where to find the value in the sorted array.
        # Virtual because it is a floating point value, not an valid index.
        # The nearest neighbours are used for interpolation
        try:
            method_props = _QuantileMethods[method]
        except KeyError:
            raise ValueError(
                f"{method!r} is not a valid method. Use one of: "
                f"{_QuantileMethods.keys()}") from None
        virtual_indexes = method_props["get_virtual_index"](values_count,
                                                            quantiles)
        virtual_indexes = np.asanyarray(virtual_indexes)

        if method_props["fix_gamma"] is None:
            supports_integers = True
        else:
            int_virtual_indices = np.issubdtype(virtual_indexes.dtype,
                                                np.integer)
            supports_integers = method == 'linear' and int_virtual_indices

        if supports_integers:
            # No interpolation needed, take the points along axis
            if supports_nans:
                # may contain nan, which would sort to the end
                arr.partition(
                    concatenate((virtual_indexes.ravel(), [-1])), axis=0,
                )
                slices_having_nans = np.isnan(arr[-1, ...])
            else:
                # cannot contain nan
                arr.partition(virtual_indexes.ravel(), axis=0)
                slices_having_nans = np.array(False, dtype=bool)
            result = take(arr, virtual_indexes, axis=0, out=out)
        else:
            previous_indexes, next_indexes = _get_indexes(arr,
                                                          virtual_indexes,
                                                          values_count)
            # --- Sorting
            arr.partition(
                np.unique(np.concatenate(([0, -1],
                                          previous_indexes.ravel(),
                                          next_indexes.ravel(),
                                          ))),
                axis=0)
            if supports_nans:
                slices_having_nans = np.isnan(arr[-1, ...])
            else:
                slices_having_nans = None
            # --- Get values from indexes
            previous = arr[previous_indexes]
            next = arr[next_indexes]
            # --- Linear interpolation
            gamma = _get_gamma(virtual_indexes, previous_indexes, method_props)
            result_shape = virtual_indexes.shape + (1,) * (arr.ndim - 1)
            gamma = gamma.reshape(result_shape)
            result = _lerp(previous,
                        next,
                        gamma,
                        out=out)
    else:
        # Weighted case
        # This implements method="inverted_cdf", the only supported weighted
        # method, which needs to sort anyway.
        weights = np.asanyarray(weights)
        if axis != 0:
            weights = np.moveaxis(weights, axis, destination=0)
        index_array = np.argsort(arr, axis=0, kind="stable")

        # arr = arr[index_array, ...]  # but this adds trailing dimensions of
        # 1.
        arr = np.take_along_axis(arr, index_array, axis=0)
        if weights.shape == arr.shape:
            weights = np.take_along_axis(weights, index_array, axis=0)
        else:
            # weights is 1d
            weights = weights.reshape(-1)[index_array, ...]

        if supports_nans:
            # may contain nan, which would sort to the end
            slices_having_nans = np.isnan(arr[-1, ...])
        else:
            # cannot contain nan
            slices_having_nans = np.array(False, dtype=bool)

        # We use the weights to calculate the empirical cumulative
        # distribution function cdf
        cdf = weights.cumsum(axis=0, dtype=np.float64)
        cdf /= cdf[-1, ...]  # normalization to 1
        # Search index i such that
        #   sum(weights[j], j=0..i-1) < quantile <= sum(weights[j], j=0..i)
        # is then equivalent to
        #   cdf[i-1] < quantile <= cdf[i]
        # Unfortunately, searchsorted only accepts 1-d arrays as first
        # argument, so we will need to iterate over dimensions.

        # Without the following cast, searchsorted can return surprising
        # results, e.g.
        #   np.searchsorted(np.array([0.2, 0.4, 0.6, 0.8, 1.]),
        #                   np.array(0.4, dtype=np.float32), side="left")
        # returns 2 instead of 1 because 0.4 is not binary representable.
        if quantiles.dtype.kind == "f":
            cdf = cdf.astype(quantiles.dtype)

        def find_cdf_1d(arr, cdf):
            indices = np.searchsorted(cdf, quantiles, side="left")
            # We might have reached the maximum with i = len(arr), e.g. for
            # quantiles = 1, and need to cut it to len(arr) - 1.
            indices = minimum(indices, values_count - 1)
            result = take(arr, indices, axis=0)
            return result

        r_shape = arr.shape[1:]
        if quantiles.ndim > 0: 
            r_shape = quantiles.shape + r_shape
        if out is None:
            result = np.empty_like(arr, shape=r_shape)
        else:
            if out.shape != r_shape:
                msg = (f"Wrong shape of argument 'out', shape={r_shape} is "
                       f"required; got shape={out.shape}.")
                raise ValueError(msg)
            result = out

        # See apply_along_axis, which we do for axis=0. Note that Ni = (,)
        # always, so we remove it here.
        Nk = arr.shape[1:]
        for kk in np.ndindex(Nk):
            result[(...,) + kk] = find_cdf_1d(
                arr[np.s_[:, ] + kk], cdf[np.s_[:, ] + kk]
            )

        # Make result the same as in unweighted inverted_cdf.
        if result.shape == () and result.dtype == np.dtype("O"):
            result = result.item()

    if np.any(slices_having_nans):
        if result.ndim == 0 and out is None:
            # can't write to a scalar, but indexing will be correct
            result = arr[-1]
        else:
            np.copyto(result, arr[-1, ...], where=slices_having_nans)
    return result