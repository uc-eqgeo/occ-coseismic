import pickle as pkl
import numpy as np
import cutde.halfspace as HS
from shapely.geometry import MultiPoint
import geopandas as gpd
import os
import pandas as pd
from time import time
import h5py as h5


# calculates green's functions at points specified in a list (or lists) of coordinates
# can also define site names, otherwise they will be numbered later on. Can make more sites than you use later for
# plotting if they are named.

############### USER INPUTS #####################
# need to run once for each green's function type (grid, sites, coast points, etc.) but can reuse for different branches
discretise_version = "_CFM"  # Tag for the directory containing the disctretised faults
mesh_version = "_EastCoastNI_10km"

steeper_dip, gentler_dip = False, False

# in list form for one coord or list of lists for multiple (in NZTM)
site_list_csv = os.path.join('..', 'sites', 'EastCoastNI_10km_grid_points.csv')
sites_df = pd.read_csv(site_list_csv)

gf_site_names = [str(site) for site in sites_df['siteId']]
gf_site_coords = np.array(sites_df[['Lon', 'Lat', 'Height']])

geojson_only = False  # True if you are generating gfs for a subset of sites that you have already prepared
#########################
gf_type = "sites"

if discretise_version[0] != '_':
    discretise_version = '_' + discretise_version

# not at actually using this part right now
if steeper_dip == True and gentler_dip == False:
    mesh_version += "_steeperdip"
elif steeper_dip == False and gentler_dip == True:
    mesh_version += "_gentlerdip"
elif steeper_dip == True and gentler_dip == True:
    print("Dip modifications are wrong. Only one statement can be True at once. Try again.")
    exit()

requested_site_coords = np.ascontiguousarray(np.array(sites_df[['Lon', 'Lat', 'Height']]))
requested_site_names = sites_df['siteId'].values

if geojson_only:
    print(f"Generating geojson for {gf_type} only")
else:
    # Load pre made_greens_functions
    gf_h5_file = f"discretised{discretise_version}/crustal_gf_dict_sites.h5"
    if not os.path.exists(gf_h5_file):
        with h5.File(gf_h5_file, "w") as gf_h5:
            gf_h5.create_dataset('site_name_list', data=np.array([], dtype='S'))
            gf_h5.create_dataset('site_coords', data=np.array([]))

    with h5.File(gf_h5_file, "r") as gf_h5:
            prepared_site_names = gf_h5['site_name_list'].asstr()[:].tolist()
            prepared_site_coords = gf_h5['site_coords'][:]

    prepare_set = set(prepared_site_names)  # Convert to set for faster lookup
    gf_ix = [ix for ix, site in enumerate(requested_site_names) if site not in prepare_set]
    all_site_names = prepared_site_names + requested_site_names[gf_ix].tolist()
    if prepared_site_coords.shape[0] == 0:
        all_site_coords = requested_site_coords[:, :2]
    else:
        all_site_coords = np.vstack([prepared_site_coords, requested_site_coords[gf_ix, :2]])
    with h5.File(gf_h5_file, "r+") as gf_h5:
        for key in ['site_name_list', 'site_coords']:
            del gf_h5[key]
        gf_h5.create_dataset('site_name_list', data=np.array(all_site_names, dtype='S'))
        gf_h5.create_dataset('site_coords', data=all_site_coords)

    # load files
    with open(f"discretised{discretise_version}/crustal_discretised_dict.pkl", "rb") as f:
        discretised_dict = pkl.load(f)

    for fault_id in discretised_dict.keys():
        triangles = discretised_dict[fault_id]["triangles"]
        rake = discretised_dict[fault_id]["rake"]

        # Identify, for this rupture, which sites have not been processed
        with h5.File(gf_h5_file, "r+") as gf_h5:
            if str(fault_id) not in gf_h5.keys():
                gf_h5.create_group(str(fault_id))
                gf_h5[str(fault_id)].create_dataset('ss', data=np.array([]))
                gf_h5[str(fault_id)].create_dataset('ds', data=np.array([]))
                gf_h5[str(fault_id)].create_dataset('rake', data=90)
                gf_h5[str(fault_id)].create_dataset('site_name_ix', data=np.array([]))
                gf_h5[str(fault_id)].create_dataset('non_zero_sites', data=np.array([]))
            
            site_name_ix = gf_h5[str(fault_id)]['site_name_ix'][:]
            if len(site_name_ix) > 0:
                prepared_site_names = np.array(all_site_names)[site_name_ix].tolist()
            else:
                prepared_site_names = []
            non_zero_ix = gf_h5[str(fault_id)]['non_zero_sites'][:]
            if len(non_zero_ix) > 0:
                dipslip = np.zeros([len(prepared_site_names)])
                dipslip[non_zero_ix] = gf_h5[str(fault_id)]['ds'][:]
            else:
                dipslip = gf_h5[str(fault_id)]['ds'][:]

        begin = time()
        prepare_set = set(prepared_site_names)  # Convert to set for faster lookup
        site_ix = np.array([ix for ix, site in enumerate(requested_site_names) if site not in prepare_set])
        if not site_ix.any():
            # All sites have been processed 
            print(f'discretised dict {fault_id} of {len(discretised_dict.keys())} prep in {time() - begin:.2f} seconds (Fault Fully pre-prepared)                ', end='\r')
            continue

        vertices = triangles.reshape(triangles.shape[0] * triangles.shape[1], 3)
        vertex_multipoint = MultiPoint(vertices)

        zero_slip_array = np.zeros((triangles.shape[0],))
        ones_slip_array = np.ones((triangles.shape[0],))

        dip_slip_array = np.ascontiguousarray(np.vstack([zero_slip_array, ones_slip_array, zero_slip_array]).T)
        strike_slip_array = np.ascontiguousarray(np.vstack([ones_slip_array, zero_slip_array, zero_slip_array]).T)

        # Index 
        gf_site_name_list = requested_site_names[site_ix].tolist()
        gf_site_coords = requested_site_coords[site_ix, :]

        # Calculate displacements for each fault
        disps_ss = HS.disp_free(obs_pts=gf_site_coords, tris=triangles, slips=strike_slip_array, nu=0.25)
        disps_ds = HS.disp_free(obs_pts=gf_site_coords, tris=triangles, slips=dip_slip_array, nu=0.25)

        disps_ss = np.hstack([dipslip, disps_ss[:, -1]])
        disps_ds = np.hstack([dipslip, disps_ds[:, -1]])
        if prepared_site_coords.shape[0] == 0:
            site_coords = gf_site_coords[:, :2]
        else:
            site_coords = np.vstack([prepared_site_coords, gf_site_coords[:, :2]])
        site_name_list = prepared_site_names + gf_site_name_list

        zero_value = 1 / (50 * 1e3)  # Zero value is the limit to store values by requiring at least 1mm of displacement from 50m of slip
        non_zero_ix = np.where((np.abs(disps_ss) + np.abs(disps_ds)) > zero_value)[0]
        disps_ss = disps_ss[non_zero_ix]
        disps_ds = disps_ds[non_zero_ix]

        if all_site_names == site_name_list:
            site_name_ix = np.arange(len(site_name_list))
        else:
            index_map = {value: idx for idx, value in enumerate(all_site_names)}  #  Create a dictionary to map the indices of all_site_names
            site_name_ix = np.array([index_map[value] for value in site_name_list if value in index_map])  # Use list comprehension to find indices

        # make displacement dictionary for outputs. only use the vertical disps. (last column)
        disp_dict = {"ss": disps_ss, "ds": disps_ds, "rake": rake, "non_zero_sites": non_zero_ix,
                    "site_name_ix": site_name_ix}

        with h5.File(gf_h5_file, "r+") as gf_h5:
            for key in disp_dict.keys():
                del gf_h5[str(fault_id)][key]
                gf_h5[str(fault_id)].create_dataset(key, data=disp_dict[key])

        if fault_id % 1 == 0:
            print(f'discretised dict {fault_id} of {len(discretised_dict.keys())} done in {time() - begin:.2f} seconds ({triangles.shape[0]} triangles per patch)    ', end='\r')
    print('')

# This geojson file will be used to control the sites of the inversion
gdf = gpd.GeoDataFrame(sites_df, geometry=gpd.points_from_xy(sites_df.Lon, sites_df.Lat), crs='EPSG:2193')
gdf.to_file(f"discretised{discretise_version}/crustal_site_locations{mesh_version}.geojson", driver="GeoJSON")

print(f"\ndiscretised{discretise_version}/crustal_site_locations{mesh_version} Complete!")