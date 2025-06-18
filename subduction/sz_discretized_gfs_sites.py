import pickle as pkl
import numpy as np
import geopandas as gpd
import pandas as pd
import cutde.halfspace as HS
import os
from time import time
import h5py as h5

"""
This script will take the discretised fault patches, and calculate the Green's functions for each site in the site list.
If the sites listed in the CSV already have a greens function calculated, then the script will skip that site.
"""
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Calculates greens functions along coastline at specified interval
# Read in the geojson file from the NSHM inversion solution
version_extension = "_version_0-1S"
# NSHM_directory = "NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MTUy"
steeper_dip, gentler_dip = False, False

# Define whch subduction zone ([_fq_]hikkerm / puysegur)
sz_zone = '_puysegur'

# in list form for one coord or list of lists for multiple (in NZTM)
csvfile = 'cube_centroids_27000_9000_buffer_0_33S_points.csv'
site_list_csv = os.path.join('..', 'sites', csvfile)
sites_df = pd.read_csv(site_list_csv)

# Names of the sites we need to prepare
gf_site_names = [str(site) for site in sites_df['siteId']]
gf_site_coords = np.array(sites_df[['Lon', 'Lat', 'Height']])

geojson_only = False  # True if you are generating gfs for a subset of sites that you have already prepared
#############################################
gf_type = "sites"

if sz_zone[0] != '_':
    sz_zone = '_' + sz_zone

if steeper_dip and gentler_dip:
    print("Dip modifications are wrong. Only one statement can be True at once. Try again.")
    exit()
elif steeper_dip:
    version_extension += "_steeperdip"
    sz_zone += "_steeperdip"
elif gentler_dip:
    version_extension += "_gentlerdip"
    sz_zone += "_gentlerdip"

if 'hikkerm' in sz_zone:
    prefix = 'sz'
elif 'puysegur' in sz_zone:
    prefix = 'py'
else:
    print("Please define a valid subduction zone (hikkerm / puysegur).")
    exit()

deblobify = False
if deblobify:
    version_extension += "_deblobify"
    sz_zone += "_deblobify"

if "_fq_" in sz_zone and version_extension[:3] != "_fq":
    version_extension = "_fq" + version_extension

requested_site_coords = np.ascontiguousarray(np.array(sites_df[['Lon', 'Lat', 'Height']]))
requested_site_names = sites_df['siteId'].values

if geojson_only:
    print(f"Generating geojson for {gf_type} only")
else:
    # Load premade greens_functions
    gf_h5_file = f"discretised{sz_zone}/{prefix}_gf_dict_sites.h5"
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

    # Load files
    with open(f"discretised{sz_zone}/{prefix}_discretised_dict.pkl",
            "rb") as f:
        discretised_dict = pkl.load(f)

    for fault_id in discretised_dict.keys():
        # Mesh information
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

        # Get DS and SS components for each triangle element, depending on the element rake
        ss_comp = np.cos(np.radians(rake))
        ds_comp = np.sin(np.radians(rake))
        total_slip_array = np.ascontiguousarray(np.zeros([triangles.shape[0], 3]))

        # Index 
        gf_site_name_list = requested_site_names[site_ix].tolist()
        gf_site_coords = requested_site_coords[site_ix, :]

        # Calculate the slip components for each triangle element
        for tri in range(triangles.shape[0]):
            ss, ds = np.linalg.lstsq(np.array([ss_comp[tri], ds_comp[tri]]).reshape([1, 2]), np.array([1]).reshape([1, 1]), rcond=None)[0]
            total_slip_array[tri, :2] = np.array([ss[0], ds[0]])

        disps = HS.disp_free(obs_pts=gf_site_coords, tris=triangles, slips=total_slip_array, nu=0.25)

        disps = np.hstack([dipslip, disps[:, -1]])
        if prepared_site_coords.shape[0] == 0:
            site_coords = gf_site_coords[:, :2]
        else:
            site_coords = np.vstack([prepared_site_coords, gf_site_coords[:, :2]])
        site_name_list = prepared_site_names + gf_site_name_list

        zero_value = 1 / (50 * 1e3)  # Zero value is the limit to store values by requiring at least 1mm of displacement from 50m of slip
        non_zero_ix = np.where(np.abs(disps) > zero_value)[0]
        disps = disps[non_zero_ix]

        if all_site_names == site_name_list:
            site_name_ix = np.arange(len(site_name_list))
        else:
            index_map = {value: idx for idx, value in enumerate(all_site_names)}  #  Create a dictionary to map the indices of all_site_names
            site_name_ix = np.array([index_map[value] for value in site_name_list if value in index_map])  # Use list comprehension to find indices

        # Set rake to 90 so that in future functions total displacement is just equal to DS
        disp_dict = {"ss": (disps * 0).astype(np.int8), "ds": disps, "non_zero_sites": non_zero_ix, "rake": 90,
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
gdf.to_file(f"discretised{sz_zone}/{prefix}_site_locations{version_extension}.geojson", driver="GeoJSON")

print(f"\n{prefix}_site_locations{version_extension} sites prepared!")