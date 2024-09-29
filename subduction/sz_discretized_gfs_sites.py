import pickle as pkl
import numpy as np
import geopandas as gpd
import pandas as pd
import cutde.halfspace as HS
from shapely.geometry import MultiPoint, LineString, Point
import os
from time import time
import shutil
import h5py as h5

"""
This script will take the discretised fault patches, and calculate the Green's functions for each site in the site list.
If the sites listed in the CSV already have a greens function calculated, then the script will skip that site.
"""

# Calculates greens functions along coastline at specified interval
# Read in the geojson file from the NSHM inversion solution
version_extension = "_25km"
# NSHM_directory = "NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MTUy"
steeper_dip, gentler_dip = False, False

# Define whch subduction zone ([_fq_]hikkerk / puysegur)
sz_zone = '_fq_hikkerk'

# in list form for one coord or list of lists for multiple (in NZTM)
csvfile = 'national_25km_grid_points.csv'
site_list_csv = os.path.join('..', 'sites', csvfile)
sites_df = pd.read_csv(site_list_csv)

# Names of the sites we need to prepare
gf_site_names = [str(site) for site in sites_df['siteId']]
gf_site_coords = np.array(sites_df[['Lon', 'Lat', 'Height']])

#############################################
gf_type = "sites"

if steeper_dip and gentler_dip:
    print("Dip modifications are wrong. Only one statement can be True at once. Try again.")
    exit()
elif steeper_dip:
    version_extension += "_steeperdip"
    sz_zone += "_steeperdip"
elif gentler_dip:
    version_extension += "_gentlerdip"
    sz_zone += "_gentlerdip"

if 'hikkerk' in sz_zone:
    prefix = 'sz'
elif 'puysegur' in sz_zone:
    prefix = 'py'
else:
    print("Please define a valid subduction zone (hikkerk / puysegur).")
    exit()

deblobify = False
if deblobify:
    version_extension += "_deblobify"
    sz_zone += "_deblobify"

if "_fq_" in sz_zone and version_extension[:3] != "_fq":
    version_extension = "_fq" + version_extension

# Load pre made_greens_functions
gf_h5_file = f"discretised{sz_zone}/{prefix}_gf_dict_sites.h5"
if not os.path.exists(gf_h5_file):
    gf_file = h5.File(gf_h5_file, "w")
    gf_file.close()

# Load files
with open(f"discretised{sz_zone}/{prefix}_discretised_dict.pkl",
          "rb") as f:
    discretised_dict = pkl.load(f)

requested_site_coords = np.ascontiguousarray(np.array(sites_df[['Lon', 'Lat', 'Height']]))

for fault_id in discretised_dict.keys():
    # Mesh information
    triangles = discretised_dict[fault_id]["triangles"]
    rake = discretised_dict[fault_id]["rake"]

    # Get DS and SS components for each triangle element, depending on the element rake
    ss_comp = np.cos(np.radians(rake))
    ds_comp = np.sin(np.radians(rake))
    total_slip_array = np.zeros([triangles.shape[0], 3])

    # Identify, for this rupture, which sites have not been processed
    with h5.File(gf_h5_file, "r+") as gf_h5:
        if str(fault_id) not in gf_h5.keys():
            gf_h5.create_group(str(fault_id))
            gf_h5[str(fault_id)].create_dataset('ss', data=np.array([]))
            gf_h5[str(fault_id)].create_dataset('ds', data=np.array([]))
            gf_h5[str(fault_id)].create_dataset('rake', data=90)
            gf_h5[str(fault_id)].create_dataset('site_name_list', data=np.array([], dtype='S'))
            gf_h5[str(fault_id)].create_dataset('site_coords', data=np.array([]))
        
        prepared_site_names = gf_h5[str(fault_id)]['site_name_list'].asstr()[:].tolist()
        prepared_site_coords = gf_h5[str(fault_id)]['site_coords'][:]
        dipslip = gf_h5[str(fault_id)]['ds'][:]

    begin = time()
    site_name_array = np.array([(ix, str(site)) for ix, site in enumerate(sites_df['siteId']) if site not in prepared_site_names])
    if len (site_name_array) == 0:
        # All sites have been processed 
        print(f'discretised dict {fault_id} of {len(discretised_dict.keys())} done in {time() - begin:.2f} seconds ({triangles.shape[0]} triangles per patch)', end='\r')
        continue
    
    # Index 
    gf_ix = [int(ix) for ix in site_name_array[:, 0]]
    gf_site_name_list = [site for site in site_name_array[:, 1]]
    gf_site_coords = requested_site_coords[gf_ix, :]

    # Calculate the slip components for each triangle element
    for tri in range(triangles.shape[0]):
        ss, ds = np.linalg.lstsq(np.array([ss_comp[tri], ds_comp[tri]]).reshape([1, 2]), np.array([1]).reshape([1, 1]), rcond=None)[0]
        total_slip_array[tri, :2] = np.array([ss[0], ds[0]])

    disps = HS.disp_free(obs_pts=gf_site_coords, tris=triangles, slips=total_slip_array, nu=0.25)

    disps = np.hstack([dipslip, disps[:, -1]])
    if prepared_site_coords.shape[0] == 0:
        site_coords = gf_site_coords
    else:
        site_coords = np.vstack([prepared_site_coords, gf_site_coords])
    site_name_list = prepared_site_names + gf_site_name_list
    site_name_list = np.array(site_name_list, dtype='S')

    # Set rake to 90 so that in future functions total displacement is just equal to DS
    disp_dict = {"ss": (disps * 0).astype(int), "ds": disps, "site_coords": site_coords[:, :2],
                 "site_name_list": site_name_list}
    
    with h5.File(gf_h5_file, "r+") as gf_h5:
        for key in disp_dict.keys():
            del gf_h5[str(fault_id)][key]
            gf_h5[str(fault_id)].create_dataset(key, data=disp_dict[key])

    if fault_id % 1 == 0:
        print(f'discretised dict {fault_id} of {len(discretised_dict.keys())} done in {time() - begin:.2f} seconds ({triangles.shape[0]} triangles per patch)', end='\r')
print('')

# This geojson file will be used to control the sites of the inversion
gdf = gpd.GeoDataFrame(sites_df, geometry=gpd.points_from_xy(sites_df.Lon, sites_df.Lat), crs='EPSG:2193')
gdf.to_file(f"discretised{sz_zone}/{prefix}_site_locations{version_extension}.geojson", driver="GeoJSON")

print(f"\n{prefix}_site_locations{version_extension} sites prepared!")