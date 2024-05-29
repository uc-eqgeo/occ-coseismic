import pickle as pkl
import numpy as np
import geopandas as gpd
import pandas as pd
import cutde.halfspace as HS
from shapely.geometry import MultiPoint, LineString, Point
import os
from time import time


# Calculates greens functions along coastline at specified interval
# Read in the geojson file from the NSHM inversion solution
version_extension = "_deblob"
# NSHM_directory = "NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MTUy"
steeper_dip, gentler_dip = False, False

# Define whch subduction zone (hikkerk / puysegur)
sz_zone = 'puysegur'

# in list form for one coord or list of lists for multiple (in NZTM)
csvfile = 'national_50km_grid_points.csv'
try:
    site_list_csv = os.path.join('/mnt/', 'c', 'Users', 'jmc753', 'Work', 'occ-coseismic', csvfile)
    sites_df = pd.read_csv(site_list_csv)
except FileNotFoundError:
    site_list_csv = os.path.join('C:\\', 'Users', 'jmc753', 'Work', 'occ-coseismic', csvfile)
    sites_df = pd.read_csv(site_list_csv)

site_coords = np.array(sites_df[['Lon', 'Lat', 'Height']])
site_name_list = [site for site in sites_df['siteId']]

#############################################
x_data = site_coords[:, 0]
y_data = site_coords[:, 1]
gf_type = "sites"

if steeper_dip and gentler_dip:
    print("Dip modifications are wrong. Only one statement can be True at once. Try again.")
    exit()
elif steeper_dip:
    version_extension += "_steeperdip"
elif gentler_dip:
    version_extension += "_gentlerdip"

if sz_zone == 'hikkerk':
    neighbours_file = '../data/hik_kerk3k_neighbours.txt'
    prefix = 'sz'
elif sz_zone == 'puysegur':
    neighbours_file = '../data/puysegur_neighbours.txt'
    prefix = 'py'
else:
    print("Please define a valid subduction zone (hikkerk / puysegur).")
    exit()

# Load files
with open(f"out_files{version_extension}/{prefix}_discretised_dict.pkl",
          "rb") as f:
    discretised_dict = pkl.load(f)

gf_dict = {}

for fault_id in discretised_dict.keys():
    triangles = discretised_dict[fault_id]["triangles"]
    rake = discretised_dict[fault_id]["rake"]

    # Get DS and SS components for each triangle element, depending on the element rake
    ss_comp = np.cos(np.radians(rake))
    ds_comp = np.sin(np.radians(rake))
    total_slip_array = np.zeros([triangles.shape[0], 3])

    begin = time()
    # Calculate the slip components for each triangle element
    for tri in range(triangles.shape[0]):
        ss, ds = np.linalg.lstsq(np.array([ss_comp[tri], ds_comp[tri]]).reshape([1, 2]), np.array([1]).reshape([1, 1]), rcond=None)[0]
        total_slip_array[tri, :2] = np.array([ss[0], ds[0]])

    disps = HS.disp_free(obs_pts=site_coords, tris=triangles, slips=total_slip_array, nu=0.25)

    # Set rake to 90 so that in future functions total displacement is just equal to DS
    disp_dict = {"ss": disps[:, -1] * 0, "ds": disps[:, -1], "rake": 90, "site_coords": site_coords,
                 "site_name_list": site_name_list, "x_data": x_data, "y_data": y_data}

    gf_dict[fault_id] = disp_dict
    if fault_id % 1 == 0:
        print(f'discretized dict {fault_id} of {len(discretised_dict.keys())} done in {time() - begin:.2f} seconds ({triangles.shape[0]} triangles per patch)')

with open(f"out_files{version_extension}/{prefix}_gf_dict_{gf_type}.pkl", "wb") as f:
    pkl.dump(gf_dict, f)
