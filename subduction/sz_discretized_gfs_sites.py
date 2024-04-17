import pickle as pkl
import numpy as np
import geopandas as gpd
import pandas as pd
import cutde.halfspace as HS
from shapely.geometry import MultiPoint, LineString, Point
import os


# Calculates greens functions along coastline at specified interval
# Read in the geojson file from the NSHM inversion solution
version_extension = "_vtesting"
#NSHM_directory = "NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MTUy"
steeper_dip, gentler_dip = False, False
# in list form for one coord or list of lists for multiple (in NZTM)
site_list_csv = os.path.join('/mnt/', 'c', 'Users', 'jmc753', 'Work', 'occ-coseismic', 'wellington_qgis_grid_points.csv')
sites_df = pd.read_csv(site_list_csv)

site_coords = np.array(sites_df[['Lon', 'Lat', 'Height']])
site_name_list = [site for site in sites_df['siteId']]


#############################################
x_data = site_coords[:, 0]
y_data = site_coords[:, 1]
gf_type = "sites"
# not using this part at the moment
# if steeper_dip == True and gentler_dip == False:
#     dip_modification_extension = "_steeperdip"
# elif gentler_dip == True and steeper_dip == False:
#     dip_modification_extension = "_gentlerdip"
# elif gentler_dip == False and steeper_dip == False:
#     dip_modification_extension = ""
if steeper_dip == True and gentler_dip == True:
    # throw an error
    print("Dip modifications are wrong. Only one statement can be True at once. Try again.")
    exit()
elif steeper_dip:
    version_extension += "_steeperdip"
elif gentler_dip:
    version_extension += "_gentlerdip"

# Load files
with open(f"out_files{version_extension}/sz_discretised_dict.pkl",
          "rb") as f:
    discretised_dict = pkl.load(f)

gf_dict_sites = {}
for rupture_id in discretised_dict.keys():
    triangles = discretised_dict[rupture_id]["triangles"]
    rake = discretised_dict[rupture_id]["rake"]

    vertices = triangles.reshape(triangles.shape[0] * triangles.shape[1], 3)
    vertex_multipoint = MultiPoint(vertices)

    pts = site_coords

    zero_slip_array = np.zeros((triangles.shape[0],))
    ones_slip_array = np.ones((triangles.shape[0],))

    dip_slip_array = np.vstack([zero_slip_array, ones_slip_array, zero_slip_array]).T
    strike_slip_array = np.vstack([ones_slip_array, zero_slip_array, zero_slip_array]).T

    # calculate displacements
    disps_ss = HS.disp_free(obs_pts=pts, tris=triangles, slips=strike_slip_array, nu=0.25)
    disps_ds = HS.disp_free(obs_pts=pts, tris=triangles, slips=dip_slip_array, nu=0.25)

    # make displacement dictionary for outputs. only use vertical disps (last column)
    disp_dict = {"ss": disps_ss[:, -1], "ds": disps_ds[:, -1], "rake": rake, "site_name_list": site_name_list,
                 "site_coords": site_coords, "x_data": x_data, "y_data": y_data}

    gf_dict_sites[rupture_id] = disp_dict


with open(f"out_files{version_extension}/sz_gf_dict_{gf_type}.pkl", "wb") as f:
    pkl.dump(gf_dict_sites, f)

