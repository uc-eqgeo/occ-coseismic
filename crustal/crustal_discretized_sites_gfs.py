import pickle as pkl
import numpy as np
import cutde.halfspace as HS
from shapely.geometry import MultiPoint
import geopandas as gpd
import os
import pandas as pd


# calculates green's functions at points specified in a list (or lists) of coordinates
# can also define site names, otherwise they will be numbered later on. Can make more sites than you use later for
# plotting if they are named.

############### USER INPUTS #####################
# need to run once for each green's function type (grid, sites, coast points, etc.) but can reuse for different branches
mesh_version = "_Model_testing"
#out_extension = f"_{mesh_version}_v1"

steeper_dip, gentler_dip = False, False

# in list form for one coord or list of lists for multiple (in NZTM)
site_list_csv = os.path.join('/mnt/', 'c', 'Users', 'jmc753', 'Work', 'occ-coseismic', 'wellington_qgis_grid_points.csv')
sites_df = pd.read_csv(site_list_csv)

site_coords = np.array(sites_df[['Lon', 'Lat', 'Height']])
site_name_list = [site for site in sites_df['siteId']]
#########################
gf_type = "sites"

# not at actually using this part right now
if steeper_dip == True and gentler_dip == False:
    mesh_version += "_steeperdip"
elif steeper_dip == False and gentler_dip == True:
    mesh_version += "_gentlerdip"
elif steeper_dip == True and gentler_dip == True:
    print("Dip modifications are wrong. Only one statement can be True at once. Try again.")
    exit()

# load files
with open(f"out_files{mesh_version}/crustal_discretized_dict.pkl", "rb") as f:
    discretised_dict = pkl.load(f)

# make geojson of site locations
site_df = gpd.GeoDataFrame({"site_name": site_name_list, "x": site_coords[:, 0], "y": site_coords[:, 1]})
site_gdf = gpd.GeoDataFrame(site_df, geometry=gpd.points_from_xy(site_coords[:, 0], site_coords[:, 1]))

# all_rectangle_outline_gs = gpd.GeoSeries(all_rectangle_polygons, crs=2193)
# all_rectangle_outline_gdf = gpd.GeoDataFrame(df_all_rectangle, geometry=all_rectangle_outline_gs.geometry, crs=2193)
site_gdf.to_file(f"out_files{mesh_version}/site_points.geojson", driver="GeoJSON", crs=2193)

gf_dict_sites = {}
for fault_id in discretised_dict.keys():
    triangles = discretised_dict[fault_id]["triangles"]
    rake = discretised_dict[fault_id]["rake"]

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

    # make displacement dictionary for outputs. only use the vertical disps. (last column)
    disp_dict = {"ss": disps_ss[:, -1], "ds": disps_ds[:, -1], "rake": rake, "site_name_list": site_name_list,
                 "site_coords": site_coords, "x_data": site_coords[:, 0], "y_data": site_coords[:, 1]}

    gf_dict_sites[fault_id] = disp_dict


with open(f"out_files{mesh_version}/crustal_gf_dict_{gf_type}.pkl", "wb") as f:
    pkl.dump(gf_dict_sites, f)


