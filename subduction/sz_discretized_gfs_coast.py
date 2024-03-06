import pickle as pkl
import numpy as np
import geopandas as gpd
import cutde.halfspace as HS
from shapely.geometry import MultiPoint, LineString, Point
import os
import rasterio.mask
from array_operations import write_tiff, write_gmt_grd
import math


# Calculates greens functions along coastline at specified interval
# Read in the geojson file from the NSHM inversion solution
extension1 = "sites_v3_Uy"
#NSHM_directory = "NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MTUy"
steeper_dip, gentler_dip = False, False
point_dist = 1000   # in meters

########
if steeper_dip == True and gentler_dip == False:
    extension2 = "_steeperdip"
elif gentler_dip == True and steeper_dip == False:
    extension2 = "_gentlerdip"
elif gentler_dip == False and steeper_dip == False:
    extension2 = ""
else:
    print("Dip modifications are wrong. Only one statement can be True at once. Try again.")
    exit()

# Load files
target_coastline = gpd.read_file("../data/coastline/wellington_coast_merged.geojson")
# distance between points along coastline

with open(f"out_files/{extension1}{extension2}/subduction_discretized_polygons_{extension1}{extension2}.pkl",
          "rb") as f:
    discretised_dict = pkl.load(f)


# find points at point_dist intervals along each line
coast_points = []
for line in target_coastline.geometry:  # shapely multiline#
    # step through each line at increasing distances
    # starts at 0 km and goes up to largest whole km
    for i in range(0, int(line.length), point_dist):
        coast_point_i = line.interpolate(i)
        coast_points.append(coast_point_i)

#### For filtering scenarios based on disp
# get x and y data for points along coastline
x_data, y_data = [], []
for point in coast_points:
    x_data.append(point.x)
    y_data.append(point.y)

xpoints, ypoints = np.array(x_data), np.array(y_data)
np.save(f"out_files/{extension1}{extension2}/xpoints_{extension1}.npy", xpoints)
np.save(f"out_files/{extension1}{extension2}/ypoints_{extension1}.npy", ypoints)
xy_coords = np.vstack((xpoints, ypoints)).T

# this just numbers the coastline points to be consistent with the other named site files.
site_name_list = range(len(xpoints))

gf_dict_coast = {}
for i in discretised_dict.keys():
    triangles = discretised_dict["triangles"]
    rake = discretised_dict["rake"]
    if len(triangles) > 0:
        vertices = triangles.reshape(triangles.shape[0] * triangles.shape[1], 3)
        vertex_multipoint = MultiPoint(vertices)

        pts = np.vstack((xpoints, ypoints, xpoints * 0.)).T

        zero_slip_array = np.zeros((triangles.shape[0],))
        ones_slip_array = np.ones((triangles.shape[0],))

        dip_slip_array = np.vstack([zero_slip_array, ones_slip_array, zero_slip_array]).T
        strike_slip_array = np.vstack([ones_slip_array, zero_slip_array, zero_slip_array]).T

        # calculate displacements
        disps_ss = HS.disp_free(obs_pts=pts, tris=triangles, slips=strike_slip_array, nu=0.25)
        disps_ds = HS.disp_free(obs_pts=pts, tris=triangles, slips=dip_slip_array, nu=0.25)

        # make displacement dictionary for outputs. only use vertical disps (last column)
        disp_dict = {"ss": disps_ss[:, -1], "ds": disps_ds[:, -1], "rake": rake, "site_name_list": site_name_list,
                     "site_coords": xy_coords}

        gf_dict_coast[i] = disp_dict


with open(f"out_files/{extension1}{extension2}/sz_gf_dict_{extension1}{extension2}.pkl", "wb") as f:
    pkl.dump(gf_dict_coast, f)

