import pickle as pkl
import numpy as np
import geopandas as gpd
import cutde.halfspace as HS
from shapely.geometry import MultiPoint, LineString, Point
import os


# Calculates greens functions along coastline at specified interval
# Read in the geojson file from the NSHM inversion solution
version_extension = "_v1_steeperdip"
#NSHM_directory = "NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MTUy"
steeper_dip, gentler_dip = True, False
# in list form for one coord or list of lists for multiple (in NZTM)
site_1_coord = np.array([1749376, 5427530, 0])   # downtown Wellington in NZTM
site_2_coord = np.array([1736455, 5427195, 0])   # South Coast, sea/shore electricity cable location
site_3_coord = np.array([1751064, 5423128, 0])   # Wellington airport
site_4_coord = np.array([1754357, 5445716, 0])   # Porirua CBD north of Ohariu fault
site_5_coord = np.array([1754557, 5445119, 0])   # Porirua CBD south of Ohariu fault
site_6_coord = np.array([1757199, 5434207, 0])   # Petone (Hutt Valley); people and office buildings
site_7_coord = np.array([1759240, 5432111, 0])   # Seaview (Hutt Valley); oil tankers loading/unloading
site_8_coord = np.array([1766726, 5471342, 0])   # Paraparaumu; west coast
site_9_coord = np.array([1758789, 5427418, 0])   # Eastbourne (eastern wellington harbour)
site_10_coord = np.array([1760183, 5410911, 0])   # Turakirae Head
site_11_coord = np.array([1779348, 5415831, 0])   # Lake Ferry (small settlement, flood infrustructure)
site_12_coord = np.array([1789451, 5391086, 0])      # Cape Palliser (marine terraces to compare)
site_13_coord = np.array([1848038, 5429751, 0])     # Flat Point (round out point spacing)

site_coords = np.vstack((site_1_coord, site_2_coord, site_3_coord, site_4_coord, site_5_coord, site_6_coord,
                   site_7_coord, site_8_coord, site_9_coord, site_10_coord, site_11_coord, site_12_coord, site_13_coord))
site_name_list = ["Wellington CBD", "South Coast", "Wellington Airport", "Porirua CBD north", "Porirua CBD south",
                  "Petone", "Seaview", "Paraparaumu", "Eastbourne", "Turakirae Head", "Lake Ferry", "Cape Palliser",
                  "Flat Point"]

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

