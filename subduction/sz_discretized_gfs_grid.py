import pickle as pkl
import numpy as np
import cutde.halfspace as HS
from shapely.geometry import MultiPoint
import rasterio.mask
from array_operations import write_tiff, write_gmt_grd
import os

##### USER INPUTS #####
version_extension = "_v1_gentlerdip"
cell_size = 4000            # in meters
x, y = 1760934, 5431096     # central location of grid; Seaview
buffer_size = 12.e4         # in meters (area around wellington to calculate displacements)

steeper_dip, gentler_dip = True, False

#######################
gf_type = "grid"

if steeper_dip and gentler_dip:
    print("Dip modifications are wrong. Only one statement can be True at once. Try again.")
    exit()

# load files: open discretized dict of subduction interface to calculate grenns functions over
with open(f"out_files{version_extension}/sz_discretised_dict.pkl", "rb") as f:
    discretised_dict = pkl.load(f)

# Grid of x and y to calculate sea surface displacements at
x_data = np.arange(round(x-buffer_size, -3), round(x+buffer_size, -3), cell_size)
y_data = np.arange(round(y-buffer_size, -3), round(y+buffer_size, -3), cell_size)

xmesh, ymesh = np.meshgrid(x_data, y_data)
points_x_test = xmesh.flatten()
points_y_test = ymesh.flatten()
pts_test = np.vstack((points_x_test, points_y_test, points_x_test * 0.)).T


# this just numbers the grid points to be consistent with the other named site files. At the moment it's not used.
# Later on it becomes important to keep track of how the grid is reshaped into a list to make sure the point number
# matches the grid cell.
site_name_list = list(range(len(x_data) * len(y_data)))

gf_dict = {}

for fault_id in discretised_dict.keys():
    triangles = discretised_dict[fault_id]["triangles"]
    rake = discretised_dict[fault_id]["rake"]

    pts = pts_test

    zero_slip_array = np.zeros((triangles.shape[0],))
    ones_slip_array = np.ones((triangles.shape[0],))

    dip_slip_array = np.vstack([zero_slip_array, ones_slip_array, zero_slip_array]).T
    strike_slip_array = np.vstack([ones_slip_array, zero_slip_array, zero_slip_array]).T

    disps_ss = HS.disp_free(obs_pts=pts, tris=triangles, slips=strike_slip_array, nu=0.25)
    disps_ds = HS.disp_free(obs_pts=pts, tris=triangles, slips=dip_slip_array, nu=0.25)


    disp_dict = {"ss": disps_ss[:, -1], "ds": disps_ds[:, -1], "rake": rake, "site_coords": pts,
                 "site_name_list": site_name_list, "x_data": x_data, "y_data": y_data}

    gf_dict[fault_id] = disp_dict
    if fault_id % 10 == 0:
        print(f'discretized dict {fault_id} of {len(discretised_dict.keys())}')


with open(f"out_files{version_extension}/sz_gf_dict_{gf_type}.pkl", "wb") as f:
    pkl.dump(gf_dict, f)

