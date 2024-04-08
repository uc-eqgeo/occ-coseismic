import pickle as pkl
import numpy as np
try:
    import cutde.halfspace as HS
except:
    print('No cutde')
# from shapely.geometry import MultiPoint
# import rasterio.mask
# from array_operations import write_tiff, write_gmt_grd
# import os
from time import time
from timeit import timeit

##### USER INPUTS #####
version_extension = "_vtesting"
cell_size = 4000            # in meters
x, y = 1760934, 5431096     # central location of grid; Seaview
buffer_size = 12.e4         # in meters (area around wellington to calculate displacements)
#x, y = 1625083, 5430914     # central location of grid; Center of New Zealand Monument

steeper_dip, gentler_dip = False, False

#######################
gf_type = "grid"

if steeper_dip and gentler_dip:
    print("Dip modifications are wrong. Only one statement can be True at once. Try again.")
    exit()

# load files: open discretized dict of subduction interface to calculate grenns functions over
with open(f"out_files{version_extension}/sz_discretised_dict.pkl", "rb") as f:
    discretised_dict = pkl.load(f)

# Grid of x and y to calculate sea surface displacements at
x_data = np.arange(round(x - buffer_size, -3), round(x + buffer_size, -3), cell_size)
y_data = np.arange(round(y - buffer_size, -3), round(y + buffer_size, -3), cell_size)

xmesh, ymesh = np.meshgrid(x_data, y_data)
points_x_test = xmesh.flatten()
points_y_test = ymesh.flatten()
pts = np.vstack((points_x_test, points_y_test, points_x_test * 0.)).T

# this just numbers the grid points to be consistent with the other named site files. At the moment it's not used.
# Later on it becomes important to keep track of how the grid is reshaped into a list to make sure the point number
# matches the grid cell.
site_name_list = list(range(len(x_data) * len(y_data)))

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

    disps = HS.disp_free(obs_pts=pts, tris=triangles, slips=total_slip_array, nu=0.25)

    # Set rake to 90 so that in future functions total displacement is just equal to DS
    disp_dict = {"ss": disps[:, -1] * 0, "ds": disps[:, -1], "rake": 90, "site_coords": pts,
                 "site_name_list": site_name_list, "x_data": x_data, "y_data": y_data}

    gf_dict[fault_id] = disp_dict
    if fault_id % 1 == 0:
        print(f'discretized dict {fault_id} of {len(discretised_dict.keys())} done in {time() - begin:.2f} seconds ({triangles.shape[0]} triangles per patch)')

with open(f"out_files{version_extension}/sz_gf_dict_{gf_type}.pkl", "wb") as f:
    pkl.dump(gf_dict, f)
