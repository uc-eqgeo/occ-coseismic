import pickle as pkl
import numpy as np
import cutde.halfspace as HS
from shapely.geometry import MultiPoint
import rasterio.mask
from array_operations import write_tiff, write_gmt_grd

### USER INPUTS ###
mesh_version = "_Model2"
steeper_dip, gentler_dip = False, False
cell_size = 4000  # in meters
x, y = 1760934, 5431096  # central location of grid; Seaview
buffer = 12.e4  # in meters (area around w ellington to calculate displacements)

#########################
gf_type = "grid"
if steeper_dip == True and gentler_dip == False:
    extension2 = "_steeperdip"
elif gentler_dip == True and steeper_dip == False:
    extension2 = "_gentlerdip"
elif gentler_dip == False and steeper_dip == False:
    extension2 = ""
else:
    print("Dip modifications are wrong. Only one statement can be True at once. Try again.")
    exit()

# load files
with open(f"out_files{mesh_version}/crustal_discretized_dict.pkl", "rb") as f:
    discretized_dict = pkl.load(f)

# Maximum distance from any patch that ruptures to calculate displacement
#buffer_size = 1.e5  # in meters

# Grid of x and y to calculate sea surface displacements at
x_data = np.arange(round(x-buffer, -3), round(x+buffer, -3), cell_size)
y_data = np.arange(round(y-buffer, -3), round(y+buffer, -3), cell_size)

xmesh, ymesh = np.meshgrid(x_data, y_data)
# xmesh is organized by grid rows (one item is one row). Each item is a duplicate of the x_data array.
# ymesh is organized by rows too (one item is one row). Each value in the item is the same value (the y_data value).
points_x_test = xmesh.flatten()
points_y_test = ymesh.flatten()
pts_test = np.vstack((points_x_test, points_y_test, points_x_test * 0.)).T

# this just numbers the grid points to be consistent with the other named site files. At the moment it's not used.
# Later on it becomes important to keep track of how the grid is reshaped into a list to make sure the point number
# matches the grid cell.
site_name_list = list(range(len(x_data) * len(y_data)))

# calcuates gfs over a buffer instead of the whole area
# will matter if calculating gfs at higher resolution
# grid of ones to mask (so that you don't calculate disps over whole grid)
#test_disps = np.ones((y_data.size, x_data.size))
# Store grid of ones as geotiff
#write_tiff(f"out_files/{extension1}{extension2}/for_mask.tif", x_data, y_data, test_disps)

#make empty greens funcitons dictionary
gf_dict = {}

# calculates greens for 1 m slip on each fault section.
counter = 0
for fault_id in discretized_dict.keys():
    counter += 1
    triangles = discretized_dict[fault_id]["triangles"]
    rake = discretized_dict[fault_id]["rake"]

    # calculate greens functions for each fault section/fault id
    # unnecessary if statement but it should catch bigger errors at the plot stage
    # if len(triangles) > 0:

   ################### chat with Andy about using a mask vs just all the grid points.
    #vertices = triangles.reshape(triangles.shape[0] * triangles.shape[1], 3)
    #vertex_multipoint = MultiPoint(vertices)
    #outline = vertex_multipoint.convex_hull.buffer(buffer_size, cap_style=2, join_style=2)

    #test_dset = rasterio.open(f"out_files/{extension1}{extension2}/for_mask.tif")
    #mask = rasterio.mask.mask(test_dset, [outline])[0][0]
    #mask_loc = np.where(mask == 1)

    #xpoints = xmesh[mask_loc]
    #ypoints = ymesh[mask_loc]

    #pts = np.vstack((xpoints, ypoints, xpoints * 0.)).T
    pts = pts_test

    zero_slip_array = np.zeros((triangles.shape[0],))
    ones_slip_array = np.ones((triangles.shape[0],))

    dip_slip_array = np.vstack([zero_slip_array, ones_slip_array, zero_slip_array]).T
    strike_slip_array = np.vstack([ones_slip_array, zero_slip_array, zero_slip_array]).T

    # should I just be using the grid point here instead of the xmesh[mask_loc] and ymesh[mask_loc]?
    disps_ss = HS.disp_free(obs_pts=pts, tris=triangles, slips=strike_slip_array, nu=0.25)
    disps_ds = HS.disp_free(obs_pts=pts, tris=triangles, slips=dip_slip_array, nu=0.25)

    #out_data_ss = np.zeros(mask.shape)
    #out_data_ss[mask_loc] = disps_ss[:, -1]

    #out_data_ds = np.zeros(mask.shape)
    #out_data_ds[mask_loc] = disps_ds[:, -1]

    # disp_dict = {"ss": out_data_ss, "ds": out_data_ds, "rake": rake, "site_coords": pts, "site_name_list": site_name_list}
    disp_dict = {"ss": disps_ss[:, -1], "ds": disps_ds[:, -1], "rake": rake, "site_coords": pts,
                 "site_name_list": site_name_list, "x_data": x_data, "y_data": y_data}

    gf_dict[fault_id] = disp_dict
    if counter % 10 == 0:
        print(f"discretized dict {counter} of {len(discretized_dict.keys())}")
    #print(f"discretized dict {fault_id}")
    # else:
    #     out_data_ss = np.zeros((y_data.size, x_data.size))
    #     out_data_ds = np.zeros((y_data.size, x_data.size))
    #     disp_dict = {"ss": out_data_ss, "ds": out_data_ds, "rake": "NaN"}
    #     gf_dict[i] = disp_dict
    #     print(f"discretised dict {i}")


with open(f"out_files{mesh_version}/crustal_gf_dict_{gf_type}.pkl", "wb") as f:
    pkl.dump(gf_dict, f)
