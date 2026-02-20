import pickle as pkl
import numpy as np
import cutde.halfspace as HS
from shapely.geometry import MultiPoint
import geopandas as gpd
import os
import pandas as pd
from time import time
import h5py as h5
import xarray as xr
from scipy.sparse import csc_array, csr_array, hstack, csr_matrix
import matplotlib.pyplot as plt

# calculates green's functions at points specified in a list (or lists) of coordinates
# can also define site names, otherwise they will be numbered later on. Can make more sites than you use later for
# plotting if they are named.
os.chdir(os.path.dirname(os.path.abspath(__file__)))
############### USER INPUTS #####################
# need to run once for each green's function type (grid, sites, coast points, etc.) but can reuse for different branches
discretise_version = "_CFM"  # Tag for the directory containing the disctretised faults
mesh_version = "_tsunami"

rake90 = False

res = 2000
minx, miny, maxx, maxy = 617058, 4158575, 2385253, 6240875  # NZTM bounds for NZ
minx = np.floor(minx / res) * res
miny = np.floor(miny / res) * res


rlons, rlats = np.arange(minx, maxx + res * 10, res * 10), np.arange(miny, maxy + res * 10, res * 10)
rxx, ryy = np.meshgrid(rlons, rlats)
recce_points = np.vstack([rxx.ravel(), ryy.ravel(), np.zeros_like(rxx.ravel())]).T

lons, lats = np.arange(minx, maxx + res, res), np.arange(miny, maxy + res, res)
xx, yy = np.meshgrid(lons, lats)
empty_grid = csr_array(np.zeros_like(xx))
full_grid = csr_array(np.zeros_like(xx))
obs_points = np.vstack([xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())]).T

geojson_only = False  # True if you are generating gfs for a subset of sites that you have already prepared
#########################
gf_type = "sites"

if discretise_version[0] != '_':
    discretise_version = '_' + discretise_version


# Load pre made_greens_functions
gf_h5_file = f"discretised{discretise_version}/crustal_tsunami_gf_dict{'_rake90' if rake90 else ''}.h5"
if os.path.exists(gf_h5_file):
    os.remove(gf_h5_file)

if not os.path.exists(gf_h5_file):
    with h5.File(gf_h5_file, "w") as gf_h5:
        gf_h5.create_dataset('grid_extent', data=np.array([minx, miny, maxx, maxy, res]))
    poly_list = []
else:
    poly_list = list(h5.File(gf_h5_file, "r").keys())
    poly_list.remove('grid_extent')

# load files
with open(f"discretised{discretise_version}/crustal_discretised_dict.pkl", "rb") as f:
    discretised_dict = pkl.load(f)

for fault_id in discretised_dict.keys():
# for fault_id in [82]:
    triangles = discretised_dict[fault_id]["triangles"]
    rake = discretised_dict[fault_id]["rake"]

    begin = time()

    vertices = triangles.reshape(triangles.shape[0] * triangles.shape[1], 3)
    vertex_multipoint = MultiPoint(vertices)

    zero_slip_array = np.zeros((triangles.shape[0],))
    ones_slip_array = np.ones((triangles.shape[0],))

    slip_array = np.ascontiguousarray(np.vstack([ones_slip_array, ones_slip_array, zero_slip_array]).T) * np.array([np.cos(np.radians(rake)), np.sin(np.radians(rake)), 1])

    # Calculate displacements for each fault
    vert_sparse = empty_grid.copy()
    disps = HS.disp_free(obs_pts=recce_points, tris=triangles, slips=slip_array, nu=0.25)
    rvert = disps[:, 2].reshape([rlats.shape[0], rlons.shape[0]])
    rvert = np.where(np.abs(rvert) < 1e-3, 0, rvert)  # Zero out very small values (less than 1mm)
    y_search, x_search = np.where(rvert)
    if x_search.shape[0] > 0:
        xmin = x_search.min() - 1 if x_search.min() > 0 else 0
        xmax = x_search.max() + 1 if x_search.max() < rlons.shape[0] - 1 else x_search.max()
        ymin = y_search.min() - 1 if y_search.min() > 0 else 0
        ymax = y_search.max() + 1 if y_search.max() < rlats.shape[0] - 1 else y_search.max()
        xmin, xmax, ymin, ymax = np.where(lons==rlons[xmin])[0][0], np.where(lons==rlons[xmax])[0][0], np.where(lats==rlats[ymin])[0][0], np.where(lats==rlats[ymax])[0][0]

        fault_lons, fault_lats = np.arange(lons[xmin], lons[xmax] + res, res), np.arange(lats[ymin], lats[ymax] + res, res)
    else:
        fault_lons, fault_lats = np.arange(minx, maxx + res, res), np.arange(miny, maxy + res, res)
        xmin, xmax, ymin, ymax = 0, fault_lons.shape[0], 0, fault_lats.shape[0]

    xx, yy = np.meshgrid(fault_lons, fault_lats)
    obs_points = np.vstack([xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())]).T


    disps = HS.disp_free(obs_pts=obs_points, tris=triangles, slips=slip_array, nu=0.25)
    vert = disps[:, 2].reshape([fault_lats.shape[0], fault_lons.shape[0]])
    vert = np.where(np.abs(vert) < 1e-3, 0, vert)  # Zero out very small values (less than 1mm)
    grid_sparse = csr_array(vert)
    vert_sparse[ymin:ymax + 1, xmin:xmax + 1] = vert
    plt.imshow(vert_sparse.toarray(), extent=(lons[0], lons[-1], lats[0], lats[-1]), vmin=-0.01, vmax=0.01, cmap='RdBu')
    plt.colorbar()
    plt.savefig(f'greens_{str(fault_id)}')
    plt.close()
    with h5.File(gf_h5_file, "r+") as gf_h5:
        gf_h5.create_group(str(fault_id))
        gf_h5[str(fault_id)].create_dataset('vertical', data=vert_sparse.data)
        gf_h5[str(fault_id)].create_dataset('vertical_indices', data=vert_sparse.indices)
        gf_h5[str(fault_id)].create_dataset('vertical_indptr', data=vert_sparse.indptr)
        gf_h5[str(fault_id)].create_dataset('shape', data=vert_sparse.shape)

    full_grid += vert_sparse
    if fault_id % 1 == 0:
        print(f'discretised dict {fault_id} of {len(discretised_dict.keys())} done in {time() - begin:.2f} seconds ({triangles.shape[0]} triangles per patch)    ', end='\r')
print('')

plt.imshow(full_grid.toarray(), extent=(lons[0], lons[-1], lats[0], lats[-1]), vmin=-np.percentile(np.abs(full_grid.data), 95), vmax=np.percentile(np.abs(full_grid.data), 95), cmap='RdBu')
plt.colorbar()
plt.show()
print('')
# # This geojson file will be used to control the sites of the inversion
# gdf = gpd.GeoDataFrame(sites_df, geometry=gpd.points_from_xy(sites_df.Lon, sites_df.Lat), crs='EPSG:2193')
# gdf.to_file(f"discretised{discretise_version}/crustal_site_locations{mesh_version}.geojson", driver="GeoJSON")

# print(f"\ndiscretised{discretise_version}/crustal_site_locations{mesh_version} Complete!")