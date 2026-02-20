import pickle as pkl
import numpy as np
import geopandas as gpd
import pandas as pd
import cutde.halfspace as HS
import os
from time import time
import h5py as h5
import xarray as xr
from scipy.sparse import csc_array, csr_array, hstack, csr_matrix
import matplotlib.pyplot as plt

"""
This script will take the discretised fault patches, and calculate the Green's functions for each site in the site list.
If the sites listed in the CSV already have a greens function calculated, then the script will skip that site.
"""
os.chdir(os.path.dirname(os.path.abspath(__file__)))

res = 2000
minx, miny, maxx, maxy = 617058, 4158575, 3289272, 7568775  # NZTM bounds for NZ
minx = np.floor(minx / res) * res
miny = np.floor(miny / res) * res

rlons, rlats = np.arange(minx, maxx + res * 10, res * 10), np.arange(miny, maxy + res * 10, res * 10)
rxx, ryy = np.meshgrid(rlons, rlats)
recce_points = np.vstack([rxx.ravel(), ryy.ravel(), np.zeros_like(rxx.ravel())]).T

# Calculates greens functions along coastline at specified interval
# Read in the geojson file from the NSHM inversion solution
version_extension = "_version_0-1S"
# NSHM_directory = "NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MTUy"
steeper_dip, gentler_dip = False, False

# Define whch subduction zone ([_fq_]hikkerm / puysegur)
sz_zone = '_hikkerk'

rake90 = False  # if True, all rakes will be set to 90 degrees (NSHM default, but not our mesh default)

geojson_only = False  # True if you are generating gfs for a subset of sites that you have already prepared
#############################################
gf_type = "sites"

if sz_zone[0] != '_':
    sz_zone = '_' + sz_zone

if steeper_dip and gentler_dip:
    print("Dip modifications are wrong. Only one statement can be True at once. Try again.")
    exit()
elif steeper_dip:
    version_extension += "_steeperdip"
    sz_zone += "_steeperdip"
elif gentler_dip:
    version_extension += "_gentlerdip"
    sz_zone += "_gentlerdip"

if 'hikker' in sz_zone:
    prefix = 'sz'
elif 'puysegur' in sz_zone:
    prefix = 'py'
else:
    print("Please define a valid subduction zone (hikkerm / puysegur).")
    exit()

# Load files
with open(f"discretised{sz_zone}/{prefix}_discretised_dict.pkl",
        "rb") as f:
    discretised_dict = pkl.load(f)

if "_fq_" in sz_zone and version_extension[:3] != "_fq":
    version_extension = "_fq" + version_extension

lons, lats = np.arange(minx, maxx + res, res), np.arange(miny, maxy + res, res)
xx, yy = np.meshgrid(lons, lats)
obs_points = np.vstack([xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())]).T
empty_grid = csr_array(np.zeros_like(xx))

with open(f"discretised{sz_zone}/{prefix}_discretised_dict.pkl",
        "rb") as f:
    discretised_dict = pkl.load(f)

h5_file = f"discretised{sz_zone}/{prefix}_tsunami_gf_dict{'_rake90' if rake90 else ''}.h5"

if not os.path.exists(h5_file):
    with h5.File(h5_file, "w") as gf_h5:
        gf_h5.create_dataset('grid_extent', data=np.array([minx, miny, maxx, maxy, res]))
    poly_list = []
else:
    poly_list = list(h5.File(h5_file, "r").keys())
    poly_list.remove('grid_extent')
for poly in discretised_dict.keys():
    if str(poly) not in poly_list:
        triangles = discretised_dict[poly]["triangles"]
        slip_array = np.zeros([triangles.shape[0], 3])
        for ix, rake in enumerate(discretised_dict[poly]['rake']):
            rake = rake if not rake90 else 90.0
            slip_array[ix, :] = [np.cos(np.radians(rake)), np.sin(np.radians(rake)), 0]
        # disps = HS.disp_free(obs_pts=obs_points, tris=triangles, slips=slip_array, nu=0.25)
        # vert = disps[:, 2].reshape([lats.shape[0], lons.shape[0]])
        # vert = np.where(np.abs(vert) < 1e-3 / 25, 0, vert)  # Zero out very small values (less than 1mm from 25m of slip)
        # vert_sparse = csr_array(vert)
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
        plt.savefig(f'greens_{str(poly)}')
        plt.close()
        with h5.File(h5_file, "r+") as gf_h5:
            gf_h5.create_group(str(poly))
            gf_h5[str(poly)].create_dataset('vertical', data=vert_sparse.data)
            gf_h5[str(poly)].create_dataset('vertical_indices', data=vert_sparse.indices)
            gf_h5[str(poly)].create_dataset('vertical_indptr', data=vert_sparse.indptr)
            gf_h5[str(poly)].create_dataset('shape', data=vert_sparse.shape)
    print(f"Calculating tsunami greens functions for discretised fault {poly}... Mean rake: {np.mean(discretised_dict[poly]['rake']) if not rake90 else 90.0}")
    if np.isnan(np.mean(discretised_dict[poly]['rake'])):
        pass