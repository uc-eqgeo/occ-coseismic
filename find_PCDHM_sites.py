import geopandas as gpd
import numpy as np
import math
from shapely.geometry import Polygon, Point
import pandas as pd

def split_cell(cell_dicts, parent_id, max_grid, min_grid, max_id, coastline, faults, fault_buffer, split_factor=2, hires_coast=False):
    print(f"Splitting {parent_id}", end='\r')
    for child in range(split_factor ** 2):
        max_id += 1
        lon0 = cell_dicts[parent_id]['lon0'] + (child % split_factor) * cell_dicts[parent_id]['resolution'] / split_factor
        lon1 = lon0 + cell_dicts[parent_id]['resolution'] / split_factor
        lat0 = cell_dicts[parent_id]['lat0'] + (child // split_factor) * cell_dicts[parent_id]['resolution'] / split_factor
        lat1 = lat0 + cell_dicts[parent_id]['resolution'] / split_factor
        cell_dict = {'id': max_id, 
                     'parent': parent_id,
                     'depth': cell_dicts[parent_id]['depth'] + 1,
                     'resolution': lon1 - lon0,
                     'lon0': lon0, 'lon1': lon1,
                     'lat0': lat0, 'lat1': lat1,
                     'children': [],
                     'split': True,
                     'write_out': False}

        # Add cell dict to cell_dicts
        cell_dicts[parent_id]['children'].append(max_id)
        cell_dicts[max_id] = cell_dict

        # Checks that the cell is onland
        poly = Polygon([(lon0, lat0), (lon1, lat0), (lon1, lat1), (lon0, lat1)])
        # If cell does not cover the coastline
        if not coastline.within(poly).any():
            # If cell does not intersect the coastline
            if not coastline.intersects(poly).any():
                # If cell is not inside coastline
                if not coastline.contains(poly).any():
                    cell_dicts[max_id]['split'] = False  # Stop splitting
                    cell_dicts[max_id]['write_out'] = False  # Don't bother writing
                    continue  # No need to do any other checks
        # Stop if at highest resolution
        if cell_dicts[max_id]['resolution'] <= min_grid:
            cell_dicts[max_id]['split'] = False  # Stop splitting
            cell_dicts[max_id]['write_out'] = True  # Write out cell
            continue  # No need to do any other checks

        lon0 -= cell_dicts[max_id]['resolution'] * fault_buffer
        lon1 += cell_dicts[max_id]['resolution'] * fault_buffer
        lat0 -= cell_dicts[max_id]['resolution'] * fault_buffer
        lat1 += cell_dicts[max_id]['resolution'] * fault_buffer

        poly = Polygon([(lon0, lat0), (lon1, lat0), (lon1, lat1), (lon0, lat1)])
        # Stop if no faults in the cell
        if not faults.within(poly).any():
            # If cell does not intersect any faults
            if not faults.intersects(poly).any():
                # If cell is acceptable resolution
                if cell_dicts[max_id]['resolution'] <= max_grid:
                    # If an inland cell and hires coast not needed
                    if coastline.contains(poly).any() and hires_coast:
                        cell_dicts[max_id]['split'] = False
                        cell_dicts[max_id]['write_out'] = True

        while cell_dicts[max_id]['split']:
            cell_dicts, max_id = split_cell(cell_dicts, max_id, max_grid, min_grid, max_id, coastline, faults, fault_buffer, split_factor, hires_coast)

    cell_dicts[parent_id]['split'] = False

    return cell_dicts, max_id

search_type = 'cube'  # 'grid', 'cube' or 'quad'

# Resolution
max_grid = 9000  # Default resolution. Min grid will be adjusted to work with this
min_grid = 3000  # Min grid is the highest resolution of the quad or cubetree. Must be reachable by halving or thirding max_grid 

grid_width = 1000e3  # Width of the grid in meters
grid_length = 1500e3 # Length of the grid in meters

# Keep as false to make sure all of coast is covered, and therefore all OCC sites can be queried in datamesh
hires_coast = True # If True, keep splitting cells that intersect the coast
coastal_trim = False  # If True, removes any centroids that are not overland, even if polygon crosses the coast

fault_buffer = 0

if search_type == 'quad':
    split_factor = 2  # How many times to split each cell
    if fault_buffer is None:
        fault_buffer = 1/2  # This is the extra fault radius around each cell used to decide if the cell is to be divided
elif search_type == 'cube':
    split_factor = 3
    if fault_buffer is None:
        fault_buffer = 1/3  # This is the extra fault radius around each cell used to decide if the cell is to be divided
elif search_type == 'grid':
    split_factor = 5
    fault_buffer = 0  # This is the extra fault radius around each cell used to decide if the cell is to be divided
    min_grid = max_grid
    
min_grid = int(max_grid / split_factor ** np.ceil(math.log(max_grid / min_grid, split_factor)))

coastline = gpd.read_file('QGIS\\nz-coastlines-and-islands-polygons-topo-1500k.gpkg')
faults = gpd.read_file('C:\\Users\\jmc753\\Work\\NZ_CFM_v1_0\\Shapefiles\\NZ_CFM_v1_0.shp')

# Ensures the bottom left corner of the grid is based on the coastline file, rounded to 1km
lon0 = np.round(coastline.bounds.minx.min(), -3) - min_grid / 2
lat0 = np.round(coastline.bounds.miny.min(), -3) - min_grid / 2

lon1 = np.ceil(grid_width / max_grid) * max_grid + lon0
lat1 = np.ceil(grid_length / max_grid) * max_grid + lat0

# This makes sure that everything is a square
length = lon1 - lon0
width = lat1 - lat0
longside = max(length, width)
# Increase longside to be compatible with the split factor
split_power = int(np.ceil(math.log(longside / max_grid, split_factor)))
longside = max_grid * split_factor ** split_power

lat1 = lat0 + longside
lon1 = lon0 + longside

print('Search Type:', search_type)
print('Max Resolution:', max_grid)
print('Min Resolution:', min_grid)
print('Fault Buffer:', fault_buffer)
print('Length/Width:', grid_length / 1e3, 'x', grid_width / 1e3, 'km')
print('Split Power:', split_factor ** split_power, f"({split_factor} ** {split_power})")
print('Starting Grid size:', longside / 1e3, 'km')

cell_dicts = {}
cell_dict = {'id': 0, 
             'parent': None, 
             'depth': 0, 
             'resolution': lon1 - lon0, 
             'lon0': lon0, 'lon1': lon1, 
             'lat0': lat0, 'lat1': lat1, 
             'children': [], 
             'split': True,
             'write_out': False}

cell_dicts[0] = cell_dict

if cell_dict['resolution'] <= max_grid:
    cell_dicts[0]['split'] = False
    cell_dicts[0]['write_out'] = True

max_id = 0
ix = 0
while cell_dicts[ix]['split']:
    cell_dicts, max_id = split_cell(cell_dicts, 0, max_grid, min_grid, max_id, coastline, faults, fault_buffer, split_factor, hires_coast)

print('')

id = []
depth = []
res = []
cell_poly = []
split = []

centroids = []
n_cells = len(cell_dicts.keys())

ix = 0
for cell in cell_dicts.keys():
    print(f"Calculating Centroids {ix}/{n_cells}", end='\r')
    if cell_dicts[cell]['write_out']:
        centroid = Point((cell_dicts[cell]['lon0'] + cell_dicts[cell]['lon1']) / 2,
                            (cell_dicts[cell]['lat0'] + cell_dicts[cell]['lat1']) / 2)
        # Drop any centroids that are not overland
        if coastal_trim and not coastline.contains(centroid).any():
            n_cells -= 1
            continue
        id.append(ix)
        depth.append(cell_dicts[cell]['depth'])
        res = cell_dicts[cell]['resolution']
        cell_poly.append(Polygon([(cell_dicts[cell]['lon0'], cell_dicts[cell]['lat0']), 
                                  (cell_dicts[cell]['lon1'], cell_dicts[cell]['lat0']), 
                                  (cell_dicts[cell]['lon1'], cell_dicts[cell]['lat1']), 
                                  (cell_dicts[cell]['lon0'], cell_dicts[cell]['lat1'])]))
        split.append(1 if cell_dicts[cell]['split'] else 0)
        centroids.append(centroid)
        ix += 1
    else:
        n_cells -= 1

print('\nSubsampling Complete')

if search_type == 'grid':
    polyname = f"national_{int(max_grid / 1000)}km_{search_type}_poly"
    centroid_name = f"national_{int(max_grid / 1000)}km_{search_type}"
else:
    outtag = f"_{str(max_grid).replace('.', '_')}_{str(min_grid).replace('.', '_')}_buffer_{str(f'{fault_buffer:.02f}').replace('.', '_')}"
    polyname = f"{search_type}_poly{outtag}"
    centroid_name = f"{search_type}_centroids{outtag}"

grid_points = gpd.GeoDataFrame({'id': id, 'depth': depth, 'resolution': res, 'geometry': cell_poly})
grid_points.set_crs(epsg=2193, inplace=True)
grid_points.to_file(f'sites\\{polyname}.geojson', driver='GeoJSON')
print(f"Written sites\\{polyname}.geojson")


centroid_gdf = gpd.GeoDataFrame({'id': id, 'depth': depth, 'resolution': res, 'geometry': centroids})
centroid_gdf.set_crs(epsg=2193, inplace=True)
centroid_gdf.to_file(f'sites\\{centroid_name}.geojson', driver='GeoJSON')
print(f"Written sites\\{centroid_name}.geojson")

centroid_df = pd.DataFrame(columns=['X', 'Y', 'id'])
centroid_df['X'] = centroid_gdf.geometry.x
centroid_df['Y'] = centroid_gdf.geometry.y
centroid_df['id'] = np.arange(centroid_df.shape[0])
centroid_df.to_csv(f'sites\\{centroid_name}.csv', index=False)
print(f"Written sites\\{centroid_name}.csv")
