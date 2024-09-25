import geopandas as gpd
import numpy as np
import math
from shapely.geometry import Polygon, Point

def split_cell(cell_dicts, parent_id, max_grid, min_grid, max_id, coastline, faults, fault_buffer, split_factor=2):
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
                    cell_dicts[max_id]['split'] = False
                    cell_dicts[max_id]['write_out'] = True

        while cell_dicts[max_id]['split']:
            cell_dicts, max_id = split_cell(cell_dicts, max_id, max_grid, min_grid, max_id, coastline, faults, fault_buffer, split_factor)

    cell_dicts[parent_id]['split'] = False

    return cell_dicts, max_id


search_type = 'cube'  # 'cube or quad'

# Max grid is the default resulution of the cubetree
max_grid = 3000

# Min grid is the minimum resolution of the cubetree.
# Must be reachable by halving or thirding max_grid 
min_grid = 1000

fault_buffer = 1/3  # This is the extra fault radius around each cell used to decide if the cell is to be divided

if search_type == 'quad':
    split_factor = 2
elif search_type == 'cube':
    split_factor = 3
    
min_grid = int(max_grid / split_factor ** np.ceil(math.log(max_grid / min_grid, split_factor)))

print('Max Resolution:', max_grid)
print('Min Resolution:', min_grid)
print('Fault Buffer:', fault_buffer)

# Based on bottom corner of NZ coastlines file used to create regular grids
lon0, lon1, lat0, lat1 = 1089971.2195, 2100000, 4747966.4462, 6200000

lon0 -= min_grid / 2
lat0 -= min_grid / 2 
lon1 = np.ceil((lon1 - lon0) / max_grid) * max_grid + lon0
lat1 = np.ceil((lat1 - lat0) / max_grid) * max_grid + lat0


# This makes sure that max_grid is the default resolution
length = lon1 - lon0
width = lat1 - lat0
longside = max(length, width)
longside = max_grid * split_factor ** np.ceil(math.log(max(length, width) / max_grid, split_factor))

lat1 = lat0 + longside
lon1 = lon0 + longside

coastline = gpd.read_file('QGIS\\nz-coastlines-and-islands-polygons-topo-1500k.gpkg')
faults = gpd.read_file('C:\\Users\\jmc753\\Work\\NZ_CFM_v1_0\\Shapefiles\\NZ_CFM_v1_0.shp')

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
    cell_dicts, max_id = split_cell(cell_dicts, 0, max_grid, min_grid, max_id, coastline, faults, fault_buffer, split_factor)

id = []
depth = []
res = []
cell_poly = []
split = []

centroids = []

ix = 0
for cell in cell_dicts.keys():
    if cell_dicts[cell]['write_out']:
        id.append(ix)
        depth.append(cell_dicts[cell]['depth'])
        res = cell_dicts[cell]['resolution']
        cell_poly.append(Polygon([(cell_dicts[cell]['lon0'], cell_dicts[cell]['lat0']), 
                                  (cell_dicts[cell]['lon1'], cell_dicts[cell]['lat0']), 
                                  (cell_dicts[cell]['lon1'], cell_dicts[cell]['lat1']), 
                                  (cell_dicts[cell]['lon0'], cell_dicts[cell]['lat1'])]))
        split.append(1 if cell_dicts[cell]['split'] else 0)
        centroids.append(Point((cell_dicts[cell]['lon0'] + cell_dicts[cell]['lon1']) / 2,
                               (cell_dicts[cell]['lat0'] + cell_dicts[cell]['lat1']) / 2))
        ix += 1

print('\nSubsampling Complete')

outtag = f"_{str(max_grid).replace('.', '_')}_{str(min_grid).replace('.', '_')}_buffer_{str(f'{fault_buffer:.02f}').replace('.', '_')}"

grid_points = gpd.GeoDataFrame({'id': id, 'depth': depth, 'resolution': res, 'geometry': cell_poly})
grid_points.set_crs(epsg=2193, inplace=True)
grid_points.to_file(f'sites\\{search_type}_poly{outtag}.geojson', driver='GeoJSON')
print(f"Written sites\\{search_type}_poly{outtag}.geojson")


centroid_gdf = gpd.GeoDataFrame({'id': id, 'depth': depth, 'resolution': res, 'geometry': centroids})
centroid_gdf.set_crs(epsg=2193, inplace=True)
centroid_gdf.to_file(f'sites\\{search_type}_centroids{outtag}.geojson', driver='GeoJSON')
print(f"Written sites\\{search_type}_centroids{outtag}.geojson")
