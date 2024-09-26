import geopandas as gpd
import pandas as pd
import numpy as np
import meshio
from shapely.geometry import Polygon, LineString, Point
import pickle as pkl
import os

"""
This script will discretise the subduction zone into patches based on the fake quakes fault geometry.
"""

#### USER INPUT #####
# Define whch subduction zone (hikkerk / puysegur)
sz_zone = 'hikkerk'

if not sz_zone in ['hikkerk', 'puysegur']:
    print("Please define a valid subduction zone (hikkerk / puysegur).")
    exit()

#Sensitivity testing for subduction interface depth
steeper_dip = False
gentler_dip = False

if steeper_dip and gentler_dip:
    print("Dip modifications are wrong. Only one statement can be True at once. Try again.")
    exit()
elif steeper_dip:
    sz_zone += "_steeperdip"
elif gentler_dip:
    sz_zone += "_gentlerdip"

#######################
def cross_3d(a, b):
    """
    Calculates cross product of two 3-dimensional vectors.
    """
    x = ((a[1] * b[2]) - (a[2] * b[1]))
    y = ((a[2] * b[0]) - (a[0] * b[2]))
    z = ((a[0] * b[1]) - (a[1] * b[0]))
    return np.array([x, y, z])

def check_triangle_normal(triangle_vertices):
    """ check if the normal vector is positive or negative. If negative, flip the order of the vertices.
    triangle_vertices is a 3x3 array of the vertices of the triangle (3 vertices, each with xyz)"""

    vector_a = triangle_vertices[1] - triangle_vertices[0]
    vector_b = triangle_vertices[1] - triangle_vertices[2]
    cross_a_b_vector = cross_3d(vector_a, vector_b)
    # Ensure that normal always points down (for proper strike convention), if not, flip the order of the vertices
    # this results in greens functions where dip slip is reverse motion, which is what we want (positive rake).
    if cross_a_b_vector[-1] > 0:
        ordered_triangle_vertices = triangle_vertices[-1::-1]
    else:
        ordered_triangle_vertices = triangle_vertices
    return ordered_triangle_vertices
####################
if sz_zone == 'hikkerk':
    prefix = 'sz'
else:
    prefix = 'py'

# %% Turn vtk file into GeoJSON file
# Read in the VTK file that contains the fault patches used by fakequakes. Currently assumes rectangular fault patches
tiles = meshio.read(f"../data/{prefix}_tiles.vtk")

assert 'quad' in tiles.cells[0].type, "Only quad cells are supported at the moment"

n_patches = tiles.cells[0].data.shape[0]
# get rectangle centroid and polygon vertices based on NSHM fault data
all_rectangle_centroids = []
all_rectangle_polygons = []
trace_lengths = []

# make dataframe for rectangle polygon attribtes (for export later)
df1 = pd.DataFrame(columns=['fault_id', 'dip_deg', 'patch_height_m', 'up_depth_km', 'low_depth_km'])
df_rectangle_centroid = pd.DataFrame(columns=['fault_id', 'depth'])

# Turn section traces into rectangular patches using the metadata in the GeoJSON file
for i in range(n_patches):
    patch_points = tiles.points[tiles.cells[0].data[i]]
    patch_points[:, 2] /= -1000  # Convert to km, positive down
    patch_centroid = np.mean(patch_points, axis=0)

    low_depth, up_depth = patch_points[:, 2].min(), patch_points[:, 2].max()

    ######## depth sensitivity testing
    if steeper_dip:
        low_depth, up_depth = patch_points[:, 2].min() * 1.15, patch_points[:, 2].max() * 1.15
    if gentler_dip:
        low_depth, up_depth = patch_points[:, 2].min() * 0.85, patch_points[:, 2].max() * 0.85
    
    patch_height = (low_depth - up_depth) * 1000.
    patch_horizontal_dist = np.sqrt(np.sum((patch_points[0, :2] - patch_points[1, :2]) ** 2))  # Horizontal distance in to down dip direction
    trace_lengths.append(np.sqrt(np.sum((patch_points[0, :2] - patch_points[3, :2]) ** 2)))  # Horizontal distance in along strike
    dip_angle = np.degrees(np.arctan(np.abs(patch_height) / patch_horizontal_dist))

    # write patch attributes to dictionary and add to bottom of data frame
    df1.loc[i] = {'fault_id': i, 'dip_deg': dip_angle, 'patch_height_m': patch_height, 'up_depth_km': up_depth, 'low_depth_km': low_depth}
    df_rectangle_centroid.loc[i] = {'fault_id': i, 'depth': patch_centroid[2]}

    all_rectangle_centroids.append(patch_centroid)
    all_rectangle_polygons.append(Polygon(patch_points[:, :2]))

trace_length = np.mean(trace_lengths)

# make directory for outputs
if not os.path.exists(f"discretised_fq_{sz_zone}"):
    os.mkdir(f"discretised_fq_{sz_zone}")


# write rectangle centroid and rectangle polygons to geojson
all_rectangle_centroids = np.array(all_rectangle_centroids)
all_rectangle_centroids_gs = gpd.GeoSeries([Point(centroid) for centroid in all_rectangle_centroids], crs=2193)
all_rectangle_centroids_gdf = gpd.GeoDataFrame(df_rectangle_centroid, geometry=all_rectangle_centroids_gs.geometry, crs=2193)
all_rectangle_centroids_gdf.to_file(
    f"discretised_fq_{sz_zone}/{prefix}_all_rectangle_centroids.geojson", driver="GeoJSON")

all_rectangle_outline_gs = gpd.GeoSeries(all_rectangle_polygons, crs=2193)
all_rectangle_outline_gdf = gpd.GeoDataFrame(df1, geometry=all_rectangle_outline_gs.geometry, crs=2193)
all_rectangle_outline_gdf.to_file(f"discretised_fq_{sz_zone}/{prefix}_all_rectangle_outlines.geojson", driver="GeoJSON")

# %%
#####
# read in triangle mesh and add the patch centroids as points
if sz_zone == 'hikkerk':
    mesh = meshio.read(f"../data/hik_kerk3k_with_rake.vtk")
else:
    mesh = meshio.read(f"../data/puysegur.vtk")

mesh_rake = mesh.cell_data["rake"][0]
mesh_triangles = mesh.cells_dict["triangle"]    # indices of vertices that make up triangles
mesh_vertices = mesh.points              # xyz of vertices
mesh_centroids = np.mean(mesh_vertices[mesh_triangles], axis=1)
mesh_centroids[:, 2] /= -1000  # Convert to km, positive down

# multiply depths by steeper/gentler constant
if steeper_dip == True:
    mesh_vertices = mesh.points * [1, 1, 1.15]
elif gentler_dip == True:
    mesh_vertices = mesh.points * [1, 1, 0.85]
else:
    mesh_vertices = mesh.points # xyz of vertices in mesh. indexed.
# array of 3 xyz arrays. (three sets of vertices to make a triangle)
triangle_vertex_arrays = mesh_vertices[mesh_triangles]

###### ensuring correct strike convention for accurate displacment calculations later
    # calculate the normal vector to each triangle. If negative, reverse ordering of triangle of vertices.
ordered_triangle_vertex_arrays = []
for triangle in triangle_vertex_arrays:
    ordered_triangle_array = check_triangle_normal(triangle)
    ordered_triangle_vertex_arrays.append(ordered_triangle_array)
ordered_triangle_vertex_arrays = np.array(ordered_triangle_vertex_arrays)

triangle_centroids = np.mean(ordered_triangle_vertex_arrays, axis=1)   # three part arrray. mean of x, mean of y, mean of z.
triangle_centroids[:, 2] /= -1000  # Convert to km, positive down

# Find the rake for each rectangle patch by finding closest triangle centroid to the rectangle centroid
rectangle_rake = []
for rectangle_centroid in all_rectangle_centroids:
    distances = np.linalg.norm(rectangle_centroid - mesh_centroids, axis=1)
    if distances.min() < trace_length: # Find all triangles within 1 patch width of the patches
        closest_triangle = np.argmin(distances)
        rectangle_rake.append(mesh_rake[closest_triangle])
    else:
        rectangle_rake.append(np.nan)
rectangle_rake = np.array(rectangle_rake)
# %%
# find the closest rectangle to each triangle centroid
closest_rectangles = []
search_radius = trace_length * 0.75  # 75% of the average trace length

for ix, triangle_centroid in enumerate(triangle_centroids):
    distances = np.linalg.norm(all_rectangle_centroids - triangle_centroid, axis=1)
    if distances.min() < trace_length: # Find all triangles within 1 patch width of the patches
        nearest_ix = np.where(distances < trace_length)[0]
        nearest = nearest_ix[np.argsort(distances[nearest_ix])]
        closest_rectangle = nearest[0]
        # if len(nearest) == 1:
        #     closest_rectangle = nearest[0]
        # else:
        #     vsep = all_rectangle_centroids[:, 2] - triangle_centroid[2]
        #     abs_sep = abs(vsep)
        #     if np.sum(distances < search_radius) == 1:  # If only one option, use that option. Alternatively, use geographically nearest if using original, blobify method
        #         closest_rectangle = np.argmin(distances)
        #     else:
        #         nearest2 = nearest[np.argsort(abs_sep[nearest])[:2]]
        #         if np.diff(abs_sep[nearest2])[0] < 1.5e3:  # If nearest 2 are within 1.5 km vertical seperation, use the geographically nearest
        #             closest_rectangle = nearest2[np.argmin(distances[nearest2])]
        #         else:
        #             closest_rectangle = nearest2[np.argmin(abs_sep[nearest2])]  # If nearest 2 are > 1.5 km vertical seperation, use the structurally nearest
        closest_rectangles.append(closest_rectangle)
    else:
        closest_rectangles.append(-1)

# # Prevent isolated triangles
if os.path.exists('../data/hik_kerk3k_with_rake_neighbours.txt') and sz_zone == 'hikkerk':
    print('Removing isolated triangles')
    with open('../data/hik_kerk3k_with_rake_neighbours.txt', 'r') as f:
        neighbours = [[int(tri) for tri in line.strip().split()] for line in f.readlines()]

    clear_isolated = True
    clear_run = 0

    while clear_isolated:
        # Count isolated
        n_isolated = 0
        for tri in range(len(closest_rectangles)):
            rect = closest_rectangles[tri]
            if rect != -1:
                if len(neighbours[tri]) == 3:
                    neigh = [closest_rectangles[neigh] for neigh in neighbours[tri] if closest_rectangles[neigh] != -1]
                    if sum([ix != rect for ix in neigh]) == 3:
                        n_isolated += 1

        # First search for triangles entirely isolated by 1 patch - replace with that patch
        isolated_triangles = []
        for tri in range(len(closest_rectangles)):
            rect = closest_rectangles[tri]
            if rect != -1:
                if len(neighbours[tri]) == 3:
                    neigh = [closest_rectangles[neigh] for neigh in neighbours[tri] if closest_rectangles[neigh] != -1]
                    if sum([ix != rect for ix in neigh]) == 3 and np.unique(neigh).shape[0] == 1:
                        isolated_triangles.append(tri)
        for tri in isolated_triangles:
            rect = closest_rectangles[tri]
            neigh = [closest_rectangles[neigh] for neigh in neighbours[tri] if closest_rectangles[neigh] != -1]
            closest_rectangles[tri] = neigh[0]
        
        # Second search for triangles entirely isolated by 2 patches - replace most patch on 2 sides
        isolated_triangles = []
        for tri in range(len(closest_rectangles)):
            rect = closest_rectangles[tri]
            if rect != -1:
                if len(neighbours[tri]) == 3:
                    neigh = [closest_rectangles[neigh] for neigh in neighbours[tri] if closest_rectangles[neigh] != -1]
                    if sum([ix != rect for ix in neigh]) == 3 and np.unique(neigh).shape[0] == 2:
                        isolated_triangles.append(tri)
        for tri in isolated_triangles:
            rect = closest_rectangles[tri]
            neigh = [closest_rectangles[neigh] for neigh in neighbours[tri] if closest_rectangles[neigh] != -1]
            closest_rectangles[tri] = np.median(neigh).astype(int)
        
        # Third search for triangles isolated by 3 patches - reassign to second nearest patch
        isolated_triangles = []
        for tri in range(len(closest_rectangles)):
            rect = closest_rectangles[tri]
            if rect != -1:
                if len(neighbours[tri]) == 3:
                    neigh = [closest_rectangles[neigh] for neigh in neighbours[tri] if closest_rectangles[neigh] != -1]
                    if sum([ix != rect for ix in neigh]) == 3 and np.unique(neigh).shape[0] == 3:
                        isolated_triangles.append(tri)
        for tri in isolated_triangles:
            distances = np.linalg.norm(all_rectangle_centroids - triangle_centroids[tri], axis=1)
            nearest_ix = np.where(distances < trace_length)[0]
            nearest = nearest_ix[np.argsort(distances[nearest_ix])]
            if len(nearest) > 1:
                closest_rectangles[tri] = nearest[1]
            else:
                closest_rectangles[tri] = nearest[0]

        # Check outcome
        isolated_triangles = []
        for tri in range(len(closest_rectangles)):
            rect = closest_rectangles[tri]
            if rect != -1:
                if len(neighbours[tri]) == 3:
                    neigh = [closest_rectangles[neigh] for neigh in neighbours[tri] if closest_rectangles[neigh] != -1]
                    if sum([ix != rect for ix in neigh]) == 3:
                        isolated_triangles.append(tri)

        clear_run += 1
        if len(isolated_triangles) == 0:
            clear_isolated = False
            print(f'No isolated triangles found after {clear_run} runs')
        elif clear_run > 3:
            clear_isolated = False
            print(f'Failed to remove all isolated triangles after {clear_run} runs limit: {len(isolated_triangles)} remain')
        else:
            print(f'Run {clear_run}: Reduced isolated triangles from {n_isolated} to {len(isolated_triangles)}')

closest_rectangles = np.array(closest_rectangles)
# %%
# Create polygons from triangles
discretised_polygons = []
discretised_dict = {}
for index in range(n_patches):
    triangles_locs = np.where(closest_rectangles == index)[0]
    triangles = ordered_triangle_vertex_arrays[triangles_locs]

    # make dictionary of triangles that go with each polygon
    triangle_polygons = [Polygon(triangle) for triangle in triangles]
    if triangle_polygons:
        dissolved_triangle_polygons = gpd.GeoSeries(triangle_polygons).unary_union
    else:
        dissolved_triangle_polygons = None
    discretised_polygons.append(dissolved_triangle_polygons)
    discretised_dict[index] = {"triangles": triangles, "rake": mesh_rake[triangles_locs], "triangle_indices": triangles_locs}  

# %%
# Create a geodataframe and geojson file from the polygons
gdf = gpd.GeoDataFrame({"rake": rectangle_rake, "geometry": discretised_polygons, "fault_id": np.arange(n_patches)}, crs=2193)
gdf.to_file(f"discretised_fq_{sz_zone}/{prefix}_discretised_polygons.geojson", driver="GeoJSON")

pkl.dump(discretised_dict, open(f"discretised_fq_{sz_zone}/{prefix}_discretised_dict.pkl", "wb"))

