import geopandas as gpd
import pandas as pd
import numpy as np
import meshio
from shapely.geometry import Polygon, LineString, Point
import pickle as pkl
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
#### USER INPUT #####
#this can be any working branch, should be the same for all.
NSHM_directory = "NZSHM22_ScaledInversionSolution-QXV0b21hdGlvblRhc2s6MTA3Njk2"

# Define whch subduction zone (hikkerm / puysegur)
sz_zone = 'puysegur'

if not sz_zone in ['hikkerm', 'puysegur']:
    print("Please define a valid subduction zone (hikkerm / puysegur).")
    exit()

if "hikkerm" in sz_zone:
    neighbours_file = '../data/hik_kerk3k_neighbours.txt'
    prefix = 'sz'
else:
    neighbours_file = '../data/puysegur_neighbours.txt'
    prefix = 'py'

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

# De-blobify outputs
deblobify = False

if deblobify:
    sz_zone += "_deblobify"
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

# Read in the geojson file from the NSHM inversion solution
if "hikkerm" in sz_zone:
    traces = gpd.read_file(f"../data/sz_solutions/{NSHM_directory}/ruptures/fault_sections.geojson").to_crs(epsg=2193)
else:
    traces = gpd.read_file(f"../data/puysegur_fault_sections.geojson").to_crs(epsg=2193)

# get rectangle centroid and polygon vertices based on NSHM fault data
all_rectangle_centroids = []
all_rectangle_polygons = []
# make dataframe forrectangle polygon attribtes (for export later)
df1 = pd.DataFrame()

df_rectangle_centroid = pd.DataFrame()

# Turn section traces into rectangular patches using the metadata in the GeoJSON file
for i, trace in traces.iterrows():
    # Convert the trace to a numpy array
    trace_array = np.array(trace.geometry.coords)
    # Convert the depths to metres
    trace_array[:, -1] *= -1000.

    ######## depth sensitivity testing

    initial_patch_height = (trace.LowDepth - trace.UpDepth) * 1000.
    initial_tandip = np.tan(np.radians(trace.DipDeg))
    patch_horizontal_dist = initial_patch_height / initial_tandip

    # set conditions for rectangular patch geometry parameters, which change with dip angle
    # for steeper dip ( fault section depths * 1.85)  # Fix based on bug found by JDE where adjustment is applied later
    # if steeper_dip == True and gentler_dip == False:
    #     low_depth, up_depth = trace.LowDepth * 1.15, trace.UpDepth * 1.15
    #     patch_height = (low_depth - up_depth) * 1000.
    #     trace_centroid = np.array([*trace.geometry.centroid.coords[0], trace.UpDepth * -1000])
    #     dip_angle = np.degrees(np.arctan(patch_height / patch_horizontal_dist))

    # # for gentler dip ( fault section depths * 0.85)
    # elif gentler_dip == True and steeper_dip == False:
    #     low_depth, up_depth = trace.LowDepth * 0.85, trace.UpDepth * 0.85
    #     patch_height = (low_depth - up_depth) * 1000.
    #     trace_centroid = np.array([*trace.geometry.centroid.coords[0], trace.UpDepth * -1000])
    #     dip_angle = np.degrees(np.arctan(patch_height / patch_horizontal_dist))

    # uses geometry from NSHM with no modifications
    #elif gentler_dip == False and steeper_dip == False:
    low_depth, up_depth = trace.LowDepth, trace.UpDepth
    # Calculate the centroid of the trace
    trace_centroid = np.array([*trace.geometry.centroid.coords[0], trace.UpDepth * -1000])
    # Calculate the height of the patch
    patch_height = (trace.LowDepth - trace.UpDepth) * 1000.
    dip_angle = trace.DipDeg

    # write patch attributes to dictionary and add to bottom of data frame
    df2 = pd.DataFrame({'fault_id': [int(trace.FaultID)], 'dip_deg': [dip_angle], 'patch_height_m': [patch_height],
                        'up_depth_km': [up_depth], 'low_depth_km': [low_depth]}, index=[int(trace.FaultID)])
    df1 = pd.concat([df1, df2])

    df2_rectangle_centroid = pd.DataFrame({'fault_id': [trace.FaultID], 'depth': [np.mean([trace.UpDepth, trace.LowDepth])]}, index=[int(trace.FaultID)])
    df_rectangle_centroid = pd.concat([df_rectangle_centroid, df2_rectangle_centroid])
    #######################

    # Calculate the strike of the trace
    trace_strike = trace_array[1] - trace_array[0]
    # Normalise the strike vector
    trace_strike = trace_strike / np.linalg.norm(trace_strike)
    # Calculate the across-strike vector, by rotating the strike vector 90 degrees
    # aka dip direction vector (x, y), Normalized.
    across_strike = np.matmul(np.array([[0, 1], [-1, 0]]), trace_strike[:-1])
    #across_strike = np.matmul(np.array([[0, 1], [-1, 0]]), trace_strike)

    # Calculate the down-dip vector by incorporating the dip angle
    cosdip = np.cos(np.radians(dip_angle))
    sindip = np.sin(np.radians(dip_angle))
    down_dip = np.array([cosdip * across_strike[0], cosdip * across_strike[1], - 1 * sindip])

    # Calculate the centroid of the patch
    rectangle_centroid = trace_centroid + (patch_height / sindip) / 2 * down_dip

    # Calculate the corners of the patch and make a shapely polygon
    dd1 = trace_array[1] + (patch_height / sindip) * down_dip
    dd2 = trace_array[0] + (patch_height / sindip) * down_dip
    rectangle_polygon = Polygon(np.vstack([trace_array, dd1, dd2]))

    # Append the patch centroid and polygon to lists
    all_rectangle_centroids.append(rectangle_centroid)
    all_rectangle_polygons.append(rectangle_polygon)

# make directory for outputs
os.makedirs(f"discretised_{sz_zone}", exist_ok=True)


# write rectangle centroid and rectangle polygons to geojson
all_rectangle_centroids = np.array(all_rectangle_centroids)
all_rectangle_centroids_gs = gpd.GeoSeries([Point(centroid) for centroid in all_rectangle_centroids], crs=2193)
all_rectangle_centroids_gdf = gpd.GeoDataFrame(df_rectangle_centroid, geometry=all_rectangle_centroids_gs.geometry, crs=2193)
all_rectangle_centroids_gdf.to_file(
    f"discretised_{sz_zone}/{prefix}_all_rectangle_centroids.geojson", driver="GeoJSON")

all_rectangle_outline_gs = gpd.GeoSeries(all_rectangle_polygons, crs=2193)
all_rectangle_outline_gdf = gpd.GeoDataFrame(df1, geometry=all_rectangle_outline_gs.geometry, crs=2193)
all_rectangle_outline_gdf.to_file(f"discretised_{sz_zone}/{prefix}_all_rectangle_outlines.geojson", driver="GeoJSON")

#####
# read in triangle mesh and add the patch centroids as points
if "hikkerm" in sz_zone:
    mesh = meshio.read(f"../data/hik_kerk3k_with_rake.vtk")
else:
    mesh = meshio.read(f"../data/puysegur.vtk")

mesh_rake = mesh.cell_data["rake"][0]
mesh_triangles = mesh.cells_dict["triangle"]    # indices of vertices that make up triangles
mesh_vertices = mesh.points              # xyz of vertices
mesh_centroids = np.mean(mesh_vertices[mesh_triangles], axis=1)

# # multiply depths by steeper/gentler constant
# if steeper_dip == True:
#     mesh_vertices = mesh.points * [1, 1, 1.15]
# elif gentler_dip == True:
#     mesh_vertices = mesh.points * [1, 1, 0.85]
# else:
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

# Find the rake for each rectangle patch by finding closest triangle centroid to the rectangle centroid
rectangle_rake = []
for rectangle_centroid in all_rectangle_centroids:
    distances = np.linalg.norm(rectangle_centroid - mesh_centroids, axis=1)
    if distances.min() < 2.e4:
        closest_triangle = np.argmin(distances)
        rectangle_rake.append(mesh_rake[closest_triangle])
    else:
        rectangle_rake.append(np.nan)
rectangle_rake = np.array(rectangle_rake)

# find the closest rectangle to each triangle centroid
closest_rectangles = []
for ix, triangle_centroid in enumerate(triangle_centroids):
    distances = np.linalg.norm(all_rectangle_centroids - triangle_centroid, axis=1)
    if distances.min() < 2.2e4:
        nearest = np.where(distances < 2.2e4)[0]
        vsep = all_rectangle_centroids[:, 2] - triangle_centroid[2]
        abs_sep = abs(vsep)
        if np.sum(distances < 2.2e4) == 1 or not deblobify:  # If only one option, use that option. Alternatively, use geographically nearest if using original, blobify method
            closest_rectangle = np.argmin(distances)
        else:
            nearest2 = nearest[np.argsort(abs_sep[nearest])[:2]]
            if np.diff(abs_sep[nearest2])[0] < 1.5e3:  # If nearest 2 are within 1.5 km vertical seperation, use the geographically nearest
                closest_rectangle = nearest2[np.argmin(distances[nearest2])]
            else:
                closest_rectangle = nearest2[np.argmin(abs_sep[nearest2])]  # If nearest 2 are > 1 km vertical seperation, use the structurally nearest
        closest_rectangles.append(closest_rectangle)
    else:
        closest_rectangles.append(-1)

# Manually correct some triangles
if os.path.exists('../data/mesh_corrections.csv') and "hikkerm" in sz_zone:
    print('Manually correcting some triangles')
    with open('../data/mesh_corrections.csv', 'r') as f:
        corrections = [[int(val) for val in line.strip().split(',')] for line in f.readlines()]

    for tri, closest_rectangle in corrections:
        closest_rectangles[tri] = closest_rectangle

# # Prevent isolated triangles
if os.path.exists('../data/hik_kerk3k_with_rake_neighbours.txt') and sz_zone == 'hikkerm':
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
            nearest_ix = np.where(distances < 2.2e4)[0]
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
elif deblobify:
    print('No neighbour file found - final output may include isolated triangles')

closest_rectangles = np.array(closest_rectangles)

# Create polygons from triangles
discretised_polygons = []
discretised_dict = {}
for index in traces.index:
    triangles_locs = np.where(closest_rectangles == index)[0]
    triangles = ordered_triangle_vertex_arrays[triangles_locs]
    if steeper_dip:
        triangles[:, :, 2] *= 1.15
    if gentler_dip:
        triangles[:, :, 2] *= 0.85

    # make dictionary of triangles that go with each polygon
    triangle_polygons = [Polygon(triangle) for triangle in triangles]
    dissolved_triangle_polygons = gpd.GeoSeries(triangle_polygons).unary_union
    discretised_polygons.append(dissolved_triangle_polygons)
    discretised_dict[index] = {"triangles": triangles, "rake": mesh_rake[triangles_locs], "triangle_indices": triangles_locs}  


# Create a geodataframe and geospon file from the polygons
gdf = gpd.GeoDataFrame({"rake": rectangle_rake, "geometry": discretised_polygons, "fault_id": traces.index})
gdf.crs = traces.crs
gdf.to_file(f"discretised_{sz_zone}/{prefix}_discretised_polygons.geojson", driver="GeoJSON")

pkl.dump(discretised_dict, open(f"discretised_{sz_zone}/{prefix}_discretised_dict.pkl", "wb"))
