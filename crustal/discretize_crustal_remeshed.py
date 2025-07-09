# %%
import geopandas as gpd
import pandas as pd
import numpy as np
import meshio
from shapely.geometry import Polygon, LineString, Point
import pickle as pkl
from crustal_helper_scripts import read_rake_csv, rake_from_traces, read_combination_csv
import glob
import os

# This script discretises the fault meshes into patches based on the "sheet in the wind" NSHM faults.
# only run once per fault geometry (can reuse for any branch) because all the faults are the same

# outputs (to discretised{Model_extension} directory)
#   crustal_discretised_dict.pkl,
#   crustal_discretised_polygons.geojson,
#   all_rectangle_outlines.geojson,
#   named_rectangle_centroids.geojson,
#   named_rectangle_polygons.geojson

os.chdir(os.path.dirname(os.path.abspath(__file__)))

###### Inputs 1
# Doesn't really matter which inversion solution because all the NSHM fault files are the same.
NSHM_directory = "NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDEz"
# provide model extension to match the mesh directory and name output directory
discretised_extension = "_CFM"

mesh_directory = f"../data/mesh2500"
# this must be the same length as the number of meshes and have some value that matches all the target fault sections
# will need to come up with a better way to do this in the future when more faults/meshes are used
target_NSHM_fault_names = ["Aotea|Evans Bay", "Dry River|Huangarua", "Fisherman", "Honeycomb",
                           "Moonshine|Otaki", "Ohariu", "Okupe", "Opouawe", "Otaraia",
                           "Pahaua", "Palliser|Kaiwhata", "Pukerua", "Riversdale", "Mana|Otaheke",
                           "Wairarapa", "Wellington Hutt", "Whareama", "Wharekauhau", "Whitemans"]
########

#### Inputs 2
target_traces = gpd.read_file(f"discretised{discretised_extension}/name_filtered_fault_sections.geojson" "").to_crs(
    epsg=2193)
all_traces = gpd.read_file(f"../data/crustal_solutions/{NSHM_directory}/ruptures/fault_sections.geojson").to_crs(epsg=2193)

# read in rake data
rake_dict = read_rake_csv("../data/wellington_alt_geom/alt_geom_rakes.csv")
if discretised_extension == "_Model2" or discretised_extension == "_Model_testing":
    rake_col = "model2_rake"
elif discretised_extension == "_Model1":
    rake_col = "model1_rake"
elif discretised_extension == "_ModelCFM":
    rake_col = "cfm_rake"
else:
    rake_dict = rake_from_traces(target_traces)
    rake_col = "cfm_rake"

# read in fault meshes and assign unique names to each one
# there is probably a better way to do this, but the names matter so this is it for now.
# stl can be visualised by command line meshio convert, and opening in paraview

# Read in all meshes from mesh directory, and add to mesh list_and mesh_name_list
stl_list = glob.glob(f"{mesh_directory}/*.stl")
mesh_list = []
mesh_name_list = []
target_NSHM_fault_names = []
for mesh in stl_list:
    mesh_list.append(meshio.read(mesh))
    mesh_name = os.path.basename(mesh).split(".")[0].replace('remeshed', 'mesh')
    mesh_name_list.append(mesh_name)
    target_NSHM_fault_names.append(mesh_name.split('_')[0].replace('-combined', 'combined'))

# Read in combination file
combine_meshes = True
if combine_meshes:
    comb_dict = read_combination_csv('../data/NZ_CFM_v1_0_rs1km_modified_connected_edited_OCC.csv')

#########

# Sensitivity testing for patch edge depth/dip angle. Skip for crustal for now.
steeper_dip = False
gentler_dip = False


##############
def cross_3d(a, b):
    """
    Calculates cross product of two 3-dimensional vectors.
    """
    x = ((a[1] * b[2]) - (a[2] * b[1]))
    y = ((a[2] * b[0]) - (a[0] * b[2]))
    z = ((a[0] * b[1]) - (a[1] * b[0]))
    return np.array([x, y, z])


def check_triangle_normal(triangle_vertices):
    """ needed for correct dip direction. check if the normal vector is positive or negative. If negative,
    flip the order of the vertices. Triangle_vertices is a 3x3 array of the vertices of the triangle (3 vertices,
    each with xyz)"""

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


named_rectangle_centroids = []
named_rectangle_polygons = []
all_rectangle_polygons = []
# make dataframe for fault section rectangle attributes (for export later)
df_named_rectangle = pd.DataFrame()
df_all_rectangle = pd.DataFrame()

df_named_rectangle_centroid = pd.DataFrame()
df_all_rectangle_centroid = pd.DataFrame()

# turn NSHM traces into rectangular patches using the metadata in the GeoJSON file
for i, trace in all_traces.iterrows():
    # Convert the trace to a numpy array
    trace_xy_array = np.array(trace.geometry.coords)
    # NSHM crustal doesn't have z coords on the traces, need to add in as zero.
    trace_array = np.zeros(shape=(2, 3))
    # pull first and last vertex from the trace
    trace_array[0] = np.append(trace_xy_array[0], trace.UpDepth)
    trace_array[1] = np.append(trace_xy_array[-1], trace.UpDepth)
    # Convert the depths to metres (not needed here since it's zero, but good practice)
    trace_array[:, -1] *= -1000.

    low_depth, up_depth = trace.LowDepth, trace.UpDepth
    # Calculate the centroid of the trace
    trace_centroid = np.array([*trace.geometry.centroid.coords[0], trace.UpDepth * -1000])
    # Calculate the height of the patch
    rectangle_height = (trace.LowDepth - trace.UpDepth) * 1000.
    dip_angle = trace.DipDeg
    extension2 = ""

    # write patch attributes to dictionary and add to bottom of data frame
    df2_rectangle = pd.DataFrame({'fault_id': [int(trace.FaultID)], 'fault_name': [trace.FaultName],
                                  'dip_deg': [dip_angle], 'rectangle_patch_height_m': [rectangle_height],
                                  'up_depth_km': [up_depth], 'low_depth_km': [low_depth],
                                  'dip_dir_deg': [trace.DipDir]}, index=[i])
    df_all_rectangle = pd.concat([df_all_rectangle, df2_rectangle])

    # # Calculate the across-strike vector, (orthogonal to strike) aka dip direction xy vector
    across_strike = np.array([np.sin(np.radians(trace.DipDir)), np.cos(np.radians(trace.DipDir))])

    # Calculate the down-dip vector by incorporating the dip angle
    cosdip = np.cos(np.radians(dip_angle))
    sindip = np.sin(np.radians(dip_angle))
    down_dip = np.array([cosdip * across_strike[0], cosdip * across_strike[1], - 1 * sindip])

    # Calculate the corners of the patch and make a shapely polygon
    dd1 = trace_array[1] + (rectangle_height / sindip) * down_dip
    dd2 = trace_array[0] + (rectangle_height / sindip) * down_dip
    rectangle_polygon = Polygon(np.vstack([trace_array, dd1, dd2]))

    # Append the patch and polygon to lists
    all_rectangle_polygons.append(rectangle_polygon)
# %%
####
all_rectangle_outline_gs = gpd.GeoSeries(all_rectangle_polygons, crs=2193)
all_rectangle_outline_gdf = gpd.GeoDataFrame(df_all_rectangle, geometry=all_rectangle_outline_gs.geometry, crs=2193)
all_rectangle_outline_gdf.to_file(f"discretised{discretised_extension}/all_rectangle_outlines.geojson",
                                  driver="GeoJSON")

# Turn name-filtered section traces into rectangular patches using the metadata in the GeoJSON file
for i, trace in target_traces.iterrows():
    # Convert the trace to a numpy array
    trace_xy_array = np.array(trace.geometry.coords)
    # NSHM crustal doesn't have z coords on the traces, need to add in as zero.
    trace_array = np.zeros(shape=(2, 3))
    # pull first and last vertex from the trace
    trace_array[0] = np.append(trace_xy_array[0], trace.UpDepth)
    trace_array[1] = np.append(trace_xy_array[-1], trace.UpDepth)
    # Convert the depths to metres (not needed here since it's zero, but good practice)
    trace_array[:, -1] *= -1000.

    # dip/depth sensitivity testing
    initial_rectangle_height = (trace.LowDepth - trace.UpDepth) * 1000.
    initial_tandip = np.tan(np.radians(trace.DipDeg))
    rectangle_horizontal_dist = initial_rectangle_height / initial_tandip

    # set conditions for rectangular patch geometry parameters, which change with dip angle
    # for steeper dip ( fault section depths * 1.15)
    # probably is a little bit broken if steeper or gentler are True
    if steeper_dip is True and gentler_dip is False:
        low_depth, up_depth = trace.LowDepth * 1.15, trace.UpDepth * 1.15
        rectangle_height = (low_depth - up_depth) * 1000.
        trace_centroid = np.array([*trace.geometry.centroid.coords[0], trace.UpDepth * -1000])
        dip_angle = np.degrees(np.arctan(rectangle_height / rectangle_horizontal_dist))
        extension2 = "_steeperdip"
    # for gentler dip ( fault section depths * 0.85)
    elif gentler_dip is True and steeper_dip is False:
        low_depth, up_depth = trace.LowDepth * 0.85, trace.UpDepth * 0.85
        rectangle_height = (low_depth - up_depth) * 1000.
        trace_centroid = np.array([*trace.geometry.centroid.coords[0], trace.UpDepth * -1000])
        dip_angle = np.degrees(np.arctan(rectangle_height / rectangle_horizontal_dist))
        extension2 = "_gentlerdip"
    # uses geometry from NSHM with no modifications
    elif gentler_dip is False and steeper_dip is False:
        low_depth, up_depth = trace.LowDepth, trace.UpDepth
        # Calculate the centroid of the trace
        trace_centroid = np.array([*trace.geometry.centroid.coords[0], trace.UpDepth * -1000])
        # Calculate the height of the patch
        rectangle_height = (trace.LowDepth - trace.UpDepth) * 1000.
        dip_angle = trace.DipDeg
        extension2 = ""
    else:
        print("Dip modifications are wrong. Only one statement can be True at once. Try again.")
        exit()

    # write patch attributes to dictionary and add to bottom of data frame
    df2_rectangle = pd.DataFrame({'fault_id': [int(trace.FaultID)], 'fault_name': [trace.FaultName],
                                  'dip_deg': [dip_angle], 'rectangle_patch_height_m': [rectangle_height],
                                  'up_depth_km': [up_depth], 'low_depth_km': [low_depth],
                                  'dip_dir_deg': [trace.DipDir]}, index=[i])
    df_named_rectangle = pd.concat([df_named_rectangle, df2_rectangle])

    df2_rectangle_centroid = pd.DataFrame({'fault_id': [trace.FaultID], 'fault_name': [trace.FaultName.replace(':', '')],
                                           'mesh_name': trace['Mesh Name']}, index=[i])
    df_named_rectangle_centroid = pd.concat([df_named_rectangle_centroid, df2_rectangle_centroid])
    #######################
    # # Calculate the across-strike vector, (orthogonal to strike) aka dip direction xy vector
    across_strike = np.array([np.sin(np.radians(trace.DipDir)), np.cos(np.radians(trace.DipDir))])

    # Calculate the down-dip vector by incorporating the dip angle
    cosdip = np.cos(np.radians(dip_angle))
    sindip = np.sin(np.radians(dip_angle))
    down_dip = np.array([cosdip * across_strike[0], cosdip * across_strike[1], - 1 * sindip])

    # Calculate the centroid of the patch
    rectangle_centroid = trace_centroid + (rectangle_height / sindip) / 2 * down_dip

    # Calculate the corners of the patch and make a shapely polygon
    dd1 = trace_array[1] + (rectangle_height / sindip) * down_dip
    dd2 = trace_array[0] + (rectangle_height / sindip) * down_dip
    rectangle_polygon = Polygon(np.vstack([trace_array, dd1, dd2]))

    # Append the patch centroid and polygon to lists
    named_rectangle_centroids.append(rectangle_centroid)
    named_rectangle_polygons.append(rectangle_polygon)

# write patch centroid and rectangle polygons to geojson
# probably an opportunity to reduce code length by combining this with above
named_rectangle_centroids = np.array(named_rectangle_centroids)
named_rectangle_centroids_gs = gpd.GeoSeries([Point(centroid) for centroid in named_rectangle_centroids], crs=2193)
named_rectangle_centroids_gdf = gpd.GeoDataFrame(
    df_named_rectangle_centroid, geometry=named_rectangle_centroids_gs.geometry, crs=2193)
named_rectangle_centroids_gdf.to_file(
    f"discretised{discretised_extension}/named_rectangle_centroids.geojson", driver="GeoJSON")

# Remove all spaces from fault names (Life's just easier that way)
named_rectangle_centroids_gdf['fault_name'] = named_rectangle_centroids_gdf['fault_name'].str.replace(' ', '')
named_rectangle_outline_gs = gpd.GeoSeries(named_rectangle_polygons, crs=2193)
named_rectangle_outline_gdf = gpd.GeoDataFrame(df_named_rectangle, geometry=named_rectangle_outline_gs.geometry, crs=2193)
named_rectangle_outline_gdf.to_file(
    f"discretised{discretised_extension}/named_rectangle_polygons.geojson", driver="GeoJSON")

#### mesh stuff
# make output geodata frame to add all discretised patch/fault info to
out_gdf = gpd.GeoDataFrame({"rake": [], "geometry": [], "fault_name": [], "fault_id": []})

# set up dictionaries. Key will be the fault id
discretised_dict = {}       # values are xyz or triangle coords for each fault ID (i.e., rectangle)
# can probably combine these two dictionaries
# discretised_rake_dict = {}

# loop over individual fault meshes, find closest rectangle patch centroid for same fault name
####### important to subset by name so you don't get the closest patch centroid from a different fault
unused_mesh = []
used_mesh = []
for i in range(len(mesh_list)):
    # fault mesh name, deal with some mesh at a time
    mesh = mesh_list[i]
    mesh_name = mesh_name_list[i]

    if np.sum(named_rectangle_centroids_gdf['mesh_name'].str.contains(target_NSHM_fault_names[i], case=False)):
        print(f"making discretised mesh for {mesh_name} ({target_NSHM_fault_names[i]})")
        used_mesh.append(target_NSHM_fault_names[i])
    else:
        continue

    mesh_triangles_indices = mesh.cells_dict["triangle"]    # indices of vertices that make up triangles

    # NOT CURRENTLY USING FOR CRUSTAL. Multiply depths by steeper/gentler constant
    if steeper_dip is True:
        mesh_vertices = mesh.points * [1, 1, 1.15]
    elif gentler_dip is True:
        mesh_vertices = mesh.points * [1, 1, 0.85]
    else:
        mesh_vertices = mesh.points  # xyz of vertices in mesh. indexed.

    # array of 3 xyz arrays. (three sets of vertices to make a triangle)
    triangle_vertex_arrays = mesh_vertices[mesh_triangles_indices]

    ###### ensuring correct strike convention for accurate displacment calculations later
    # calculate the normal vector to each triangle. If negative, reverse ordering of triangle of vertices.
    ordered_triangle_vertex_arrays = []
    for triangle in triangle_vertex_arrays:
        ordered_triangle_array = check_triangle_normal(triangle)
        ordered_triangle_vertex_arrays.append(ordered_triangle_array)

    ordered_triangle_vertex_arrays = np.array(ordered_triangle_vertex_arrays)

    triangle_centroids = np.mean(ordered_triangle_vertex_arrays, axis=1)   # three part array. means of x, y, and z.

    # find the closest fault rectangle patch to each triangle centroid
    closest_rectangles = []
    closest_rectangle_fault_ids = []    # index (fault ID) for closest rectangle centroid to each triangle centroid
    for triangle_centroid in triangle_centroids:
        # search_terms = target_NSHM_fault_names[i]
        # subset centroids by fault name that matches mesh name (prevents grabbing the wrong fault)
        rectangle_centroid_gdf_i = named_rectangle_centroids_gdf[
            named_rectangle_centroids_gdf['mesh_name'].str.contains(target_NSHM_fault_names[i], case=False)]
        # get coordinates of patch centroids
        # needs to be in array format to work with triangle centroids
        patch_centroids_i = np.vstack([np.array(value.coords) for value in
                                       rectangle_centroid_gdf_i.geometry.values])

        # find closest patch centroid to each triangle
        distances = np.linalg.norm(patch_centroids_i - triangle_centroid, axis=1)
        if distances.min() < 10.e4:
            closest_patch = np.argmin(distances)    # gets list index from subset of patches
            # finds fault id of closest patch within centroids that are subset by name
            closest_fault_id = rectangle_centroid_gdf_i.fault_id.values[closest_patch]
            # closest_fault_id = named_rectangle_centroids_gdf.fault_id.values[closest_patch]
            closest_rectangle_fault_ids.append(closest_fault_id)
        else:
            closest_rectangle_fault_ids.append(-1)  # ??

    # array of: index (fault ID) for closest rectangle centroid to each triangle centroid
    closest_rectangle_fault_ids = np.array(closest_rectangle_fault_ids)

    # Create fault polygons from triangles
    mesh_polygons = []

    # for adding to discretised polygon geojson attributes
    polygon_rakes = []
    polygon_fault_ids = []

    # make discretised patches (patch of triangles)
    for j, trace in target_traces.iterrows():
        # get the triangles that are closest to the trace of interest (i.e., index of closest rectangle matches fault
        # index)
        triangles_locs = np.where(closest_rectangle_fault_ids == trace.FaultID)
        triangles = ordered_triangle_vertex_arrays[triangles_locs]

        # skip the trace if there's no matching mesh in the rake dictionary
        trace_id = trace.FaultID
        if trace_id in rake_dict.keys():
            # extract rake value from rake dictionary based on FaultID
            rake = rake_dict[trace.FaultID][rake_col]

            # skip discretizing a rectangular patch if we don't have a mesh that matches it
            if len(triangles) > 0:
                # make dictionary of triangles that go with each polygon
                triangle_polygons = [Polygon(triangle) for triangle in triangles]
                dissolved_polygons = gpd.GeoSeries(triangle_polygons).unary_union
                mesh_polygons.append(dissolved_polygons)

                # Find the rake for each patch
                polygon_rakes.append(rake)

                # add patch index
                polygon_fault_ids.append(trace.FaultID)

                # set the triangles and rake for each fault ID
                discretised_dict[trace.FaultID] = {"triangles": triangles, "rake": rake, "triangle_indices": triangles_locs}
        else:
            print('\t{} not found in rake file. Skipping...'.format(target_NSHM_fault_names[i]))

    # get patch rakes and fault ids to add to output geojson file
    polygon_rakes = np.array(polygon_rakes)
    polygon_fault_ids = np.array(polygon_fault_ids)
    mesh_name_array = np.full(polygon_fault_ids.shape, mesh_name)

    # Create a geodataframe from the discretised polygons
    gdf = gpd.GeoDataFrame({"rake": polygon_rakes, "geometry": mesh_polygons, "fault_name": mesh_name_array, "fault_id":
                            polygon_fault_ids})

    out_gdf = pd.concat([out_gdf, gdf])

missing_meshes = list(set(list(target_traces['Mesh Name'])) - set(used_mesh))
missing_meshes.sort()
if len(missing_meshes) > 0:
    for missing in missing_meshes:
        print('Expected {} .stl file not found in meshdir. Faults not created'.format(missing))

# write discretised polygons to geojson
out_gdf.crs = target_traces.crs
out_gdf.to_file(f"discretised{discretised_extension}/crustal_discretised_polygons.geojson",
                driver="GeoJSON")

# make pickle with triangle vertices
pkl.dump(discretised_dict, open(f"discretised{discretised_extension}/crustal_discretised_dict.pkl", "wb"))
