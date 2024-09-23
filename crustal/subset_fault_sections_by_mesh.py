import os
import geopandas as gpd
from shapely.geometry import Point
from crustal_helper_scripts import read_combination_csv
from itertools import product

# This script extracts the fault sections from the NSHM within a defined region of interest.
# Specific faults can also be included or excluded based on their name.
# Outputs a geojson with filtered traces.
# Do this prior to discretise_crustal, because there's no point trying to discretise patches when we don't have a mesh
# Only have to do once no matter the NSHM branch since the faults don't change

#######INPUTS
# find only the rupture scenarios that use faults we have meshes for
NSHM_directory = "NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDEz"  # geologic, mid B and N, C 4.2
model_extension = "_CFM_test"         # "_Model1" or "_Model2" or "_CFM"

# Define region of interest
minLon, maxLon, minLat, maxLat = 160.5, 179.0, -48.0, -34.0

# Convert to RoI from Lon/Lat to NZ grid if needed
if abs(minLon) <= 180:
    P1, P2 = gpd.GeoSeries([Point(minLon, minLat), Point(maxLon, maxLat)], crs=4326).to_crs(epsg=2193)
    minLon, minLat = P1.coords[0]
    maxLon, maxLat = P2.coords[0]

# Define fault names to capture that may not be in RoI.
# Choose a unique part of each name in fault sections, avoid using hyphens and stuff to remove ambiguity.
# It's ok to "capture" more faults than you have meshes for at this stage.
target_fault_names = ["Aotea", "Huangarua", "Fisherman", "Honeycomb", "Moonshine", "Otaki", "Ohariu", "Okupe",
                      "Opouawe", "Uruti", "Otaraia", "Pahaua", "Palliser", "Pukerua", "Riversdale", "Shepherds Gully",
                      "Mana", "Otaheke", "Wairarapa", "Wellington Hutt", "Whareama", "Wharekauhau", "Whitemans"]
target_fault_names = []
# Define fault names to exclude that may be in RoI.
# If fault appears in target_fault_names and exclude_fault_names, it will be included.
exclude_fault_names = []

exclude_fault_names = [fault for fault in exclude_fault_names if fault not in target_fault_names]

#############
# load all fault sections
traces = gpd.read_file(f"../data/crustal_solutions/{NSHM_directory}/ruptures/fault_sections.geojson").to_crs(epsg=2193)

# Read in combination file
# Combination file aims to identify which faults, which are seperate in the NSHM, have been combined into a single mesh .stl file
combine_meshes = True
if combine_meshes:
    comb_dict = read_combination_csv('../data/NZ_CFM_v1_0_rs1km_modified_connected_edited_OCC.csv', saveSpaces=True)
    component_dict = {}

    for comb, comp_list in zip(comb_dict.keys(), comb_dict.values()):
        for comp in comp_list:
            component_dict[comp] = comb

    # Check if any target faults are faults in combined meshes of a different name, and include the other faults
    for target, key in product(target_fault_names, component_dict.keys()):
        if target in key:
            target_fault_names += comb_dict[component_dict[key]]

    # Check if any target faults are combined meshes, and thus include the component faults
    for target, key in product(target_fault_names, comb_dict.keys()):
        if target in key:
            target_fault_names += comb_dict[key]

    target_fault_names = list(set([fault.rstrip(' 0123456789') for fault in target_fault_names]))   # remove numbers from mesh name
    target_fault_names.sort()

    # Check if any exclude faults are faults in combined meshes of a different name, and include the other faults
    for target, key in product(exclude_fault_names, component_dict.keys()):
        if target in key:
            exclude_fault_names += comb_dict[component_dict[key]]

    # Check if any exclude faults are combined meshes, and thus include the component faults
    for target, key in product(exclude_fault_names, comb_dict.keys()):
        if target in key:
            exclude_fault_names += comb_dict[key]

    exclude_fault_names = list(set([fault.rstrip(' 0123456789') for fault in exclude_fault_names]))   # remove numbers from mesh name
    exclude_fault_names.sort()

# Add mesh file name to traces gdf
mesh_names = []
for i, trace in traces.iterrows():
    if trace.ParentName.replace(':', '') not in component_dict.keys():
        mesh_name = trace.ParentName.replace(' ', '').replace('&', '&amp;').replace(':', '')
    else:
        mesh_name = component_dict[trace.ParentName.replace(':', '')].replace(' ', '').replace('|', '-') + 'combined'

    mesh_names += [mesh_name]

traces['Mesh Name'] = mesh_names

# find all fault ids that have a name match to the target name list
filtered_trace_ids = []
filtered_trace_names = []
for i, row in traces.iterrows():
    for fault_name in target_fault_names:
        if fault_name in row.FaultName.replace(':', ''):
            filtered_trace_ids.append(row.FaultID)
            filtered_trace_names.append(fault_name)

# Check target faults are included
for fault_name in target_fault_names:
    if fault_name not in filtered_trace_names:
        print(f"Specifically requested {fault_name} not in fault section geojson. Check you spelt it correctly or it was included in NSHM")

# find all fault ids within region of interest
for i, row in traces.cx[minLon:maxLon, minLat:maxLat].iterrows():
    # if row.FaultName not in exclude_fault_names:
    #     filtered_trace_ids.append(row.FaultID)
    filtered_trace_ids.append(row.FaultID)

# remove duplicates and sort
filtered_trace_ids = list(set(filtered_trace_ids))
filtered_trace_ids.sort()

# Remove excluded traces
exclude_ids = []
for i, row in traces.loc[filtered_trace_ids].iterrows():
    for fault_name in exclude_fault_names:
        if fault_name in row.FaultName.replace(':', ''):
            exclude_ids.append(i)

filtered_trace_ids = [fault_id for fault_id in filtered_trace_ids if fault_id not in exclude_ids]

# subset fault sections by target rupture id
filtered_traces_gdf = traces[traces.FaultID.isin(filtered_trace_ids)]

# make directory for outputs if it doesn't already exist
if not os.path.exists(f"discretised{model_extension}"):
    os.mkdir(f"discretised{model_extension}")

filtered_traces_gdf.to_file(f"discretised{model_extension}/name_filtered_fault_sections.geojson", driver="GeoJSON")
