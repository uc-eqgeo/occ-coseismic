import os
import geopandas as gpd
from shapely.geometry import Point

# This script extracts the fault sections from the NSHM within a defined region of interest.
# Specific faults can also be included or excluded based on their name.
# Outputs a geojson with filtered traces.
# Do this prior to discretize_crustal, because there's no point trying to discretize patches when we don't have a mesh
# Only have to do once no matter the NSHM branch since the faults don't change

#######INPUTS
# find only the rupture scenarios that use faults we have meshes for
NSHM_directory = "NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDEz"  # geologic, mid B and N, C 4.2
model_extension = "_Model_testing"         # "_Model1" or "_Model2" or "_CFM"

# Define region of interest
minLon, maxLon, minLat, maxLat = 174.5, 176.5, -42.0, -40.0
#minLon, maxLon, minLat, maxLat = 160.5, 176.5, -48.0, -40.0

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

# Define fault names to exclude that may be in RoI.
# If fault appears in target_fault_names and exclude_fault_names, it will be included.
exclude_fault_names = ["Aotea", "Huangarua", "Fisherman", "Honeycomb", "Moonshine", "Otaki", "Ohariu", "Okupe",
                       "Opouawe", "Uruti", "Otaraia", "Pahaua", "Palliser", "Pukerua", "Riversdale", "Shepherds Gully",
                       "Mana", "Otaheke", "Wairarapa", "Wellington Hutt", "Whareama", "Wharekauhau", "Whitemans"]

exclude_fault_names = [fault for fault in exclude_fault_names if fault not in target_fault_names]

#############
# load all fault sections
traces = gpd.read_file(f"../data/crustal_solutions/{NSHM_directory}/ruptures/fault_sections.geojson").to_crs(epsg=2193)

# find all fault ids that have a name match to the target name list
filtered_trace_ids = []
filtered_trace_names = []
for i, row in traces.iterrows():
    for fault_name in target_fault_names:
        if fault_name in row.FaultName:
            filtered_trace_ids.append(row.FaultID)
            filtered_trace_names.append(fault_name)

# Check target faults are included
for fault_name in target_fault_names:
    if fault_name not in filtered_trace_names:
        print(f"Specifically requested {fault_name} not in fault section geojson. Check you spelt it correctly")

# find all fault ids within region of interest that are not specifically excluded
for i, row in traces.cx[minLon:maxLon, minLat:maxLat].iterrows():
    if row.FaultName not in exclude_fault_names:
        filtered_trace_ids.append(row.FaultID)

# remove duplicates and sort
filtered_trace_ids = list(set(filtered_trace_ids))
filtered_trace_ids.sort()

# subset fault sections by target rupture id
filtered_traces_gdf = traces[traces.FaultID.isin(filtered_trace_ids)]

# make directory for outputs if it doesn't already exist
if not os.path.exists(f"out_files{model_extension}"):
    os.mkdir(f"out_files{model_extension}")

filtered_traces_gdf.to_file(f"out_files{model_extension}/name_filtered_fault_sections.geojson",
                            driver="GeoJSON")