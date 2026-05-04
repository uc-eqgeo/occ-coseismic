# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:13:28 2024

Little script to convert data downloaded from the searise project website into a site format for occ-searise NSHM
@author: jmc753
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import os
from shapely.geometry import Point

searise_sites = ['.\\sites\\national_500m_gridS.geojson']  # Put in multiple csv or geojsons if you want to combine into one output file
data_format = 'qgis' # 'qgis' for qgis exports, 'searise' for searise exports, 'hamling' for Hamling VLM coast sites from paper
out_csv_file = None  # If none, automatically set to the input file name with '_points' appended
extra_suffixes = []  # Extra suffixes to append to the output file name (Works if you have subsets of 1 CSV file, e.g. northern and southern sections of a fault model, or south islands only)

split_regionally = False  # Whether to split the outputs into North and South Island files as well
coast_buffer = 0  # km offshore buffer
coast_trim = False  # Whether to trim all sites to be within coast buffer
coast_file = ".\\data\\coastline\\nz-70sqkm_island_coastlines-polygons-topo-150k.gpkg"

if len(searise_sites) > 1 and len(extra_suffixes) > 0:
    print("Warning: Multiple site files provided, but extra suffixes will not be used. Output file will be a combination of all input files.")
    extra_suffixes = ['']
elif '' not in extra_suffixes:
    extra_suffixes = [''] + extra_suffixes  # Ensure the first suffix is empty to handle the case where no suffix is needed

for suffix in extra_suffixes:
    out_pd = pd.DataFrame(columns=['siteId', 'Lon', 'Lat', 'Height'])
    for site_file in searise_sites:
        file_type = os.path.splitext(site_file)[1]
        site_file = site_file.replace(file_type, suffix + file_type)  # Append the suffix to the CSV file name if provided
        if not os.path.exists(site_file):
            print(f"File {site_file} does not exist. Skipping.")
            continue
        if file_type.lower() == '.csv':
            data = pd.read_csv(site_file)  # Read in the CSV file, appending the suffix if provided
        elif file_type.lower() == '.geojson':
            data = gpd.read_file(site_file)
            data['geometry'] = data.geometry.centroid  # Get polygon centroids if polygon geojson used instead of point
            if data.crs == 4326:
                data = data.to_crs(2193)

        if data_format == 'searise':  # For searise point exports
            data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.lon, data.lat), crs='EPSG:4326')
            data.geometry = data.geometry.to_crs('EPSG:2193')  # Convert to NZTM
            coord_name = False
            reset_id = True
            sort_values = True
        elif data_format == 'hamling':  # For Hamling VLM coast sites from paper
            data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.Lon, data.Lat), crs='EPSG:4326')
            data.rename(columns={'Site ID': 'siteId'}, inplace=True)
            data.geometry = data.geometry.to_crs('EPSG:2193')  # Convert to NZTM
            coord_name = False
            reset_id = True
            sort_values = False
        else:  # For QGIS point exports
            if file_type.lower() == '.csv':
                if data.X.max() > 180:  # If the data is in NZTM
                    data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.X, data.Y), crs='EPSG:2193')
                else:  # If the data is in Lat/Lon
                    data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.X, data.Y), crs='EPSG:4326').to_crs('EPSG:2193')  # Convert to NZTM
            if 'id' in data.columns:
                data.rename(columns={'id': 'siteId'}, inplace=True)
                coord_name = True
                reset_id = False
                sort_values = True
            else:
                site_col = [col for col in data.columns if 'site' in col.lower()]
                if len(site_col) == 0:
                    data['siteId'] = ''
                    coord_name = True
                else:
                    data.rename(columns={site_col[0]: 'siteId'}, inplace=True)
                    coord_name = False
                sort_values = True
                reset_id = False


        data['Lon'] = np.round(data.geometry.x, 1)
        data['Lat'] = np.round(data.geometry.y, 1)
        data['Height'] = 0

        out_pd = pd.concat([out_pd if not out_pd.empty else None, data[['siteId', 'Lon', 'Lat', 'Height']]])

    if out_pd.empty:
        print("No valid site data found in the provided files. Exiting.")
        continue
    if sort_values:
        out_pd = out_pd.sort_values(by=['Lat', 'Lon']).reset_index(drop=True)  # Sort based on Latitude, then longitude
        if coord_name:
            out_pd['siteId'] = [f"{round(out_pd.loc[ix, 'Lon'])}_{round(out_pd.loc[ix, 'Lat'])}" for ix in range(out_pd.shape[0])]  # Set siteId to be based on NZTM location
        elif reset_id:
            out_pd['siteId'] = np.array(out_pd.index) # Reset siteIds

    if any([coord_name, reset_id]):
        ix = np.unique(out_pd['siteId'].to_numpy(), return_index=True)[1]  # Remove duplicate siteIds for different searise scenarios
        out_pd = out_pd[['siteId', 'Lon', 'Lat', 'Height']].iloc[ix].reset_index(drop=True)

    if out_csv_file is None:
        out_file = '.\\sites\\' + ''.join([file.split('\\')[-1].replace(os.path.splitext(file)[1], '_') for file in searise_sites]).strip('_') + suffix + '_points.csv'
    else:
        out_file = out_csv_file

    out_gpd = gpd.GeoDataFrame(out_pd, geometry=gpd.points_from_xy(out_pd.Lon, out_pd.Lat), crs='EPSG:2193')  # Convert to GeoDataFrame for spatial operations

    if coast_trim:
        print("Trimming sites to within {} km offshore...".format(coast_buffer))
        coast_gpd = gpd.read_file(coast_file)
        if coast_buffer > 0:
            coast_gpd['geometry'] = coast_gpd.geometry.buffer(coast_buffer * 1e3)  # Buffer specified distance around the coastline
        out_gpd = gpd.sjoin(out_gpd, coast_gpd, predicate='within')

    out_gpd[['siteId', 'Lon', 'Lat', 'Height']].to_csv(out_file, index=False)
    print(f"\tOutput file saved as: {out_file}")
    out_gpd.to_file(out_file.replace('.csv', '.geojson'), driver='GeoJSON')
    print(f"\tOutput GeoJSON file saved as: {out_file.replace('.csv', '.geojson')}")

if len(extra_suffixes) > 1:
    split_regionally = False

if split_regionally:
    print("Splitting outputs into Hikurangi and Puysegur sections...")
    wellington = Point([1749150, 5428092]) # Wellington coordinates in NZTM
    te_anau = Point([1186710, 4957633])  # Te Anau coordinates in NZTM
    distance = 350  # Distance South of Wellington in km to include for hikurangi

    # For Hikurangi, find all centroids north of 350km south of Wellington
    northern_section = out_gpd[(out_gpd.geometry.y > wellington.y) | (out_gpd.distance(wellington) < distance * 1e3)]
    northern_section.to_file(out_file.replace('.csv', 'N.geojson'), driver='GeoJSON')
    print(f"\tWritten {out_file.replace('.csv', 'N.geojson')}")
    northern_section[['siteId', 'Lon', 'Lat', 'Height']].to_csv(out_file.replace('.csv', 'N.csv'), index=False)
    print(f"\tWritten {out_file.replace('.csv', 'N.csv')}")

    # For Puysegur, find all centroids within 350km of Te Anau
    southern_section = out_gpd[(out_gpd.distance(te_anau) < distance * 1e3)]
    southern_section.to_file(out_file.replace('.csv', 'S.geojson'), driver='GeoJSON')
    print(f"\tWritten {out_file.replace('.csv', 'S.geojson')}")
    southern_section[['siteId', 'Lon', 'Lat', 'Height']].to_csv(out_file.replace('.csv', 'S.csv'), index=False)
    print(f"\tWritten {out_file.replace('.csv', 'S.csv')}")
