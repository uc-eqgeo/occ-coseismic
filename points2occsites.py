# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:13:28 2024

Little script to convert data downloaded from the searise project website into a site format for occ-searise NSHM
@author: jmc753
"""

import pandas as pd
import geopandas as gpd
import numpy as np

searise_csv = ['.\\sites\\cube_centroids_27000_9000_buffer_0_33S.csv']
data_format = 'qgis' # 'qgis' for qgis exports, 'searise' for searise exports, 'hamling' for Hamling VLM coast sites from paper
out_csv_file = None  # If none, automatically set to the input file name with '_points' appended

out_pd = pd.DataFrame(columns=['siteId', 'Lon', 'Lat', 'Height'])

for csv_file in searise_csv:
    data = pd.read_csv(csv_file)

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
        if data.X.max() > 180:  # If the data is in NZTM
            data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.X, data.Y), crs='EPSG:2193')
        else:  # If the data is in Lat/Lon
            data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.X, data.Y), crs='EPSG:4326')
            data.geometry = data.geometry.to_crs('EPSG:2193')  # Convert to NZTM
        if 'id' in data.columns:
            data.rename(columns={'id': 'siteId'}, inplace=True)
            coord_name = True
            reset_id = False
            sort_values = True
        else:
            site_col = [col for col in data.columns if 'site' in col.lower()]
            data.rename(columns={site_col[0]: 'siteId'}, inplace=True)
            sort_values = True
            coord_name = False
            reset_id = False


    data['Lon'] = data.geometry.x
    data['Lat'] = data.geometry.y
    data['Height'] = 0

    out_pd = pd.concat([out_pd, data[['siteId', 'Lon', 'Lat', 'Height']]])

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
    out_csv_file = '.\\sites\\' + ''.join([csv.split('\\')[-1].replace('.csv', '_') for csv in searise_csv]) + 'points.csv'

out_pd.to_csv(out_csv_file, index=False)
