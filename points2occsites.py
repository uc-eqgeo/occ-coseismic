# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:13:28 2024

Little script to convert data downloaded from the searise project website into a site format for occ-searise NSHM
@author: jmc753
"""

import pandas as pd
import geopandas as gpd
import numpy as np

searise_csv = '.\\sites\\national_2km_grid.csv'

data = pd.read_csv(searise_csv)

if 'lon' in data.columns:  # For searise point exports
    data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.lon, data.lat), crs='EPSG:4326')
    data.geometry = data.geometry.to_crs('EPSG:2193')  # Convert to NZTM
    coord_name = False
    sort_values = True
elif 'Lon' in data.columns:  # For Hamling VLM coast sites from paper
    data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.Lon, data.Lat), crs='EPSG:4326')
    data.rename(columns={'Site ID': 'siteId'}, inplace=True)
    data.geometry = data.geometry.to_crs('EPSG:2193')  # Convert to NZTM
    coord_name = False
    sort_values = False
else:  # For QGIS point exports
    if data.X.max() > 180:  # If the data is in NZTM
        data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.X, data.Y), crs='EPSG:2193')
    else:  # If the data is in Lat/Lon
        data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.X, data.Y), crs='EPSG:4326')
        data.geometry = data.geometry.to_crs('EPSG:2193')  # Convert to NZTM
    data.rename(columns={'id': 'siteId'}, inplace=True)
    coord_name = True
    sort_values = True

data['Lon'] = data.geometry.x
data['Lat'] = data.geometry.y

ix = np.unique(data['siteId'].to_numpy(), return_index=True)[1]  # Remove duplicate siteIds for different searise scenarios
data = data[['siteId', 'Lon', 'Lat']].iloc[ix]

data['Height'] = 0
if sort_values:
    data = data.sort_values(by=['Lat', 'Lon']).reset_index(drop=True)  # Sort based on Latitude, then longitude
    if coord_name:
        data['siteId'] = [f"{round(data.loc[ix, 'Lon'])}_{round(data.loc[ix, 'Lat'])}" for ix in range(data.shape[0])]  # Set siteId to be based on NZTM location
    else:
        data['siteId'] = np.array(data.index) # Reset siteIds

searise_out = searise_csv.replace('.csv', '_points.csv')

data.to_csv(searise_out, index=False)
