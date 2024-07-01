# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:13:28 2024

Little script to convert data downloaded from the searise project website into a site format for occ-searise NSHM
@author: jmc753
"""

import pandas as pd
import geopandas as gpd
import numpy as np

searise_csv = 'NZ_VLM_final_May24.csv'

data = pd.read_csv(searise_csv)

if 'lon' in data.columns:  # For searise point exports
    data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.lon, data.lat), crs='EPSG:4326')
    coord_name = False
    sort_values = True
elif 'Lon' in data.columns:  # For Hamling VLM coast sites from paper
    data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.Lon, data.Lat), crs='EPSG:4326')
    data.rename(columns={'Site ID': 'siteId'}, inplace=True)
    coord_name = False
    sort_values = False
else:  # For QGIS point exports
    data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.X, data.Y), crs='EPSG:4326')
    data.rename(columns={'id': 'siteId'}, inplace=True)
    coord_name = True
    sort_values = True

data['Lon'] = data.geometry.x
data['Lat'] = data.geometry.y
breakpoint()
ix = np.unique(data['siteId'].to_numpy(), return_index=True)[1]  # Remove duplicate siteIds for different searise scenarios
data = data[['siteId', 'Lon', 'Lat']].iloc[ix]

data['Height'] = 0
if sort_values:
    data = data.sort_values(by=['Lat', 'Lon']).reset_index(drop=True)  # Sort based on Latitude, then longitude
    if coord_name:
        data['siteId'] = [f"{round(data.loc[ix, 'Lon']/1e3)}_{round(data.loc[ix, 'Lat']/1e3)}" for ix in range(data.shape[0])]  # Set siteId to be based on NZTM location
    else:
        data['siteId'] = np.array(data.index) # Reset siteIds

searise_out = searise_csv.replace('.csv', '_points.csv')

data.to_csv(searise_out, index=False)
