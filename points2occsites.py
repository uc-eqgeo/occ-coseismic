# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:13:28 2024

Little script to convert data downloaded from the searise project website into a site format for occ-searise NSHM
@author: jmc753
"""

import pandas as pd
import geopandas as gpd
import numpy as np

searise_csv = 'national_50km_grid.csv'

data = pd.read_csv(searise_csv)

if 'lon' in data.columns:  # For searise point exports
    data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.lon, data.lat), crs='EPSG:4326')
else:  # For QGIS point exports
    data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.X, data.Y), crs='EPSG:4326')
    data.rename(columns={'id': 'siteId'}, inplace=True)

data['Lon'] = data.geometry.x
data['Lat'] = data.geometry.y

ix = np.unique(data['siteId'].to_numpy(), return_index=True)[1] # Remove duplicate siteIds for different searise scenarios
data = data[['siteId', 'Lon', 'Lat']].iloc[ix].reset_index(drop=True)

data['Height'] = 0
data['siteId'] = np.array(data.index) # Reset siteIds

searise_out = searise_csv.replace('.csv', '_points.csv')

data.to_csv(searise_out, index=False)
