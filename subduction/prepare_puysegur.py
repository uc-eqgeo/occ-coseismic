# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:16:32 2024
Script for reading in the Puysegur SZ geometry, and converting into a geojson format

@author: jmc753
"""

import pandas as pd
import geopandas as gpd
from shapely import LineString, Polygon
import math


GNS_file = '../data/sz_solutions/GNS SR2022-31 ESup_DFM_4_Puysegur-0p7.csv'

in_cols = ['FaultID', 'FaultName', 'SlipRate', 'DipDeg', 'UpDepth', 'LowDepth', 'Lon1', 'Lat1', 'Depth1', 'Lon2', 'Lat2', 'Depth2']

df = pd.read_csv(GNS_file, header=0, names=in_cols)

gdf1 = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Lon1, df.Lat1, df.UpDepth), crs="EPSG:4326")
gdf2 = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Lon2, df.Lat2, df.UpDepth), crs="EPSG:4326")

traces = [LineString([p1, p2]) for p1, p2 in zip(gdf2['geometry'], gdf1['geometry'])]

gdf = gdf1.drop(columns=['Lon1', 'Lat1', 'Depth1', 'Lon2', 'Lat2', 'Depth2', 'geometry'])
gdf = gpd.GeoDataFrame(gdf, geometry=traces)

gdf.to_file('../data/sz_solutions/puysegur_fault_sections.geojson', driver="GeoJSON")
