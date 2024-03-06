import geopandas as gpd
import pathlib
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box
from typing import Union
from matplotlib import pyplot as plt

coast_shp_name = ("coast_and_interface/nz-coastlines-and-islands-polygons-topo-150k.shp")
coast_shp = pathlib.Path(__file__).parent / coast_shp_name

min_x1 = 800000
min_y1 = 4000000
max_x2 = 3200000
max_y2 = 7500000

min_x1_wgs = 160.
max_x2_wgs = 185.
min_y1_wgs = -51.
max_y2_wgs = -33.


def clip_coast_with_trim(x1: Union[int, float], y1: Union[int, float], x2: Union[int, float], y2: Union[int, float],
                         wgs: bool = False):
    """
    To clip coastline into area of interest
    :param x1: Bottom-left easting (NZTM, metres)
    :param y1: Bottom-left northing
    :param x2: Top-right easting
    :param y2: Top-right northing
    :return:
    """
    if not wgs:
        conditions = [x1 >= min_x1, y1 >= min_y1, x2 <= max_x2, y2 <= max_y2, x1 < x2, y1 < y2]
        assert all(conditions), "Check coordinates"

    if wgs:
        boundary = gpd.GeoSeries(Polygon(([x1, y1], [x1, y2], [x2, y2], [x2, y1])), crs=4326)
    else:
        boundary = gpd.GeoSeries(Polygon(([x1, y1], [x1, y2], [x2, y2], [x2, y1])), crs=2193)

    coast_df = gpd.GeoDataFrame.from_file(coast_shp)

    if wgs:
        coast_df.to_crs(epsg=4326, inplace=True)
    trimmed_df = gpd.clip(coast_df, boundary)
    poly_ls = []
    for item in trimmed_df.geometry:
        if isinstance(item, Polygon):
            poly_ls.append(item)
        elif isinstance(item, MultiPolygon):
            poly_ls += list(item)
    polygon_geoseries = gpd.GeoSeries(poly_ls, crs=2193)

    return polygon_geoseries


def clip_coast(x1: Union[int, float], y1: Union[int, float], x2: Union[int, float], y2: Union[int, float],
               wgs: bool = False):
    """
    To clip coastline into area of interest
    :param x1: Bottom-left easting (NZTM, metres)
    :param y1: Bottom-left northing
    :param x2: Top-right easting
    :param y2: Top-right northing
    :return:
    """

    if not wgs:
        conditions = [x1 >= min_x1, y1 >= min_y1, x2 <= max_x2, y2 <= max_y2, x1 < x2, y1 < y2]
        assert all(conditions), "Check coordinates"

    coast_df = gpd.GeoDataFrame.from_file(coast_shp)

    if wgs:
        coast_df.to_crs(epsg=4326, inplace=True)
    trimmed_df = coast_df.cx[x1:x2, y1:y2]

    poly_ls = []
    for item in trimmed_df.geometry:
        if isinstance(item, Polygon):
            poly_ls.append(item)
        elif isinstance(item, MultiPolygon):
            poly_ls += list(item)
    polygon_geoseries = gpd.GeoSeries(poly_ls, crs=2193)

    return polygon_geoseries

def plot_gis_lines(gis_file: Union[str, pathlib.Path], ax: plt.Axes, color: str, linewidth: int = 0.3, clip_bounds: list = None,
                   linestyle: str = "-"):
    data = gpd.read_file(gis_file)
    if clip_bounds is not None:
        clipping_poly = box(*clip_bounds)
        clipped_data = gpd.clip(data, clipping_poly)
    else:
        clipped_data = data

    clipped_data.plot(color=color, ax=ax, linewidth=linewidth, linestyle=linestyle)

def plot_gis_polygons(gis_file: Union[str, pathlib.Path], ax: plt.Axes, edgecolor: str, linewidth: int = 0.3, clip_bounds: list = None,
                      linestyle: str = "-", facecolor="none"):
    data = gpd.read_file(gis_file)
    if clip_bounds is not None:
        clipping_poly = box(*clip_bounds)
        clipped_data = gpd.clip(data, clipping_poly)
    else:
        clipped_data = data

    clipped_data.plot(edgecolor=edgecolor, ax=ax, linewidth=linewidth, linestyle=linestyle, facecolor=facecolor)






def plot_coast(ax: plt.Axes, clip_boundary: list = None, edgecolor: str = "0.5", facecolor: str = "none", linewidth: int = 0.3,
               trim_polygons=True, wgs: bool = False):
    if clip_boundary is None:
        if wgs:
            x1, y1, x2, y2 = [min_x1_wgs, min_y1_wgs, max_x2_wgs, max_y2_wgs]
        else:
            x1, y1, x2, y2 = [min_x1, min_y1, max_x2, max_y2]
    else:
        assert isinstance(clip_boundary, list)
        assert len(clip_boundary) == 4
        x1, y1, x2, y2 = clip_boundary
    if trim_polygons:
        clipped_gs = clip_coast_with_trim(x1, y1, x2, y2, wgs=wgs)
    else:
        clipped_gs = clip_coast(x1, y1, x2, y2, wgs=wgs)

    clipped_gs.plot(ax=ax, edgecolor=edgecolor, facecolor=facecolor, linewidth=linewidth)
    if wgs:
        aspect = 1/np.cos(np.radians(np.mean([y1, y2])))
        ax.set_aspect(aspect)
    return x1, y1, x2, y2












