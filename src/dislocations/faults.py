from __future__ import annotations

import numpy as np
from typing import Union, List, Any
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import split
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as mpl_patch
from matplotlib import cm
from okada4py import okada92
from geopandas import GeoSeries
from itertools import product
from dislocations.utilities import geopandas_polygon_to_gmt
from copy import deepcopy


# from icp_error.io.shapefile_operations import geopandas_polygon_to_gmt

mu = np.array([30.0e9])
nu = np.array([0.25])


def calculate_strike(x1: Union[float, int], y1: Union[float, int], x2: Union[float, int], y2: Union[float, int]):
    """
    Function to calculate bearing from point (x1, y1) to point (x2, y2).
    Only for projected coordinates
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return:
    """
    x_diff = x2 - x1
    y_diff = y2 - y1

    bearing = 90. - np.degrees(np.arctan2(y_diff, x_diff))
    while bearing < 0.:
        bearing += 360.

    while bearing >= 360.:
        bearing -= 360.

    return bearing


def isect_line_plane(p0: Union[Point, np.ndarray], u: np.ndarray, p_co: Union[Point, np.ndarray],
                     p_no: np.ndarray, epsilon=1e-6):
    """
    p0, p1: Define the line.
    p_co, p_no: define the plane:
        p_co Is a point on the plane (plane coordinate).
        p_no Is a normal vector defining the plane direction;
             (does not need to be normalized).

    Return a Vector or None (when the intersection can't be found).
    """

    dot = np.dot(p_no, u)

    if abs(dot) > epsilon:
        # The factor of the point between p0 -> p1 (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # Otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: infront of p1.
        w = p0 - p_co
        fac = - 1 * np.dot(p_no, w) / dot
        u *= fac
        return p0 + u
    else:
        # The segment is parallel to plane.
        return None


def approximate_listric_profile(original_fault: ListricFault, tile_size: Union[float, int], interpolation_dist: float = 10.):
    extended_listric = deepcopy(original_fault)
    extended_listric.add_lower_patch(dip=original_fault.lowest_patch.dip, patch_width=2 * tile_size)

    ext_cross_sec_dists, ext_cross_sec_depths, _ = extended_listric.calc_fault_cross_section()
    extended_along_dip_line = LineString(np.column_stack((ext_cross_sec_dists, ext_cross_sec_depths)))

    interpolated_points = [extended_along_dip_line.interpolate(dist) for dist in
                           np.arange(0., extended_along_dip_line.length + interpolation_dist,
                                     interpolation_dist)]

    tile_corners = [interpolated_points[0]]
    while tile_corners[-1].y > original_fault.lowest_patch.bottom_z:
        top_point = tile_corners[-1]
        top_point_index = interpolated_points.index(top_point)
        remaining_points = interpolated_points[top_point_index + 1:]

        dists = np.abs(np.array([(point.distance(top_point) - tile_size) for point in remaining_points]))
        bottom_point = remaining_points[np.argmin(dists)]
        tile_corners.append(bottom_point)

    if abs(tile_corners[-1].y - original_fault.lowest_patch.bottom_z) > abs(
            tile_corners[-2].y - original_fault.lowest_patch.bottom_z):
        tile_corners = tile_corners[:-1]

    patch_heights = np.array([(top.y - bot.y) for top, bot in zip(tile_corners, tile_corners[1:])])
    patch_surface_widths = np.array([(bot.x - top.x) for top, bot in zip(tile_corners, tile_corners[1:])])

    return patch_heights, patch_surface_widths


class GreensFunctionComponent:
    def __init__(self, displacements: np.ndarray, shape: tuple = None):
        assert displacements.shape[-1] == 3, "Expecting array with 3 columns"
        if displacements.shape == (3,):
            self.ux, self.uy, self.uz = [displacements[i] for i in range(3)]
        elif shape is None:
            self.ux, self.uy, self.uz = [displacements[:, i] for i in range(3)]
        else:
            if len(shape) == 1:
                assert shape[0] == displacements.shape[0], "Displacement array not of expected shape"
                self.ux, self.uy, self.uz = [displacements[:, i] for i in range(3)]
            else:
                assert displacements.shape[0] == shape[0] * shape[1], "Displacement array not of expected shape"
                self.ux, self.uy, self.uz = [displacements[:, i].reshape(shape) for i in range(3)]


class GreensFunctions:
    def __init__(self, strike_slip: GreensFunctionComponent, dip_slip: GreensFunctionComponent):
        self._strike_slip = strike_slip
        self._dip_slip = dip_slip

    @property
    def strike_slip(self):
        return self._strike_slip

    @property
    def dip_slip(self):
        return self._dip_slip

    @property
    def ss(self):
        return self._strike_slip

    @property
    def ds(self):
        return self._dip_slip


class Patch:
    def __init__(self, top_z_tolerance: Union[float, int] = 10,
                 patch_i_index: int = None, patch_j_index: int = None, patch_index: int = None):
        self.top_z_tolerance = top_z_tolerance
        self._centroid, self._dip, self._dip_direction, self._length, self._width = (None,) * 5
        self._strike_slip, self._dip_slip = (None,) * 2
        self._i = patch_i_index
        self._j = patch_j_index
        self._patch_index = patch_index

    # Getters and setters for 7 parameters necessary to define patch geometry
    @property
    def centroid(self):
        return self._centroid

    @centroid.setter
    def centroid(self, array: Union[list, tuple, set, np.ndarray]):
        assert len(array) == 3, "Specify X, Y, Z of centroid (in metres)"
        assert array[-1] < 0, "Centroid Z must be negative (below sea level)"
        self._centroid = np.array([array[0], array[1], array[2]])

    @property
    def centroid_depth(self):
        assert self.centroid[-1] < 0, "Z should be negative"
        return -1 * self.centroid[-1] + 10

    @property
    def dip(self):
        return self._dip

    @dip.setter
    def dip(self, dip_value: Union[float, int]):
        assert 0 <= dip_value <= 90, "Dip is in degrees and should be between zero and 90!"
        self._dip = dip_value

    @property
    def dip_direction(self):
        return self._dip_direction

    @dip_direction.setter
    def dip_direction(self, bearing: Union[float, int]):
        while bearing < 0:
            bearing += 360
        while bearing > 360:
            bearing -= 360
        self._dip_direction = bearing

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, length_value: Union[float, int]):
        assert length_value > 0, "Length must be positive (and in this case in metres)!"
        self._length = length_value

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, width_value: Union[float, int]):
        assert width_value > 0, "Width must be positive (and in this case in metres)!"
        self._width = width_value

    @property
    def area(self):
        assert all([x is not None for x in (self.width, self.length)])
        return self.width * self.length

    # Getter and setter for strike (setter sets dip direction)
    @property
    def strike(self):
        strike_bearing = self.dip_direction - 90
        if strike_bearing < 0:
            strike_bearing += 360
        return strike_bearing

    @strike.setter
    def strike(self, strike_bearing: Union[float, int]):
        self.dip_direction = strike_bearing + 90

    # Getters for patch id
    @property
    def i(self):
        return self._i

    @i.setter
    def i(self, index: int):
        assert index >= 0
        self._i = index

    @property
    def j(self):
        return self._j

    @j.setter
    def j(self, index: int):
        assert index >= 0
        self._j = index

    @property
    def patch_index(self):
        return self._patch_index

    @patch_index.setter
    def patch_index(self, index: int):
        assert index >= 0
        self._patch_index = index

    # Useful vectors
    @property
    def down_dip_vector(self):
        assert self.dip > 0
        dip_radians = np.radians(self.dip)
        direction_radians = np.radians(self.dip_direction)
        cos_dip = np.cos(dip_radians)
        vector_x = np.sin(direction_radians) * cos_dip
        vector_y = np.cos(direction_radians) * cos_dip
        vector_z = -1 * np.sin(dip_radians)

        return np.array([vector_x, vector_y, vector_z])

    @property
    def normal_vector(self):
        dip_radians = np.radians(self.dip)
        direction_radians = np.radians(self.dip_direction)
        sin_dip = np.sin(dip_radians)
        vector_x = np.sin(direction_radians) * sin_dip
        vector_y = np.cos(direction_radians) * sin_dip
        vector_z = np.cos(dip_radians)

        return np.array([vector_x, vector_y, vector_z])

    @property
    def plane_equation_coefficients(self):
        a, b, c = self.normal_vector
        d = np.dot(self.normal_vector, self.down_dip_vector)
        return a, b, c, d

    @property
    def along_strike_vector(self):
        strike_radians = np.radians(self.strike)
        return np.array([np.sin(strike_radians), np.cos(strike_radians), 0])

    @property
    def across_strike_vector(self):
        direction_radians = np.radians(self.dip_direction)
        return np.array([np.sin(direction_radians), np.cos(direction_radians), 0])

    # Top and bottom of patch
    @property
    def top_z(self):
        sin_dip = np.sin(np.radians(self.dip))
        z = self.centroid[-1] + sin_dip * self.width / 2
        if abs(z) < self.top_z_tolerance:
            return 0
        else:
            return z

    @property
    def bottom_z(self):
        sin_dip = np.sin(np.radians(self.dip))
        z = self.centroid[-1] - sin_dip * self.width / 2
        return z

    @property
    def top_centre(self):
        centre = self.centroid - (self.width / 2) * self.down_dip_vector
        if abs(centre[-1]) < self.top_z_tolerance:
            centre[-1] = 0
        return centre

    @top_centre.setter
    def top_centre(self, array: Union[list, tuple, set, np.ndarray]):
        assert len(array) == 3, "Specify X, Y, Z of top_centre (in metres)"
        assert array[-1] <= 0, "top_centre Z must be zero or negative (below sea level)"
        top_centre_array = np.array([array[0], array[1], array[2]])
        centroid_array = top_centre_array + (self.width / 2) * self.down_dip_vector

        self._centroid = centroid_array

    @property
    def bottom_centre(self):
        return self.centroid + (self.width / 2) * self.down_dip_vector

    @property
    def dip_slip(self):
        return self._dip_slip

    @dip_slip.setter
    def dip_slip(self, value: Union[float, int]):
        self._dip_slip = value

    @property
    def strike_slip(self):
        return self._strike_slip

    @strike_slip.setter
    def strike_slip(self, value: Union[float, int]):
        self._strike_slip = value

    @property
    def total_slip(self):
        return np.sqrt(self.dip_slip**2 + self.strike_slip**2)

    @property
    def rake(self):
        np.degrees(np.arctan2(self.strike_slip, self.dip_slip))
        return

    def set_slip_rake(self, slip: Union[float, int], rake: Union[float, int]):
        """
        :param slip:
        :param rake:
        :return:
        """
        self.dip_slip = slip * np.sin(np.radians(rake))
        self.strike_slip = slip * np.cos(np.radians(rake))

    def slipvec_to_rake(self, slipvec: float):
        """
        Program to perform the 'inverse' of slipvec.
        Arguments: strike, dip azimuth of slip vector (all degrees)
        Returns rake (degrees)
        """

        angle = self.strike - slipvec
        if angle < -180.:
            angle = 360. + angle
        elif angle > 180.:
            angle = angle - 360.

        if angle == 90.:
            rake = 90.
        elif angle == -90.:
            rake = -90.
        else:
            strike_par = np.cos(np.radians(angle))
            strike_perp = np.sin(np.radians(angle)) / np.cos(np.radians(self.dip))
            rake = np.degrees(np.arctan2(strike_perp, strike_par))
        return rake

    @property
    def corner_array(self):
        corner_ls = []
        for xi, yi in zip((-1, -1, 1, 1, -1), (-1, 1, 1, -1, -1)):
            x_offset = xi * (self.length / 2) * self.along_strike_vector
            y_offset = yi * (self.width / 2) * self.down_dip_vector
            corner = self.centroid + x_offset + y_offset
            corner_ls.append(corner)

        return np.array(corner_ls)

    @property
    def corner_polygon(self):
        return Polygon(self.corner_array, )

    def polygon_to_shp(self, shp_name: str, epsg: int = 2193):
        """
        Export geometry as shapefile, usually in nztm
        :param shp_name:
        :param epsg:
        :return:
        """
        assert shp_name[-4:] == ".shp"
        gs = GeoSeries(self.corner_polygon)
        if epsg is not None:
            gs.crs = "epsg:{:d}".format(epsg)

        gs.to_file(shp_name)

    def polygon_to_gmt(self, gmt_file: str):
        assert gmt_file[-4:] == ".gmt"
        gs = GeoSeries(self.corner_polygon)

        geopandas_polygon_to_gmt(gs, gmt_file)

    @property
    def top_corner(self):
        x_offset = -1 * (self.length / 2) * self.along_strike_vector
        y_offset = -1 * (self.width / 2) * self.down_dip_vector
        corner = self.centroid + x_offset + y_offset
        return corner

    def patch_collection(self, patch_colour: str = "b", patch_alpha: float = 0.5, line_colour: str = "k",
                         line_width: Union[float, int] = 1):
        x, y, z = self.corner_array.T.tolist()
        collection_list = [list(zip(x, y, z))]
        patch_collection = Poly3DCollection(collection_list, alpha=patch_alpha, facecolors=patch_colour)
        line_collection = Line3DCollection(collection_list, linewidths=line_width, colors=line_colour)
        return patch_collection, line_collection

    def greens_functions(self, site_x: Union[int, float, list, tuple, np.ndarray],
                         site_y: Union[int, float, list, tuple, np.ndarray],
                         site_z: Union[int, float, list, tuple, np.ndarray] = None):
        if any(isinstance(site_x, a) for a in (int, float)):
            site_x = np.array([site_x], dtype=float)
            site_y = np.array([site_y], dtype=float)

        else:
            site_x = np.array(site_x, dtype=float)
            site_y = np.array(site_y, dtype=float)

        x_with_shape = np.array(site_x)
        x_shape = x_with_shape.shape
        x_array, y_array = x_with_shape.flatten(), np.array(site_y).flatten()
        assert x_array.shape == y_array.shape, "X and Y arrays must be the same shape"
        if site_z is None:
            z_array = np.zeros(x_array.shape)
        else:
            z_array = np.array(site_z).flatten()
            assert x_array.shape == z_array.shape, "X and Z arrays must be the same shape"
        xc, yc, depth, length, width, dip, strike = [np.array([float(a)]) for a in (self.centroid[0], self.centroid[1],
                                                                                    self.centroid_depth, self.length,
                                                                                    self.width, self.dip, self.strike)]
        unit1 = np.array([1.0])
        unit0 = np.array([0.0])
        ds_u, _, _, _, _ = okada92(x_array, y_array, z_array, xc, yc, depth, length, width, dip, strike, unit0, unit1,
                                   unit0, mu, nu)
        ss_u, _, _, _, _ = okada92(x_array, y_array, z_array, xc, yc, depth, length, width, dip, strike, unit1, unit0,
                                   unit0, mu, nu)

        ds_array = np.reshape(ds_u, (len(x_array), 3))
        ss_array = np.reshape(ss_u, (len(x_array), 3))

        shape = None if len(x_shape) == 1 else x_shape
        ds_gf = GreensFunctionComponent(ds_array, shape)
        ss_gf = GreensFunctionComponent(ss_array, shape)

        return GreensFunctions(ss_gf, ds_gf)

    @classmethod
    def from_centroid(cls, centroid_x: Union[float, int] = None, centroid_y: Union[float, int] = None,
                      centroid_z: Union[float, int] = None, centroid_array: Union[list, tuple, set, np.ndarray] = None,
                      patch_length: Union[float, int] = None, patch_width: Union[float, int] = None,
                      dip: Union[float, int] = None, dip_direction: Union[float, int] = None,
                      strike: Union[float, int] = None, top_z_tolerance: Union[float, int] = 10,
                      patch_i_index: int = None, patch_j_index: int = None, patch_index: int = None):
        patch_i = cls(top_z_tolerance=top_z_tolerance, patch_i_index=patch_i_index, patch_j_index=patch_j_index,
                      patch_index=patch_index)
        # make sure all centroid arguments are provided and that none are duplicated
        centroid_conditions = [a is not None for a in [centroid_x, centroid_array]]
        assert all([any(centroid_conditions), not all(centroid_conditions)]), "Provide array or x,y,z individually"
        if centroid_x is not None:
            assert all([a is not None for a in [centroid_y, centroid_z]]), "Specify x, y, z of centroid"
            patch_i.centroid = np.array([centroid_x, centroid_y, centroid_z])
        else:
            patch_i.centroid = centroid_array

        # Make sure strike or dip direction are specified (but not both)
        strike_conditions = [a is not None for a in [strike, dip_direction]]
        assert all([any(strike_conditions), not all(strike_conditions)]), "Specify strike or dip dir. (but not both)"
        if strike is not None:
            patch_i.strike = strike
        else:
            patch_i.dip_direction = dip_direction

        # Other parameters
        other_conditions = [a is not None for a in [patch_length, patch_width, dip]]
        assert all(other_conditions), "Specify all of length, width and dip"
        patch_i.length, patch_i.width, patch_i.dip = [patch_length, patch_width, dip]

        return patch_i

    @classmethod
    def from_top_dimensions(cls, top_centre_x: Union[float, int] = None, top_centre_y: Union[float, int] = None,
                            top_centre_z: Union[float, int] = None,
                            top_centre_array: Union[list, tuple, set, np.ndarray] = None,
                            patch_length: Union[float, int] = None, patch_width: Union[float, int] = None,
                            bottom_z: Union[float, int] = None, surface_width: Union[float, int] = None,
                            patch_height: Union[float, int] = None,
                            dip: Union[float, int] = None, dip_direction: Union[float, int] = None,
                            strike: Union[float, int] = None, top_z_tolerance: Union[float, int] = 10,
                            patch_i_index: int = None, patch_j_index: int = None, patch_index: int = None):
        """
        To create patch from top centre, strike, length and other parameters
        :param top_centre_x:
        :param top_centre_y:
        :param top_centre_z:
        :param top_centre_array:
        :param patch_length:
        :param patch_width:
        :param bottom_z:
        :param surface_width:
        :param patch_height:
        :param dip:
        :param dip_direction:
        :param strike:
        :param top_z_tolerance:
        :param patch_i_index:
        :param patch_j_index:
        :param patch_index:
        :return:
        """

        patch_i = cls(top_z_tolerance=top_z_tolerance, patch_i_index=patch_i_index, patch_j_index=patch_j_index,
                      patch_index=patch_index)
        # Make sure strike or dip direction are specified (but not both)
        strike_conditions = [a is not None for a in [strike, dip_direction]]
        assert all([any(strike_conditions), not all(strike_conditions)]), "Specify strike or dip dir. (but not both)"
        if strike is not None:
            patch_i.strike = strike
        else:
            patch_i.dip_direction = dip_direction

        if dip is None:
            assert all([a is not None for a in [surface_width, patch_height]])
            dip = np.degrees(np.arctan(patch_height / surface_width))

        # Patch length and dip
        other_conditions = [a is not None for a in [patch_length, dip]]
        assert all(other_conditions), "Specify all of length and dip"
        patch_i.length, patch_i.dip = [patch_length, dip]

        # make sure all top surface point arguments are provided and that none are duplicated
        top_centre_conditions = [a is not None for a in [top_centre_x, top_centre_array]]
        assert all([any(top_centre_conditions), not all(top_centre_conditions)]), "Provide array or x,y,z individually"

        # Use one of several parameters to set patch width
        # Check that conflicting parameters are not set
        width_conditions = [a is not None for a in [patch_width, bottom_z, patch_height, surface_width]]
        assert all([any(width_conditions), not all(width_conditions)]), "Specify ONE constraint on patch width"
        top_z = top_centre_z if top_centre_z is not None else top_centre_array[-1]

        if patch_width is not None:
            patch_i.width = patch_width
        else:
            # Calculate patch_width from other parameters
            if patch_height is not None:
                assert patch_height > 0., "Patch height must be positive"
                calculated_width = patch_height / np.sin(np.radians(patch_i.dip))

            elif surface_width is not None:
                assert surface_width > 0., "Surface width must be positive"
                calculated_width = surface_width / np.cos(np.radians(patch_i.dip))

            else:
                assert bottom_z < top_z, "Bottom must be deeper than top of patch"
                calculated_width = (top_z - bottom_z) / np.sin(np.radians(patch_i.dip))
            patch_i.width = calculated_width

        # Define centroid of patch from top_centre
        if top_centre_x is not None:
            assert all([a is not None for a in [top_centre_y, top_centre_z]]), "Specify x, y, z of top_centre"
            patch_i.top_centre = np.array([top_centre_x, top_centre_y, top_centre_z])
        else:
            patch_i.top_centre = top_centre_array

        return patch_i

    @classmethod
    def from_top_endpoints(cls, x1: Union[float, int], y1: Union[float, int], x2: Union[float, int],
                           y2: Union[float, int], dip: Union[float, int], top_z: Union[float, int] = 0,
                           bottom_z: Union[float, int] = None, surface_width: Union[float, int] = None,
                           patch_width: Union[float, int] = None, dip_direction: Union[float, int] = None,
                           strike: Union[float, int] = None, patch_height: Union[float, int] = None,
                           strike_tolerance: Union[float, int] = 1, top_z_tolerance: Union[float, int] = 10,
                           patch_i_index: int = None, patch_j_index: int = None, patch_index: int = None):
        """

        :param x1:
        :param y1:
        :param x2:
        :param y2:
        :param dip:
        :param top_z:
        :param bottom_z:
        :param surface_width:
        :param patch_width:
        :param dip_direction:
        :param strike:
        :param patch_height:
        :param strike_tolerance:
        :param top_z_tolerance:
        :param patch_i_index:
        :param patch_j_index:
        :param patch_index:
        :return:
        """

        top_cen_x = 0.5 * (x1 + x2)
        top_cen_y = 0.5 * (y1 + y2)

        patch_length = np.sqrt((x2 - x1)**2 + (y2-y1)**2)

        # Calculate strike using supplied coordinates
        preliminary_strike = calculate_strike(x1, y1, x2, y2)

        # compare preliminary strike with any other supplied strike or dip direction

        strike_conditions = [a is not None for a in [strike, dip_direction]]
        assert not all(strike_conditions), "Only specify at most one of strike/dip_direction"

        # If nothing else specified, use preliminary_strike
        if not any(strike_conditions):
            final_strike = preliminary_strike

        # If value for strike specified, compare with preliminary
        elif strike is not None:
            # If the same, use preliminary
            if abs(preliminary_strike - strike) <= strike_tolerance:
                final_strike = preliminary_strike
            # If 180 degrees different, use supplied strike value
            elif abs(abs(preliminary_strike - strike) - 180) <= strike_tolerance:
                final_strike = strike
            else:
                raise ValueError("Specified strike incompatible with line orientation")

        # Otherwise, dip_direction must have been specified
        else:
            # Convert to equivalent strike value for ease of comparison
            strike_from_dd = dip_direction - 90.
            while strike_from_dd <= 0.:
                strike_from_dd += 360.
            # If consistent, use preliminary
            if abs(preliminary_strike - strike_from_dd) <= strike_tolerance:
                final_strike = preliminary_strike
            # If 180 degrees different, use supplied strike value
            elif abs(abs(preliminary_strike - strike_from_dd) - 180) <= strike_tolerance:
                final_strike = strike_from_dd
            else:
                raise ValueError("Specified dip_direction incompatible with line orientation")

        patch_i = cls.from_top_dimensions(top_centre_array=(top_cen_x, top_cen_y, top_z), bottom_z=bottom_z,
                                          surface_width=surface_width, patch_width=patch_width,
                                          dip=dip, strike=final_strike, patch_height=patch_height,
                                          patch_length=patch_length, top_z_tolerance=top_z_tolerance,
                                          patch_i_index=patch_i_index, patch_j_index=patch_j_index,
                                          patch_index=patch_index)

        return patch_i

    def tsunami_greens_function(self, site_x: float, site_y: float, site_z: float = None):
        if any(isinstance(site_x, a) for a in (int, float)):
            site_x = np.array([site_x], dtype=float)
            site_y = np.array([site_y], dtype=float)
        xc, yc, depth, length, width, dip, strike = [np.array([float(a)]) for a in (self.centroid[0], self.centroid[1],
                                                                                    self.centroid_depth, self.length,
                                                                                    self.width, self.dip, self.strike)]
        x_with_shape = np.array(site_x)
        x_array, y_array = x_with_shape.flatten(), np.array(site_y).flatten()
        assert x_array.shape == y_array.shape, "X and Y arrays must be the same shape"

        if site_z is None:
            z_array = np.zeros(x_array.shape)
        else:
            z_array = np.array(site_z).flatten()
            assert x_array.shape == z_array.shape, "X and Z arrays must be the same shape"

        unit1 = np.array([1.0])
        unit0 = np.array([0.])
        ds_u, _, _, _, _ = okada92(x_array, y_array, z_array, xc, yc, depth, length, width, dip, strike, unit0, unit1,
                                   unit0, mu, nu)
        return ds_u[-1]

    def projected_line(self, projection_centre_x: Union[int, float], projection_centre_y: Union[int, float],
                       projection_strike: Union[int, float]):
        strike_radians = np.radians(projection_strike)
        strike_vector = np.array([np.sin(strike_radians), np.cos(strike_radians)])

        top_centre_vec = np.array([projection_centre_x, projection_centre_y]) - self.top_centre[:2]
        bot_centre_vec = np.array([projection_centre_x, projection_centre_y]) - self.bottom_centre[:2]
        top_centre_distance = np.dot(strike_vector, top_centre_vec)
        bot_centre_distance = np.dot(strike_vector, bot_centre_vec)

        out_array = np.array([[top_centre_distance, self.top_z],
                              [bot_centre_distance, self.bottom_z]])
        return out_array


class ListricFault:
    def __init__(self, top_patch: Patch = None, top_z_tolerance: Union[float, int] = 10, patch_i: int = None):
        self._top_patch = top_patch
        self.top_z_tolerance = top_z_tolerance
        self._lower_patches = []
        self.i = patch_i

    @property
    def top_centre(self):
        assert self.top_patch is not None, "Set top patch first"
        return self.top_patch.top_centre

    @property
    def dip_direction(self):
        assert self.top_patch is not None, "Set top patch first"
        return self.top_patch.dip_direction

    @property
    def strike(self):
        assert self.top_patch is not None, "Set top patch first"
        return self.top_patch.strike

    @property
    def length(self):
        assert self.top_patch is not None, "Set top patch first"
        return self.top_patch.length

    @property
    def patches(self):
        assert self.top_patch is not None, "Set top patch first"
        return [self.top_patch] + self.lower_patches

    @property
    def num_patches(self):
        return len(self.patches)

    @property
    def top_patch(self):
        return self._top_patch

    @top_patch.setter
    def top_patch(self, patch: Patch):
        if patch.j is not None:
            assert patch.j == 0
        else:
            patch.j = 0
        if len(self._lower_patches) > 0:
            print("Resetting top patch may mess up all lower patches!")
            print("Better to start new fault object.")
        self._top_patch = patch

    @property
    def lower_patches(self):
        return self._lower_patches

    @property
    def lowest_patch(self):
        assert self.top_patch is not None, "Set top patch first"
        if len(self.lower_patches) == 0:
            return self.top_patch
        else:
            return self.lower_patches[-1]

    @property
    def patch_corners(self):
        return [patch.corner_array for patch in self.patches]

    @property
    def patch_corner_polygons(self):
        return [Polygon(patch.corner_array) for patch in self.patches]

    def polygon_to_shp(self, shp_name: str, epsg: int = 2193):
        """
        Export geometry as shapefile, usually in nztm
        :param shp_name:
        :param epsg:
        :return:
        """
        assert shp_name[-4:] == ".shp"
        gs = GeoSeries(self.patch_corner_polygons)
        if epsg is not None:
            gs.crs = "epsg:{:d}".format(epsg)

        gs.to_file(shp_name)

    def polygon_to_gmt(self, gmt_file: str):
        assert gmt_file[-4:] == ".gmt"
        gs = GeoSeries(self.patch_corner_polygons)

        geopandas_polygon_to_gmt(gs, gmt_file)

    def patch_collection(self, patch_colour: str = "b", patch_alpha: float = 0.5, line_colour: str = "k",
                         line_width: Union[float, int] = 1):
        collection_list = []
        for corners in self.patch_corners:
            x, y, z = corners.T.tolist()
            collection_list.append(list(zip(x, y, z)))
        patch_collection = Poly3DCollection(collection_list, alpha=patch_alpha, facecolors=patch_colour)
        line_collection = Line3DCollection(collection_list, linewidths=line_width, colors=line_colour)
        return patch_collection, line_collection

    def add_lower_patch(self, patch_width: Union[float, int] = None,
                        bottom_z: Union[float, int] = None, surface_width: Union[float, int] = None,
                        patch_height: Union[float, int] = None,
                        dip: Union[float, int] = None):
        patch_top_centre = self.lowest_patch.bottom_centre
        new_patch = Patch.from_top_dimensions(top_centre_array=patch_top_centre, patch_length=self.length,
                                              patch_width=patch_width, bottom_z=bottom_z, surface_width=surface_width,
                                              patch_height=patch_height, dip=dip, dip_direction=self.dip_direction,
                                              patch_j_index=self.num_patches, patch_i_index=self.i)
        self._lower_patches.append(new_patch)

    def calc_fault_cross_section(self):
        """takes a listric fault and outputs
        along_dip_vertices: XYZ vertices of cross section line (using top/botton patch centers)
        cross_sec_dists: horizontal distances along cross section line for each vertex
        cross_sec_depths: Z coordinate for each vertex"""

        along_dip_vertices = []
        cross_sec_dists = []
        cross_sec_depths = []
        # find top center coords for each patch, plus bottom center for bottom patch
        # Shapely needs input as a list of tuples. Tuples can be (x,y) or (x,y,z)
        for patch in self.patches:
            along_dip_vertices.append(patch.top_centre)  # check if tuple needed
            cross_sec_depths.append(patch.top_z)
        along_dip_vertices.append(self.lowest_patch.bottom_centre)  # check if tuple needed
        cross_sec_depths.append(self.lowest_patch.bottom_z)
        #print("Cross section depths: " + str(cross_sec_depths))

        # find horizontal distances to vertices (distance axis in cross-section)
        for vertex in along_dip_vertices:
            difference_vector = vertex[:-1] - self.top_centre[:-1]  # this is the vector
            cross_sec_dists.append(np.linalg.norm(difference_vector))  # finds magnitude of difference vector

        return cross_sec_dists, cross_sec_depths, along_dip_vertices

    def calc_centroid_cross_section(self):
        """takes a listric fault and outputs
        cross_sec_dist: the horizontal distance and Z coordinate for each patch centroid
        cross_sec_depths: the Z coordinate for each centroid"""
        centroid_dists = []
        centroid_depths = []
        centroids = []
        centroid_line_depths = []
        centroid_line_dists = []
        centroid_line_verts = []

        # find top center coords for each patch, centroid, plus bottom center for bottom patch
        for patch in self.patches:
            centroid_depths.append((patch.centroid_depth - 10) * -1)
            centroids.append(patch.centroid)
            centroid_line_depths.append(patch.top_centre[2])
            centroid_line_verts.append(patch.top_centre)
            centroid_line_depths.append((patch.centroid_depth - 10) * -1)
            centroid_line_verts.append(patch.centroid)
        centroid_line_depths.append(self.lowest_patch.bottom_z)
        centroid_line_verts.append(self.lowest_patch.bottom_centre)

        # find horizontal distances to centroid (distance axis in cross-section)
        for centroid in centroids:
            difference_vector1 = centroid[:-1] - self.top_centre[:-1]  # this is the vector
            centroid_dists.append(np.linalg.norm(difference_vector1))  # finds magnitude of difference vector

        # find horizontal distances to centroids and patch vertices (distance axis in cross-section)
        for vert in centroid_line_verts:
            difference_vector2 = vert[:-1] - self.top_centre[:-1]  # this is the vector
            centroid_line_dists.append(np.linalg.norm(difference_vector2))  # finds magnitude of difference vector

        cross_sec_line_coords = list(zip(centroid_line_dists, centroid_line_depths))
        centroid_cross_sec_verts = list(zip(centroid_dists, centroid_depths))

        return centroid_cross_sec_verts, cross_sec_line_coords

    def cross_section_linestring(self):
        cross_sec_dists, cross_sec_depths, _ = self.calc_fault_cross_section()
        return LineString(np.column_stack((cross_sec_dists, cross_sec_depths)))

    def cross_section_heights_surface_widths(self):
        cross_sec_dists, cross_sec_depths, _ = self.calc_fault_cross_section()
        cross_sec_depths = np.array(cross_sec_depths)
        cross_sec_dists = np.array(cross_sec_dists)
        patch_surface_widths = cross_sec_dists[1:] - cross_sec_dists[:-1]
        patch_heights = cross_sec_depths[:-1] - cross_sec_depths[1:]

        return patch_heights, patch_surface_widths

    def taper_slip_along_dip(self, slip, cross_sec_dists, cross_sec_depths):
        """makes a list of slip values for the centroids of each patch.
        Based on the distance along dip of the fault and a
        triangular slip distribution from the centre"""

        # make an empty list of line vertices and cross-section distances
        along_dip_vertices = []
        # find top center coords for each patch, plus bottom center for bottom patch
        # Shapely needs input as a list of tuples. Tuples can be (x,y) or (x,y,z)
        for patch in self.patches:
            along_dip_vertices.append(patch.top_centre) # check if tuple needed
            #cross_sec_depths.append(patch.top_z)
        along_dip_vertices.append(self.lowest_patch.bottom_centre) # check if tuple is needed

        # find horizontal distances to vertices
        for vertex in along_dip_vertices:
           difference_vector = vertex[:-1] - self.top_centre[:-1]  # this is the vector
           cross_sec_dists.append(np.linalg.norm(difference_vector)) # finds magnitude of difference vector

        # make shapely line from vertices
        along_dip_line = LineString(along_dip_vertices)
        centre_dist = along_dip_line.length/2

        # make list of patch centroids
        ls_patch_centroids = []
        for patch in self.patches:
            ls_patch_centroids.append(tuple(patch.centroid))

        # slip per patch
        # calculate along-dip distance to centroids
        along_dip_slip_vals = [] #empty list of slip amounts
        for centroid in ls_patch_centroids:
            # find distance along fault (along dip)
            dist = along_dip_line.project(Point(centroid))
            # find along-fault distance from fault centre
            dist_from_centre = abs(dist - centre_dist)
            # multiply peak slip by ratio
            # defined by traingular distribution; peak at midpoint
            patch_slip = slip * ((centre_dist - dist_from_centre)/centre_dist)
            along_dip_slip_vals.append(patch_slip)

        return along_dip_slip_vals #, cross_sec_depths, cross_sec_dists

    #def taper_slip_along_edges(self, fault_trace, slip):

    def projected_outline(self, projection_centre_x: Union[int, float], projection_centre_y: Union[int, float],
                          projection_strike: Union[int, float]):
        all_points = np.vstack([patch.projected_line(projection_centre_x, projection_centre_y, projection_strike)
                                for patch in self.patches])
        return np.unique(np.round(all_points, decimals=4), axis=0)

    @classmethod
    def from_depth_profile(cls, patch_length: Union[float, int], patch_heights: np.ndarray = None,
                           patch_dips: np.ndarray = None, patch_surface_widths = None,
                           top_centre_x: Union[float, int] = None, top_centre_y: Union[float, int] = None,
                           top_centre_z: Union[float, int] = None,
                           top_centre_array: Union[list, tuple, set, np.ndarray] = None,
                           dip_direction: Union[float, int] = None,
                           strike: Union[float, int] = None, i_index: int = 0):
        assert len(patch_heights) is not None
        assert len(patch_heights) > 1
        if patch_dips is not None:
            assert len(patch_heights) == len(patch_dips)
            assert patch_surface_widths is None
        else:
            assert len(patch_heights) == len(patch_surface_widths)

        listric = cls(patch_i=i_index)

        if patch_dips is not None:
            top_patch = Patch.from_top_dimensions(patch_height=patch_heights[0], dip=patch_dips[0],
                                                  patch_length=patch_length, top_centre_x=top_centre_x,
                                                  top_centre_y=top_centre_y, top_centre_z=top_centre_z,
                                                  top_centre_array=top_centre_array, dip_direction=dip_direction,
                                                  strike=strike, patch_j_index=0, patch_i_index=i_index)
        else:
            top_patch = Patch.from_top_dimensions(patch_height=patch_heights[0], surface_width=patch_surface_widths[0],
                                                  patch_length=patch_length, top_centre_x=top_centre_x,
                                                  top_centre_y=top_centre_y, top_centre_z=top_centre_z,
                                                  top_centre_array=top_centre_array, dip_direction=dip_direction,
                                                  strike=strike, patch_j_index=0, patch_i_index=i_index)
        listric.top_patch = top_patch

        if patch_dips is not None:
            for patch_height, patch_dip in zip(patch_heights[1:], patch_dips[1:]):
                listric.add_lower_patch(patch_height=patch_height, dip=patch_dip)
        else:
            for patch_height, surface_width in zip(patch_heights[1:], patch_surface_widths[1:]):
                listric.add_lower_patch(patch_height=patch_height, surface_width=surface_width)
        return listric

    @classmethod
    def from_depth_profile_square_tile(cls, tile_size: Union[float, int], patch_heights: np.ndarray = None,
                           patch_dips: np.ndarray = None, patch_surface_widths = None,
                           top_centre_x: Union[float, int] = None, top_centre_y: Union[float, int] = None,
                           top_centre_z: Union[float, int] = None,
                           top_centre_array: Union[list, tuple, set, np.ndarray] = None,
                           dip_direction: Union[float, int] = None,
                           strike: Union[float, int] = None, i_index: int = 0, interpolation_dist: float = 10.):
        original = cls.from_depth_profile(patch_length=tile_size, patch_heights=patch_heights, patch_dips=patch_dips,
                                          patch_surface_widths=patch_surface_widths, top_centre_x=top_centre_x,
                                          top_centre_y=top_centre_y, top_centre_z=top_centre_z,
                                          top_centre_array=top_centre_array, dip_direction=dip_direction,
                                          strike=strike, i_index=i_index)

        approximate_heights, approximate_surf_wid = approximate_listric_profile(original, tile_size,
                                                                                interpolation_dist=interpolation_dist)
        approximated = cls.from_depth_profile(patch_length=tile_size, patch_heights=approximate_heights,
                                              patch_surface_widths=approximate_surf_wid, top_centre_x=top_centre_x,
                                              top_centre_y=top_centre_y, top_centre_z=top_centre_z,
                                              top_centre_array=top_centre_array, dip_direction=dip_direction,
                                              strike=strike, i_index=i_index)

        return approximated

    def update_i_index(self, new_i: int):
        for patch in self.patches:
            patch.i = new_i

    def translate_fault_along_strike(self, distance: float, new_i_index: int):
        new_top_centre = self.top_centre + distance * self.top_patch.along_strike_vector
        cross_sec_heights, cross_sec_widths = self.cross_section_heights_surface_widths()

        new_fault = self.from_depth_profile(self.top_patch.length, patch_heights=cross_sec_heights,
                                            patch_surface_widths=cross_sec_widths, top_centre_array=new_top_centre,
                                            strike=self.strike, i_index=new_i_index)

        return new_fault


    def tile_along_strike(self, num_either_side:int):
        to_translate = np.arange(-num_either_side * self.top_patch.length,
                                 (num_either_side + 1) * self.top_patch.length,
                                 self.top_patch.length)
        fault_list = []
        for i, shift in enumerate(to_translate):
            new_fault = self.translate_fault_along_strike(shift, i)


            fault_list.append(new_fault)

        return MultiListric(fault_list)


class MultiListric:
    def __init__(self, sub_faults: List[ListricFault], vertical_only: bool = False):
        assert len(sub_faults) > 1
        self.faults = sub_faults
        self.vertical_only = vertical_only
        self.max_i = max([fault.i for fault in self.faults])
        self.min_i = min([fault.i for fault in self.faults])
        self.min_j = min([patch.j for patch in self.faults[0].patches])
        self.max_j = max([patch.j for patch in self.faults[0].patches])

        self.patches = []
        for fault in self.faults:
            self.patches += fault.patches

        self.patch_dict = {}
        self.ij_dict = {}
        for patch_i, patch in enumerate(self.patches):
            self.patch_dict[patch_i] = patch
            patch.patch_index = patch_i
            self.ij_dict[(patch.i, patch.j)] = patch_i

        self._end_indices, self._start_indices, self._top_indices, self._bot_indices = (None,) * 4
        self.laplacian = None

    @property
    def start_indices(self):
        if self._start_indices is None:
            self._start_indices = np.array([patch_i for (patch_i, patch) in self.patch_dict.items()
                                            if patch.i == self.min_i])
        return self._start_indices

    @property
    def end_indices(self):
        if self._end_indices is None:
            self._end_indices = np.array([patch_i for (patch_i, patch) in self.patch_dict.items()
                                            if patch.i == self.max_i])
        return self._end_indices

    @property
    def top_indices(self):
        if self._top_indices is None:
            self._top_indices = np.array([patch_i for (patch_i, patch) in self.patch_dict.items()
                                            if patch.j == self.min_j])
        return self._top_indices

    @property
    def bot_indices(self):
        if self._bot_indices is None:
            self._bot_indices = np.array([patch_i for (patch_i, patch) in self.patch_dict.items()
                                          if patch.j == self.max_j])
        return self._bot_indices



    def compute_laplacian(self, double: bool = False):
        laplacian = np.zeros((len(self.patches), len(self.patches)))

        for patch_index, patch in self.patch_dict.items():
            adjacents = []
            for neighbour in [patch.i - 1, patch.i + 1]:
                if (neighbour, patch.j) in self.ij_dict.keys():
                    adjacents.append(self.ij_dict[(neighbour, patch.j)])
            for neighbour in [patch.j - 1, patch.j + 1]:
                if (patch.i, neighbour) in self.ij_dict.keys():
                    adjacents.append(self.ij_dict[(patch.i, neighbour)])

            for adjacent_i in adjacents:
                laplacian[patch_index, adjacent_i] = 1.0
            laplacian[patch_index, patch_index] = -1.0 * len(adjacents)

        if not double:
            self.laplacian = laplacian
        else:
            self.laplacian = np.hstack([laplacian, laplacian])

    @property
    def patch_corners(self):
        return [patch.corner_array for patch in self.patches]

    @property
    def patch_corner_polygons(self):
        return [Polygon(patch.corner_array) for patch in self.patches]

    def polygon_to_shp(self, shp_name: str, epsg: int = 2193):
        """
        Export geometry as shapefile, usually in nztm
        :param shp_name:
        :param epsg:
        :return:
        """
        assert shp_name[-4:] == ".shp"
        gs = GeoSeries(self.patch_corner_polygons)
        if epsg is not None:
            gs.crs = "epsg:{:d}".format(epsg)

        gs.to_file(shp_name)

    def polygon_to_gmt(self, gmt_file: str):
        assert gmt_file[-4:] == ".gmt"
        gs = GeoSeries(self.patch_corner_polygons)

        geopandas_polygon_to_gmt(gs, gmt_file)

    def patch_collection(self, patch_colour: str = "b", patch_alpha: float = 0.5, line_colour: str = "k",
                         line_width: Union[float, int] = 1):
        collection_list = []
        for corners in self.patch_corners:
            x, y, z = corners.T.tolist()
            collection_list.append(list(zip(x, y, z)))
        patch_collection = Poly3DCollection(collection_list, alpha=patch_alpha, facecolors=patch_colour)
        line_collection = Line3DCollection(collection_list, linewidths=line_width, colors=line_colour)
        return patch_collection, line_collection

    def matplotlib_patch_list2d(self):
        collection_list = []
        for corners in self.patch_corners:
            x, y, z = corners.T.tolist()
            collection_list.append(mpl_patch(list(zip(x, y))))
        return collection_list

    def patch_collection2d(self, patch_alpha: float = 0.5, line_colour: str = "k",
                           line_width: Union[float, int] = 1, cmap=cm.viridis):
        collection_list = self.matplotlib_patch_list2d()
        patch_collection = PatchCollection(collection_list, alpha=patch_alpha, linewidths=line_width,
                                           edgecolors=line_colour, cmap=cmap)

        return patch_collection

    def is_edge(self, patch: Patch, top_edge: bool = False):
        if top_edge:
            return any([patch.i == self.max_i,
                        patch.i == self.min_i,
                        patch.j == self.min_j,
                        patch.j == self.max_j])
        else:
            return any([patch.i == self.max_i,
                        patch.i == self.min_i,
                        patch.j == self.max_j])

    def edge_patch_bool(self, top_edge: bool = False):
        return np.array([self.is_edge(patch, top_edge=top_edge) for patch in self.patches], dtype=int)

    @property
    def xbounds(self):
        minx = min([patch.corner_array[:, 0].min() for patch in self.patches])
        maxx = max([patch.corner_array[:, 0].max() for patch in self.patches])
        return minx, maxx

    @property
    def ybounds(self):
        miny = min([patch.corner_array[:, 1].min() for patch in self.patches])
        maxy = max([patch.corner_array[:, 1].max() for patch in self.patches])
        return miny, maxy






class MultiPatchFault:
    def __init__(self, patches: Union[list, tuple, set]):
        assert len(patches) > 1, "Only one patch, don't bother with MultiPatch"
        assert all([isinstance(sub_patch, Patch) for sub_patch in patches])
        self.patches = list(patches)
        self._n_patches_along = None
        self._n_patches_down = None
        self._exclude = None

    @property
    def patch_indices(self):
        return [sub_patch.patch_index for sub_patch in self.patches]

    @property
    def patch_along_distances(self):
        top_zero = self.patches[0].top_corner
        distance_list = []
        for patch in self.patches:
            vector = patch.centroid - top_zero
            distance = np.dot(patch.along_strike_vector, vector)
            distance_list.append(distance)

        return distance_list

    def subdivide_patch_evenly(self, patch: Patch, n_along: int, n_down: int, start_down_index: int):
        pass

    @staticmethod
    def subdivide_patch_size(patch: Patch, along_width: Union[float, int], down_width: Union[float, int],
                             start_down_index: int = 0, min_size: float = 0.5):
        width_test = (patch.width >= down_width * (1 + min_size))
        length_test = (patch.length >= along_width * (1 + min_size))

        assert all([width_test, length_test]), "Suggests only one sub-patch. Why are you trying to subdivide?"

        num_along_float = patch.length / along_width
        num_down_float = patch.width / down_width

        num_along_int = np.int(np.round(num_along_float))
        num_down_int = np.int(np.round(num_down_float))

        if np.isclose(num_along_int, num_along_float):
            num_along_normal = num_along_int
            last_along_length = 0
        else:
            num_along_normal = num_along_int - 1
            last_along_length = num_along_float - num_along_normal

        if np.isclose(num_down_int, num_down_float):
            num_down_normal = num_down_int
            last_down_width = 0
        else:
            num_down_normal = num_down_int - 1
            last_down_width = num_down_float - num_down_normal

        sub_patch_ls = []
        for i, j in product(np.arange(num_along_normal), np.arange(num_down_normal)):
            patch_i = i
            patch_j = start_down_index + j
            patch_index = patch_i + patch_j * num_along_int
            along_offset = (i + 0.5) * along_width * patch.along_strike_vector
            down_offset = (j + 0.5) * down_width * patch.down_dip_vector
            sp_centroid = patch.top_corner + along_offset + down_offset
            sub_patch = Patch.from_centroid(centroid_array=sp_centroid, patch_length=along_width,
                                            patch_width=down_width, dip=patch.dip, strike=patch.strike,
                                            patch_i_index=patch_i, patch_j_index=patch_j, patch_index=patch_index)
            sub_patch_ls.append(sub_patch)

        if not np.isclose(last_along_length, 0):
            patch_i = num_along_normal
            along_offset = (num_along_normal * along_width + 0.5 * last_along_length) * patch.along_strike_vector
            for j in np.arange(num_down_normal):
                patch_j = start_down_index + j
                patch_index = patch_i + patch_j * num_along_int
                down_offset = (j + 0.5) * down_width * patch.down_dip_vector
                sp_centroid = patch.top_corner + along_offset + down_offset
                sub_patch = Patch.from_centroid(centroid_array=sp_centroid, patch_length=last_along_length,
                                                patch_width=down_width, dip=patch.dip, strike=patch.strike,
                                                patch_i_index=patch_i, patch_j_index=patch_j, patch_index=patch_index)
                sub_patch_ls.append(sub_patch)

        if not np.isclose(last_down_width, 0):
            patch_j = start_down_index + num_down_normal
            down_offset = (num_down_normal * down_width + 0.5 * last_down_width) * patch.down_dip_vector
            for i in np.arange(num_along_normal):
                patch_i = i
                patch_index = patch_i + patch_j * num_along_int
                along_offset = (i + 0.5) * along_width * patch.along_strike_vector
                sp_centroid = patch.top_corner + along_offset + down_offset
                sub_patch = Patch.from_centroid(centroid_array=sp_centroid, patch_length=along_width,
                                                patch_width=last_down_width, dip=patch.dip, strike=patch.strike,
                                                patch_i_index=patch_i, patch_j_index=patch_j, patch_index=patch_index)
                sub_patch_ls.append(sub_patch)

        if not any([np.isclose(x, 0) for x in [last_along_length, last_down_width]]):
            patch_i = num_along_normal
            patch_j = start_down_index + num_down_normal
            patch_index = patch_i + patch_j * num_along_int
            along_offset = (num_along_normal * along_width + 0.5 * last_along_length) * patch.along_strike_vector
            down_offset = (num_down_normal * down_width + 0.5 * last_down_width) * patch.down_dip_vector
            sp_centroid = patch.top_corner + along_offset + down_offset
            sub_patch = Patch.from_centroid(centroid_array=sp_centroid, patch_length=along_width,
                                            patch_width=last_down_width, dip=patch.dip, strike=patch.strike,
                                            patch_i_index=patch_i, patch_j_index=patch_j, patch_index=patch_index)
            sub_patch_ls.append(sub_patch)

        sorted_patches = sorted(sub_patch_ls, key=lambda x: x.patch_index)

        return sorted_patches

    def projected_outline(self, projection_centre_x: Union[int, float], projection_centre_y: Union[int, float],
                          projection_strike: Union[int, float]):
        all_points = np.vstack([patch.projected_line(projection_centre_x, projection_centre_y, projection_strike)
                                for patch in self.patches])
        return np.unique(np.round(all_points, decimals=4), axis=0)
