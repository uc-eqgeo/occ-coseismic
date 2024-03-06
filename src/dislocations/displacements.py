import numpy as np
from typing import Union
from dislocations.faults import ListricFault, Patch
import os
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd
from dislocations.utilities import write_tiff


class MultiPatch:
    def __init__(self, patches_or_faults: Union[Patch, ListricFault, list, tuple, set]):
        if isinstance(patches_or_faults, (Patch, ListricFault)):
            comprehension_list = [patches_or_faults]
        else:
            comprehension_list = patches_or_faults
        self._patch_ls = []
        # assert all([isinstance(a, (Patch, ListricFault)) for a in comprehension_list])
        for item in comprehension_list:
            if isinstance(item, Patch):
                self._patch_ls.append(item)
            elif isinstance(item, ListricFault):
                self._patch_ls += item.patches

        self._x = None
        self._y = None
        self._z = None

        self._greens_functions = None

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def dip_slip(self):
        return (patch.dip_slip for patch in self.patch_ls)

    @property
    def strike_slip(self):
        return (patch.strike_slip for patch in self.patch_ls)

    @property
    def patch_ls(self):
        return self._patch_ls

    @property
    def greens_functions(self):
        if self._greens_functions is None:
            self.collect_greens_functions()
        return self._greens_functions

    def collect_greens_functions(self):
        assert all([a is not None for a in (self.x, self.y, self.z)]), "No sites read in yet"
        self._greens_functions = [patch.greens_functions(self.x, self.y, self.z) for patch in self.patch_ls]

    @property
    def dip_slip_displacements(self):
        assert not any([a is None for a in self.dip_slip]), "Set slip on patches"
        ds_gf = [gf.ds for gf in self.greens_functions]
        combined_ux = sum([patch.dip_slip * gf.ux for patch, gf in zip(self.patch_ls, ds_gf)])
        combined_uy = sum([patch.dip_slip * gf.uy for patch, gf in zip(self.patch_ls, ds_gf)])
        combined_uz = sum([patch.dip_slip * gf.uz for patch, gf in zip(self.patch_ls, ds_gf)])
        return [combined_ux, combined_uy, combined_uz]

    @property
    def strike_slip_displacements(self):
        assert not any([a is None for a in self.dip_slip]), "Set slip on patches"
        ss_gf = [gf.ss for gf in self.greens_functions]

        combined_ux = sum([patch.strike_slip * gf.ux for patch, gf in zip(self.patch_ls, ss_gf)])
        combined_uy = sum([patch.strike_slip * gf.uy for patch, gf in zip(self.patch_ls, ss_gf)])
        combined_uz = sum([patch.strike_slip * gf.uz for patch, gf in zip(self.patch_ls, ss_gf)])
        return [combined_ux, combined_uy, combined_uz]

    @property
    def total_displacements(self):
        return [(ss_u + ds_u) for ss_u, ds_u in zip(self.strike_slip_displacements, self.dip_slip_displacements)]

    def greens_functions_array(self, rake: Union[float, int] = None, vertical_only: bool = False):
        if rake is not None:
            assert 0. <= rake <= 360.
        if vertical_only:
            strike_slip = np.array([gf.ss.uz for gf in self.greens_functions])
            dip_slip = np.array([gf.ds.uz for gf in self.greens_functions])
        else:
            strike_slip = np.array([[gf.ss.ux, gf.ss.uy, gf.ss.uz] for gf in self.greens_functions])
            dip_slip = np.array([[gf.ds.ux, gf.ds.uy, gf.ds.uz] for gf in self.greens_functions])

        if rake is not None:
            disps = strike_slip * np.cos(np.radians(rake)) + dip_slip * np.sin(np.radians(rake))
            return disps

        else:
            return strike_slip, dip_slip


class DisplacementTable(MultiPatch):
    def __init__(self, patches_or_faults: Union[Patch, ListricFault, list, tuple, set],
                 x_sites: Union[int, float, list, tuple, np.ndarray],
                 y_sites: Union[int, float, list, tuple, np.ndarray],
                 z_sites: Union[int, float, list, tuple, np.ndarray] = None):
        super(DisplacementTable, self).__init__(patches_or_faults=patches_or_faults)

        x_array, y_array = np.array(x_sites), np.array(y_sites)
        assert x_array.shape == y_array.shape, "Arrays must have same shape"
        self._x, self._y = x_sites, y_sites

        if z_sites is None:
            self._z = np.zeros(self.x.shape)
        else:
            self._z = z_sites

    @classmethod
    def from_csv(cls, patches_or_faults: Union[Patch, ListricFault, list, tuple, set],
                 csv_file: str, xy_or_xyz: str = "xy"):
        assert os.path.exists(csv_file)
        data = np.loadtxt(csv_file)
        num_columns = data.shape[-1]
        assert num_columns >= 2, "Should have 2 or 3 columns"
        assert xy_or_xyz.lower() in ("xy", "xyz"), "Specify type of array"
        if xy_or_xyz.lower() == "xy":
            table = cls.from_xy_array(patches_or_faults, data)
        else:
            table = cls.from_xyz_array(patches_or_faults, data)
        return table

    @classmethod
    def from_xyz_array(cls, patches_or_faults: Union[Patch, ListricFault, list, tuple, set],
                       array: np.ndarray):
        num_columns = array.shape[-1]
        assert num_columns >= 3, "Should have 3 columns"
        array_float = np.array(array, dtype=np.float)
        table = cls(patches_or_faults, array_float[:, 0], array_float[:, 1], z_sites=array_float[:, 2])
        return table

    @classmethod
    def from_xy_array(cls, patches_or_faults: Union[Patch, ListricFault, list, tuple, set],
                      array: np.ndarray):
        num_columns = array.shape[-1]
        assert num_columns >= 2, "Should have 2 columns"
        array_float = np.array(array, dtype=np.float)
        table = cls(patches_or_faults, array_float[:, 0], array_float[:, 1])
        return table

    def displacements_array(self, vertical_only: bool = False, station_z: bool = False, header: bool = False):
        ux, uy, uz = self.total_displacements
        if vertical_only:
            if station_z:
                out_tuple = (self.x, self.y, self.z, uz)
                header_row = ["stationX", "stationY", "stationZ", "uZ"]
            else:
                out_tuple = (self.x, self.y, uz)
                header_row = ["stationX", "stationY", "uZ"]
        else:
            if station_z:
                out_tuple = (self.x, self.y, self.z, ux, uy, uz)
                header_row = ["stationX", "stationY", "stationZ", "uX", "uY", "uZ"]
            else:
                out_tuple = (self.x, self.y, ux, uy, uz)
                header_row = ["stationX", "stationY", "uX", "uY", "uZ"]

        out_array = np.vstack(out_tuple).T
        if not header:
            return out_array
        else:
            return out_array, header_row

    def write_displacements_csv(self, filename: str, vertical_only: bool = False, station_z: bool = False,
                                delimiter=" ", fmt="%.4f", header: bool = False):
        if header:
            out_array, header_row = self.displacements_array(vertical_only=vertical_only, station_z=station_z,
                                                             header=header)
            header_str = " ".join(header_row)
            np.savetxt(filename, out_array, fmt=fmt, delimiter=delimiter, header=header_str)
        else:
            out_array = self.displacements_array(vertical_only=vertical_only, station_z=station_z,
                                                 header=header)
            np.savetxt(filename, out_array, fmt=fmt, delimiter=delimiter)

    def write_displacements_shp(self, filename: str, vertical_only: bool = False, station_z: bool = False,
                                epsg: int = 2193):
        if epsg not in [2193, 4326, 32759, 32760, 27200]:
            print("EPSG:{:d} Not a recognised NZ coordinate system...".format(epsg))
            print("Writing anyway...")
        out_array, header_row = self.displacements_array(vertical_only=vertical_only, station_z=station_z,
                                                         header=True)
        dataframe = pd.DataFrame(out_array, columns=header_row)
        geom = [Point(xi, yi) for xi, yi in zip(self.x, self.y)]
        out_gdf = gpd.GeoDataFrame(dataframe, geometry=geom, crs={"init": "epsg:{:d}".format(epsg)})
        out_gdf.to_file(filename)


class DisplacementGrid(MultiPatch):
    def __init__(self, patches_or_faults: Union[Patch, ListricFault, list, tuple, set],
                 x_sites: Union[int, float, list, tuple, np.ndarray],
                 y_sites: Union[int, float, list, tuple, np.ndarray],
                 z_sites: Union[int, float, list, tuple, np.ndarray] = None):
        super(DisplacementGrid, self).__init__(patches_or_faults=patches_or_faults)

        x_array, y_array = np.array(x_sites), np.array(y_sites)
        # assert x_array.shape == y_array.shape, "Arrays must have same shape"
        if x_array.ndim == 1:
            print("Detected a 1D array, making a grid...")
            self._x_values = x_array
            self._y_values = y_array
            self._x, self._y = np.meshgrid(x_array, y_array)
        else:
            assert x_array.ndim == 2, "Can only support 1D or 2D arrays"
            self._x, self._y = x_array, y_array
            # Get x values and y values
            min_x, max_x = np.min(self._x), np.max(self._x)
            self._x_values = np.linspace(min_x, max_x, num=self._x.shape[1])
            min_y, max_y = np.min(self._y), np.max(self._y)
            self._x_values = np.linspace(min_y, max_y, num=self._x.shape[0])

        if z_sites is None:
            self._z = np.zeros(self._x.shape)
        else:
            z_array = np.array(z_sites)
            assert z_array.shape == self._x.shape
            self._z = z_array

    @property
    def x_values(self):
        return self._x_values

    @property
    def y_values(self):
        return self._y_values

    def write_displacement_tiff(self, filename: str, direction: str = "vert", epsg: int = 2193):
        assert direction in ["east", "e", "north", "n", "vertical", "vert", "v"]
        if direction in ["east", "e"]:
            z_values = self.total_displacements[0]
        elif direction in ["north", "n"]:
            z_values = self.total_displacements[1]
        else:
            z_values = self.total_displacements[2]

        assert z_values.ndim in [1, 2]
        if z_values.ndim == 1:
            z_out = z_values.reshape((len(self.y_values), len(self.x_values)))
        else:
            z_out = z_values

        write_tiff(filename=filename, x=self.x_values, y=self.y_values, z=z_out, epsg=epsg)

    @classmethod
    def from_raster(cls):
        pass

    @classmethod
    def from_bounds(cls, patches_or_faults: Union[Patch, ListricFault, list, tuple, set],
                    x_min: Union[float, int], x_max: Union[float, int], x_step: Union[float, int],
                    y_min: Union[float, int], y_max: Union[float, int], y_step: Union[float, int] = None):
        assert y_max > y_min
        assert x_max > x_min
        assert x_step > 0
        if y_step is None:
            y_step = x_step
        else:
            assert y_step > 0

        x_range = x_max - x_min

        x_values = np.arange(x_min, x_max, x_step)
        y_values = np.arange(y_min, y_max, y_step)

        return cls(patches_or_faults, x_values, y_values)







