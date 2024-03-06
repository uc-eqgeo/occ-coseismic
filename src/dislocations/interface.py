from utilities import read_grid
from typing import Union
import numpy as np

interface_filename = "coast_and_interface/interface_nztm.tif"


def read_interface(x1: Union[float, int] = None, y1: Union[float, int] = None, x2: Union[float, int] = None,
                   y2: Union[float, int] = None):
    no_args = all([a is None for a in (x1, y1, x2, y2)])
    all_args = all([a is not None for a in (x1, y1, x2, y2)])
    assert any([no_args, all_args])

    # Read grid in
    x, y, z = read_grid(interface_filename, nan_threshold=500)
    z *= 1000.
    if y[0] > y[-1]:
        y = y[::-1]
        z = z[::-1, :]

    if no_args:
        x_mesh, y_mesh = np.meshgrid(x, y)
        return x_mesh, y_mesh, z

    else:
        assert all([x2 > x1, y2 > y1])
        i1 = np.argmax(x >= x1)
        i2 = np.argmin(x < x2)
        j1 = np.argmax(y >= y1)
        j2 = np.argmin(y < y2)

        cut_x = x[i1:i2]
        cut_y = y[j1:j2]
        cut_z = z[j1:j2, i1:i2]
        x_mesh, y_mesh = np.meshgrid(cut_x, cut_y)
        return x_mesh, y_mesh, cut_z
