Site Selection
==============

POSTAL reads in location data as a list of discrete sites in a csv file.
If you are processing a gridded area, then each grid cell must be converted to a point.
Columns in the site csv are 'SiteId', 'Lon', 'Lat', and 'Height' (Height isn't needed by the scripts, I just can't remember if things will break if it's not included).
Points can be downloaded as a csv from QGIS or from the `Takiwa searise map <https://searise.takiwa.co/map/6233f47872b8190018373db9/embed>`_, then converted into the correct format using points2occ.py.
This script will assign names to sites exported from QGIS based on their NZTM location, to the nearest 1km.

I have written a script that will come up with site spacings based on quad- and cube-tree subsampling of faults, but that's currently on a seperate branch to be worried about at some later date

