Preprocessing
=============

The aim of the preporcessing is to 
- Select the sites for which you want to run your models
- Discretise the fault models onto the higher resolution mesh
- Prepare Greens Functions for each site that you are wanting to run

Site Selection
--------------

Sites are read in from a csv file in the ./sites directory, containing the columns siteId, Lon, Lat, Height.
SiteId must be unique for each site.
Lon and Lat are currently in NZTM, and height should just be 0.
If you are looking to process a grid, then each pixel in the grid must have it's own entry in the csv file.

``./points2occsites.py`` is a script to convert csv files of points exported from QGIS or downloaded from `Takiwa Searise <https://searise.takiwa.co/map/6233f47872b8190018373db9/embed>`_ into the correct format.
If sites do not have a siteId, then this script will assign one based on the location of the site, rounded to the nearerst 1km.


Crustal Fault Discretisation
----------------------------
.. include:: ../crustal/README.md

Subduction Zone Discretisation
------------------------------
Which should be straightforwards

.. include:: ../subduction/README.rst

Until you are ready for the POSTAL scripts