Description:
First, the python script "shp_to_lulc" extracts the extent (bounding box) of shapefiles given in the coordinate system EPSG:4326 (WGS84).
Then it connects to the OSM Landuse/Landcover Database (if you want to connect to an other database change the code accordingly) to fetch all LULC data available there (from the naturals table). The data obtained is stored in a folder (named results) in .shp format. The script can process an arbitary number of shapefiles, as long as the coordinate system is correct and the extent of the shapefiles is not too large.

Usage:
Please input the path (like: D:\path\to\shps\) to a folder, where input shapefiles are stored, when executing the script. Resulting shapefiles are stored next to the location of the script.

Note: 
With minor edits, requirements like the coordinate system and LULC-classes can be changed.

Library requirements:
•	psycopg2 (tested with 2.8.4)
•	Python 3.7.X (tested with 3.7.6)
•	GDAL (tested with 3.0.2)
