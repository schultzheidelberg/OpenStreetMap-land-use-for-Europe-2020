The setup and database-export instructions are maintained in README.md.

Core raster requirements: requirements.txt
CPU model requirements: requirements-training.txt
Optional PostGIS client: requirements-database.txt
Testing and lint: requirements-dev.txt

GDAL Python bindings must match the installed native GDAL library. See the
Ubuntu 22.04 installation example in README.md; do not pip-install an arbitrary
GDAL version over an existing system installation.

shp_to_lulc.py accepts explicit input/output directories and uses standard
libpq connection settings (PGHOST, PGDATABASE, PGUSER, PGSERVICE, .pgpass).
It does not contain database credentials or prompt when imported.
