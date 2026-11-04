"""Export OSM land-use polygons for WGS84 shapefile extents from PostGIS.

Connection settings come from libpq's PG* environment variables or a service file.
Importing this module never opens a database connection or prompts for input.
"""

import argparse
import logging
import math
import os
import tempfile
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# Preserve the original project's OSM taxonomy (including coastal wetlands).
OSM_CLASSES = {
    "urban_fabric": ("1.1", ("garages", "residential")),
    "indu_comm": (
        "1.2",
        ("railway", "industrial", "commercial", "retail", "harbour", "port", "lock", "marina"),
    ),
    "mine_dump": ("1.3", ("quarry", "construction", "landfill", "brownfield")),
    "artificial_veg": (
        "1.4",
        (
            "stadium",
            "recreation_ground",
            "golf_course",
            "sports_center",
            "playground",
            "pitch",
            "village_green",
            "allotments",
            "cemetery",
            "park",
            "zoo",
            "track",
            "garden",
            "raceway",
        ),
    ),
    "arable_land": (
        "2.1",
        ("greenhouse_horticulture", "greenhouse", "farmland", "farm", "farmyard"),
    ),
    "permanent_crops": ("2.2", ("vineyard", "orchard")),
    "pastures": ("2.3", ("meadow",)),
    "forests": ("3.1", ("forest", "wood")),
    "shrub": ("3.2", ("grass", "greenfield", "scrub", "heath", "grassland")),
    "open_spaces": ("3.3", ("cliff", "fell", "sand", "scree", "beach", "mud", "glacier", "rock")),
    "inland_wetlands": ("4.1", ("marsh", "wetland")),
    "costal_wetland": ("4.2", ("salt_pond", "tidal")),
    "water_bodies": ("5.0", ("water", "riverbank", "reservoir", "basin", "dock", "canal", "pond")),
}
TYPE_TO_CLASS = {
    osm_type: class_name
    for class_name, (_, osm_types) in OSM_CLASSES.items()
    for osm_type in osm_types
}

# Clip intersecting polygons instead of silently dropping polygons crossing a tile.
QUERY = """
WITH bounds AS (
    SELECT ST_Transform(ST_MakeEnvelope(%s, %s, %s, %s, 4326), 3857) AS geom
)
SELECT ST_AsBinary(ST_Multi(ST_CollectionExtract(
           ST_Intersection(nat.the_geom, bounds.geom), 3))), nat.type
FROM public.naturals AS nat CROSS JOIN bounds
WHERE ST_Intersects(nat.the_geom, bounds.geom) AND nat.type = ANY(%s)
"""


def iter_bbox(minX, minY, maxX, maxY, size=4.0):
    """Yield query boxes incrementally, without gaps or zero-area cells."""
    values = (minX, minY, maxX, maxY, size)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Bounding box coordinates and size must be finite")
    if size <= 0 or minX >= maxX or minY >= maxY:
        raise ValueError("Bounding box must have positive area and size must be positive")
    nx = math.ceil((maxX - minX) / size)
    ny = math.ceil((maxY - minY) / size)
    for x in range(nx):
        for y in range(ny):
            yield (
                minX + x * size,
                minY + y * size,
                min(minX + (x + 1) * size, maxX),
                min(minY + (y + 1) * size, maxY),
            )


def split_bbox(minX, minY, maxX, maxY, size=4.0):
    """Compatibility helper for callers needing a list; the exporter streams boxes."""
    return list(iter_bbox(minX, minY, maxX, maxY, size))


def iter_extents(shapefiles, tile_size=4.0):
    """Yield unique names and WGS84 boxes, with no process-wide directory changes."""
    from osgeo import ogr, osr

    expected = osr.SpatialReference()
    expected.ImportFromEPSG(4326)
    expected.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    for path in shapefiles:
        dataset = ogr.Open(str(path))
        if dataset is None:
            raise ValueError(f"Cannot open shapefile: {path}")
        try:
            layer = dataset.GetLayer()
            crs = layer.GetSpatialRef()
            if crs is None:
                raise ValueError(f"Missing CRS: {path}")
            crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            if not crs.IsSame(expected):
                raise ValueError(f"Input must use EPSG:4326: {path}")
            for index, feature in enumerate(layer, start=1):
                geometry = feature.GetGeometryRef()
                if geometry is None or geometry.IsEmpty():
                    raise ValueError(f"Empty geometry in {path}, feature {index}")
                left, right, bottom, top = geometry.GetEnvelope()
                for tile, box in enumerate(iter_bbox(left, bottom, right, top, tile_size), start=1):
                    yield f"{Path(path).stem}_feature_{index}_tile_{tile}", box
        finally:
            dataset = None


def export_to_shape(rows, output):
    """Stream database rows to a temporary shapefile; preserve existing outputs."""
    from osgeo import ogr, osr

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if any(output.parent.glob(output.stem + ".*")):
        raise FileExistsError(f"Output or sidecar already exists: {output}")
    driver = ogr.GetDriverByName("ESRI Shapefile")
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(3857)
    count = 0
    with tempfile.TemporaryDirectory(prefix=".osm-export-", dir=output.parent) as scratch:
        temporary = Path(scratch) / output.name
        dataset = driver.CreateDataSource(str(temporary))
        if dataset is None:
            raise RuntimeError(f"Cannot create shapefile: {temporary}")
        try:
            layer = dataset.CreateLayer(output.stem, crs, ogr.wkbMultiPolygon)
            if layer is None:
                raise RuntimeError("Cannot create output layer")
            for name, width in (("class", 24), ("code", 4)):
                field = ogr.FieldDefn(name, ogr.OFTString)
                field.SetWidth(width)
                if layer.CreateField(field) != 0:
                    raise RuntimeError(f"Cannot create field {name}")
            for wkb, osm_type in rows:
                if wkb is None:
                    continue
                geometry = ogr.CreateGeometryFromWkb(bytes(wkb))
                if geometry is None:
                    raise ValueError("Database returned an invalid geometry")
                if geometry.IsEmpty():
                    continue
                class_name = TYPE_TO_CLASS[osm_type]
                feature = ogr.Feature(layer.GetLayerDefn())
                feature.SetField("class", class_name)
                feature.SetField("code", OSM_CLASSES[class_name][0])
                feature.SetGeometry(geometry)
                if layer.CreateFeature(feature) != 0:
                    raise RuntimeError("Cannot write output feature")
                count += 1
        finally:
            dataset = None
        # Shapefiles have multiple sidecars: publish only after writing succeeds.
        # Link creation refuses concurrent replacement; rollback only our new links.
        published = []
        try:
            for part in sorted(Path(scratch).iterdir()):
                destination = output.parent / part.name
                os.link(part, destination)
                published.append(destination)
        except BaseException:
            for destination in published:
                destination.unlink()
            raise
    return count


def export_extents(connection, extents, output_dir, batch_size=1000):
    """Use a server-side cursor so result sets are not loaded with fetchall()."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    results = []
    for index, (name, box) in enumerate(extents):
        output = Path(output_dir) / f"{name}.shp"
        with connection.cursor(name=f"osm_export_{index}") as cursor:
            cursor.itersize = batch_size
            cursor.execute(QUERY, (*box, list(TYPE_TO_CLASS)))
            count = export_to_shape(cursor, output)
        LOGGER.info("Wrote %s (%d polygons)", output, count)
        results.append(output)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory containing WGS84 .shp files")
    parser.add_argument("output_dir", type=Path, help="Destination for EPSG:3857 shapefiles")
    parser.add_argument(
        "--tile-size", type=float, default=4.0, help="Maximum query extent in degrees (default: 4)"
    )
    parser.add_argument("--batch-size", type=int, default=1000, help="Rows fetched at a time")
    args = parser.parse_args(argv)
    if args.tile_size <= 0 or not math.isfinite(args.tile_size) or args.batch_size < 1:
        parser.error("tile-size and batch-size must be positive and finite")
    shapefiles = sorted(args.input_dir.glob("*.shp"))
    if not shapefiles:
        parser.error(f"No .shp files in {args.input_dir}")
    import psycopg2

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    connection = psycopg2.connect("")
    try:
        with connection:
            export_extents(
                connection,
                iter_extents(shapefiles, args.tile_size),
                args.output_dir,
                args.batch_size,
            )
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
