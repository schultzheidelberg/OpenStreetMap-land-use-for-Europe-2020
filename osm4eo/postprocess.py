"""Single-band land-use postprocessing with explicit inputs and bounded reads."""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import geometry_mask, geometry_window
from rasterio.warp import transform_geom
from rasterio.windows import Window

from .labels import LABEL_MAP
from .outputs import atomic_raster_output

CLASS_TO_OSM = {value: key for key, value in LABEL_MAP.items()}
WINDOW_SIZE = 512
GDAL_CACHE_BYTES = 256 * 1024 * 1024


def _windows(width, height):
    for row in range(0, height, WINDOW_SIZE):
        for column in range(0, width, WINDOW_SIZE):
            yield Window(
                column, row, min(WINDOW_SIZE, width - column), min(WINDOW_SIZE, height - row)
            )


def _check_classification(dataset):
    if dataset.count != 1:
        raise ValueError("Expected a single-band classification raster")
    if np.dtype(dataset.dtypes[0]).kind not in "iu":
        raise ValueError("Classification rasters must contain integer class IDs")


def _profile(dataset, *, nodata=None):
    profile = dataset.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        compress="deflate",
        tiled=True,
        blockxsize=WINDOW_SIZE,
        blockysize=WINDOW_SIZE,
        BIGTIFF="IF_SAFER",
        NUM_THREADS="1",
    )
    if nodata is not None:
        profile["nodata"] = nodata
    return profile


def _background(dataset):
    """Return a storage fill value; validity is recorded separately in the mask."""
    return dataset.nodata if dataset.nodata is not None else 0


def extract_lu(input_path, output_path, class_id, *, overwrite=False):
    """Keep one class; other pixels become the source nodata value (or zero)."""
    with rasterio.Env(GDAL_CACHEMAX=GDAL_CACHE_BYTES, GDAL_NUM_THREADS="1"):
        with rasterio.open(input_path) as source:
            _check_classification(source)
            limits = np.iinfo(source.dtypes[0])
            nodata = _background(source)
            if not limits.min <= class_id <= limits.max:
                raise ValueError("class_id must be representable in the source dtype")
            with atomic_raster_output([input_path], output_path, overwrite=overwrite) as temporary:
                with rasterio.open(temporary, "w", **_profile(source)) as target:
                    for window in _windows(source.width, source.height):
                        data = source.read(1, window=window, masked=True)
                        keep = (data.data == class_id) & ~np.ma.getmaskarray(data)
                        target.write(
                            np.where(keep, data.data, nodata).astype(data.dtype), 1, window=window
                        )
                        target.write_mask(keep, window=window)


def clip_extent(input_path, mask_path, output_path, *, mask_value=1, overwrite=False):
    """Apply a mask only when its CRS, shape and pixel grid match exactly."""
    with rasterio.Env(GDAL_CACHEMAX=GDAL_CACHE_BYTES, GDAL_NUM_THREADS="1"):
        with rasterio.open(input_path) as source, rasterio.open(mask_path) as mask:
            _check_classification(source)
            if (
                source.shape != mask.shape
                or source.crs != mask.crs
                or source.transform != mask.transform
            ):
                raise ValueError(
                    "Mask and classification must have the same shape, CRS and pixel grid"
                )
            nodata = _background(source)
            with atomic_raster_output(
                [input_path, mask_path], output_path, overwrite=overwrite
            ) as temporary:
                with rasterio.open(temporary, "w", **_profile(source)) as target:
                    for window in _windows(source.width, source.height):
                        data = source.read(1, window=window, masked=True)
                        mask_data = mask.read(1, window=window, masked=True)
                        keep = (
                            (mask_data.data == mask_value)
                            & ~np.ma.getmaskarray(mask_data)
                            & ~np.ma.getmaskarray(data)
                        )
                        target.write(
                            np.where(keep, data.data, nodata).astype(data.dtype), 1, window=window
                        )
                        target.write_mask(keep, window=window)


def transfer_labels(input_path, output_path, *, overwrite=False):
    """Map model classes 1..12 to OSM codes without remapping newly written IDs."""
    with rasterio.Env(GDAL_CACHEMAX=GDAL_CACHE_BYTES, GDAL_NUM_THREADS="1"):
        with rasterio.open(input_path) as source:
            _check_classification(source)
            nodata = source.nodata
            # A nodata marker cannot also identify a valid output class.
            if nodata in CLASS_TO_OSM.values():
                nodata = 0
            with atomic_raster_output([input_path], output_path, overwrite=overwrite) as temporary:
                with rasterio.open(temporary, "w", **_profile(source, nodata=nodata)) as target:
                    for window in _windows(source.width, source.height):
                        data = source.read(1, window=window, masked=True)
                        result = data.data.copy()
                        for model_class, osm_class in CLASS_TO_OSM.items():
                            result[data.data == model_class] = osm_class
                        valid = ~np.ma.getmaskarray(data)
                        result[~valid] = nodata if nodata is not None else 0
                        target.write(result, 1, window=window)
                        target.write_mask(valid, window=window)


@contextmanager
def _gdal_settings(gdal):
    cache = gdal.GetCacheMax()
    threads = gdal.GetConfigOption("GDAL_NUM_THREADS")
    gdal.SetCacheMax(GDAL_CACHE_BYTES)
    gdal.SetConfigOption("GDAL_NUM_THREADS", "1")
    try:
        yield
    finally:
        gdal.SetCacheMax(cache)
        gdal.SetConfigOption("GDAL_NUM_THREADS", threads)


def sieve_raster(input_path, output_path, size=64, *, connectivity=8, overwrite=False):
    """Sieve globally using GDAL bands; never split connected regions by window.

    GDAL's component bookkeeping still grows with the number of polygons. Highly
    fragmented rasters can therefore need substantial memory despite band I/O.
    """
    if size < 1 or connectivity not in (4, 8):
        raise ValueError("size must be positive and connectivity must be 4 or 8")
    from osgeo import gdal

    with rasterio.Env(GDAL_CACHEMAX=GDAL_CACHE_BYTES, GDAL_NUM_THREADS="1"):
        with rasterio.open(input_path) as source:
            _check_classification(source)
            if source.dtypes[0] not in ("uint8", "int16", "uint16", "int32"):
                raise ValueError("Sieve supports uint8, int16, uint16 or int32 classifications")
            with atomic_raster_output([input_path], output_path, overwrite=overwrite) as temporary:
                with rasterio.open(temporary, "w", **_profile(source)) as target:
                    for window in _windows(source.width, source.height):
                        target.write(source.read(1, window=window), 1, window=window)
                        target.write_mask(source.read_masks(1, window=window), window=window)
                with _gdal_settings(gdal):
                    input_dataset = gdal.Open(str(input_path), gdal.GA_ReadOnly)
                    output_dataset = gdal.Open(str(temporary), gdal.GA_Update)
                    if input_dataset is None or output_dataset is None:
                        raise OSError("GDAL could not open the sieve input/output")
                    try:
                        input_band = input_dataset.GetRasterBand(1)
                        code = gdal.SieveFilter(
                            input_band,
                            input_band.GetMaskBand(),
                            output_dataset.GetRasterBand(1),
                            size,
                            connectivity,
                        )
                        if code != 0:
                            raise RuntimeError(f"GDAL sieve failed with status {code}")
                        output_dataset.FlushCache()
                    finally:
                        input_band = None
                        output_dataset = None
                        input_dataset = None


def merge_extents(input_paths, output_path, *, overwrite=False):
    """Mosaic aligned integer tiles in bounded windows, using first-valid priority."""
    input_paths = list(input_paths)
    if not input_paths:
        raise ValueError("At least one input raster is required")
    with rasterio.Env(GDAL_CACHEMAX=GDAL_CACHE_BYTES, GDAL_NUM_THREADS="1"):
        with rasterio.open(input_paths[0]) as first:
            _check_classification(first)
            profile = _profile(first)
            background = _background(first)
            reference_crs, reference_transform = first.crs, first.transform
            reference_dtype = first.dtypes[0]
            reference_nodata = first.nodata
        tiles = []
        for path in input_paths:
            with rasterio.open(path) as source:
                _check_classification(source)
                transform = source.transform
                if (
                    source.crs != reference_crs
                    or source.dtypes[0] != reference_dtype
                    or source.nodata != reference_nodata
                    or transform.b != 0
                    or transform.d != 0
                    or transform.a <= 0
                    or transform.e >= 0
                    or transform.a != reference_transform.a
                    or transform.e != reference_transform.e
                ):
                    raise ValueError(
                        "Mosaic inputs must share CRS, dtype, nodata and north-up pixel resolution"
                    )
                offsets = (
                    (transform.c - reference_transform.c) / transform.a,
                    (transform.f - reference_transform.f) / transform.e,
                )
                if any(not np.isclose(value, round(value), atol=1e-7, rtol=0) for value in offsets):
                    raise ValueError("Mosaic inputs must lie on the same pixel grid")
                column, row = (round(value) for value in offsets)
                tiles.append((path, column, row, source.width, source.height))
        left = min(column for _, column, _, _, _ in tiles)
        top = min(row for _, _, row, _, _ in tiles)
        width = max(column + width for _, column, _, width, _ in tiles) - left
        height = max(row + height for _, _, row, _, height in tiles) - top
        profile.update(
            width=width,
            height=height,
            transform=rasterio.windows.transform(
                Window(left, top, width, height), reference_transform
            ),
        )
        with atomic_raster_output(input_paths, output_path, overwrite=overwrite) as temporary:
            with rasterio.open(temporary, "w", **profile) as target:
                for window in _windows(width, height):
                    result = np.full(
                        (int(window.height), int(window.width)), background, dtype=reference_dtype
                    )
                    occupied = np.zeros(result.shape, dtype=bool)
                    for path, column, row, tile_width, tile_height in tiles:
                        column, row = column - left, row - top
                        start_column = max(int(window.col_off), column)
                        start_row = max(int(window.row_off), row)
                        end_column = min(int(window.col_off + window.width), column + tile_width)
                        end_row = min(int(window.row_off + window.height), row + tile_height)
                        if start_column >= end_column or start_row >= end_row:
                            continue
                        source_window = Window(
                            start_column - column,
                            start_row - row,
                            end_column - start_column,
                            end_row - start_row,
                        )
                        rows = slice(start_row - int(window.row_off), end_row - int(window.row_off))
                        columns = slice(
                            start_column - int(window.col_off), end_column - int(window.col_off)
                        )
                        region = result[rows, columns]
                        region_occupied = occupied[rows, columns]
                        if region_occupied.all():
                            continue
                        with rasterio.open(path) as source:
                            data = source.read(1, window=source_window, masked=True)
                        # A class value can equal the declared nodata when an
                        # explicit mask marks it valid. Track coverage separately.
                        take = ~region_occupied & ~np.ma.getmaskarray(data)
                        region[take] = data.data[take]
                        region_occupied |= take
                        if occupied.all():
                            break
                    target.write(result, 1, window=window)
                    target.write_mask(occupied, window=window)


def _read_geometries(path, geometry_crs):
    with open(path, encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("type") == "FeatureCollection":
        geometries = [feature.get("geometry") for feature in document.get("features", [])]
    elif document.get("type") == "Feature":
        geometries = [document.get("geometry")]
    else:
        geometries = [document]
    geometries = [geometry for geometry in geometries if geometry is not None]
    if not geometries:
        raise ValueError("GeoJSON contains no geometries")
    if geometry_crs is None:
        geometry_crs = (document.get("crs") or {}).get("properties", {}).get("name", "EPSG:4326")
    return geometries, geometry_crs


def clip_geojson(input_path, geometry_path, output_path, *, geometry_crs=None, overwrite=False):
    """Crop to every GeoJSON feature, reprojecting from its declared CRS or WGS84."""
    geometries, geometry_crs = _read_geometries(geometry_path, geometry_crs)
    with rasterio.Env(GDAL_CACHEMAX=GDAL_CACHE_BYTES, GDAL_NUM_THREADS="1"):
        with rasterio.open(input_path) as source:
            _check_classification(source)
            if source.crs is None:
                raise ValueError("Input raster needs a CRS for GeoJSON clipping")
            geometries = [
                transform_geom(geometry_crs, source.crs, geometry) for geometry in geometries
            ]
            try:
                crop = geometry_window(source, geometries)
            except rasterio.errors.WindowError as error:
                raise ValueError("GeoJSON geometries do not overlap the raster") from error
            nodata = _background(source)
            profile = _profile(source)
            profile.update(
                width=int(crop.width),
                height=int(crop.height),
                transform=source.window_transform(crop),
            )
            with atomic_raster_output(
                [input_path, geometry_path], output_path, overwrite=overwrite
            ) as temporary:
                with rasterio.open(temporary, "w", **profile) as target:
                    for window in _windows(target.width, target.height):
                        source_window = Window(
                            window.col_off + crop.col_off,
                            window.row_off + crop.row_off,
                            window.width,
                            window.height,
                        )
                        data = source.read(1, window=source_window, masked=True)
                        inside = geometry_mask(
                            geometries,
                            out_shape=data.shape,
                            invert=True,
                            transform=source.window_transform(source_window),
                        )
                        keep = inside & ~np.ma.getmaskarray(data)
                        target.write(
                            np.where(keep, data.data, nodata).astype(data.dtype), 1, window=window
                        )
                        target.write_mask(keep, window=window)


def main(operation, argv=None):
    """CLI shared by the numbered pipeline entry points."""
    descriptions = {
        "extract": "Extract one land-use class",
        "sieve": "Remove small connected class regions",
        "mask": "Mask a classification using an aligned raster",
        "labels": "Convert model classes 1..12 to OSM land-use codes",
        "merge": "Mosaic aligned classification tiles",
        "geojson": "Clip to all geometries in a GeoJSON file",
    }
    parser = argparse.ArgumentParser(description=descriptions[operation])
    parser.add_argument("input", type=Path, nargs="+" if operation == "merge" else None)
    if operation == "mask":
        parser.add_argument("mask", type=Path)
        parser.add_argument("--mask-value", type=int, default=1)
    if operation == "geojson":
        parser.add_argument("geometry", type=Path)
        parser.add_argument(
            "--geometry-crs", help="Override GeoJSON CRS (default: declared CRS or EPSG:4326)"
        )
    if operation == "merge":
        parser.add_argument("--output", type=Path, required=True)
    else:
        parser.add_argument("output", type=Path)
    if operation == "extract":
        parser.add_argument("--class-id", type=int, required=True)
    if operation == "sieve":
        parser.add_argument("--size", type=int, default=64)
        parser.add_argument("--connectivity", type=int, choices=(4, 8), default=8)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output")
    args = parser.parse_args(argv)
    try:
        if operation == "extract":
            extract_lu(args.input, args.output, args.class_id, overwrite=args.overwrite)
        elif operation == "sieve":
            sieve_raster(
                args.input,
                args.output,
                args.size,
                connectivity=args.connectivity,
                overwrite=args.overwrite,
            )
        elif operation == "mask":
            clip_extent(
                args.input,
                args.mask,
                args.output,
                mask_value=args.mask_value,
                overwrite=args.overwrite,
            )
        elif operation == "labels":
            transfer_labels(args.input, args.output, overwrite=args.overwrite)
        elif operation == "merge":
            merge_extents(args.input, args.output, overwrite=args.overwrite)
        else:
            clip_geojson(
                args.input,
                args.geometry,
                args.output,
                geometry_crs=args.geometry_crs,
                overwrite=args.overwrite,
            )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Saved {args.output}")
