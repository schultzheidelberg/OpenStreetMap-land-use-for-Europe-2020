"""Publish a validated, single-file GeoTIFF without exposing partial results."""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import rasterio
from rasterio.windows import Window


@contextmanager
def atomic_raster_output(input_paths, output_path, overwrite=False):
    """Yield a temporary TIFF, then validate and atomically publish it.

    Existing outputs require explicit replacement. Inputs can never be replaced,
    even through symbolic links or hard links. External masks/overviews/PAM
    sidecars are rejected because a single rename cannot publish them atomically.
    """
    output = Path(output_path).absolute()
    for input_path in input_paths:
        source = Path(input_path)
        if source.resolve() == output.resolve() or (
            source.exists() and output.exists() and source.samefile(output)
        ):
            raise ValueError("The output must differ from every input file")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}; use --overwrite to replace it")
    output_sidecars = [Path(f"{output}{suffix}") for suffix in (".aux.xml", ".msk", ".ovr")]
    if any(path.exists() for path in output_sidecars):
        raise ValueError("Destination has existing sidecars; select a new single-file output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{output.stem}-", suffix=".tif", dir=output.parent)
    os.close(descriptor)
    temporary = Path(name)
    sidecars = [Path(f"{temporary}{suffix}") for suffix in (".aux.xml", ".msk", ".ovr")]
    try:
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            yield temporary
        if any(path.exists() for path in sidecars):
            raise ValueError(
                "Output has external sidecars; store masks and metadata inside the TIFF"
            )
        with rasterio.open(temporary) as check:
            if check.driver != "GTiff" or min(check.width, check.height, check.count) < 1:
                raise ValueError("Output must be a nonempty GeoTIFF")
            check.read(1, window=Window(0, 0, 1, 1))
            check.read(1, window=Window(check.width - 1, check.height - 1, 1, 1))
        if any(path.exists() for path in output_sidecars):
            raise ValueError("Destination acquired external sidecars while processing")
        if overwrite:
            os.replace(temporary, output)
        else:
            # Atomic no-clobber publication if another process creates the
            # destination while this job runs. Both paths share a filesystem.
            os.link(temporary, output)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
        for sidecar in sidecars:
            sidecar.unlink(missing_ok=True)
