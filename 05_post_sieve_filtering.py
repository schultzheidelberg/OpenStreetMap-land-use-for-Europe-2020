"""Sieve a classification: python 05_post_sieve_filtering.py input.tif output.tif."""

# Original implementation: Zhaoyan Wu.

from osm4eo.postprocess import main, sieve_raster


def seieve(in_file, out_file, size):
    """Compatibility alias retaining the historical misspelled function name."""
    return sieve_raster(in_file, out_file, size)


if __name__ == "__main__":
    main("sieve")
