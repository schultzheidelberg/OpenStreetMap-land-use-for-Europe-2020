"""Mosaic tiles: python 08_post_extent_merge.py a.tif b.tif --output mosaic.tif."""

from osm4eo.postprocess import main
from osm4eo.postprocess import merge_extents as extent_merge

__all__ = ["extent_merge"]


if __name__ == "__main__":
    main("merge")
