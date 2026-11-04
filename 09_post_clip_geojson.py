"""Clip boundaries: python 09_post_clip_geojson.py input.tif boundaries.geojson output.tif."""

from osm4eo.postprocess import clip_geojson, main

__all__ = ["clip_geojson"]


if __name__ == "__main__":
    main("geojson")
