"""Restore OSM codes: python 07_post_label_transfer.py input.tif output.tif."""

from osm4eo.labels import LABEL_MAP as transform_map
from osm4eo.postprocess import main
from osm4eo.postprocess import transfer_labels as transform_label

__all__ = ["transform_map", "transform_label"]


if __name__ == "__main__":
    main("labels")
