"""Extract a class: python 04_postproce_extract_LU.py input.tif output.tif --class-id 12."""

from osm4eo.postprocess import extract_lu, main


def extract_LU_layers(img_input, img_saved, LU_class):
    """Compatibility entry point for the original extraction function."""
    return extract_lu(img_input, img_saved, LU_class)


if __name__ == "__main__":
    main("extract")
