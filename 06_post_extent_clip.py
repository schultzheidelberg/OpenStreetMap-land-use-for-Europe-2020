"""Apply an aligned mask: python 06_post_extent_clip.py input.tif mask.tif output.tif."""

from osm4eo.postprocess import clip_extent, main


def extent_clip(img_input, img_saved, mask):
    """Compatibility entry point retaining the original positional order."""
    return clip_extent(img_input, mask, img_saved)


if __name__ == "__main__":
    main("mask")
