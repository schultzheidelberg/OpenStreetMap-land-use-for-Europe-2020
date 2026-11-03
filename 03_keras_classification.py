"""Predict land-use classes from an explicitly selected raster and trained model."""

from osm4eo.inference import classify_raster, load_predictor, main


def prediction(s2_image, classified_image, model_path, **kwargs):
    """Compatibility entry point; the model path is now an explicit argument."""
    return classify_raster(s2_image, classified_image, load_predictor(model_path), **kwargs)


if __name__ == "__main__":
    main()
