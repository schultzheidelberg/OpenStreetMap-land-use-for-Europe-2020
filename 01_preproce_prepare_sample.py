"""Prepare numeric training samples from one explicitly selected image/label pair."""

import argparse

from osm4eo.sampling import prepare_samples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Sentinel-2 GeoTIFF")
    parser.add_argument("--labels", required=True, help="Aligned original OSM-code GeoTIFF")
    parser.add_argument("--train-output", required=True, help="Training .npz output")
    parser.add_argument("--validation-output", required=True, help="Validation .npz output")
    parser.add_argument("--samples-per-class", type=int, default=5000)
    parser.add_argument("--patch-size", type=int, default=12)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=512, help="Bounded raster read size")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        train, validation = prepare_samples(
            args.image,
            args.labels,
            args.train_output,
            args.validation_output,
            samples_per_class=args.samples_per_class,
            patch_size=args.patch_size,
            train_fraction=args.train_fraction,
            seed=args.seed,
            window_size=args.window_size,
            overwrite=args.overwrite,
        )
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(f"Saved {train} training and {validation} validation samples")


if __name__ == "__main__":
    main()
