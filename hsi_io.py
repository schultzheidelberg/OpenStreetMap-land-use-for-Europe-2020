"""Raster/sample I/O plus optional compatibility helpers for historic datasets."""

import zipfile
from pathlib import Path

import numpy as np
import rasterio as rio


def loadtif(filename):
    """Read a raster as rows, columns, bands, preserving its native dtype."""
    with rio.open(filename) as source:
        return np.moveaxis(source.read(), 0, -1)


def savetif(X, infilename, outfilename, *, overwrite=False):
    """Write a categorical uint16 raster using the reference pixel grid."""
    from osm4eo.outputs import atomic_raster_output

    values = np.asarray(X)
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 65535):
        raise ValueError("Classification values must be finite and within uint16 range")
    if not np.all(values == np.floor(values)):
        raise ValueError("Classification values must be integer class IDs")
    with rio.open(infilename) as source:
        if values.shape != (source.height, source.width):
            raise ValueError("Classification dimensions must match the reference raster")
        profile = source.profile.copy()
        profile.update(count=1, dtype="uint16", nodata=0, compress="deflate", predictor=2)
    with atomic_raster_output([infilename], outfilename, overwrite=overwrite) as temporary:
        with rio.open(temporary, "w", **profile) as destination:
            destination.write(values.astype(np.uint16), 1)


def window_sample_houston(X, num, filename):
    """Legacy eager 12px symmetric sampling helper; new pipelines use osm4eo.sampling."""
    if extension(filename) not in {"tif", "tiff"}:
        raise ValueError("Window sampling requires a GeoTIFF")
    if num < 1 or num > len(X):
        raise ValueError("Sample count must be positive and no larger than the input index")
    image = loadtif(filename).astype(np.float32)
    minimum, maximum = image.min(), image.max()
    if maximum - minimum > 0.1:
        image = 2 * (image - minimum) / (maximum - minimum) - 1
    padded = np.pad(image, ((6, 6), (6, 6), (0, 0)), mode="symmetric")
    patches = []
    for index in X.head(num).index:
        if index < 0 or index >= image.shape[0] * image.shape[1]:
            raise ValueError("Sample index outside the raster")
        row, col = divmod(int(index), image.shape[1])
        patches.append(padded[row : row + 12, col : col + 12])
    return np.stack(patches)


def _text_words(filename):
    """Read plain text or a ZIP containing one unambiguous data member."""
    filename = Path(filename)
    if not zipfile.is_zipfile(filename):
        return filename.read_text().split()
    with zipfile.ZipFile(filename) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        matches = [name for name in members if Path(name).name == filename.stem]
        if len(matches) == 1:
            member = matches[0]
        elif len(members) == 1:
            member = members[0]
        else:
            raise ValueError("ZIP must contain one data file or a member matching its stem")
        return archive.read(member).decode("utf-8").split()


def import_data(filename):
    """Read a GeoTIFF or legacy text with a `bands rows cols` header."""
    if extension(filename) in {"tif", "tiff"}:
        image = loadtif(filename).astype(np.float32)
        rows, cols, bands = image.shape
        minimum, maximum = image.min(), image.max()
        if maximum - minimum > 0.1:
            image = 2 * (image - minimum) / (maximum - minimum) - 1
        return bands, rows, cols, image.reshape(rows * cols, bands)
    words = _text_words(filename)
    bands, rows, cols = map(int, words[:3])
    values = np.asarray(words[3:], dtype=np.float32)
    if min(bands, rows, cols) < 1 or len(values) != bands * rows * cols:
        raise ValueError("Data dimensions do not match the legacy text header")
    if cols == 1:
        data = values.reshape(bands, rows).T
    else:
        data = values.reshape(cols, rows, bands).transpose(1, 0, 2).reshape(rows * cols, bands)
    return bands, rows, cols, data


def import_labels(filename):
    """Read legacy MAT or text labels with a `rows cols` header."""
    if extension(filename) == "mat":
        import hdf5storage

        arrays = [
            value
            for key, value in hdf5storage.loadmat(filename).items()
            if not key.startswith("__") and isinstance(value, np.ndarray)
        ]
        if len(arrays) != 1 or arrays[0].ndim != 2:
            raise ValueError("MAT labels must contain one two-dimensional data array")
        rows, cols = arrays[0].shape
        return rows, cols, arrays[0].reshape(-1)
    words = _text_words(filename)
    rows, cols = map(int, words[:2])
    labels = np.asarray(words[2:], dtype=np.float32)
    if min(rows, cols) < 1 or len(labels) != rows * cols:
        raise ValueError("Label dimensions do not match the legacy text header")
    return rows, cols, labels


def export_data(filename, X, nbands):
    """Write one complete legacy dataset in the band-major order import_data expects."""
    data = np.asarray(X)
    if data.ndim != 2 or data.shape[1] != nbands or nbands < 1:
        raise ValueError("Data must have shape (pixels, nbands)")
    with open(filename, "w") as handle:
        handle.write(f"{nbands} {len(data)} 1\n")
        np.savetxt(handle, data.T.reshape(-1), fmt="%.15f")


def split_train_test(X, ratio):
    if not 0 < ratio < 1:
        raise ValueError("Train ratio must be between 0 and 1")
    count = int(len(X) * ratio)
    return X[:count], X[count:]


def _validate_samples(patches, labels):
    patches = np.asarray(patches, dtype=np.float32)
    labels = np.asarray(labels)
    if patches.ndim != 4 or any(size < 1 for size in patches.shape):
        raise ValueError("Patches must be a nonempty (samples, rows, columns, bands) array")
    if labels.ndim != 1 or len(labels) != len(patches):
        raise ValueError("Labels must have one value per patch")
    if not np.all(np.isfinite(patches)):
        raise ValueError("Patches must contain only finite values")
    if not (
        np.issubdtype(labels.dtype, np.integer) or np.issubdtype(labels.dtype, np.floating)
    ) or not np.all(np.isfinite(labels)):
        raise ValueError("Labels must be numeric class IDs")
    if not np.all((labels >= 1) & (labels <= 12) & (labels == np.floor(labels))):
        raise ValueError("Labels must be integer class IDs 1..12 (zero is reserved)")
    return patches, labels.astype(np.int64)


def _legacy_samples(samples):
    samples = np.asarray(samples, dtype=object)
    if samples.ndim != 2 or samples.shape[1] != 2 or not len(samples):
        raise ValueError("Legacy samples must be a nonempty (samples, 2) object array")
    return _validate_samples(np.stack(samples[:, 0]), np.asarray(samples[:, 1], dtype=float))


def load_samples(filename, *, allow_legacy_pickle=False):
    """Load numeric NPZ; old object-array NPY requires explicit trust opt-in."""
    filename = Path(filename)
    if filename.suffix.lower() == ".npz":
        with np.load(filename, allow_pickle=False) as data:
            if not {"patches", "labels"}.issubset(data.files):
                raise ValueError("Sample NPZ must contain patches and labels arrays")
            return _validate_samples(data["patches"], data["labels"])
    if filename.suffix.lower() == ".npy" and allow_legacy_pickle:
        return _legacy_samples(np.load(filename, allow_pickle=True))
    raise ValueError(
        "Use numeric .npz samples; trusted old .npy files require --allow-legacy-pickle"
    )


def train_test_preprocess(Train, Test):
    """Compatibility adapter for already-loaded legacy object-array samples."""
    x_train, y_train = _legacy_samples(Train)
    x_test, y_test = _legacy_samples(Test)
    if x_train.shape[1:] != x_test.shape[1:]:
        raise ValueError("Training and validation patch shapes must match")
    return x_train, y_train[:, None], x_test, y_test[:, None]


def export_labels(filename, y):
    """Write a complete text label array and its dimensions."""
    labels = np.asarray(y)
    if labels.ndim == 1:
        labels = labels[:, None]
    if labels.ndim != 2:
        raise ValueError("Labels must be a one- or two-dimensional array")
    with open(filename, "w") as handle:
        handle.write(f"{labels.shape[0]} {labels.shape[1]}\n")
        np.savetxt(handle, labels, fmt="%.0f")


def extension(filename):
    return Path(filename).suffix.lower().lstrip(".")


def load_zerodata(filename):
    import pandas as pd

    return pd.read_csv(filename, delimiter="\t")


def save_zerodata(zerodata, filename):
    """save data from zero class"""
    print("zerodata.shape = ", zerodata.shape)
    if zerodata.shape[0] > 0:
        zerodata.to_csv(filename, sep="\t")
    else:
        print("zerodata.shape = ", zerodata.shape)
        print("No 'zerodata' to save!")


def load_train_test(train_filename, train_labels_filename, test_filename, test_labels_filename):
    """Import legacy datasets and remove the reserved zero label."""
    import pandas as pd

    def load_frame(data_path, label_path):
        bands, rows, cols, data = import_data(data_path)
        label_rows, label_cols, labels = import_labels(label_path)
        if (rows, cols) != (label_rows, label_cols):
            raise ValueError("Data and label dimensions differ")
        frame = pd.DataFrame(data)
        frame["labels"] = labels.astype(np.int64)
        return bands, rows, cols, frame

    bands, rows, cols, train = load_frame(train_filename, train_labels_filename)
    test_bands, _, _, test = load_frame(test_filename, test_labels_filename)
    if bands != test_bands:
        raise ValueError("Training and test band counts differ")
    zerodata = pd.concat([train[train.labels == 0], test[test.labels == 0]])
    train = train[train.labels != 0]
    test = test[test.labels != 0]
    return (
        bands,
        rows,
        cols,
        train.iloc[:, :-1],
        test.iloc[:, :-1],
        train.iloc[:, -1],
        test.iloc[:, -1],
        zerodata,
    )


def debug_print(filename, arr):
    with open(filename, "w") as handle:
        for element in arr:
            handle.write(f"{element}\n")


def save_train_history(hist_dict, filename):
    """save train history"""
    import pandas as pd

    hist_df = pd.DataFrame(hist_dict)
    if extension(filename) == "csv":
        with open(filename, "w") as f:
            hist_df.to_csv(f, sep="\t")
    elif extension(filename) == "mat":
        from scipy.io import savemat

        savemat(
            filename,
            {
                "loss": hist_df["loss"].to_numpy(),
                "accuracy": hist_df["accuracy"].to_numpy(),
                "val_loss": hist_df["val_loss"].to_numpy(),
                "val_accuracy": hist_df["val_accuracy"].to_numpy(),
            },
        )
    elif extension(filename) == "txt":
        with open(filename, "w") as f:
            f.write("loss\taccuracy\tval_loss\tval_accuracy\n")
            np.savetxt(f, hist_df.to_numpy(), delimiter="\t")
