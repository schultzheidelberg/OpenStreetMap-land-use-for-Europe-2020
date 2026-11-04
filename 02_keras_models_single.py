"""Train the original 13-output ResNet from explicit sample files."""

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from hsi_io import load_samples


def lr_schedule(epoch, initial_rate=0.01):
    """Learning Rate Schedule

    Learning rate is reduced after epochs 80, 150, 200, and 300.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = initial_rate
    if epoch > 300:
        lr *= 0.5e-3
    elif epoch > 200:
        lr *= 1e-3
    elif epoch > 150:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print("Learning rate: ", lr)
    return lr


def resnet_layer(
    inputs,
    num_filters=16,
    kernel_size=5,
    strides=1,
    activation="relu",
    batch_normalization=True,
    conv_first=True,
):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D

    conv = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding="same")

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v2(input_shape, depth, num_classes=13):
    """Build the historic three-stage ResNet v2 with 1x1/5x5/1x1 bottlenecks.

    Depth is 9n+2 (normally 29), and input patches are normally 12x12.
    Thirteen outputs preserve IDs 0..12, with zero reserved for unlabeled pixels.
    """
    from tensorflow import keras
    from tensorflow.keras.layers import (
        Activation,
        AveragePooling2D,
        BatchNormalization,
        Dense,
        Flatten,
        Input,
    )
    from tensorflow.keras.models import Model

    if depth < 11 or (depth - 2) % 9 != 0:
        raise ValueError("depth must be 9n+2 with n >= 1 (e.g. 29 or 56)")
    if len(input_shape) != 3 or min(input_shape[:2]) < 9 or input_shape[2] < 1:
        raise ValueError("ResNet requires (rows, columns, bands), with rows/columns >= 9")
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs, num_filters=num_filters_in, conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = "relu"
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2  # downsample

            # bottleneck residual unit
            y = resnet_layer(
                inputs=x,
                num_filters=num_filters_in,
                kernel_size=1,
                strides=strides,
                activation=activation,
                batch_normalization=batch_normalization,
                conv_first=False,
            )
            y = resnet_layer(inputs=y, num_filters=num_filters_in, conv_first=False)
            y = resnet_layer(inputs=y, num_filters=num_filters_out, kernel_size=1, conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters_out,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False,
                )
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = AveragePooling2D(pool_size=3)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, activation="softmax", kernel_initializer="he_normal")(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def classification_metrics(labels, predicted):
    """Report fixed-ID metrics; absent true classes do not cause zero division."""
    accuracy = float(np.mean(labels == predicted))
    confusion = np.bincount(labels * 13 + predicted, minlength=13 * 13).reshape(13, 13)
    expected = float(np.dot(confusion.sum(axis=0), confusion.sum(axis=1)) / len(labels) ** 2)
    accuracy_by_class = {
        str(label): float(np.mean(predicted[labels == label] == label))
        for label in range(1, 13)
        if np.any(labels == label)
    }
    return {
        "accuracy": accuracy,
        "cohen_kappa": (accuracy - expected) / (1 - expected) if expected < 1 else None,
        "class_recall": accuracy_by_class,
        "mean_present_class_recall": float(np.mean(list(accuracy_by_class.values()))),
    }


def train_model(args):
    """Train only when explicitly invoked; preserve existing output by default."""
    output = Path(args.output_model)
    if output.suffix not in {".keras", ".h5"}:
        raise ValueError("Output model must end in lowercase .keras or .h5")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}")
    if output.resolve() in {Path(args.train).resolve(), Path(args.validation).resolve()}:
        raise ValueError("Output model must differ from input sample files")
    if (
        args.epochs < 1
        or args.batch_size < 1
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        raise ValueError(
            "Epochs and batch size must be positive; learning rate must be finite and positive"
        )
    if args.depth < 11 or (args.depth - 2) % 9:
        raise ValueError("ResNet depth must be 9n+2 with n >= 1")
    train_path, validation_path = Path(args.train), Path(args.validation)
    if train_path.resolve() == validation_path.resolve() or (
        train_path.exists() and validation_path.exists() and train_path.samefile(validation_path)
    ):
        raise ValueError("Training and validation files must differ")
    x_train, y_train = load_samples(args.train, allow_legacy_pickle=args.allow_legacy_pickle)
    x_validation, y_validation = load_samples(
        args.validation, allow_legacy_pickle=args.allow_legacy_pickle
    )
    if x_train.shape[1:] != x_validation.shape[1:]:
        raise ValueError("Training and validation patch shapes must match")
    if min(x_train.shape[1:3]) < 9:
        raise ValueError("Training patches must be at least 9 by 9 pixels for this ResNet")

    # Keep TensorFlow optional for preprocessing, tests, imports, and --help.
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "Training requires TensorFlow; install requirements-training.txt"
        ) from error
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    keras = tf.keras
    keras.utils.set_random_seed(args.seed)
    model = resnet_v2(x_train.shape[1:], args.depth, num_classes=13)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Nadam(
            learning_rate=args.learning_rate, beta_1=0.9, beta_2=0.999
        ),
        metrics=["accuracy"],
    )
    model.summary()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="osm4eo-training-", dir=output.parent) as temporary:
        checkpoint_path = Path(temporary) / "best.keras"
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                str(checkpoint_path), monitor="val_accuracy", save_best_only=True, mode="max"
            )
        ]
        if args.learning_rate_schedule:
            callbacks.append(
                keras.callbacks.LearningRateScheduler(
                    lambda epoch: lr_schedule(epoch, args.learning_rate)
                )
            )
        model.fit(
            x_train,
            keras.utils.to_categorical(y_train, 13),
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=callbacks,
            validation_data=(x_validation, keras.utils.to_categorical(y_validation, 13)),
        )
        best_model = keras.models.load_model(checkpoint_path, compile=False)
        # Optimized prediction graphs can mask non-finite weights behind finite
        # scores. Check one tensor at a time before accepting the checkpoint.
        for weight in best_model.weights:
            if not np.isfinite(weight.numpy()).all():
                raise ValueError("Best model contains non-finite weights; training diverged")
        start = time.monotonic()
        scores = np.asarray(best_model.predict(x_validation, batch_size=args.batch_size, verbose=0))
        if scores.shape != (len(x_validation), 13) or not np.isfinite(scores).all():
            raise ValueError("Best model must return finite validation scores with shape (N, 13)")
        predicted = np.argmax(scores, axis=1)
        elapsed = time.monotonic() - start
        final_path = Path(temporary) / output.name
        best_model.save(final_path)
        if args.overwrite:
            os.replace(final_path, output)
        else:
            os.link(final_path, output)
            final_path.unlink()
    metrics = classification_metrics(y_validation, predicted)
    metrics["prediction_seconds"] = elapsed
    print(json.dumps(metrics, indent=2))
    print(f"Saved best validation model to {output}")
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Training .npz sample file")
    parser.add_argument("--validation", required=True, help="Validation .npz sample file")
    parser.add_argument("--output-model", required=True, help="Best model .keras or .h5 path")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--depth", type=int, default=29)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--learning-rate-schedule",
        action="store_true",
        help="Opt in to the historical schedule (previously defined but unused)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-legacy-pickle",
        action="store_true",
        help="Trust and load old object-array .npy samples; never use for untrusted files",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        train_model(args)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
