from typing import Any, Callable, List, Optional, Tuple, Union
import logging
from zipfile import ZipFile
import os

import tensorflow as tf
from tensorflow.keras import Model, layers
from .download import download_from_url

def _bf16_to_float32(x: tf.Tensor) -> tf.Tensor:
    # Convert bfloat16 to float32 for numpy compatibility
    return tf.cast(x, tf.float32) if x.dtype == tf.bfloat16 else x

def conv_sequence(
    out_channels: int,
    activation: Optional[Union[str, Callable]] = None,
    bn: bool = False,
    padding: str = "same",
    kernel_initializer: str = "he_normal",
    **kwargs: Any,
) -> List[layers.Layer]:
    """Builds a convolutional-based layer sequence

    >>> from tensorflow.keras import Sequential
    >>> from doctr.models import conv_sequence
    >>> module = Sequential(conv_sequence(32, 'relu', True, kernel_size=3, input_shape=[224, 224, 3]))

    Args:
    ----
        out_channels: number of output channels
        activation: activation to be used (default: no activation)
        bn: should a batch normalization layer be added
        padding: padding scheme
        kernel_initializer: kernel initializer
        **kwargs: additional arguments to be passed to the convolutional layer

    Returns:
    -------
        list of layers
    """
    # No bias before Batch norm
    kwargs["use_bias"] = kwargs.get("use_bias", not bn)
    # Add activation directly to the conv if there is no BN
    kwargs["activation"] = activation if not bn else None
    conv_seq = [layers.Conv2D(out_channels, padding=padding, kernel_initializer=kernel_initializer, **kwargs)]

    if bn:
        conv_seq.append(layers.BatchNormalization())

    if (isinstance(activation, str) or callable(activation)) and bn:
        # Activation function can either be a string or a function ('relu' or tf.nn.relu)
        conv_seq.append(layers.Activation(activation))

    return conv_seq



def load_pretrained_params(
    model: Model,
    url: Optional[str] = None,
    hash_prefix: Optional[str] = None,
    overwrite: bool = False,
    internal_name: str = "weights",
    **kwargs: Any,
) -> None:
    """Load a set of parameters onto a model

    >>> from doctr.models import load_pretrained_params
    >>> load_pretrained_params(model, "https://yoursource.com/yourcheckpoint-yourhash.zip")

    Args:
    ----
        model: the keras model to be loaded
        url: URL of the zipped set of parameters
        hash_prefix: first characters of SHA256 expected hash
        overwrite: should the zip extraction be enforced if the archive has already been extracted
        internal_name: name of the ckpt files
        **kwargs: additional arguments to be passed to `doctr.utils.data.download_from_url`
    """
    if url is None:
        logging.warning("Invalid model URL, using default initialization.")
    else:
        archive_path = download_from_url(url, hash_prefix=hash_prefix, cache_subdir="models", **kwargs)

        # Unzip the archive
        params_path = archive_path.parent.joinpath(archive_path.stem)
        if not params_path.is_dir() or overwrite:
            with ZipFile(archive_path, "r") as f:
                f.extractall(path=params_path)

        # Load weights
        model.load_weights(f"{params_path}{os.sep}{internal_name}")
