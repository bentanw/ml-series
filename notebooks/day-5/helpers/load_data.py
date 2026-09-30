"""Small, notebook-friendly loader for Google's Quick, Draw! bitmap data."""

from urllib.parse import quote

import numpy as np
import requests


BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"


def _read_bytes(stream, byte_count):
    chunks = []
    while byte_count:
        chunk = stream.read(byte_count)
        if not chunk:
            raise EOFError("The drawing file ended earlier than expected.")
        chunks.append(chunk)
        byte_count -= len(chunk)
    return b"".join(chunks)


def _load_category(name, sample_count):
    url = f"{BASE_URL}/{quote(name)}.npy"

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        response.raw.decode_content = True

        version = np.lib.format.read_magic(response.raw)
        if version == (1, 0):
            read_header = np.lib.format.read_array_header_1_0
        elif version == (2, 0):
            read_header = np.lib.format.read_array_header_2_0
        else:
            raise ValueError(f"Unsupported NumPy file version: {version}")

        shape, fortran_order, dtype = read_header(response.raw)
        if fortran_order:
            raise ValueError("Fortran-ordered arrays are not supported.")

        count = min(sample_count, shape[0])
        pixels_per_image = int(np.prod(shape[1:]))
        value_count = count * pixels_per_image
        data = _read_bytes(response.raw, value_count * dtype.itemsize)

    images = np.frombuffer(data, dtype=dtype, count=value_count)
    return images.reshape(count, 28, 28).copy()


def load_quickdraw(class_names, samples_per_class=5_000, seed=42):
    """Return shuffled, normalized images and integer labels for selected classes."""
    image_batches = []
    label_batches = []

    for label, name in enumerate(class_names):
        print(f"Loading {name} drawings...")
        images = _load_category(name, samples_per_class)
        image_batches.append(images)
        label_batches.append(np.full(len(images), label, dtype=np.int64))

    images = np.concatenate(image_batches).astype(np.float32) / 255.0
    labels = np.concatenate(label_batches)

    order = np.random.default_rng(seed).permutation(len(images))
    return images[order], labels[order]
