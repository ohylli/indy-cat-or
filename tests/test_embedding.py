"""Tests for the embed stage.

The embedder wraps a real pretrained DINOv2, so these need the weights
(auto-downloaded on first run) and are marked ``slow``; deselect with
``pytest -m "not slow"``. They check the contract callers rely on -- shapes,
dtype, raw (un-normalized) output, and that batching matches single calls --
not the embedding values themselves.
"""

import numpy as np
import pytest
from PIL import Image

from indycat.embedding import Embedder

# Every test here loads real DINOv2 weights.
pytestmark = pytest.mark.slow


def blank_image(width: int = 100, height: int = 100) -> Image.Image:
    return Image.new("RGB", (width, height))


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    # Force CPU so the test runs on a GPU-less CI box; module-scoped so the
    # weights load once for the whole file.
    return Embedder(device="cpu")


def test_embed_returns_one_raw_float32_vector(embedder: Embedder) -> None:
    vector = embedder.embed(blank_image())
    assert vector.shape == (embedder.embedding_dim,)
    assert vector.dtype == np.float32
    # Raw, not L2-normalized: a unit vector would have norm ~1.
    assert np.linalg.norm(vector) > 1.5


def test_embed_batch_shape(embedder: Embedder) -> None:
    batch = embedder.embed_batch([blank_image(), blank_image(), blank_image()])
    assert batch.shape == (3, embedder.embedding_dim)
    assert batch.dtype == np.float32


def test_empty_batch_returns_empty_array(embedder: Embedder) -> None:
    batch = embedder.embed_batch([])
    assert batch.shape == (0, embedder.embedding_dim)
    assert batch.dtype == np.float32


def test_embed_matches_embed_batch(embedder: Embedder) -> None:
    # The single-image path is embed_batch under the hood; same input must give
    # the same vector regardless of which entry point is used.
    image = blank_image()
    single = embedder.embed(image)
    batched = embedder.embed_batch([image])[0]
    np.testing.assert_array_equal(single, batched)


def test_same_image_embeds_deterministically(embedder: Embedder) -> None:
    # Inference-only model in eval mode: identical input -> identical output.
    image = blank_image(120, 80)
    np.testing.assert_array_equal(embedder.embed(image), embedder.embed(image))
