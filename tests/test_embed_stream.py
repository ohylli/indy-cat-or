"""Tests for ``embed_stream`` -- the streaming batch embedder in ``_common``.

These exercise the batching/skip/alignment logic with a stub embedder, so they
need no model weights and stay fast. The stub records the size of every
``embed_batch`` call (proving peak live crops never exceed one batch) and
encodes each crop's width into its vector (proving crops stay aligned with their
rows across batch boundaries).
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from _common import embed_stream
from indycat.embedding import Embedder


class StubEmbedder(Embedder):
    """An Embedder that skips weight loading and fakes ``embed_batch``.

    Each crop becomes a vector whose entries are all the image *width*, so a
    returned vector can be traced back to the exact crop that produced it. Every
    batch size handed to ``embed_batch`` is recorded in ``batch_sizes``.
    """

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim
        self.batch_sizes: list[int] = []

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed_batch(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        self.batch_sizes.append(len(images))
        return np.array(
            [[float(img.width)] * self._dim for img in images], dtype=np.float32
        )


def crop(width: int) -> Image.Image:
    """A 1px-tall image whose width is its identity tag."""
    return Image.new("RGB", (width, 1))


def test_batches_and_preserves_order() -> None:
    embedder = StubEmbedder()
    items = [(f"row{w}", crop(w)) for w in (10, 11, 12, 13, 14)]
    rows, vectors = embed_stream(embedder, items, batch_size=2, total=5, desc="t")

    # 5 crops at batch_size 2 -> forward passes of 2, 2, 1 (never more than one
    # batch of crops live at once).
    assert embedder.batch_sizes == [2, 2, 1]
    assert rows == ["row10", "row11", "row12", "row13", "row14"]
    assert vectors.shape == (5, embedder.embedding_dim)
    # Column 0 is each crop's width: order preserved across batch boundaries.
    np.testing.assert_array_equal(vectors[:, 0], [10, 11, 12, 13, 14])


def test_skips_are_excluded_from_output_but_do_not_break_batches() -> None:
    embedder = StubEmbedder()
    # None entries (no-cat skips) are interleaved with kept crops.
    items: list[tuple[str, Image.Image] | None] = [
        ("a", crop(10)),
        None,
        ("b", crop(11)),
        ("c", crop(12)),
        None,
    ]
    rows, vectors = embed_stream(embedder, items, batch_size=2, total=5, desc="t")

    # Only the 3 kept crops are embedded; skips contribute no row, no vector.
    assert embedder.batch_sizes == [2, 1]
    assert rows == ["a", "b", "c"]
    assert vectors.shape == (3, embedder.embedding_dim)
    np.testing.assert_array_equal(vectors[:, 0], [10, 11, 12])


def test_empty_stream_returns_empty_array() -> None:
    embedder = StubEmbedder()
    items: list[tuple[str, Image.Image] | None] = []
    rows, vectors = embed_stream(embedder, items, batch_size=2, total=0, desc="t")
    assert rows == []
    assert vectors.shape == (0, embedder.embedding_dim)
    assert vectors.dtype == np.float32


def test_all_skips_returns_empty_array() -> None:
    embedder = StubEmbedder()
    items: list[tuple[str, Image.Image] | None] = [None, None]
    rows, vectors = embed_stream(embedder, items, batch_size=2, total=2, desc="t")
    assert rows == []
    assert vectors.shape == (0, embedder.embedding_dim)
    assert embedder.batch_sizes == []
