"""Decide stage: score an embedding against the Indy gallery.

The final pipeline stage (``... -> embed -> decide``). The live decision is
*verification by similarity threshold against the Indy gallery only*: a query
embedding is compared to the gallery vectors and the best aggregated cosine
similarity is the score. (Choosing the threshold is calibration's job, not this
module's -- this is the pure scoring core the calibrate/evaluate drivers build
on; see ``docs/calibration_design.md`` Sec. 1 and 5.)

Like the rest of the core this is I/O-free and UI-agnostic: it takes plain numpy
arrays and returns plain data. Two design points from the handoff live here:

* **Normalize at decide time.** Stored gallery vectors are raw (un-normalized)
  by design, to keep the linear-probe escalation path open. Cosine similarity is
  a dot product *after* L2-normalizing both sides, which is done here.
* **``max`` aggregation is the default.** A query's score is the single best
  cosine similarity to any gallery vector -- Indy's identity is concentrated in
  head and tail, so a query showing one aspect should match the gallery photos
  that share it rather than be diluted by the rest. ``mean-top3`` is the measured
  alternative, selectable per call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

#: How a query's per-gallery similarities collapse to one score.
Aggregation = Literal["max", "mean-top3"]
AGGREGATIONS: tuple[Aggregation, ...] = ("max", "mean-top3")

#: Number of top matches averaged by the ``mean-top3`` aggregation.
_TOP_K = 3
#: Floor for a vector norm, so a (degenerate) zero vector never divides by zero.
_EPS = 1e-12


def l2_normalize(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """L2-normalize a single vector ``(dim,)`` or a batch ``(n, dim)``.

    Normalizing both gallery and query turns cosine similarity into a plain dot
    product. Zero-norm rows (degenerate, not expected from DINOv2) are divided by
    a tiny floor instead of zero, so they stay zero rather than become ``nan``.
    """
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    normalized: NDArray[np.float32] = (vectors / np.maximum(norms, _EPS)).astype(
        np.float32
    )
    return normalized


@dataclass(frozen=True)
class Gallery:
    """The Indy reference vectors to score against, L2-normalized.

    Construct via :meth:`from_raw` from the stored (raw) embeddings; ``vectors``
    are then unit-norm so a dot product is cosine similarity. ``names`` are the
    ``source_filename`` of each row, so a match can name the best gallery photo.
    """

    names: tuple[str, ...]
    vectors: NDArray[np.float32]

    @classmethod
    def from_raw(cls, names: list[str], raw_vectors: NDArray[np.float32]) -> Gallery:
        """Build a gallery from raw (un-normalized) stored vectors."""
        if len(names) != raw_vectors.shape[0]:
            raise ValueError(
                f"names ({len(names)}) and vectors ({raw_vectors.shape[0]}) "
                "must be row-aligned"
            )
        if raw_vectors.shape[0] == 0:
            raise ValueError("a gallery cannot be empty")
        return cls(names=tuple(names), vectors=l2_normalize(raw_vectors))


@dataclass(frozen=True)
class Match:
    """The outcome of scoring one query against a gallery.

    ``best_index``/``best_name`` always point at the single closest gallery
    vector (``argmax`` of the cosine similarities), regardless of aggregation;
    ``score`` is the aggregated value the threshold is compared to.
    """

    score: float
    best_index: int
    best_name: str


def aggregate(similarities: NDArray[np.float32], aggregation: Aggregation) -> float:
    """Collapse per-gallery similarities to one score.

    ``max`` is the single best match; ``mean-top3`` averages the top
    ``min(3, n)`` (so it degrades gracefully on a gallery smaller than three).
    """
    if similarities.size == 0:
        raise ValueError("cannot aggregate an empty similarity array")
    if aggregation == "max":
        return float(similarities.max())
    if aggregation == "mean-top3":
        k = min(_TOP_K, similarities.size)
        top_k = np.sort(similarities)[-k:]
        return float(top_k.mean())
    raise ValueError(
        f"unknown aggregation {aggregation!r}; expected one of {AGGREGATIONS}"
    )


def score(
    query_raw: NDArray[np.float32],
    gallery: Gallery,
    aggregation: Aggregation = "max",
) -> Match:
    """Score one raw query embedding against the gallery.

    The query is L2-normalized, dotted with the (already-normalized) gallery to
    get per-vector cosine similarities, then reduced by ``aggregation``. The
    reported best match is the closest single vector.
    """
    query = l2_normalize(query_raw)
    similarities = gallery.vectors @ query
    best_index = int(np.argmax(similarities))
    return Match(
        score=aggregate(similarities, aggregation),
        best_index=best_index,
        best_name=gallery.names[best_index],
    )


def score_many(
    queries_raw: NDArray[np.float32],
    gallery: Gallery,
    aggregation: Aggregation = "max",
) -> list[Match]:
    """Score a batch of raw query embeddings ``(n, dim)`` against the gallery."""
    return [score(query, gallery, aggregation) for query in queries_raw]
