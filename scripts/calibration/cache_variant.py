"""Match a loaded embeddings cache against a frozen artifact's embedding identity.

When grading a frozen artifact, the test/eval caches are dictated by the
artifact -- there is no ``--model``/``--margin`` flag to select them (invariant
#2 of ``docs/embeddings_provenance.md``). These helpers turn the artifact's
:class:`~calibration.artifact.EmbeddingIdentity` into the cache directory it
implies and assert a loaded sidecar matches it, so a held-out exam embedded on
different footing than the gallery fails loud rather than scoring a query against
a mismatched gallery (the silently-wrong accuracy the provenance design exists to
prevent).

Shared by ``evaluate.py`` (Oxford test set) and ``evaluate_catbreeds.py`` (the
cat-breeds eval) so the match logic cannot diverge between them.
"""

from __future__ import annotations

from _common import EmbeddingsMeta, EmbeddingsVariant
from calibration.artifact import EmbeddingIdentity
from calibration.manifest import SplitConfigError


def artifact_variant(embedding: EmbeddingIdentity) -> EmbeddingsVariant:
    """The cache variant directory the frozen artifact dictates a set was embedded in.

    ``EmbeddingsVariant`` carries the three directory axes (``model_id``,
    ``detect``, ``margin``); ``margin`` is irrelevant under ``--no-detect`` so it
    is recorded ``None`` in the artifact -- coerce it to ``crop_slug``'s harmless
    ``0.0`` for the path (``nocrop`` ignores it anyway).
    """
    return EmbeddingsVariant(
        model_id=embedding.model_id,
        detect=embedding.detect,
        margin=embedding.margin if embedding.margin is not None else 0.0,
    )


def artifact_variant_key(
    embedding: EmbeddingIdentity,
) -> tuple[str, bool, float | None, float | None]:
    """The artifact's normalized identity, matching :meth:`EmbeddingsMeta.variant_key`.

    ``(model_id, detect, margin, min_confidence)`` with ``margin``/``min_confidence``
    forced to ``None`` when ``detect`` is false -- so the artifact compares equal to
    the sidecars it was frozen against (the artifact already stores them ``None``,
    but normalize here too so the comparison is symmetric with the cache side).
    """
    if not embedding.detect:
        return (embedding.model_id, embedding.detect, None, None)
    return (
        embedding.model_id,
        embedding.detect,
        embedding.margin,
        embedding.min_confidence,
    )


def assert_cache_matches_artifact(
    meta: EmbeddingsMeta, embedding: EmbeddingIdentity, dataset: str
) -> None:
    """Loud if a loaded cache's variant != the artifact's frozen embedding.

    A cache embedded with a different backbone or crop than the gallery would
    compare a query against the gallery on mismatched footing, exactly the
    silently-wrong accuracy the provenance design prevents, so a drift is a hard
    error.
    """
    if meta.variant_key() != artifact_variant_key(embedding):
        raise SplitConfigError(
            f"{dataset} cache variant {meta.variant_key()} != the artifact's "
            f"frozen embedding identity {artifact_variant_key(embedding)}; it was "
            "embedded on different footing than the gallery"
        )
