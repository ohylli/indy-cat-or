"""Score a manifest's roles against the gallery, via the ``indycat.decision`` core.

The bridge between a split manifest (images named by ``source_filename``) and the
UI-agnostic decide API: look a name up in the embeddings cache, score it against
the gallery, and return :class:`ScoredImage` rows the metrics/renderers consume.
A name the manifest references but the cache lacks is a loud failure, never a
silently shrunk role.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from calibration.metrics import ScoredImage
from indycat.decision import Aggregation, Gallery, score


def build_name_to_vector(
    names: list[str], vectors: NDArray[np.float32]
) -> dict[str, NDArray[np.float32]]:
    """Map each ``source_filename`` to its embedding row."""
    return {name: vectors[i] for i, name in enumerate(names)}


def select_vectors(
    names: list[str], name_to_vector: dict[str, NDArray[np.float32]]
) -> NDArray[np.float32]:
    """Stack the vectors for ``names``; a name absent from the cache is loud.

    A manifest references images by ``source_filename``; one missing from the
    embeddings cache means the manifest and the cache disagree (a re-embed or an
    Oxford no-cat miss), which must surface rather than silently shrink the role.
    """
    missing = [name for name in names if name not in name_to_vector]
    if missing:
        raise KeyError(
            f"{len(missing)} manifest image(s) absent from the embeddings cache: "
            f"{missing[:5]}{' ...' if len(missing) > 5 else ''}"
        )
    return np.stack([name_to_vector[name] for name in names])


def score_role(
    role_names: list[str],
    name_to_vector: dict[str, NDArray[np.float32]],
    gallery: Gallery,
    aggregation: Aggregation,
    breeds: dict[str, str] | None = None,
) -> list[ScoredImage]:
    """Score every image in a role against the gallery."""
    missing = [name for name in role_names if name not in name_to_vector]
    if missing:
        raise KeyError(
            f"{len(missing)} manifest image(s) absent from the embeddings cache: "
            f"{missing[:5]}{' ...' if len(missing) > 5 else ''}"
        )
    scored: list[ScoredImage] = []
    for name in role_names:
        match = score(name_to_vector[name], gallery, aggregation)
        breed = breeds.get(name) if breeds is not None else None
        scored.append(ScoredImage(name, match.score, match.best_name, breed))
    return scored
