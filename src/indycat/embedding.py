"""Embed stage: turn a cat crop into a DINOv2 embedding vector.

This is the second pipeline stage (... -> crop -> embed -> ...). Like the
detector it is I/O-free: it takes already-opened PIL images and returns plain
numpy arrays, so callers own where images come from and where vectors go.

The backbone is a frozen, pretrained DINOv2 vision transformer used in
*inference only* -- it is never trained. It is loaded through Hugging Face
``transformers`` so the matching image processor (resize + normalization)
ships with the weights and the variant is swappable by name (a larger DINOv2,
or DINOv3 later) without touching this interface.

Vectors are returned raw -- NOT L2-normalized. Normalization belongs to the
decide stage (cosine similarity), and keeping the stored gallery the model's
literal output leaves the linear-probe escalation path open.
"""

from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class Embedder:
    """Frozen DINOv2 backbone that maps an image to an embedding vector.

    Default model: ``facebook/dinov2-base`` (768-dim embeddings). The single
    image-level vector is DINOv2's pooled CLS token -- the conventional image
    embedding. Whether mean-pooling the patch tokens identifies Indy better is
    an open question to measure later; it would be an internal change here, not
    a change to ``embed``.
    """

    def __init__(
        self,
        model: str = "facebook/dinov2-base",
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoImageProcessor.from_pretrained(model)
        self._model = AutoModel.from_pretrained(model)
        self._model.eval().to(self.device)

    @property
    def embedding_dim(self) -> int:
        """Length of each vector (768 for dinov2-base)."""
        return int(self._model.config.hidden_size)

    def embed(self, image: Image.Image) -> NDArray[np.float32]:
        """One image -> one raw vector of shape ``(embedding_dim,)``."""
        vector: NDArray[np.float32] = self.embed_batch([image])[0]
        return vector

    def embed_batch(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        """N images -> array of shape ``(N, embedding_dim)`` in one forward pass.

        The GPU-efficient path: preprocessing and the transformer run on the
        whole batch at once. This is what makes embedding the ~2000 Oxford
        crops a matter of seconds rather than minutes.
        """
        if not images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        # The processor handles each model's own resize + normalization, so we
        # never hand-code preprocessing. RGB guards against alpha/palette input.
        rgb = [image.convert("RGB") for image in images]
        inputs = self._processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
        # pooler_output is DINOv2's CLS-derived image-level embedding.
        embeddings: NDArray[np.float32] = (
            outputs.pooler_output.cpu().numpy().astype(np.float32)
        )
        return embeddings
