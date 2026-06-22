# Embedding provenance: model + crop identity through the pipeline

## Why

We are about to experiment with different DINOv2 variants (and later DINOv3).
Today the embedding model id is an *input* that gets thrown away: `Embedder`
takes `model=` but never stores it, the gallery builders pass it through to a
print only, and `scripts/predict_app/app.py` hardcodes `Embedder()`
(dinov2-base) with **no link to which model actually built the artifact's
gallery**. The same blind spot exists for the crop settings (detect on/off and
the margin).

The failure mode this design prevents: scoring a query embedded with model A
(or crop setting A) against a gallery built with model B. A dimension change
(base→large) would crash, but a same-dim swap
(`dinov2-base` → `dinov2-with-registers-base`, both 768-dim) would *not* — it
would silently produce wrong accuracy numbers, exactly what the project's data
rules exist to avoid. The crop setting matters just as much: the whole reason
Oxford is re-detected instead of reusing its shipped boxes is to keep positives
and negatives on **identical footing**; a gallery embedded crop-on/margin-0.1
against negatives embedded crop-off would bias the threshold with nothing to
catch it.

## Design invariants

1. **Identity is stored once per dataset** — a sidecar YAML beside
   `embeddings.npy` — and **once in the calibration artifact**. Folder and file
   names are only human handles; they are never parsed for identity.
2. **Two embedding sources that meet are asserted-equal** on
   `(model_id, detect, margin, min_confidence)`, the same loud-on-drift
   discipline as the existing row-count (`load_cached_embeddings`) and gallery
   fingerprint (`load_artifact`) checks.
3. **Crop settings change *which* images are embedded** (detect-off embeds the
   Oxford no-cat misses that detect-on skips), so they change the split — but
   **the embedding model does not** (embedding never fails; only detection
   misses, and detection depends on the detector + `min_confidence` + the
   detect toggle, not on the embedding model). Therefore crop settings must flow
   into the manifest / report / artifact *names*; the model only needs to flow
   into identity.

## Folder layout

Model first, crop variant nested under it:

```
data/embeddings/indy/facebook--dinov2-base/crop-m0.1/{embeddings.npy, metadata.csv, embeddings.meta.yaml}
data/embeddings/oxford/facebook--dinov2-base/crop-m0.1/{...}            # detect on,  margin 0.1
data/embeddings/oxford/facebook--dinov2-base/nocrop/{...}               # detect off
data/artifacts/calibration-…-dinov2-base-crop-m0.1.yaml (+ .gallery.npy)
```

Model-first nesting matches the primary experiment axis (the backbone). Crop
variants sit as siblings under each model, so "crop vs no-crop for this model"
is an easy comparison and "delete this model" is one `rm -r`. The no-arg default
run lands in `facebook--dinov2-base/crop-m0.1/`.

### Slugs

- `model_slug("facebook/dinov2-base")` → `facebook--dinov2-base` (the `/` would
  otherwise be a path separator; keeping the org disambiguates future ids like
  `facebook/dinov3-vitb16-…`).
- `crop_slug(detect, margin)` → `crop-m0.1` (detect on, `:g`-formatted margin)
  or `nocrop` (detect off — margin omitted because it does not apply).

## Sidecar contents (`embeddings.meta.yaml`)

```yaml
format_version: 1
model_id: facebook/dinov2-base
embedding_dim: 768
normalized: false
detect: true
margin: 0.1            # null when detect off
min_confidence: 0.25   # null when detect off; sidecar-only, NOT in the folder name
row_count: 33
```

`min_confidence` is recorded for honesty (it can change which crop is selected)
but is **not** in the folder name — it is not an axis we vary; if we ever sweep
it, we promote it into the handle then.

The **variant-match key** = `(model_id, detect, margin, min_confidence)`, with
`margin`/`min_confidence` normalized to `None` when `detect` is false (so two
`nocrop` caches match regardless of an irrelevant recorded margin).

## Step-by-step changes

### 1. `src/indycat/embedding.py`
Store `self.model_id = model` in `__init__` (stops discarding the name).
`embedding_dim` already exists. No behavior change.

### 2. `scripts/_common.py` — shared variant + sidecar layer (pure, unit-testable)
```python
def model_slug(model_id: str) -> str
def crop_slug(detect: bool, margin: float) -> str

@dataclass(frozen=True)
class EmbeddingsVariant:                 # model_id, detect, margin
    def subdir(self) -> Path             # model_slug / crop_slug
    def dir(self, dataset_root: Path) -> Path

@dataclass(frozen=True)
class EmbeddingsMeta:                     # the sidecar's data model
    def variant_key(self) -> tuple        # normalized identity for equality

SIDECAR_NAME = "embeddings.meta.yaml"
def write_embeddings_meta(meta, out_dir) -> Path
def read_embeddings_meta(out_dir) -> EmbeddingsMeta              # loud if missing
def load_embeddings_variant(out_dir) -> tuple[names, vectors, EmbeddingsMeta]
    # composes load_cached_embeddings + read_embeddings_meta,
    # adds meta.row_count vs npy-rows cross-check (loud)
```
`load_cached_embeddings` stays as-is (still the row-count guard);
`load_embeddings_variant` wraps it.

### 3. Both builders — default path + write sidecar
`scripts/build_indy_gallery.py`, `scripts/build_oxford_negatives.py`:
- `--out-dir` default becomes `None`; when `None`, derive via
  `EmbeddingsVariant(model, not no_detect, margin).dir(REPO_ROOT/"data"/"embeddings"/<dataset>)`.
  Keeps the no-arg run working and makes it impossible to drop large vectors into
  the base folder by accident.
- After embedding, `write_embeddings_meta(...)` from `embedder.model_id`,
  `embedder.embedding_dim`, `detect=not no_detect`, `margin`, `min_confidence`,
  `normalized=False`, `row_count=len(rows)`.
- Oxford's `catalog.csv` stays where it is (describes the raw dataset, not a run).

### 4. `scripts/calibration/manifest.py` — variant-aware paths
- Replace the four `INDY_*` / `OXFORD_*` path constants with an
  `EMBEDDINGS_ROOT` + resolver taking an `EmbeddingsVariant` (reuse `_common`).
  Add `DEFAULT_MODEL = "facebook/dinov2-base"`, `DEFAULT_DETECT = True`,
  `DEFAULT_MARGIN = 0.1` so the zero-arg baseline still resolves to the
  dinov2-base/crop-m0.1 dir.
- `load_indy_metadata` / `load_oxford_metadata` take the resolved metadata path
  (already parameterized; just stop defaulting to the removed constant).

### 5. `scripts/calibration/cli.py` (calibrate) — select + assert the variant
- Add **scoring** flags (compose with `--manifest`; *not* in `_GENERATION_FLAGS`):
  `--model` (default `DEFAULT_MODEL`), `--no-detect`, `--margin` (default `0.1`).
- Resolve indy+oxford variant dirs; load both via `load_embeddings_variant`;
  **assert `indy_meta.variant_key() == oxford_meta.variant_key()`** →
  `SplitConfigError` (the identical-footing guard).
- Thread the variant slug into `default_manifest_name` / `default_report_name` /
  `default_artifact_name` (required for manifests, since crop settings change the
  split; harmless-but-clearer for model).
- Pass the variant meta into `build_artifact`.

### 6. `scripts/calibration/artifact.py` — record identity, bump to v2
- `ARTIFACT_FORMAT_VERSION = 2`.
- Add an operative `embedding` block to `CalibrationArtifact`:
  `model_id, embedding_dim, detect, margin, min_confidence`. Serialize /
  deserialize in `artifact_to_dict` / `_artifact_from_dict`.
- In `load_artifact`, add a loud cross-check: `embedding_dim == raw_vectors.shape[1]`
  (catches a base artifact paired with large vectors even if the fingerprint file
  was swapped wholesale).

### 7. `scripts/calibration/evaluate.py` — variant from the artifact
- Resolve the test-set cache dirs from
  `artifact.embedding.{model_id, detect, margin}` (not the removed constants);
  load via `load_embeddings_variant`.
- **Assert the loaded sidecars' `variant_key()` matches the artifact's**
  embedding identity → loud. No new CLI flag — the variant is dictated by the
  frozen artifact.

### 8. Predict app — close the live loop
- `scripts/predict_app/app.py`: `load_embedder(model_id)` →
  `Embedder(model=model_id)`, `@st.cache_resource` keyed by `model_id`; call with
  `artifact.model_id`. The detect toggle defaults to `artifact.detect`
  (overridable, like the threshold slider).
- `scripts/predict_app/predict.py`: add a `margin: float` param to `classify`,
  pass it into `detect_and_crop(image, detector, margin)` (today it silently uses
  the function default `0.1`); the app passes `artifact.margin`. Optional guard:
  assert `embedder.embedding_dim == gallery` width at bundle build.

### 9. Tests
- New: `model_slug` / `crop_slug` / `EmbeddingsVariant` paths, sidecar
  round-trip, `load_embeddings_variant` row-count + missing-sidecar errors, the
  calibrate dual-sidecar mismatch, the evaluate artifact-vs-cache mismatch,
  artifact v2 round-trip + dim cross-check.
- Update: `test_artifact` (format v2 + new block), `test_calibrate`,
  `test_evaluate` (fixtures write into a variant dir + sidecar), `test_predict`
  (margin threading; the fakes subclass `Embedder`, so adding `model_id` is
  trivial).

### 10. Docs
- `CLAUDE.md`: update the gallery / calibration / predict bullets for the new
  layout, sidecar, and variant flags.
- `docs/calibration_design.md`: artifact format v2 + the new CLI flags + the
  variant-match assertions.
- Reference this file from `docs/dinov3_setup.md`.

## The assertion net (summary)

| Where | Check | On failure |
| --- | --- | --- |
| Builder | writes the sidecar (source of truth) | — |
| `load_embeddings_variant` | sidecar present; `row_count` vs `.npy` rows | loud |
| Calibrate | indy sidecar `variant_key` == oxford sidecar `variant_key` | loud |
| `load_artifact` | `embedding_dim` == vector width | loud |
| Evaluate | test caches' `variant_key` == artifact embedding identity | loud |
| Predict | `embedder.embedding_dim` == gallery width (optional) | loud |

## Migration

Decision: **delete the existing flat caches and re-run the builders.** The old
`data/embeddings/{indy,oxford}/` files have no sidecar and do not sit in a
variant subdir, so the loaders would fail loudly anyway (by design). Re-running
is cheap (`build_indy` is 35 images; `build_oxford` is the only slow one,
minutes on the 3070) and produces real sidecars. After migration, any artifact
built before v2 must be regenerated too.

## Sequencing (each step independently runnable)

1 → 2 (unit tests, no behavior change) → 3 (run `build_indy`, then
`build_oxford --limit 16` as a smoke test) → 4 + 5 + 6 (zero-arg `calibrate.py`
against the rebuilt caches, writing a v2 artifact) → 7 (`evaluate.py`) → 8 (run
the app / `test_predict`) → 9 + 10.
