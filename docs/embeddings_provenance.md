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

1. **Identity is stored in the data, never parsed from names.** It lives in
   three places: a sidecar YAML beside each dataset's `embeddings.npy`, the
   **calibration artifact**, and the **split-manifest header** (the variant its
   frozen filename lists were drawn against). Folder and file names are only
   human handles. The manifest carrying its own variant — not just encoding it
   in its filename — is what lets a replay be cross-checked rather than trusted.
2. **Identities that meet are asserted-equal** on
   `(model_id, detect, margin, min_confidence)` — two caches scored together, a
   cache against the manifest or artifact that selects rows from it, and a
   loaded cache against the variant the CLI *asked* for. Same loud-on-drift
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
data/splits/three_way-…-dinov2-base-crop-m0.1.yaml                      # variant in name AND header
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

The builder writes `margin`/`min_confidence` as `null` **at write time** when
`detect` is off (not merely normalized away in the match key), so a `nocrop`
sidecar never carries a misleading `margin: 0.1`.

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
  `normalized=False`, `row_count=len(rows)` — passing `margin`/`min_confidence`
  as `None` when `detect` is off so the on-disk sidecar matches the variant key.
- Oxford's `catalog.csv` stays where it is (describes the raw dataset, not a run).

### 4. `scripts/calibration/manifest.py` — variant-aware paths + variant in the header
- Replace the four `INDY_*` / `OXFORD_*` path constants with an
  `EMBEDDINGS_ROOT` + resolver taking an `EmbeddingsVariant` (reuse `_common`).
  Add `DEFAULT_MODEL = "facebook/dinov2-base"`, `DEFAULT_DETECT = True`,
  `DEFAULT_MARGIN = 0.1` so the zero-arg baseline still resolves to the
  dinov2-base/crop-m0.1 dir.
- `load_indy_metadata` / `load_oxford_metadata` take the resolved metadata path
  (already parameterized; just stop defaulting to the removed constant).
- **Record the variant in the manifest header, not only its filename.** Bump
  `MANIFEST_FORMAT_VERSION` to `2` and add an `embedding` block to
  `SplitManifest` / `GenerationParams` (`model_id, detect, margin,
  min_confidence`), serialized in `manifest_to_dict` and read back in
  `manifest_from_dict`. It is set at generation from the loaded sidecars' shared
  variant (calibrate, step 5, owns the cache loading; `manifest.py` owns no
  I/O). This keeps identity in the data — invariant #1 — instead of trusting a
  filename, and is what the replay cross-check in step 5 asserts against.
  `load_manifest` already rejects a stale `format_version` loudly, so old v1
  manifests fail by design (see Migration).

### 5. `scripts/calibration/cli.py` (calibrate) — select + assert the variant
- Add **scoring** flags (compose with `--manifest`; *not* in `_GENERATION_FLAGS`):
  `--model` (default `DEFAULT_MODEL`), `--no-detect`, `--margin` (default `0.1`).
- Resolve indy+oxford variant dirs from the requested `(model, detect, margin)`;
  load both via `load_embeddings_variant`. Three loud (`SplitConfigError`)
  checks:
  - **each loaded sidecar's `(model_id, detect, margin)` == the CLI-requested
    variant** — catches `--margin 0.2` landing on a 0.1 cache, or a sidecar
    whose contents disagree with its folder.
  - **`indy_meta.variant_key() == oxford_meta.variant_key()`** — the
    identical-footing guard, and the *only* available check for `min_confidence`
    (it has no calibrate flag, so it cannot be checked against the CLI request).
  - on `--manifest` replay, **the manifest header's variant ==
    the loaded sidecars' `variant_key()`** — so a manifest generated `nocrop`
    (whose Oxford lists include the no-cat-miss filenames) replayed with crop-on
    flags fails cleanly here, instead of dying on a raw `KeyError` deep in
    `build_name_to_vector`. On generate, write the shared variant into the
    manifest header.
- Thread the variant slug into `default_manifest_name` / `default_report_name` /
  `default_artifact_name` (required for manifests, since crop settings change the
  split; harmless-but-clearer for model).
- Pass the variant meta into `build_artifact`.

### 6. `scripts/calibration/artifact.py` — record identity, bump to v2
- `ARTIFACT_FORMAT_VERSION = 2`.
- Add an operative `embedding` block to `CalibrationArtifact`:
  `model_id, embedding_dim, detect, margin, min_confidence`. Serialize /
  deserialize in `artifact_to_dict` / `_artifact_from_dict`.
- In `load_artifact`, add a loud cross-check: `embedding_dim == raw_vectors.shape[1]`.
  This is belt-and-suspenders, **not** the same-dim-swap guard: a same-dim swap
  (base → with-registers-base) passes it. The same-dim swap is caught by
  `model_id` flowing into the artifact (predict loads `Embedder(artifact.model_id)`)
  and by the calibrate/evaluate `variant_key` asserts; the dim check only adds a
  catch for a base artifact paired with large vectors after a wholesale file swap.

### 7. `scripts/calibration/evaluate.py` — variant from the artifact
- Resolve the test-set cache dirs from
  `artifact.embedding.{model_id, detect, margin}` (not the removed constants);
  load via `load_embeddings_variant`.
- **Assert the loaded sidecars' `variant_key()` matches the artifact's**
  embedding identity → loud. No new CLI flag — the variant is dictated by the
  frozen artifact.

### 8. Predict app — close the live loop
- `scripts/predict_app/app.py`:
  - `load_embedder(model_id)` → `Embedder(model=model_id)`, `@st.cache_resource`
    keyed by `model_id`; call with `artifact.model_id`.
  - `load_detector(min_confidence)` → `CatDetector(min_confidence=min_confidence)`,
    `@st.cache_resource` keyed by `min_confidence`; call with
    `artifact.min_confidence`. Today `load_detector` builds a bare `CatDetector()`,
    so the live detector silently uses the default threshold rather than the one
    the gallery was built with — the same identical-footing gap the margin fix
    below closes.
  - The detect toggle defaults to `artifact.detect` (overridable, like the
    threshold slider).
- `scripts/predict_app/predict.py`: add a `margin: float` param to `classify`,
  pass it into `detect_and_crop(image, detector, margin)` (today it silently uses
  the function default `0.1`); the app passes `artifact.margin`. With this plus
  the `min_confidence` fix, the live detect→crop→embed matches the gallery's
  full `(model, min_confidence, margin)`, not just its model. Optional guard:
  assert `embedder.embedding_dim == gallery` width at bundle build.

### 9. Data-review app — follow the new layout
- `scripts/data_review/common.py`: replace the hardcoded
  `OXFORD_METADATA_CSV = data/embeddings/oxford/metadata.csv` with a resolve over
  the shared `EmbeddingsVariant` (reuse `_common` + the manifest `DEFAULT_*`),
  pointing at the **baseline crop-on variant** (`DEFAULT_MODEL`, `detect=True`,
  `DEFAULT_MARGIN`). Detection misses depend on the detector + `min_confidence` +
  the detect toggle, **not** the embedding model (invariant #3), so the baseline
  crop-on metadata is the right and stable source whatever model we are
  experimenting with.
- `misses.py` ("Oxford detect misses" = catalog minus embedding metadata) reads
  the resolved path; its existing "run `build_oxford`" empty-state message stays
  (now: build the baseline variant first). `mapping.py` / `crops.py` are
  unaffected — they read `images/indy/mapping.csv` and
  `data/crops/indy/detections.csv`, not the embedding caches.

### 10. Tests
- New: `model_slug` / `crop_slug` / `EmbeddingsVariant` paths, sidecar
  round-trip, `load_embeddings_variant` row-count + missing-sidecar errors, the
  calibrate requested-vs-sidecar mismatch and the dual-sidecar mismatch, the
  manifest v2 round-trip + the `--manifest` replay variant mismatch, the evaluate
  artifact-vs-cache mismatch, artifact v2 round-trip + dim cross-check.
- Update: `test_artifact` (format v2 + new block), `test_split_manifest`
  (format v2 + embedding header), `test_calibrate`, `test_evaluate` (fixtures
  write into a variant dir + sidecar), `test_predict` (margin threading; the
  fakes subclass `Embedder`, so adding `model_id` is trivial — make sure they
  set it, since they don't call `super().__init__`).
- The data-review app has no unit tests; verify step 9 by running it (the
  Playwright review workflow), not in pytest.

### 11. Docs
- `CLAUDE.md`: update the gallery / calibration / predict / data-review bullets
  for the new layout, sidecar, and variant flags.
- `docs/calibration_design.md`: manifest format v2 + artifact format v2 + the
  new CLI flags + the variant-match assertions.
- Reference this file from `docs/dinov3_setup.md`.

## The assertion net (summary)

| Where | Check | On failure |
| --- | --- | --- |
| Builder | writes the sidecar (source of truth) | — |
| `load_embeddings_variant` | sidecar present; `row_count` vs `.npy` rows | loud |
| Calibrate | each sidecar `(model, detect, margin)` == CLI-requested variant | loud |
| Calibrate | indy sidecar `variant_key` == oxford sidecar `variant_key` | loud |
| Calibrate (replay) | manifest header variant == loaded sidecars' `variant_key` | loud |
| `load_artifact` | `embedding_dim` == vector width | loud |
| Evaluate | test caches' `variant_key` == artifact embedding identity | loud |
| Predict | `embedder.embedding_dim` == gallery width (optional) | loud |

## Migration

Decision: **delete the existing flat caches *and* the stale v1
manifests/artifacts, then re-run.** The old `data/embeddings/{indy,oxford}/`
files have no sidecar and do not sit in a variant subdir, so the loaders would
fail loudly anyway (by design). Re-running is cheap (`build_indy` is 35 images;
`build_oxford` is the only slow one, minutes on the 3070) and produces real
sidecars.

Also clear `data/splits/*.yaml` and `data/artifacts/*` (not just regenerate):
manifests bump to v2 and artifacts to v2, so old ones fail `load_manifest` /
`load_artifact` loudly — but a leftover v1 `*.yaml` in `data/artifacts/` would
still be listed by the predict app's `find_artifacts`, and selecting it would
raise inside `load_bundle` as an uncaught Streamlit traceback. Deleting them
keeps that surface clean. After migration, regenerate the manifest + artifact
with a zero-arg `calibrate.py --policy target-fpr --artifact`.

## Sequencing (each step independently runnable)

1 → 2 (unit tests, no behavior change) → 3 (run `build_indy`, then
`build_oxford --limit 16` as a smoke test) → 4 + 5 + 6 (zero-arg `calibrate.py`
against the rebuilt caches, writing a v2 manifest + artifact) → 7
(`evaluate.py`) → 8 (run the app / `test_predict`) → 9 (run the data-review app,
Playwright) → 10 + 11.
