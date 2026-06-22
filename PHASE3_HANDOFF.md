# Phase 3 handoff — cat-breeds dataset (Kaggle ma7555)

**Temporary doc.** Delete it once Phase 3 is merged. It hands off the finishing
work (HTML report, tests, docs, cleanup) for the cat-breeds evaluation feature to
a fresh session. Phases 1 (build) and 2 (eval) are **done, validated, lint/type
clean, all 175 existing tests pass, and committed to `main`** (see the commit
whose subject is `Add cat-breeds (Kaggle ma7555) eval-negatives build +
evaluation`). Phase 3 is the remaining work below.

## What this feature does

Adds a second negatives source alongside Oxford: the whole **cat-breeds (Kaggle
ma7555) dataset** as a held-out false-positive stress test. Download → stratified
sample → detect+crop+embed (same footing as the gallery) → grade the frozen
calibration artifact against every cat-breeds cat as a negative, with the Indy
held-out `test` photos as positives so recall and FPR show side by side.

It is purely an **evaluation** dataset for now — never enters the gallery or
calibration (split discipline preserved). The dataset is large (126,607 images,
67 "breeds", many of them coat-pattern not breed labels — noisy, as the user's
sighted friend flagged) and includes a Norwegian Forest Cat slice.

## Measured baseline result (record it in the docs)

Against the frozen `dinov2-base / crop-m0.1` artifact, 9,719 cat-breeds cats
(per-breed cap 200, seed 20240601), frozen threshold 0.7199, aggregation `max`:

| Metric | Value |
| --- | --- |
| Recall (Indy) | 0.900 (9/10; miss = `standing_floor_side_07` @ 0.706) |
| FPR (all) | 0.013 (124 / 9719) |
| FPR (long-haired look-alikes, 14-breed set) | 0.040 |
| **FPR (NFC only)** | **0.167** |
| FPR (other / shorthairs) | 0.004 |

Per-breed: every false positive is long-haired. Worst: Norwegian Forest Cat
0.167, Cymric 0.118, Ragamuffin 0.099, Turkish Van 0.070, Siberian 0.059,
Domestic Long Hair 0.050. Flat-faced longhairs (Himalayan/Birman/Persian) and all
shorthairs ≈ 0. Signal: the system is confused specifically by the bushy
semi-longhair NFC body type Indy is, not by long fur in general. This is the
"real exam" — these breeds are unseen during calibration.

## Files created / modified this session (the surface to finish)

**Created**
- `scripts/build_catbreeds_negatives.py` — builder (kagglehub download into
  `images/cat-breeds/`, `--per-breed-limit` stratified seeded sample, corrupt-image
  skip, variant-nested cache + sidecar identical to Oxford's). Clone of
  `build_oxford_negatives.py`.
- `scripts/calibration/cache_variant.py` — shared `artifact_variant`,
  `artifact_variant_key`, `assert_cache_matches_artifact` (factored out of
  `evaluate.py`; both evaluators import them).
- `scripts/calibration/evaluate_catbreeds.py` — eval logic + CLI `main`.
- `scripts/calibration/evaluate_catbreeds_report.py` — **text** report;
  `CATBREEDS_LOOKALIKE_BREEDS` (14 long-haired breeds, folder-name spelled) +
  `NFC_BREED`; NFC broken out; **no drift table** (calibration look-alikes are
  Oxford, a different set, so the comparison would mislead).
- `scripts/evaluate_catbreeds.py` — thin shim.

(The throwaway Phase-0 probe `experiments/peek_catbreeds.py` was already removed.)

**Modified**
- `.gitignore` — added `/images/cat-breeds/` (1.9 GB download; mirrors the Oxford
  ignore).
- `pyproject.toml` — added `kagglehub` dep + `kagglehub.*` to the mypy
  `ignore_missing_imports` override. `uv.lock` updated.
- `scripts/calibration/metrics.py` — `build_sweep` gained
  `lookalike_breeds: frozenset[str] = LOOKALIKE_BREEDS` (default keeps Oxford
  behavior; calibrate/evaluate untouched).
- `scripts/calibration/manifest.py` — added `catbreeds_variant_dir`.
- `scripts/calibration/evaluate.py` — removed the 3 private variant helpers (now
  in `cache_variant.py`); renamed `_assert_same_experiment` →
  `assert_same_experiment` (public, reused by the cat-breeds eval). Tests only
  use `run_evaluation`/`main`, so this was safe; verified green.

**Data (gitignored, already built locally)**
- `images/cat-breeds/` (raw, + `catalog.csv`).
- `data/embeddings/catbreeds/facebook--dinov2-base/crop-m0.1/` (9,719 rows +
  sidecar + metadata.csv with a `breed` column).

## Phase 3 tasks

### A. HTML report for the cat-breeds eval (the user specifically noticed `--help` has no `--html`)

`evaluate_catbreeds.py` currently emits text + optional `--scores-out` only. Add
`--html` to match `evaluate.py` (bare flag auto-names into `REPORTS_DIR`, or an
explicit path — copy the `_HTML_AUTO` sentinel pattern + `default_report_name`
from `evaluate.py`).

Write `scripts/calibration/evaluate_catbreeds_report_html.py` mirroring
`evaluate_report_html.py`, but:
- Use `CATBREEDS_LOOKALIKE_BREEDS` in the rates table + add an NFC-only row (pass
  `lookalike_breeds=` into `build_sweep`; read NFC FPR from `build_breed_sweep`,
  same as the text report).
- **Omit the drift table** (consistent with the text report).
- Reuse `report_common` primitives (`HTML_STYLE`, `scoped_table`, `fmt_html`,
  `figure_list`) so the accessible markup stays in one place.

**Gotcha — error-list crop images.** `figure_list` resolves a negative's image by
`source_filename` inside one flat `*_image_dir`. **Cat-breeds images are nested
per breed** (`images/cat-breeds/images/<breed>/<filename>.jpg`), so a flat lookup
won't find them. Also there are **124 false positives** — embedding 124 images is
heavy and noisy for a screen reader. Pick one (recommend in this order):
1. **Text-only error lists in the HTML** (skip `figure_list`; render the same
   scored rows as a `<ol>` like the text report). Simplest, screen-reader-first,
   sidesteps the nested-path problem entirely. **Recommended.**
2. Extend `figure_list` to accept a `name -> Path` resolver (a `{source_filename:
   breed}` map builds `images/<breed>/<file>`), and cap the FP list (e.g. top
   `RISK_ROWS`) with a logged "+N more" note (no silent truncation).

Confirm the choice with the user — option 1 keeps scope small and matches the
project's text-first ethos.

### B. Tests (mirror `tests/test_evaluate.py` + the builder tests)

- **Builder** (`tests/test_build_catbreeds.py`): `list_cat_images` (folder→breed,
  sorted, bare unique filename), `sample_per_breed` (cap respected, 0 =
  unlimited, deterministic per seed, re-sorted), the corrupt-image skip path
  (feed a broken file → counted, not embedded), `write_metadata`/`write_catalog`
  shape. Don't hit the network: build a tiny fake `<root>/images/<breed>/*.jpg`
  tree in `tmp_path`. `ensure_dataset` (kagglehub) needs no test (thin wrapper);
  if you want, monkeypatch `kagglehub.dataset_download`.
- **Eval** (`tests/test_evaluate_catbreeds.py`): reuse `test_evaluate.py`'s
  fixture style (write tiny indy + catbreeds variant caches + sidecars + a
  manifest + an artifact via the existing builders/helpers). Cover: happy-path
  `run_evaluation` (positives recall + all-rows negatives + per-breed/NFC rates),
  the **variant-mismatch** assertion (`assert_cache_matches_artifact` loud when
  the catbreeds sidecar's variant ≠ artifact), the **same-experiment** guard
  (manifest gallery ≠ artifact gallery → loud), empty-catbreeds-cache loud, empty
  Indy-test loud. Also a focused `metrics` test: `build_sweep` with a custom
  `lookalike_breeds` partitions correctly (and the default is unchanged).
- **HTML** test if you add the module: a smoke render asserting the NFC row and
  the look-alike breeds appear and no drift table.
- Run: `uv run pytest`, `uv run ruff check`, `uv run ruff format`, `uv run mypy`.

### C. Docs

- `CLAUDE.md` — add a "Cat-breeds negatives" bullet under the pipeline section
  (builder, the eval, the variant-nested cache under
  `data/embeddings/catbreeds/`), and mention `scripts/evaluate_catbreeds.py`.
  Keep it terse, in the established voice.
- `docs/calibration_design.md` — document the external-negatives evaluation mode
  and **record the measured baseline result above** (the project treats each data
  addition as a measured experiment).
- Note the deferred follow-up (below) wherever the staging plan lives.

### D. Cleanup

- Delete this `PHASE3_HANDOFF.md`.

## Key design decisions already made (don't relitigate)

- **Footing must match the artifact.** The builder defaults to the baseline
  variant (`dinov2-base`, detect on, margin 0.1, min_confidence 0.25) so the cache
  matches the gallery; the eval asserts the loaded sidecar's `variant_key` ==
  the artifact's frozen embedding identity (loud on drift). No `--model`/`--margin`
  flag on the eval — the variant is dictated by the frozen artifact.
- **Sampling.** Stratified per-breed cap (default 200, seed 20240601), all breeds
  kept (every cat is a valid negative), drop count logged (no silent caps).
  `--per-breed-limit 0` = unlimited; `--limit N` = global smoke test.
- **Corrupt images.** This Petfinder scrape has broken JPEGs (Oxford doesn't);
  decode failures are counted + skipped, never embedded as garbage — same
  recoverable discipline as a detector miss (file in catalog, absent from
  metadata). 1 corrupt + 283 detect misses in the full run.
- **Look-alike set + NFC breakout** chosen by the user (long-haired set, NFC
  separate). `load_oxford_metadata` is reused as a generic breed reader (it only
  needs `source_filename` + `breed`).

## How to re-run / verify

```powershell
# build is already done locally; to rebuild:
uv run python scripts/build_catbreeds_negatives.py            # ~5 min, 3070
# the eval (no rebuild needed):
uv run python scripts/evaluate_catbreeds.py --artifact `
  data/artifacts/calibration-three_way-seed20240601-g15-c10-t10-facebook--dinov2-base-crop-m0.1-target-fpr.yaml
uv run pytest
```

## Deferred (NOT this PR — its own experiment)

The 16.7% NFC FPR makes a **hand-curated NFC slice into calibration** worthwhile,
but that's a new experiment with image-level disjointness bookkeeping (the exact
images used in calibration must leave the eval set). Keep it separate.

## Commit (Phase 1–2 already committed)

Phases 1–2 were committed to `main` (subject: `Add cat-breeds (Kaggle ma7555)
eval-negatives build + evaluation`) — source only; `images/cat-breeds/` and
`data/` stay gitignored. Commit Phase 3 as its own follow-up.
