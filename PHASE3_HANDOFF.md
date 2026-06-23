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

### A. HTML report for the cat-breeds eval — **DONE** (committed as its own follow-up)

`evaluate_catbreeds.py` now has `--html` matching `evaluate.py` (bare flag
auto-names into `REPORTS_DIR` via `default_report_name`, or an explicit path; same
`_HTML_AUTO` sentinel pattern). `html_out` threads through `run_evaluation`.

New `scripts/calibration/evaluate_catbreeds_report_html.py` mirrors
`evaluate_report_html.py`:
- Uses `CATBREEDS_LOOKALIKE_BREEDS` in the rates table + an **FPR (NFC only)** row
  (passes `lookalike_breeds=` into `build_sweep`; reads NFC FPR from
  `build_breed_sweep`, same as the text report).
- **No drift table** (consistent with the text report).
- Reuses `report_common` primitives (`HTML_STYLE`, `scoped_table`, `fmt_html`,
  `figure_list`).

**Image decision (chosen with the user): option 2.** `figure_list` (in
`report_common.py`) gained an optional `candidate_resolver: Callable[[str], str]`
(and `figure` an optional `rel_path`) — backward-compatible, every existing caller
unchanged. The cat-breeds report passes a resolver that maps the bare
`source_filename` to `<breed>/<file>` under the new `CATBREEDS_IMAGE_DIR`
(`images/cat-breeds/images`, added to `manifest.py`), so the nested per-breed
images resolve. The FP figure list is capped at a module constant
`MAX_FP_FIGURES = 20` (easily editable) with a "+N more not shown; see the text
report or the --scores-out CSV" note (no silent truncation — the full list stays
in the text report and CSV). False negatives stay uncapped (tiny).

Smoke test added: `tests/test_evaluate_catbreeds_report_html.py` (NFC row, scoped
tables, no drift table, the FP cap + note, the nested resolver + forward-slash
src). End-to-end against the real cache reproduces the documented baseline
(FPR NFC only = 0.167, 20 shown + "+104 more", 1 FN).

### B. Tests (mirror `tests/test_evaluate.py` + the builder tests) — **DONE**

- **Test-enabling refactor.** The corrupt-image skip path lived in a closure
  inside `build_catbreeds_negatives.main()` (and `main()` takes no `argv`), so it
  was not unit-testable. Lifted `detect_crop_stream(cats, detector, margin,
  misses, corrupt)` to module level (body verbatim, loops over `cats`; `main()`
  now calls it). Behavior identical — confirmed by the full suite.
- **Builder** (`tests/test_build_catbreeds.py`): `list_cat_images` (folder→breed,
  sorted, bare unique filename, non-dir entries ignored), `sample_per_breed` (cap
  respected, 0/negative = unlimited, deterministic per seed, re-sorted, seed
  changes the pick), `write_catalog`/`write_metadata` shape (breed column, index
  alignment), and the **corrupt-image skip** through `detect_crop_stream`
  (detector=None; a broken `b"not an image"` `.jpg` → counted in `corrupt[0]`,
  not yielded) plus a detector-miss case (stubbed `detect_and_crop` → `[]` →
  `misses[0]`). All synthetic in `tmp_path`; no network. `ensure_dataset`
  untested (thin kagglehub wrapper).
- **Eval** (`tests/test_evaluate_catbreeds.py`): reuses `test_evaluate.py`'s
  fixture style (tiny indy + **catbreeds** variant caches + sidecars + manifest +
  artifact). Covers: happy-path `run_evaluation` (recall + all-rows negatives +
  per-breed/NFC rates), HTML + scores-CSV outputs, the **variant-mismatch**
  assertion (`"different footing"`), the **same-experiment** guard (`"different
  experiments"`), empty-catbreeds-cache loud (`"is empty"`), empty-Indy-test
  loud, plus `main()` end-to-end HTML + missing-manifest `SystemExit`. Also the
  focused `metrics` test: `build_sweep` with a custom `lookalike_breeds`
  partitions correctly, and the default still uses Oxford's `LOOKALIKE_BREEDS`.
- **HTML** smoke test — **DONE** in Task A
  (`tests/test_evaluate_catbreeds_report_html.py`): NFC row, look-alike breeds, no
  drift table, the FP cap + note, the nested resolver.
- Verified: `uv run pytest` (201 passed), `uv run ruff check`, `uv run ruff
  format`, `uv run mypy` all clean.

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
