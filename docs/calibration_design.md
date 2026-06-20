# Decide stage & calibration — design

A design document for the **decide** stage of the pipeline (`image → detect → crop → embed → decide`) and, in particular, for the **calibration** tool that sets its threshold. It captures the decisions made in a design session and is the basis for implementation in later sessions.

It is intentionally **high-level — mostly "what", not "how"**. It records the methodology and the shape of the tools, not code. Several points are explicit candidates for refinement once real results come in; those are collected at the end. The authoritative project brief remains [`../project_handoff.md`](../project_handoff.md); this document elaborates its "Decision step" and "Evaluation" sections without replacing them.

## Status

What exists so far against this design:

- **Code layout.** The calibration code is a package, `scripts/calibration/`: `manifest.py` (split generation/load), `scoring.py` (role scoring over `indycat.decision`), `metrics.py` (pure measurement — distributions, sweep, risk lists, shared by both renderers and the future `evaluate.py`), `report_text.py` / `report_html.py` (the two renderings), and `cli.py` (the command). `scripts/calibrate.py` is a thin shim re-exporting `calibration.cli.main`, so `uv run python scripts/calibrate.py` is unchanged. `scripts/_common.py` (shared with the gallery builders) stays at the top level.
- **Split manifest generator — done.** `scripts/calibration/manifest.py` is the reusable unit (§3, §6): it loads the embedded Indy/Oxford `metadata.csv` (joining Indy to `mapping.csv` for the `prefer` flags), generates a `three_way` split, and writes/loads a YAML manifest. The three guarantees are implemented and tested: **test split drawn first** (invariant to gallery/calibration counts), **counts validated against rows actually embedded** (over-ask is a hard error), and **materialized lists used verbatim at load** (disjointness, format version, and breed-summary consistency asserted loudly). Oxford selection is breed-stratified; the `prefer` knob is plumbed but off for the baseline.
- **Scoring core — done (V0).** `src/indycat/decision.py` is the UI-agnostic decide API (§1, §5): `l2_normalize`, a `Gallery` that L2-normalizes the raw stored vectors on construction, `aggregate` (`max` default / `mean-top3`), and `score`/`score_many` returning a `Match` (aggregated score + the single best-match gallery photo, which is `argmax` regardless of aggregation). Pure numpy, no I/O. Covered by `tests/test_decision.py`. The embeddings-cache reader `load_cached_embeddings` (the I/O inverse of the metadata writer, loud on row-count drift) lives in `scripts/_common.py` for reuse by `evaluate.py`.
- **Calibration report — done (V0).** `scripts/calibration/metrics.py` (measurement) and `scripts/calibration/report_text.py` (rendering) produce the textual report of §4, **V0 subset — distributions only, no threshold**: the positive/negative distribution table (`n mean min p50 p95 max`), the lowest-positive-vs-highest-negative overlap line, a **look-alike-vs-easy** breakdown (look-alike = the four long-haired breeds **Maine_Coon, Ragdoll, Birman, Persian**), per-breed negative scores sorted by max, and the highest-scoring-negatives / lowest-scoring-positives risk lists. `--scores-out` writes the per-image CSV joined with provenance. An **optional HTML rendering** (`scripts/calibration/report_html.py`: `render_report_html`/`write_report_html`) emits the same report as a self-contained semantic-HTML document — headings, lists, and tables for screen-reader navigation — that additionally **embeds the actual photos** the risk lists name (each candidate beside the gallery photo it best matched) plus the full gallery, via relative `images/…` paths with the filename as both `alt` text and a visible caption. Covered by `tests/test_calibration_report.py`. Measured on the baseline split, the distributions separate clearly (Indy positives mean ≈0.83 vs Oxford negatives ≈0.26) with the long-haired breeds forming the high negative tail and a real overlap (a handful of Maine Coon / Ragdoll / Persian crops scoring above the weakest positive) — exactly the signal V0 exists to surface.
- **Threshold sweep — done (V1).** `scripts/calibration/metrics.py` adds the trade-off curve of §5's V1 on top of the V0 distributions (rendered by `report_text.py`/`report_html.py`): a `SweepRow`, `sweep_thresholds` (a **round, data-ranged** grid of cutoffs — multiples of `--sweep-step`, default `0.05`, spanning just below the min to just above the max observed score so the grid brackets the full trade-off), `build_sweep` (per-cutoff FPR over all / look-alike / easy negatives plus recall-on-Indy, `score >= cutoff` convention, empty group → NaN → dash), and `build_breed_sweep` (per-breed FPR per cutoff, breeds worst-first). The textual report gains a **main sweep table** (`cutoff / FPR(all) / FPR(look) / FPR(easy) / recall`) and a **separate per-breed FPR table**; both are mirrored into the HTML report. **No cutoff is chosen** — V1 only makes the trade-off visible. Covered by `tests/test_calibration_report.py`.
- **Automated threshold pick — done (V2).** `scripts/calibration/metrics.py` adds the policy layer of §5's V2: `PickPolicy` (`target-fpr` / `youdens-j` / `equal-error`), `TargetGroup` (`overall` / `look-alike`), `candidate_cutoffs` (a **fine** grid — midpoints between adjacent distinct observed scores plus bracketing endpoints, so the chosen threshold is precise and the `>=` convention stays unambiguous, unlike V1's round display grid), and `pick_threshold`, which scores that grid via `build_sweep` and selects one `SweepRow`: `target-fpr` takes the lowest cutoff (max recall) whose FPR over the chosen group is ≤ the budget; `youdens-j` maximises `recall − FPR(all)`; `equal-error` minimises `|FPR(all) − (1 − recall)|` (ties break toward the higher cutoff = fewer false positives). It returns a `ThresholdChoice` (the chosen row + a human-readable rationale) and raises loudly on empty positives/negatives or a look-alike target with no look-alike negatives — never a NaN-driven silent pick. Both renderers gain an optional `choice` param that adds a **"Chosen threshold"** section after the sweep tables; without `--policy` the output is the unchanged V1 report. The CLI adds `--policy`, `--target-fpr` (default `0.05`), and `--target-fpr-group` (default `look-alike`), all composing with `--manifest`. Covered by `tests/test_calibration_report.py` and `tests/test_calibrate.py`. Measured on the baseline split, `--policy target-fpr` picks cutoff ≈0.72 (FPR(look-alike) ≈0.048, recall 1.0) and the parameter-free policies land at the same ≈0.72 knee — the distributions separate cleanly enough that all three agree. **V2 only reports the chosen threshold; it does not freeze an artifact** (that is V3).
- **CLI — done for V0+V1+V2.** `scripts/calibration/cli.py` is the command (entry point `scripts/calibrate.py`, generation folded in, per §4): generation flags, a `--seed`/`--random-seed` group, `--manifest` replay (mutually exclusive with generation flags), `--generate-only`, plus the scoring options `--aggregation {max,mean-top3}`, `--scores-out`, `--sweep-step` (the V1 cutoff granularity), and the V2 pick options `--policy {target-fpr,youdens-j,equal-error}`, `--target-fpr`, `--target-fpr-group {overall,look-alike}` (all composing with `--manifest`). An optional `--html` flag also writes the HTML report (bare flag auto-names into `data/reports/` like the manifest auto-names into `data/splits/`; an explicit path is also accepted) while still printing the text report. Manifests are written under `data/splits/` (gitignored). A normal run generates + saves the manifest **and** scores it, printing the V0 distributions + V1 sweep (+ the V2 chosen threshold when `--policy` is given); the `--manifest` replay path scores too; `--generate-only` stops after writing the manifest. `test` is never read. Covered by `tests/test_calibrate.py`.
- **Not yet started:** §5's V3 (the frozen calibration artifact + `max`-vs-`mean-top3` comparison) and the separate `evaluate.py`. `leave_one_out` is unimplemented (the `strategy` string is stored; only `three_way` is generated).

## 1. The decision (the live path)

The live decision is unchanged from the handoff: **verification by similarity threshold against the Indy gallery only.** A new image is detected, cropped, and embedded; its embedding is compared to the Indy gallery embeddings; the best aggregated similarity is compared to a fixed threshold. Above means Indy, below means not. The negatives never participate in the live decision — their entire role is calibration and evaluation.

### Scoring function

- **Normalize at decide time.** Stored gallery vectors are raw (un-normalized) by design — that keeps the linear-probe escalation path open. The decide stage L2-normalizes both the query embedding and the gallery vectors, so cosine similarity is a dot product.
- **Aggregation: `max` is the default.** The score of a query against the gallery is the single best (maximum) cosine similarity to any gallery vector. This suits Indy's pose variety and the fact that his identity is concentrated in head and tail: a query showing one aspect should match the gallery photos that share it, not be diluted by the ones that don't. `mean-top3` (mean of the top three matches) is the measured alternative, compared during calibration — not chosen up front.
- **Multiple crops in one photo.** When the detector returns more than one cat, the photo's score is the `max` over its crops ("is Indy in this photo" = "is any crop Indy"). This is a *live-decode* concern only: the stored galleries hold one vector per image, so calibration operates per stored embedding.

The decide output is a textual result — score, threshold, verdict, and **which gallery photo was the best match** (plus that photo's `position`/`view` from `mapping.csv`). This makes a verdict inspectable without looking at the image, consistent with the screen-reader-first principle used by the detector.

## 2. Calibration philosophy

**Calibration is a measurement tool, not a number-picker.** The threshold is a judgement call that cannot be made well before the two score distributions — Indy positives vs. non-Indy negatives — have actually been seen. So the early job of calibration is to *show where the distributions separate*, not to emit a cutoff. Automated threshold-picking is added deliberately late, once the shape of the data (and the size of the overlap) is understood, because even *which* picking policy is appropriate depends on what the distributions look like and on the project's false-positive-first priority.

### What calibration measures

- **Negative scores** — every calibration-split Oxford cat scored against the Indy gallery. These should cluster low; the long-haired look-alike breeds form the high tail that determines where the threshold must sit.
- **Positive scores** — held-back Indy photos (the `calibration` role, **disjoint from the gallery**, so there is no self-match) scored against the gallery. These should cluster high.

The threshold lives in the gap between them. The size and contents of any overlap are the primary thing calibration exists to reveal.

### Metrics

- **Primary: false-positive rate on look-alikes** — non-Indy cats wrongly scored as Indy, over all non-Indy cats, with the look-alikes (long-haired breeds) called out specifically. **Report this honestly:** under the breed-stratified split the look-alike *breeds* appear on both sides (image-disjoint, breed-overlapping — see §3), so this number measures FPR on *look-alike breeds that were also seen during calibration*. It is **not** the unseen-breed generalization test the handoff's "real exam" language implies — that role belongs to the future, fully held-out NFC slice (§7). The report (and later `evaluate.py`) labels the metric accordingly so it is never read as stronger than it is.
- **Tracked alongside: recall on Indy** — to confirm that fewer false positives are not bought by missing Indy himself.
- **Breakdown:** the report breaks negatives down **by breed group (look-alike vs. easy) and per individual breed**. Per-breed numbers show exactly which breeds drive the false-positive risk (and confirm or refute the "Maine Coon / Ragdoll / Birman are the hard ones" hypothesis), rather than hiding it inside an aggregate.

## 3. The split manifest

A **manifest** defines one reproducible experiment: which images play which role. It is the unit of an experiment.

### Single file (Option 1)

Each experiment is a single self-contained YAML listing all roles. The risk of one-file manifests — that regenerating for a new experiment silently moves the test set — is handled **in the generator**, not by convention: see "test split first" below.

### Strategy

The manifest names a `strategy`:

- **`three_way`** (now) — the setup pool is carved into `gallery` and `calibration`.
- **`leave_one_out`** (later) — there is no fixed `calibration` role; the calibrate step rotates over a single pool, scoring each Indy photo against a gallery built from the others. This is why strategy lives in the manifest rather than being hard-coded in the tool.

### Generate dynamically, materialize the result

- **Generation is dynamic** — `seed` + counts/percentages + stratify-by-breed logic decide membership. Breed balancing lives in the *code*, not in stored config.
- **Counts are validated against what is actually embedded, and over-asking fails loudly.** Requested role counts are checked against the rows present in `metadata.csv` (not against an assumed 35 Indy / full Oxford), and an impossible request is a hard error, never a silently-truncated split. This matters because the zero-arg baseline (`15 + 10 + 10`) consumes *every* Indy photo with no margin: if a photo ever fails to embed or is dropped, the default must refuse to run rather than quietly produce a smaller split — consistent with the project's no-silently-wrong-numbers stance.
- **The artifact is materialized** — a generated run writes out the *resolved* filename lists (by `source_filename`) for both Indy and Oxford, test and setup. The generation parameters are also recorded in the header for audit, but the body holds the actual frozen membership.

So the seed is *how a split is created*; the materialized list is *what is saved*. Loading a manifest uses the frozen lists **verbatim — it never recomputes.**

Pure-dynamic re-derivation (seed + percentages only, no stored list) is deliberately **not** used: it makes reproducibility hostage to `metadata.csv` staying byte-identical, so a re-embed or a change in the Oxford miss count would silently yield a different split — including a drifting test exam. Materializing the result is cheap insurance against exactly the "silently-wrong numbers" failure the project guards against, and it costs nothing in usability because the dynamic generation path is still the entry point. It also makes splits **diffable** (did the test set move? — a set-intersection check) and **self-contained** (no hidden dependency on current `metadata.csv`).

### Test split first

The generator computes the **test set first**, as a pure function of `(image set, seed, test-count)` only — never touching gallery/calibration counts. This gives the invariant that keeps strategy/size comparisons valid:

> Hold `--seed` and `--test` fixed and vary `--gallery`/`--calibration` freely → identical exam. Change `--seed` → a new exam, drawn knowingly.

When a *guaranteed* fixed exam is wanted (e.g. across many experiments), the written manifest is reused directly rather than regenerated.

### What the manifest contains

- **Header / provenance:** strategy, seed, generation parameters (counts/percentages, breed filter), and a **per-breed count summary per role** — so the breed balance is inspectable at a glance without reading the full lists (screen-reader-friendly).
- **Body:** the materialized lists. Indy — `gallery` / `calibration` / `test` (small, meaningful names, hand-pickable). Oxford — `test` / `setup`, referenced by `source_filename`.
- **Not in the manifest:** the scoring aggregation (`max` vs `mean-top3`) and the threshold. Those describe the calibration *result*, not the data split, so the same manifest can be re-run under different aggregations. (See §5.)

The manifest references only images that are actually embedded (rows in `metadata.csv`), not the catalog — Oxford's no-cat misses have no vector to score. Disjointness of the role sets is **asserted mechanically at load**; a non-empty intersection is a loud failure.

### Indy vs. Oxford selection

- **Indy** is always materialized (35 photos, meaningful names). The automated selector is random-but-seeded; an optional `prefer` knob can bias the gallery toward `head_visible` / `tail_visible` photos (text fields in `mapping.csv`), since those carry the most identifying information. **`prefer` is off for the baseline.** Because the test split is drawn first (§"Test split first"), skimming the head/tail-visible photos into the gallery leaves the *calibration positives* with the weaker photos — distorting the very positive distribution V0 exists to measure. So biased-gallery is strictly a later, labelled experiment (§7); the default run keeps selection unbiased. A **manual** path — hand-edit the YAML, optionally aided by the existing `data_review` app surfacing `mapping.csv` attributes — produces a manifest in the same format. Hand-picked-vs-random gallery is itself a measurable experiment.
- **Oxford** is selected automatically, stratified by breed so each role gets a representative breed mix and the look-alike tail is never lopsided. Manual Oxford picking is deferred — Oxford's breed labels are trustworthy, so "hard look-alikes" = the long-haired breeds, fully automatable; manual selection only becomes relevant for a future community NFC set with unreliable labels.

## 4. The calibration tool — outside view

Generation is **folded into the calibrator**: one command, not a two-step dance.

### Invocation

```
# zero-arg: built-in defaults + built-in seed -> identical every time
calibrate

# specify the split
calibrate --gallery 15 --calibration 10 --test 10 --seed 42

# fresh random seed (the drawn seed is recorded in the written manifest)
calibrate --random-seed

# replay an exact prior split
calibrate --manifest splits/run-<...>.yaml
```

- `--manifest` and the generation flags are **mutually exclusive**: a manifest means "use exactly this, ignore generation."
- A **generated** run writes its resolved manifest to disk (path printed); a **loaded** run does not rewrite.
- `--aggregation max | mean-top3` selects the scoring choice (default `max`).
- `--scores-out <csv>` optionally writes per-image scores for inspection.
- `--random-seed` still **records the seed it drew** into the written manifest, so even an exploratory run is reproducible afterward.

### The default run

The zero-arg run is the **stage-1 baseline**: `gallery 15 / calibration 10 / test 10` Indy; **all** Oxford breeds at 70/30 stratified by breed; `aggregation = max`; built-in seed. So `calibrate` with no arguments means "the baseline experiment," bit-for-bit repeatable.

### Reproducibility, precisely

The seed reproduces *intent*; the materialized manifest reproduces *exact images*. These diverge if `metadata.csv` changes underneath a seed-only run. For anything that must stay locked over the life of the project, **the written manifest is the source of truth**; the seed is the day-to-day convenience.

### Output (textual, screen-reader-first)

A textual report to stdout — illustrative shape:

```
Calibration: <manifest>   (aggregation=max)
  Gallery:    15 Indy photos
  Positives:  10 Indy photos (calibration)
  Negatives:  N Oxford cats, 12 breeds

Score distribution (cosine to best gallery match):
                n      mean   min    p50    p95    max
  Indy (pos)    ...
  Oxford (neg)  ...

  Lowest positive ...  vs  highest negative ...  ->  OVERLAP ...

FPR by breed group / breed:   look-alike vs easy, then per breed
Highest-scoring negatives (false-positive risks):  score, name, breed
Lowest-scoring positives (recognition risks):      score, name
```

The per-image `--scores-out` CSV joins each score with `metadata.csv` provenance (filename, breed, Indy position/view) so the worst false-positive risks and hardest-to-recognize positives are directly inspectable.

## 5. Incremental staging

Built in stages, each validating before adding features:

- **V0 — distributions only.** Builds the core scoring (`src/indycat/decision.py`: normalize + aggregate against gallery) and the calibrate driver that emits the report above. **No threshold is chosen.** Answers the first question worth answering: *do the distributions separate at all?* If they do not, that is learned immediately, before any threshold machinery is built on a foundation that does not hold.
- **V1 — trade-off curve.** A threshold sweep: a table of `cutoff → FPR (overall, look-alike, per breed) , recall-on-Indy`. The trade-off becomes visible; still human-read.
- **V2 — automated pick by explicit policy.** Given the now-understood shape, a `--policy` flag (e.g. target-FPR / Youden's J / equal-error) emits a chosen threshold. This is where the judgement is encoded — deliberately last.
- **V3 — freeze it.** Write the calibration artifact that the decide stage consumes (contents specified below), and add aggregation comparison (`max` vs `mean-top3`) in one run.

The scoring code lives in the core (`decision.py`); calibrate is a thin driver over it. Building V0 therefore also delivers and validates the core decide API.

### V3 — the calibration artifact

The artifact is the frozen output of calibration and the input to both the live decide stage and `evaluate.py`. It has **two jobs, kept visibly separate in the file**: the *operative* fields decide actually reads to produce a verdict, and the *provenance + curve* that make a frozen number auditable rather than a bare magic constant. Writing it is essentially serializing V2's `ThresholdChoice` (chosen `SweepRow` + rationale) plus the sweep and a binding to the gallery it was calibrated against.

**Bundled, not referenced.** The artifact ships the gallery vectors with it, so a deployed decide stage needs nothing else — consistent with the handoff's "store Indy's gallery as vectors, keep any deployed version lean, don't ship his photos." Concretely a **pair**: a human-readable `calibration.yaml` (operative fields + provenance + curve + gallery manifest) and a companion `gallery.npy` holding the raw, un-normalized gallery vectors row-aligned to that manifest (raw so `decision.Gallery` keeps L2-normalizing on construction; floats are not base64'd into the YAML). A `fingerprint` over the vectors is kept in the YAML regardless, so an accidental vector/threshold mismatch is a loud failure, not a silently-wrong verdict.

**Operative (read by decide):**

- **`threshold`** — the cutoff, at full precision off V2's fine `candidate_cutoffs` grid (not a rounded display cutoff).
- **`aggregation`** — `max` | `mean-top3`. The threshold is only meaningful under the aggregation that produced it, so decide must score with the same one. This is why aggregation is deliberately *out* of the manifest (§3) but *in* here.
- **`comparison`** — recorded explicitly as `">="` (`score >= threshold` → Indy). Fixed in code, but a frozen artifact should not leave the convention implicit.
- **`gallery`** — the binding that makes the threshold valid only for *this* gallery: the companion `vectors` file, a `fingerprint` (loud drift check), `count`, and per-row `source_filename` + `position`/`view` so decide can still name the best-matching gallery photo (§1's inspectable output) without `mapping.csv`.

**Provenance + curve (audit, not consumed at decide time):**

- **`chosen_by`** — `manifest` path, `seed`, `policy`, and its params (`target_fpr`, `target_fpr_group`): V2's rationale, serialized.
- **`metrics_at_threshold`** — the chosen `SweepRow` (`fpr_all`, `fpr_look_alike`, `fpr_easy`, `recall_indy`, `n_pos`, `n_neg`): what the number buys.
- **`aggregation_comparison`** — V3's `max`-vs-`mean-top3` run: each candidate's threshold + headline metrics + the `winner`, justifying the `aggregation` field above.
- **`sweep`** — the full V1 curve, so the frozen threshold reads as a point on a visible trade-off.
- **`format_version`** — asserted loudly at load, like the manifest.

Illustrative `calibration.yaml`:

```yaml
format_version: 1

# operative — read by decide
threshold: 0.7213
aggregation: max
comparison: ">="
gallery:
  vectors: gallery.npy            # raw vectors, row-aligned to images below
  fingerprint: sha256:...         # loud drift check
  count: 15
  images:
    - {source_filename: ..., position: ..., view: ...}

# provenance — audit, not consumed
chosen_by: {manifest: data/splits/run-....yaml, seed: 12345,
            policy: target-fpr, target_fpr: 0.05, target_fpr_group: look-alike}
metrics_at_threshold: {fpr_all: 0.012, fpr_look_alike: 0.048, fpr_easy: 0.0,
                       recall_indy: 1.0, n_pos: 10, n_neg: 712}
aggregation_comparison:
  max:       {threshold: 0.72, fpr_look_alike: 0.048, recall_indy: 1.0}
  mean-top3: {threshold: 0.66, fpr_look_alike: 0.061, recall_indy: 1.0}
  winner: max

# the curve — trade-off context
sweep:
  - {cutoff: 0.50, fpr_all: ..., fpr_look_alike: ..., fpr_easy: ..., recall_indy: ...}
```

**Not in the artifact:** any `test`-role data or test numbers — those are `evaluate.py`'s output (§6 boundary). The artifact is calibrate's frozen *input* to evaluate; it never carries the exam. Written under `data/artifacts/` (gitignored, like manifests under `data/splits/`); YAML keeps it diffable and screen-reader-navigable. `evaluate.py` reads it back and asserts `format_version` and `fingerprint` before scoring the test set.

## 6. Boundaries & discipline

- **Calibrate never touches `test`.** A separate, later `evaluate.py` reads the `test` role plus a frozen calibration artifact and reports the honest numbers at the frozen threshold. Calibration uses only `gallery` and `calibration` roles.
- **Split discipline** (from the handoff, restated): disjoint at the *image* level (breed-level overlap is fine and desirable); the test exam fixed up front and never used during setup; the test exam must include the look-alike slice.
- **Core vs. driver:** the scoring/decision logic is UI-agnostic core in `src/indycat/`; the calibrate and (future) evaluate tools and the split generator live in `scripts/`, with the generator factored as a reusable unit (not tangled into calibrate's measurement job) so a standalone generate command or `evaluate.py` can reuse it.

## 7. Open / to revisit

Treated as adjustable, decided by measured results rather than assumption:

- **Aggregation winner** — `max` vs `mean-top3`, chosen from measured separation.
- **Threshold-picking policy** — target-FPR vs balanced vs equal-error, chosen once the distribution shape is known (V2).
- **Leave-one-out strategy** — second strategy to add after the three-way baseline, for comparison at fixed test exam.
- **Separate test seed** — a fully decoupled "exam can't move even if `--seed` changes" scheme; deferred, since materializing the manifest already locks the exam. Build only if the single-seed invariant proves error-prone.
- **Gallery selection** — random vs `head_visible`/`tail_visible`-biased vs hand-picked, as a measurable comparison.
- **Look-alike slice source** — currently held-out Oxford long-haired breeds; to be supplemented by a dedicated (label-caveated) Norwegian Forest Cat / look-alike set in a later data stage.
