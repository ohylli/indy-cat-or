# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Early implementation. The first two pipeline stages exist:

- **Detect & crop** — `src/indycat/detection.py`: `CatDetector` takes an opened PIL image (callers own I/O), and `detect_and_crop` composes detect+crop into the shared pipeline prefix. `scripts/detect_indy_gallery.py` drives it over the Indy photos as a textual sanity-check tool; `tests/test_detection.py` covers it. Measured on the 35 photos, yolo11n missed 8 cats and yolo11m 2, so **yolo11x is the default detector**.
- **Embed** — `src/indycat/embedding.py`: `Embedder` wraps a frozen DINOv2 (base) via HF `transformers`; `embed`/`embed_batch` return **raw, un-normalized** float32 vectors (normalization is deferred to decide). `tests/test_embedding.py` covers it. `scripts/build_indy_gallery.py` is the real gallery producer — it detects on the fly (does **not** depend on the sanity-check script) and writes `data/embeddings/indy/embeddings.npy` row-aligned with `metadata.csv`. Shared script helpers (`load_image`, `iter_images`) live in `scripts/_common.py`.

torch is installed with CUDA from the pytorch-cu128 index (PyPI's Windows wheel is CPU-only — keep the `tool.uv` index pinning); GPU verified working. The **decide stage is not yet written**; its module layout is still open.

## Tooling & commands

Managed with **uv**. The `indycat` package is installed editable, so `import indycat` works from any script without path hacks. `uv run` auto-syncs the venv from `uv.lock` first — there is no manual venv activation.

```powershell
uv add <pkg>                  # add a runtime dependency (deps start empty; add as needed)
uv add --dev <pkg>            # add a dev-only tool (e.g. pytest, ruff)
uv run python scripts/foo.py  # run a script inside the managed venv
uv run pytest                 # run tests
uv sync                       # sync venv to the lockfile explicitly
```

Lint/format with **ruff** and type-check with **mypy** (`uv run ruff check`, `uv run ruff format`, `uv run mypy`); run them on code you write. Type annotations are expected. Config lives in `pyproject.toml`.

## File structure

- `src/indycat/` — **the core**: the importable, UI-agnostic detect→crop→embed→decide pipeline. Currently `detection.py` and `embedding.py` (decide module still to come). The core never imports from `scripts/` or `experiments/`.
- `scripts/` — **keepers**: reusable, maintained entry points that drive the core (build gallery, embed datasets, calibrate, evaluate, UI launchers).
- `experiments/` — **throwaway**: exploratory one-offs, the "each data addition is a measured experiment" scratch space. Committed (shared dependency set); promote anything worth keeping into `scripts/`.
- `tests/` — pytest suite.
- `data/` — datasets and embedding caches; **contents gitignored** (large, reproducible). `data/embeddings/` holds cached vectors.
- `images/indy/` — the 35 Indy photos (tracked) plus `mapping.csv`.
- `project_handoff.md` — the authoritative design brief.

`project_handoff.md` is the authoritative design document. Read it before making architectural decisions. Its decisions are described as a "considered starting point, not fixed requirements" — several are explicitly flagged as candidates for revision once real results come in (see its "Things most likely to change" section). Treat them as defaults to justify deviating from, not constraints.

## What this project is

A binary classifier that answers one question: **is the specific cat "Indy" in this image — or not** (the name is a pun on "indicator"). This is fine-grained recognition of a single individual, closer to face recognition than to ordinary image classification. "Is there a cat" is the easy part; "is this *this particular* cat, versus a similar-looking cat of the same general type" is the actual problem.

## Architecture

The core is a four-stage pipeline: **image → detect → crop → embed → decide**.

1. **Detect & crop** — a COCO-pretrained YOLO-family detector finds the cat ("cat" is a built-in COCO class, no training) and the image is cropped to the cat region plus a small margin, before embedding. This exists to stop the system latching onto backgrounds given so few Indy photos. Must be **toggleable** so its effect on accuracy can be measured rather than assumed. Handle two edge cases explicitly: **no cat detected** (report it, do not silently embed the full frame) and **multiple cats detected** (embed each crop separately — "is Indy in this photo" becomes "is any crop Indy").
2. **Embed** — a frozen, pretrained **DINOv2** vision transformer (base size as the default starting point) used in **inference only — never trained**. It turns each crop into an embedding vector. The backbone is swappable (smaller/larger variant, or DINOv3) without changing the rest of the approach.
3. **Decide** — **verification by similarity threshold**: compute cosine similarity of the new image's embedding against the **Indy gallery only**, take the best match (or mean of top few), compare to a fixed cutoff. That is the entire live decision. The negatives never participate in it — their role is *calibration and evaluation* (choosing the threshold where the Indy and non-Indy score distributions separate). This was deliberately chosen over kNN voting because of class imbalance (~15 Indy refs vs ~2000 negatives would swamp the vote). The escalation path if the threshold is insufficient is a **linear probe** on the frozen embeddings — adopt it based on measured results, not assumption.

Precompute and store Indy's gallery embeddings as vectors; do not recompute from his images at runtime (keeps any deployed version lean and avoids shipping his photos to a server).

## Non-negotiable design principles

- **Core recognition logic must be separate from any UI.** The detect-crop-embed-decide pipeline lives in plain, importable Python modules that know nothing about how they are presented. Any UI (CLI, Streamlit, or a hand-built Flask/FastAPI app) is a thin, disposable layer that calls into that core. This is what makes the UI choice low-risk and reversible.

- **The developer relies on a screen reader.** This shapes real decisions, not just etiquette:
  - Prefer **textual, inspectable outputs** over anything requiring visual inspection. The detector's output (count of cats + per-box confidence) lets gallery crops be sanity-checked without looking at them — lean on this pattern generally.
  - Prefer **pre-labeled datasets organized by breed**, because a breed label is a text check on folder/label names. Avoid raw image-search sources as a primary dataset.
  - For UI, accessible semantic HTML with live regions is the gold standard; Streamlit has known accessibility limits and is to be evaluated in practice, with a hand-built accessible web UI as the fallback.

## Data rules (these prevent silently-wrong accuracy numbers)

- **Positives:** 35 Indy photos on hand in `images/indy/`, renamed to descriptive `<position>_<location>_<view>_<NN>.<ext>` names (the `NN` counter is a stable per-photo handle). `images/indy/mapping.csv` is the source of truth: it records each photo's original filename plus inspectable `position,location,view,head_visible,tail_visible,notes` fields — use `head_visible`/`tail_visible` to pick gallery photos without looking at the images. Plan ≈15 for the gallery, ≈20 held out for testing. Variety (angle, distance, lighting, background) matters more than count; photos clearly showing his **head markings and tail** carry the most identifying information.
- **Negatives:** start from the **Oxford-IIIT Pet dataset** (use *all* its cat images; under the threshold decision their count is free at prediction time). Its long-haired breeds (Maine Coon, Ragdoll, Birman) are built-in hard negatives. **Run the same YOLO detector over Oxford rather than reusing its shipped bounding boxes** — those are head-only ROI boxes, which would mismatch Indy's full-body crops and bias the threshold; re-detecting keeps positives and negatives on identical footing (the gallery builder already detects on the fly, so the negatives script reuses the same path). Optionally add a Norwegian Forest Cat set later — but **community dataset breed labels are unreliable** (NFC/Maine Coon/Siberian confusion is rampant), so describe such a slice honestly as "look-alike long-haired cats," not guaranteed NFCs.
- **Split discipline:**
  - **Disjoint at the image level.** The same photograph must never be used both to set up the system (gallery, calibration) and to test it. Breed-level overlap is fine and desirable (Maine Coons on both sides), the same *photo* is not. Split once up front (e.g. 70% calibration / 30% locked test) and never move anything across.
  - **Indy's 35 photos are hand-curated and contain no burst shots, so no per-session split is needed.** The general occasion-based rule (keep all near-duplicate shots from one session on the same side) does not apply to this set — image-level disjointness above is enough. Revisit this only if a future batch of Indy photos includes bursts from a single session.
  - Fix a **held-out test set up front and never touch it during setup.** It must include a slice of **look-alike long-haired cats that never appear in the gallery or calibration negatives** — that slice is the real exam.

## Evaluation

Work **incrementally — treat each data addition as an experiment whose effect is measured.** Rough staging: (1) Indy + generic Oxford negatives baseline → (2) add Oxford long-haired breeds → (3) add a dedicated NFC set → (4) extend only if results call for it.

Primary metric: **false-positive rate on look-alikes** (non-Indy cats wrongly called Indy, over all non-Indy cats) — expected to fall as same-breed negatives are added. Track it alongside **recall on Indy** to confirm fewer false positives aren't bought by missing Indy himself.

## Environment

Windows 10, NVIDIA RTX 3070 (8 GB VRAM). Inference-only on the heavy model, so 8 GB is comfortable. The GPU speeds up batch-embedding during experiments but nothing strictly requires it; a free cloud GPU is a viable fallback for the one-time dataset-embedding step. Shell is PowerShell (Bash also available). If hosting is later wanted, the binding resource is **memory (~1–2 GB), not compute** — Hugging Face Spaces is the leading free-tier candidate.
