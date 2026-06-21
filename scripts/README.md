# scripts/

Keeper entry points — reusable, maintained scripts that drive the core
`indycat` package: building the Indy gallery, embedding negative datasets,
running calibration and evaluation, and any UI launchers.

These import `indycat`; the core never imports from here. Run them inside the
managed environment, e.g.:

```powershell
uv run python scripts/build_indy_gallery.py
```

## `data_review/` — Streamlit data-review app

A screen-reader-friendly review of the project data: the `images/indy/mapping.csv`
labels (a heading-led "Rows" view and a "Data grid" view) and the detect-and-crop
results ("Crop review"). Split into focused modules (`mapping.py`, `crops.py`,
shared helpers in `common.py`) behind a thin entry point. Run:

```powershell
uv run streamlit run scripts/data_review/data_review.py
```

## `predict_app/` — Streamlit "Is it Indy?" app

The deliverable UI: upload a photo and get a text-first verdict — is this the cat
Indy, or not — with the detection confidence, crop, closest gallery match, and the
score-vs-threshold margin. The recognition logic lives in the streamlit-free
`predict.py` (`classify` composes detect→crop→embed→score and returns plain data);
`app.py` is the thin Streamlit layer. The live gallery and threshold come from a
frozen calibration artifact in `data/artifacts/` — produce one with
`scripts/calibrate.py` first. Run:

```powershell
uv run streamlit run scripts/predict_app/app.py
```
