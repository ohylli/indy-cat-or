# scripts/

Keeper entry points — reusable, maintained scripts that drive the core
`indycat` package: building the Indy gallery, embedding negative datasets,
running calibration and evaluation, and any UI launchers.

These import `indycat`; the core never imports from here. Run them inside the
managed environment, e.g.:

```powershell
uv run python scripts/build_gallery.py
```

## `data_review/` — Streamlit data-review app

A screen-reader-friendly review of the project data: the `images/indy/mapping.csv`
labels (a heading-led "Rows" view and a "Data grid" view) and the detect-and-crop
results ("Crop review"). Split into focused modules (`mapping.py`, `crops.py`,
shared helpers in `common.py`) behind a thin entry point. Run:

```powershell
uv run streamlit run scripts/data_review/data_review.py
```
