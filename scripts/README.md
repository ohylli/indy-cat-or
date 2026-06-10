# scripts/

Keeper entry points — reusable, maintained scripts that drive the core
`indycat` package: building the Indy gallery, embedding negative datasets,
running calibration and evaluation, and any UI launchers.

These import `indycat`; the core never imports from here. Run them inside the
managed environment, e.g.:

```powershell
uv run python scripts/build_gallery.py
```
