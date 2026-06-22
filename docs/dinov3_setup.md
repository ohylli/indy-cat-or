# DINOv3 backbone — access, setup, and swap notes

A memo for when we want to A/B the embedder against DINOv3. The current default
is `facebook/dinov2-base` (Apache 2.0, ungated). DINOv3 is a *gated, custom-license*
upgrade, so getting it takes a few one-time browser + auth steps before the
one-line code swap.

Verified against the installed stack: `transformers` 5.12.1, `huggingface_hub`
1.19.0 (CLI is `hf`, not the old `huggingface-cli`). As of this writing we are
**not** logged in and no `HF_TOKEN` is set — so all the access steps below apply.

## Code compatibility (already verified — no code changes needed)

DINOv3 **ViT** models are drop-in compatible with `src/indycat/embedding.py`:

- `AutoModel` resolves `dinov3-vit*` repos to `DINOv3ViTModel`; `AutoImageProcessor`
  resolves the matching `DINOv3ViTImageProcessor`.
- `DINOv3ViTModel.forward` returns a `BaseModelOutputWithPooling` whose
  `pooler_output` is the CLS token after the final layernorm
  (`sequence_output[:, 0, :]`) — exactly what `Embedder.embed_batch` reads at
  `embedding.py:83`. `config.hidden_size` is present, so `embedding_dim` works.
- So the *only* code change is the `model=` argument.

Caveat: this is verified for the **ViT** line only. `DINOv3ConvNextModel` is a
different class whose output shape was **not** confirmed — do not assume the
ConvNeXt variants are drop-in. (Why ViT over ConvNeXt: DINO self-supervised
features are patch/attention-based and stronger at the fine-grained, part-based
"his head markings + tail vs. a look-alike" problem this project is.)

## Available DINOv3 ViT models

Pretrained on Meta's LVD-1689M. `embedding_dim` is what the gallery `.npy`
becomes (caches are model-specific and must be rebuilt on a swap).

| Model id | Size | `embedding_dim` |
|---|---|---|
| `facebook/dinov3-vits16-pretrain-lvd1689m`      | small        | 384  |
| `facebook/dinov3-vits16plus-pretrain-lvd1689m`  | small+       | 384  |
| `facebook/dinov3-vitb16-pretrain-lvd1689m`      | base         | 768  |
| `facebook/dinov3-vitl16-pretrain-lvd1689m`      | large        | 1024 |
| `facebook/dinov3-vith16plus-pretrain-lvd1689m`  | huge+        | 1280 |
| `facebook/dinov3-vit7b16-pretrain-lvd1689m`     | 7B (overkill) | large |

There are also satellite-imagery (`-sat-`) variants pretrained on different data —
**not** what we want for pet photos.

Recommended starting point: **`facebook/dinov3-vitb16-pretrain-lvd1689m`** (base,
768-dim) for an apples-to-apples comparison against `dinov2-base` — same dims,
isolates "v2 vs v3" from "bigger model." Add `large` if testing capacity too.

## Setup steps (one-time)

1. **HF account** — sign up / log in at https://huggingface.co (free, email-verified).

2. **Accept the license (the gate).** Visit the model page, e.g.
   https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m — while logged
   in, fill in the "agree to share your contact information" form and submit.
   Approval is normally instant. **Access is per-repo**: repeat for `large`,
   `small`, etc. if you want them. Accepting the form *is* accepting the contract.

3. **Create an access token** at https://huggingface.co/settings/tokens →
   **New token**, **Read** scope is enough. If you make a *fine-grained* token
   instead, enable **"Read access to contents of all public/gated repos you can
   access"** or gated downloads will 403. Copy the `hf_...` value (shown once).

4. **Authenticate this machine** (pick one):
   - Interactive (persists in the HF cache — recommended). Run it yourself in the
     session so you can paste the token: `! hf auth login`
   - Env var (per-shell, not persisted): `$env:HF_TOKEN = "hf_xxx"` in PowerShell.

5. **Verify access resolved:**
   ```
   hf auth whoami                                   # prints username, not "Not logged in"
   uv run python -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('facebook/dinov3-vitb16-pretrain-lvd1689m').hidden_size)"
   ```
   - Prints `768` → fully set; `Embedder(model='facebook/dinov3-vitb16-pretrain-lvd1689m')` will work.
   - `GatedRepoError` / 401 / 403 → token not seen or license not accepted (revisit 2–4).

## License — read before relying on it commercially

- **Not open source.** Custom **"DINOv3 License,"** not Apache/MIT. This is a
  regression from DINOv2 (Apache 2.0, ungated) — only justify it with measured
  accuracy gains, not convenience.
- Read the repo's LICENSE for: the attached **Acceptable Use Policy**;
  **derivative/distillation** clauses; **attribution/redistribution** terms; and
  any **scale/commercial threshold** (Llama-style MAU cap — unconfirmed for
  DINOv3, check it).
- **For this project specifically** it almost certainly doesn't bite: we ship
  *embedding vectors*, not the weights and not Indy's photos, and operate at
  hobby scale. The terms mainly constrain redistributing weights or massive-scale
  operation. Still — skim the actual LICENSE once.

## After access lands — the swap

1. Rebuild **both** galleries (Indy positives + Oxford negatives) with the new
   model — caches are model-specific.
2. Re-run calibration + evaluation and compare the headline metric
   (**false-positive rate on look-alikes**, alongside Indy recall) against the
   dinov2-base baseline. Let the numbers decide the backbone.
