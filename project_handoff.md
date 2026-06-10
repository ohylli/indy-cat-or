# indy-cat-or — Project Handoff

A high-level brief to onboard Claude Code. It describes the problem, the approach we have sketched on claude.ai, and the reasoning behind each choice. It intentionally contains no code or commands; those are to be worked out during implementation. Treat the decisions here as a considered starting point, not fixed requirements. Several are explicit candidates for revision once real results come in, and the final section lists the ones most likely to change.

## Project name

The project is called **indy-cat-or**. It is a pun on "indicator" (Indy / cat / or), and the pun is meaningful rather than decorative: the system is a binary classifier, so the "or" reflects the actual decision it makes — Indy, or not Indy. The name doubles as a description of the architecture.

## Goal

The project recognizes whether one specific cat, named Indy (short for Indiana), appears in a given image. This is **fine-grained recognition of a single individual**, closer in spirit to face recognition than to ordinary image classification. The task is not "is there a cat" but "is this *this particular* cat." That framing drives most of the choices below.

Indy belongs to a friend, so this is a personal hobby project. There is no production pressure; the priority is a working, understandable system that can be iterated on and demonstrated.

## The subject: Indy, and why his appearance matters

Indy is a long-haired Norwegian Forest Cat. His coat is predominantly white, with dark markings concentrated in a few places: a dark cap sitting over and between the ears that comes to a point down the forehead, a smaller dark patch near one eye, and a dramatic, fully fluffed dark tail set against an otherwise white body. He has green eyes, a pink nose, and tufted ("lynx") ear tips.

Two consequences follow from this, and they shape the data strategy:

First, his identity is **concentrated in high-contrast features** — the head markings and the tail — while much of his body is plain white fluff that carries little identifying information. Images that clearly show his head and tail are therefore more valuable than ones where those are hidden or turned away.

Second, his overall *pattern type* — a white-and-dark long-haired bicolor — is **not rare**. Many Norwegian Forest Cats and Maine Coons share roughly that look, including the bushy tail. So the genuinely hard part of the problem is not distinguishing Indy from cats that look nothing like him, but distinguishing him from similar-looking cats of the same general type. This is the single most important idea behind the dataset plan.

## Approach: embedding-based recognition

The planned approach is embedding-based ("metric learning"), chosen over fine-tuning a classifier because it needs very little data of Indy, is light on compute, and is robust for individual recognition.

The idea: a large pretrained vision model is used only to turn each image into an embedding — a vector that represents the image as a point in a high-dimensional space, where visually similar images land close together. The model itself is not trained; it is used in inference only. We build a small **gallery** of reference embeddings of Indy, and classify a new image by how close its embedding is to that gallery versus to embeddings of other cats.

The planned model is **DINOv2** (a base-size vision transformer as the starting point), because it is a stable, well-supported model that is strong specifically on fine-grained visual distinctions. This is a default, not a commitment: a smaller variant is a reasonable fallback if memory or speed becomes an issue, a larger one if accuracy falls short, and the newer DINOv3 is an easy swap to try later. The embedding approach does not depend on the exact backbone.

### Detect-and-crop stage

Before embedding, each image passes through a **detect-and-crop stage**. The embedding model represents the whole image, background included, and with only a handful of Indy photos the most likely failure mode is the system latching onto backgrounds rather than the cat. The fix is architectural: a small pretrained object detector (a COCO-trained YOLO-family model — "cat" is one of COCO's built-in classes, so no training is needed here either) locates the cat, and the embedding is computed from the cropped cat region, expanded by a small margin, rather than the full frame. The full pipeline is therefore: image → detect → crop → embed → decide.

For the Oxford negatives this costs nothing extra, since that dataset ships with bounding boxes already. For Indy's photos and for new images at prediction time, the detector generates the box automatically in milliseconds. Its output is textual — a count of cats found and a confidence number per box — which means the gallery crops can be sanity-checked without visual inspection ("35 photos in, 35 single-cat detections above 0.8 confidence"), a relevant property given the developer works with a screen reader. Two edge cases need explicit handling: no cat detected (report it rather than silently embedding the full frame) and multiple cats detected (embed each crop separately, so "is Indy in this photo" becomes "is any crop Indy"). The stage should be toggleable so its effect on accuracy can be measured rather than assumed.

### Decision step

The decision is **verification by similarity threshold**: for a new image, compute its cosine similarity to the Indy gallery embeddings only, take the best match (or the mean of the top few), and compare against a fixed cutoff — above means Indy, below means not. That is the entire decision; the negatives never participate in it. Their role is *calibration and evaluation*: before deployment, the gallery-similarity score is computed for many known non-Indy cats and for held-back Indy photos, and the threshold is chosen where the two score distributions separate. Moving the threshold trades the two error types against each other — higher means fewer false alarms but more missed Indys — which maps directly onto the evaluation metrics below.

This formulation was chosen over the more textbook **k-Nearest Neighbors voting** (pool all labeled reference points, let the k nearest vote) because of class imbalance: with roughly 15 Indy references against ~2,000 negatives, the negatives fill the embedding space densely and swamp the vote by sheer numbers, even when the *closest* matches are Indy. Shrinking k to compensate makes the answer hinge on one or two reference photos, which is fragile. The threshold method measures the thing that actually matters — how close the closest Indy matches are — and sidesteps the imbalance entirely, while remaining just as cheap and training-free.

If threshold accuracy proves insufficient, the escalation path is a **linear probe** (a small classifier trained on top of the frozen embeddings; at that point class balance becomes a real consideration). The progression should be driven by measured results rather than assumed.

## Environment

Development is on **Windows 10 with an NVIDIA RTX 3070 (8 GB VRAM)**. Because the heavy model runs in inference only — no training of the large network — 8 GB is comfortable, with room to spare. The GPU will speed up embedding batches of images during experiments, but nothing here strictly requires it. A free cloud GPU (for example a hosted notebook) is a viable fallback for the one-time step of embedding a dataset if a local setup is inconvenient at any point.

## Data

**Positives (Indy).** Because the pretrained model does the heavy lifting, the gallery is few-shot: roughly 10 to 20 varied photos is expected to be plenty. **35 photos are currently on hand**, which is comfortable for a start — roughly 15 for the gallery and 20 held out for testing, which makes recall measurable in 5% steps rather than coarser ones. More can be requested from Indy's owner later if evaluation calls for it; the held-out side is where extra volume helps most. Variety matters far more than count — different angles, distances, lighting, and especially different backgrounds — and photos that show his head and tail clearly are the most useful.

When splitting the 35, split by **occasion or scene rather than shuffling individual photos**: burst shots from the same session are near-duplicates, and placing them on both sides of the split makes recall look artificially perfect. All shots from one session land on the same side. File timestamps encode the session, so this check needs no visual inspection.

**Negatives (other cats).** The plan is to start with the **Oxford-IIIT Pet dataset**: a well-labeled, openly licensed (CC BY-SA 4.0 — the share-alike term is irrelevant for a hobby project but applies if anything derived is published) set with on the order of a couple of thousand cat images across twelve breeds. Crucially, several of those breeds — Maine Coon, Ragdoll, Birman — are long-haired cats that resemble Indy, so they serve as built-in harder negatives, while the short-haired breeds give easy general coverage.

Use **all of Oxford's cat images**, not a curated subset. Under the threshold decision, negatives never participate in the live decision, so their count costs nothing at prediction time; embedding them is a one-time step of a few minutes on the GPU, after which they are just rows of numbers on disk. Composition matters more than count, in favor of breadth: the hard look-alikes are what actually determine where the threshold must sit, while the easy short-haired negatives confirm the false-positive rate on ordinary cats is near zero and guard against the subtle failure of learning "long-haired = suspicious" rather than "Indy-like = suspicious."

Later, if needed, add a dataset that includes **Norwegian Forest Cats** specifically (larger community breed collections on Kaggle or Hugging Face are the likely source), since same-breed cats are the most demanding negatives of all and the closest match to Indy's look. One honest caveat, confirmed by spot-checking one such Kaggle set with Indy's owner: **community dataset labels are unreliable** — Norwegian Forest Cat, Maine Coon, and Siberian confusion is rampant. For negatives this barely matters (a mislabeled Maine Coon is still a valid hard negative — it is still not Indy); it only matters for claims about a slice being same-breed, so such a slice should be described honestly as "look-alike long-haired cats" rather than guaranteed Norwegian Forest Cats.

A practical, accessibility-relevant note: prefer **pre-labeled datasets organized by breed**. The breed label is text, so the set can be assembled and filtered without having to visually inspect each image — presence of a breed is a text check on folder or label names, not a visual one. Oxford's labels are trustworthy; community sets, per the caveat above, less so. Raw image-search sources are best avoided as a primary source for this reason. When assembling negatives, keep a reasonable spread across breeds rather than a pile of one, so the system learns "not Indy" rather than "not one particular breed."

## Incremental development and evaluation

The intended way of working is incremental, treating data additions as an experiment whose effect is measured rather than assumed. A rough staging:

1. Indy plus only the generic Oxford negatives — a quick baseline.
2. Add the Oxford long-haired breeds as harder negatives.
3. Add a dedicated Norwegian Forest Cat set.
4. Extend further only if results call for it.

To make this measurable, fix a **held-out test set up front and never use it during setup**. It should include some Indy photos and some other cats, and — importantly — a slice of look-alike long-haired cats (ideally Norwegian Forest Cats, subject to the label-quality caveat above) that never appear in the gallery or calibration negatives. That look-alike slice is the real exam.

The setup and test sides must be **disjoint at the image level**: the same photo must never be used both to set up the system (gallery, threshold calibration) and to test it, or the measured accuracy comes out optimistically wrong. Split the Oxford images once, up front — for example 70% available for calibration, 30% locked away for testing — and never move anything across. "Image level" contrasts with breed level: it is fine, desirable even, for Maine Coons to appear on both sides of the split, just never the same photograph. For Indy's photos the same rule applies plus the session-based splitting described under Data, to keep near-duplicates from leaking across.

The most informative metric here is the **false-positive rate on look-alikes**: the number of non-Indy cats wrongly identified as Indy, divided by the total number of non-Indy cats. This is the number expected to fall as same-breed negatives are added. Track it alongside **recall on Indy** (how many of Indy's own photos are correctly recognized) to confirm that fewer false positives are not being bought at the cost of missing Indy himself. A plausible and interesting dynamic to watch for: the early stages look strong because the negatives are easy, the look-alike slice then exposes the weakness, and adding same-breed negatives closes the gap. Whether it plays out that way is exactly the open question worth observing.

## Code organization

The one structural principle worth holding to: keep the **core recognition logic separate from any user interface**. The detect-crop-embed-decide pipeline should live in plain, importable Python modules that know nothing about how they are presented. Whatever UI sits on top — command line, Streamlit, or a custom web app — should be a thin layer that calls into that core.

This keeps the UI a disposable detail rather than a commitment, so it can be swapped freely without touching the recognition code. A related practical step: precompute Indy's gallery embeddings once and store the resulting vectors, rather than recomputing from his images at runtime, which also keeps any deployed version lean and avoids shipping his photos to a server.

## User interface

Three complementary options, in rough order of when they are useful:

A **command-line tool** is the simplest and most accessible for personal quick testing and the fastest development loop — point it at images, get verdicts. This is likely the primary tool while building.

**Streamlit** is the quickest path to a shareable visual demo (image in, prediction out, in very little code). One caveat to keep in mind: Streamlit has known accessibility limitations and gives the developer little control over its rendered output, which matters here because the developer relies on a screen reader. The plan is to try it and judge how usable it feels in practice.

**Update (2026-06-10): Streamlit's accessibility was evaluated in practice and came back positive.** A small review tool (`experiments/review_mapping.py`, written to let Indy's owner sanity-check the mapping CSV labels) renders the data two ways — a heading-led list of photos and an `st.dataframe` data grid — and both proved navigable with the developer's screen reader, including working table-navigation keys in the grid. Two rough edges surfaced, both manageable: `st.image` does not announce alt text (the per-photo heading supplies the context instead), and `st.dataframe`'s `ImageColumn` initially read out the entire base64 data URL for each image cell — resolved by serving the images through Streamlit's static file serving so the cell exposes a short path instead. The honest caveat is scope: this exercised static tables and images, not the more dynamic patterns the real app will need (a live region announcing a verdict, file upload). On this evidence Streamlit is a viable UI and the hand-built HTML fallback is not required on accessibility grounds for these patterns — though the dynamic pieces are worth a quick re-check when they are built.

A **hand-built web UI** (a small Flask or FastAPI app serving accessible HTML) is the fallback if Streamlit's accessibility proves too limiting. It is more work but gives full control over the markup — proper semantic structure and a live region that announces the result — and aligns with the developer's interest and experience in accessible interfaces.

The "core separate from UI" principle above is what makes trying Streamlit first low-risk: if it does not work out, the recognition code carries over unchanged.

## Deployment

If a hosted version is wanted, the inference is light, but the resource that matters is **memory, not compute**. CPU-only inference is fine for occasional single-image uploads; the real constraint is that the model plus its framework occupies on the order of 1 to 2 GB of RAM, which rules out the very smallest free tiers.

The best free fit appears to be **Hugging Face Spaces**, whose free tier offers generous memory (well beyond what the model needs), removing the only real constraint, and which can host both a Streamlit app and a custom Docker-based app — so it covers whichever UI is chosen. Free Spaces sleep when idle, meaning the first visitor after a quiet period waits a few seconds for startup, which is acceptable for a demo. **Streamlit Community Cloud** is an alternative specifically for the Streamlit version, though its free tier is tighter on memory, in which case a smaller model variant or a move to Spaces is the answer. These are starting suggestions; the deployment target is open.

## Things most likely to change

Listed explicitly so they are treated as adjustable, not settled:

- **Model choice and size** — DINOv2 base is a starting default; smaller, larger, or DINOv3 are all on the table depending on accuracy, speed, and memory.
- **Decision method** — similarity threshold against the gallery first; a linear probe is the escalation, and whether and when to adopt it should follow from measured accuracy, not be assumed.
- **Detect-and-crop** — expected to help and architecturally cheap, but its actual effect on accuracy is to be measured (the stage is toggleable for exactly this reason).
- **How much data is enough** — the photo counts and dataset stages are estimates; real test results decide whether more (or more targeted) data is needed.
- **UI** — Streamlit versus a custom accessible UI is to be decided by how Streamlit's accessibility actually feels in use. An initial in-practice evaluation (see the UI section's 2026-06-10 update) found Streamlit acceptable for static table/image views; the more dynamic patterns the real app needs remain to be confirmed.
- **Deployment target** — only relevant if hosting is wanted at all, and the specific platform is a soft recommendation.

The throughline: the embedding approach and the incremental, measure-as-you-go method are the stable core of the plan. Most everything layered on top is a reasonable default chosen to get started quickly, and is meant to be revised as the implementation reveals what actually works.