# Across-Site Drawing Explorer

An interactive web page for exploring **4,387 children's line drawings** of **12 object
categories** produced by kids aged **4–9** at four international data-collection sites —
**San Jose** (USA), **Beijing** (China), **New Delhi** (India), and **Kisumu** (Kenya) —
for the study *"Structure and diversity in children's line drawings of object categories
across four international contexts"* (Du, Sepuri, Maheshwari, Zhu, Arieda & Long, CCN 2026).
Built in the style of the
[`drawing-explorer`](https://github.com/vislearnlab/drawing-explorer) /
[`sea-animals-draw-explorer`](https://github.com/vislearnlab/sea-animals-draw-explorer) pages.

Each drawing is placed in a 2-D map of its **CLIP (ViT-B/32)** embedding — the exact
embeddings from the paper's [`across-site-draw`](https://github.com/vislearnlab/across-site-draw)
analysis (`data/emb_df.parquet`). The page layers on interactive views of the paper's
cross-site results.

## What it shows

**Main map** — a t-SNE of all 4,387 drawings' 512-d CLIP embeddings.

- **Layout:** *Unified map* (one shared t-SNE) or *By site* (four within-site t-SNEs side
  by side, so you can compare how category structure is arranged at each site — the
  "shared category geometry" question).
- **Color by:** site · category · age · recognizability (continuous scales use **viridis**,
  matching the paper's figures).
- **Show:** dots **or the actual drawings** as marks; optional **labeled category centroids**.
- **Hover** any point for the drawing; **click** to pin it and **▶ replay** the strokes
  in the order they were drawn (tablet sites only — Kisumu is scanned paper).
- **Filter:** site chips, category chips (+ biological/semantic group chips), age range,
  and CLIP-recognized-only. Everything updates the footer and side panels together.

**Category structure by site (RDMs)** — the paper's 12×12 representational dissimilarity
matrices (cosine distance between category-mean CLIP embeddings), one per site, with
category axis labels and a shared viridis scale. Hover a cell to compare that category
pair across all four sites.

**Cross-site divergence by category** — mean cosine distance between the four sites'
category-means, per object. Reproduces the paper's result that **rabbit, tree, and cat**
diverge most across contexts (highlighted).

**Recognizability rises with age** — mean CLIP target similarity by age, per site,
recomputed live from whatever's currently filtered.

Views can be linked/shared via URL params, e.g.
`?layout=site&color=cat&marks=draw&centroids=1`.

## Run locally

```bash
python3 -m http.server 8000
# open http://127.0.0.1:8000/
```

`index.html` is self-contained (no build step, no dependencies); it reads `points.json`,
`strokes.json`, and the PNGs in `drawings/`.

## Where the drawings come from

The layout, recognizability, RDMs, and divergence all come from the paper's
`emb_df.parquet` (no server needed). Only the **thumbnails** touch external sources, and
each site's are **re-rendered at high resolution from the raw strokes** (except Kisumu,
which was collected on paper):

| Site | Thumbnail source | Rendering |
|------|------------------|-----------|
| San Jose | `devphotodraw` `all_strokes.csv` (free-drawing / `S` condition) | strokes → PNG + vector |
| Beijing | `devphotodraw` `all_strokes.csv` (free-drawing / `S` condition) | strokes → PNG + vector |
| New Delhi | `kiddraw.india_run_v1` MongoDB stroke docs | strokes → PNG + vector |
| Kisumu | scanned PNGs on the lab volume | cropped/upscaled raster |

Each `emb_df` row is linked to its exact source image via the `url` stored in the CLIP
embedding stores (`across-site-draw/data/embeddings/*.docs`), matched by embedding — this
resolves the Beijing `participant_id` collisions (dropped `IPAD` prefix) that metadata
alone can't. San Jose/Beijing/New Delhi drawings are stored as normalized **vector stroke
paths** in `strokes.json` for crisp in-browser rendering and stroke replay.

## Files

- `index.html` — the self-contained explorer.
- `points.json` — t-SNE layouts (unified + per-site) + per-drawing site/category/age/
  recognizability + per-site RDMs, category divergence, and recognizability-by-age.
- `strokes.json` — normalized SVG stroke paths for the 2,947 tablet-site drawings.
- `drawings/` — 400 px high-res PNG per drawing (universal thumbnail).
- `render_lib.py` — SVG-path parser + high-res stroke renderer (PNG + normalized vector).
- `dump_stores.py` — decodes the `.docs` CLIP embedding stores to `stores.npz`.
- `build_data.py` — regenerates `drawings/` + `points.json` + `strokes.json`.

## Rebuilding the data

Needs `numpy`, `pandas`, `pyarrow`, `pillow`, `pymongo`, `scikit-learn`, and (for
`dump_stores.py` only) `docarray>=0.40` + `protobuf` in a side venv. Requires the lab VPN
(for the New Delhi MongoDB) and the mounted lab volume (for Kisumu PNGs).

```bash
# 1) decode the CLIP embedding stores -> stores.npz  (docarray venv)
python -m venv docenv && ./docenv/bin/pip install "docarray>=0.40" protobuf numpy
./docenv/bin/python dump_stores.py /path/to/across-site-draw/data/embeddings stores.npz

# 2) Mongo connection string for the New Delhi images (do NOT commit this)
echo 'mongodb://USER:PASS@vislearnlab.ucsd.edu:27017/?authSource=admin' > auth.txt

# 3) build (reuses PNGs already on disk; only re-hits Mongo/volume for missing ones)
python3 build_data.py
```

Credentials live in a git-ignored `auth.txt` (or the `ASD_MONGO_URI` env var) and are
never committed. Paths to the source repos/volume can be overridden via the `ASD_REPO`,
`DEVPHOTODRAW`, `KISUMU_DIR`, and `STORES_NPZ` env vars (see the top of `build_data.py`).

## Data & paper

> *How does the representational geometry of children's drawings differ across four
> international contexts, and which categories converge or diverge?* — children (ages 4–9)
> drew 12 common object categories at sites in the USA, China, India, and Kenya.
> See [`across-site-draw`](https://github.com/vislearnlab/across-site-draw).
