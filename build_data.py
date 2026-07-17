#!/usr/bin/env python3
"""
Build drawings/ + points.json + strokes.json for the Across-Site Drawing Explorer.

Dataset: 4,387 children's line drawings of 12 object categories produced by kids
aged 4-9 at four international sites (San Jose USA, Beijing China, New Delhi India,
Kisumu Kenya) -- the CCN 2026 "Structure and diversity in children's line drawings"
study. The scientific spine is data/emb_df.parquet from the `across-site-draw`
repo: per-drawing 512-d CLIP (ViT-B/32) embeddings + recognizability.

This script attaches a high-resolution THUMBNAIL to each of those 4,387 drawings by
re-rendering the RAW STROKES (so the images are sharp, not the tiny tablet exports):

  * San Jose + Beijing  <- devphotodraw all_strokes.csv (free-drawing / "S" cond)
  * New Delhi           <- kiddraw.india_run_v1 MongoDB stroke docs
  * Kisumu              <- scanned PNGs on the lab volume (paper-based: no strokes)

Each drawing is linked to its exact source image via the `url` stored in the
CLIP embedding stores (data/embeddings/*.docs); dump_stores.py decodes those to
stores.npz, and we match emb_df rows to urls by embedding (handles the Beijing
participant-id collisions that metadata alone can't resolve).

Outputs:
  * drawings/<id>.png   -- 400px high-res render per drawing (universal thumbnail)
  * points.json         -- t-SNE layout + per-drawing scores + cross-site aggregates
  * strokes.json        -- normalized SVG stroke paths for the 3 tablet sites
                           (crisp in-browser vector rendering; Kisumu is raster-only)

Env / inputs (override via env vars):
  ASD_REPO      across-site-draw checkout      (default: ../across-site-draw)
  DEVPHOTODRAW  devphotodraw checkout          (default: ../devphotodraw)
  KISUMU_DIR    kisumu transformed_drawings    (default: lab volume)
  ASD_MONGO_URI mongo connection (India)       (default: auth.txt line 1)
  STORES_NPZ    stores.npz from dump_stores.py (default: ./stores.npz)
"""
import os
import io
import re
import json
import base64

import numpy as np
import pandas as pd
from PIL import Image

import render_lib as R

HERE = os.path.dirname(os.path.abspath(__file__))
ASD_REPO = os.environ.get("ASD_REPO", os.path.join(HERE, "..", "across-site-draw"))
DEVPHOTO = os.environ.get("DEVPHOTODRAW", os.path.join(HERE, "..", "devphotodraw"))
KISUMU_DIR = os.environ.get(
    "KISUMU_DIR",
    "/Volumes/vislearnlab/experiments/drawing/data/kisumu/transformed_drawings")
STORES_NPZ = os.environ.get("STORES_NPZ", os.path.join(HERE, "stores.npz"))
DRAW_DIR = os.path.join(HERE, "drawings")

EMB_PARQUET = os.path.join(ASD_REPO, "data", "emb_df.parquet")
STROKES_CSV = os.path.join(
    DEVPHOTO, "data", "compiled", "strokes_preprocessed", "all_strokes.csv")
TRANS_CSV = os.path.join(ASD_REPO, "data", "nel-translations", "en-hi-zh-sw.csv")

CATEGORIES = ["airplane", "bike", "bird", "car", "cat", "chair",
              "cup", "hat", "house", "rabbit", "tree", "watch"]
CAT_IDX = {c: i for i, c in enumerate(CATEGORIES)}
SITES = ["San Jose", "Beijing", "New Delhi", "Kisumu"]
SITE_IDX = {s: i for i, s in enumerate(SITES)}
# semantic groupings for filter chips
GROUPS = {"Animals": ["bird", "cat", "rabbit"],
          "Vehicles": ["airplane", "bike", "car"],
          "Furniture": ["chair"],
          "Household": ["cup", "hat", "watch", "house"],
          "Nature": ["tree"]}
RENDER_SIZE = 400


def png_id(url):
    """Stable, unique per-drawing id from the source image basename."""
    b = os.path.basename(str(url))
    b = b[:-4] if b.lower().endswith(".png") else b
    return re.sub(r"[^A-Za-z0-9_.-]", "-", b)


# --------------------------------------------------------------- identity spine
def load_spine():
    """emb_df (4,387 rows) with each row's exact source `url` attached by matching
    its 512-d embedding to the CLIP embedding stores (stores.npz)."""
    emb = pd.read_parquet(EMB_PARQUET)
    z = np.load(STORES_NPZ, allow_pickle=True)
    urls, embs = z["urls"], z["embs"].astype(np.float16)
    smap = {}
    for i in range(len(urls)):
        smap.setdefault(embs[i].tobytes(), []).append(i)
    matched = []
    for row in emb.itertuples(index=False):
        idxs = smap.get(np.asarray(row.embedding, dtype=np.float16).tobytes())
        matched.append(str(urls[idxs[0]]) if idxs else None)
    emb = emb.assign(url=matched)
    miss = emb.url.isna().sum()
    if miss:
        raise SystemExit(f"{miss} emb_df rows could not be matched to a source url")
    emb["id"] = emb.url.apply(png_id)
    return emb


# ------------------------------------------------------- stroke sources per site
def devphoto_strokes():
    """(pid, category) -> ordered list of stroke `d` strings, free-drawing cond."""
    st = pd.read_csv(STROKES_CSV)
    st = st[st.condition == "S"].copy()
    st["pid"] = st.filename.str.replace(".png", "", regex=False).str.split("_").str[-1]
    st = st.sort_values(["pid", "category", "stroke_count"])
    out = {}
    for (pid, cat), g in st.groupby(["pid", "category"]):
        out[(str(pid), cat)] = g["svg"].tolist()
    return out


def india_strokes():
    """(sessionId, category) and (PID, category) -> ordered stroke `d` strings."""
    import pymongo as pm
    uri = os.environ.get("ASD_MONGO_URI")
    if not uri and os.path.exists(os.path.join(HERE, "auth.txt")):
        uri = open(os.path.join(HERE, "auth.txt")).readline().strip()
    coll = pm.MongoClient(uri, serverSelectionTimeoutMS=15000)["kiddraw"]["india_run_v1"]
    by_sess, by_pid = {}, {}
    cur = coll.find({"dataType": "stroke"},
                    {"sessionId": 1, "participantID": 1, "category": 1,
                     "svg": 1, "trialNum": 1, "startStrokeTime": 1})
    rows = list(cur)
    def sortkey(d):
        return (d.get("trialNum") or 0, d.get("startStrokeTime") or 0)
    rows.sort(key=sortkey)
    for d in rows:
        cat = re.sub(r"^(a|an) ", "", d.get("category", ""))
        by_sess.setdefault((d.get("sessionId"), cat), []).append(d["svg"])
        by_pid.setdefault((str(d.get("participantID", "")).upper(), cat), []).append(d["svg"])
    return by_sess, by_pid


# ----------------------------------------------------------------- render a row
def render_row(row, dev, ind_sess, ind_pid):
    """Return (png_bytes_or_None, svg_paths_or_None) for one emb_df row."""
    site, cat, pid = row.location, row.drawing_category, str(row.participant_id)
    strokes = None
    if site in ("San Jose", "Beijing"):
        strokes = dev.get((pid, cat))
    elif site == "New Delhi":
        m = re.search(r"(india_run_v1\d+)\.png$", os.path.basename(row.url))
        if m:
            strokes = ind_sess.get((m.group(1), cat))
        if not strokes:
            strokes = ind_pid.get((pid.upper(), cat))
    if site == "Kisumu":
        src = os.path.join(KISUMU_DIR, os.path.basename(row.url))
        if not os.path.exists(src):
            return None, None
        im = Image.open(src).convert("RGB")
        # pad to square on white, then resize -> uniform thumbnail
        w, h = im.size
        s = max(w, h)
        bg = Image.new("RGB", (s, s), (255, 255, 255))
        bg.paste(im, ((s - w) // 2, (s - h) // 2))
        buf = io.BytesIO()
        bg.resize((RENDER_SIZE, RENDER_SIZE), Image.LANCZOS).save(buf, "PNG", optimize=True)
        return buf.getvalue(), None
    if not strokes:
        return None, None
    png = R.render_png(strokes, size=RENDER_SIZE)
    paths, _ = R.normalized_svg(strokes)
    return png, paths


# --------------------------------------------------------------- aggregate stats
def emb_matrix(emb):
    return np.stack(emb.embedding.apply(lambda v: np.asarray(v, np.float32)).values)


def tsne_layout(X, label=""):
    from sklearn.manifold import TSNE
    n = len(X)
    perp = max(10, min(50, n // 100))
    print(f"  t-SNE{label} on {n}x{X.shape[1]} (perplexity={perp}, cosine) ...")
    emb = TSNE(n_components=2, perplexity=perp, init="pca", metric="cosine",
               learning_rate="auto", random_state=0).fit_transform(X)
    lo, hi = emb.min(0), emb.max(0)
    emb = 30 + (emb - lo) / (hi - lo + 1e-9) * 940
    return np.round(emb, 2)


def per_site_layouts(emb, X):
    """A separate within-site t-SNE per site; returns xs, ys aligned to emb rows
    (each point positioned inside its own site's 0..1000 layout). Powers the
    'category x site' small-multiples view of shared category structure."""
    xs = np.zeros(len(emb)); ys = np.zeros(len(emb))
    site_arr = emb.location.values
    for site in SITES:
        sel = np.where(site_arr == site)[0]
        xy = tsne_layout(X[sel], label=f" [{site}]")
        xs[sel] = xy[:, 0]; ys[sel] = xy[:, 1]
    return np.round(xs, 2), np.round(ys, 2)


def cosine_dist_matrix(means):
    M = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-9)
    sim = M @ M.T
    return np.clip(1 - sim, 0, 2)


def site_rdms(emb, X):
    """12x12 cosine-distance RDM of category-mean embeddings, per site."""
    rdms = {}
    cat_arr = emb.drawing_category.map(CAT_IDX).values
    site_arr = emb.location.values
    for site in SITES:
        means = np.zeros((12, X.shape[1]))
        for ci in range(12):
            sel = (site_arr == site) & (cat_arr == ci)
            means[ci] = X[sel].mean(0) if sel.any() else 0
        rdms[site] = np.round(cosine_dist_matrix(means), 4).tolist()
    return rdms


def category_divergence(emb, X):
    """Mean cross-site cosine distance between site category-means, per category."""
    cat_arr = emb.drawing_category.map(CAT_IDX).values
    site_arr = emb.location.values
    site_means = {}
    for site in SITES:
        m = np.zeros((12, X.shape[1]))
        for ci in range(12):
            sel = (site_arr == site) & (cat_arr == ci)
            m[ci] = X[sel].mean(0) if sel.any() else np.nan
        site_means[site] = m
    div = []
    for ci in range(12):
        ds = []
        for a in range(len(SITES)):
            for b in range(a + 1, len(SITES)):
                va, vb = site_means[SITES[a]][ci], site_means[SITES[b]][ci]
                if np.isfinite(va).all() and np.isfinite(vb).all():
                    va = va / (np.linalg.norm(va) + 1e-9)
                    vb = vb / (np.linalg.norm(vb) + 1e-9)
                    ds.append(1 - float(va @ vb))
        div.append(round(float(np.mean(ds)), 4) if ds else None)
    return div


def recog_by_age(emb):
    """Per (site, age): mean target_similarity + n. Also overall by age."""
    out = {}
    for site in SITES:
        s = emb[emb.location == site]
        pts = []
        for age in range(4, 10):
            a = s[s.age == age]
            if len(a):
                pts.append([age, round(float(a.target_similarity.mean()), 4), int(len(a))])
        out[site] = pts
    return out


# ------------------------------------------------------------------------- main
def main():
    os.makedirs(DRAW_DIR, exist_ok=True)
    print("loading spine (emb_df + url match) ...")
    emb = load_spine()
    print(f"  {len(emb)} drawings: " +
          ", ".join(f"{s} {int((emb.location==s).sum())}" for s in SITES))

    # reuse a previous strokes.json so a rebuild that only changes layout/scores
    # doesn't have to re-hit Mongo or re-parse strokes for cached PNGs.
    old_svg = {}
    sj_path = os.path.join(HERE, "strokes.json")
    if os.path.exists(sj_path):
        old_svg = json.load(open(sj_path))
        print(f"  reusing {len(old_svg)} cached vector strokes")

    # Lazily load stroke sources only if some drawing actually needs (re)rendering.
    _src = {}
    def sources():
        if not _src:
            print("loading stroke sources ...")
            _src["dev"] = devphoto_strokes()
            print(f"  devphotodraw: {len(_src['dev'])} (pid,cat) stroke groups")
            _src["ind_sess"], _src["ind_pid"] = india_strokes()
            print(f"  india mongo: {len(_src['ind_sess'])} (session,cat) groups")
        return _src["dev"], _src["ind_sess"], _src["ind_pid"]

    print("rendering thumbnails ...")
    svg_store = {}
    rec = {k: [] for k in ("id", "site", "cat", "age", "pid", "recog",
                           "target_sim", "has_svg")}
    n_png = n_svg = n_miss = 0
    ids_order = []
    for j, row in enumerate(emb.itertuples(index=False)):
        did = row.id
        path = os.path.join(DRAW_DIR, f"{did}.png")
        paths = None
        if os.path.exists(path):
            # reuse existing PNG; recover its vector strokes from cache if possible
            if row.location != "Kisumu":
                paths = old_svg.get(did)
                if paths is None:
                    _, paths = render_row(row, *sources())
        else:
            png, paths = render_row(row, *sources())
            if png is None:
                n_miss += 1
                continue
            with open(path, "wb") as f:
                f.write(png)
            n_png += 1
        ids_order.append(j)
        rec["id"].append(did)
        rec["site"].append(SITE_IDX[row.location])
        rec["cat"].append(CAT_IDX[row.drawing_category])
        rec["age"].append(int(row.age))
        rec["pid"].append(str(row.participant_id))
        rec["recog"].append(1 if bool(row.recognized) else 0)
        rec["target_sim"].append(round(float(row.target_similarity), 4))
        if paths:
            svg_store[did] = paths
            rec["has_svg"].append(1)
            n_svg += 1
        else:
            rec["has_svg"].append(0)
        if (j + 1) % 500 == 0:
            print(f"    {j + 1}/{len(emb)} ({n_png} new png, {n_miss} missing)")
    print(f"  rendered {n_png} new PNGs, {n_svg} with vector strokes, {n_miss} missing")

    kept = emb.iloc[ids_order].reset_index(drop=True)
    X = emb_matrix(kept)
    xy = tsne_layout(X, label=" [all]")
    rec["x"] = xy[:, 0].tolist()
    rec["y"] = xy[:, 1].tolist()
    xs, ys = per_site_layouts(kept, X)
    rec["xs"] = xs.tolist()
    rec["ys"] = ys.tolist()
    rec["n"] = len(kept)

    print("computing cross-site structure ...")
    rdms = site_rdms(kept, X)
    div = category_divergence(kept, X)
    recage = recog_by_age(kept)

    translations = {}
    if os.path.exists(TRANS_CSV):
        tr = pd.read_csv(TRANS_CSV)
        # en uses 'bicycle' where we use 'bike'
        tr["en"] = tr["en"].replace({"bicycle": "bike"})
        for _, r in tr.iterrows():
            if r["en"] in CAT_IDX:
                translations[r["en"]] = {"hi": r.get("hi"), "zh": r.get("zh"),
                                         "sw": r.get("sw")}

    out = dict(
        categories=CATEGORIES, sites=SITES, groups=GROUPS,
        draw_dir="drawings", render_size=RENDER_SIZE,
        translations=translations,
        points=rec, rdms=rdms, divergence=div, recog_by_age=recage,
    )
    with open(os.path.join(HERE, "points.json"), "w") as f:
        json.dump(out, f, separators=(",", ":"))
    with open(os.path.join(HERE, "strokes.json"), "w") as f:
        json.dump(svg_store, f, separators=(",", ":"))
    sz = os.path.getsize(os.path.join(HERE, "points.json")) / 1e6
    ssz = os.path.getsize(os.path.join(HERE, "strokes.json")) / 1e6
    print(f"wrote points.json ({sz:.1f} MB), strokes.json ({ssz:.1f} MB), "
          f"{rec['n']} drawings")


if __name__ == "__main__":
    main()
