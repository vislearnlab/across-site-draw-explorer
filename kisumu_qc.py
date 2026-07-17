#!/usr/bin/env python3
"""
Flag Kisumu `transformed_drawings` that the lab preprocessing corrupted.

For each Kisumu drawing in the paper's analysis set (emb_df.parquet), compare the
ink coverage of the processed `transformed_drawings` PNG (what fed the CLIP
embeddings) against the clean source scan. When the transform turned a clean,
mostly-white scan into a heavily-inked black blob, the transformed image — and
therefore that drawing's CLIP embedding in the paper — is bad.

Metric per drawing:
  transformed_ink = fraction of dark pixels in transformed_drawings/<name>
  source_ink      = fraction of dark pixels in the clean source (resized/original)
  A drawing is flagged INCORRECT when the transform added a lot of ink that isn't
  in the source: transformed_ink is high AND far above source_ink.

Writes kisumu_transform_qc.csv (all 1440 rows, worst first) + prints a summary.
"""
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd

ASD_REPO = os.environ.get("ASD_REPO", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "across-site-draw"))
KVOL = "/Volumes/vislearnlab/experiments/drawing/data/kisumu"
STORES_NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stores.npz")


def dark_frac(path, thr=140):
    if not os.path.exists(path):
        return None
    a = np.asarray(Image.open(path).convert("L"))
    return float((a < thr).mean())


def main():
    emb = pd.read_parquet(os.path.join(ASD_REPO, "data", "emb_df.parquet"))
    emb = emb[emb.location == "Kisumu"].reset_index(drop=True)
    # attach each row's exact source filename via the embedding-store url match
    z = np.load(STORES_NPZ, allow_pickle=True)
    urls, embs = z["urls"], z["embs"].astype(np.float16)
    smap = {}
    for i in range(len(urls)):
        if "kisumu" in str(urls[i]).lower():
            smap.setdefault(embs[i].tobytes(), str(urls[i]))
    rows = []
    for r in emb.itertuples(index=False):
        url = smap.get(np.asarray(r.embedding, dtype=np.float16).tobytes())
        if not url:
            continue
        name = os.path.basename(url)
        tf = dark_frac(os.path.join(KVOL, "transformed_drawings", name))
        src = dark_frac(os.path.join(KVOL, "resized_drawings", name))
        if tf is None:
            continue
        added = tf - (src if src is not None else 0)
        # corrupted transform: lots of ink the clean scan doesn't have
        flag = (tf > 0.22 and added > 0.12) or tf > 0.45
        rows.append(dict(filename=name, participant_id=r.participant_id,
                         category=r.drawing_category, age=int(r.age),
                         transformed_ink=round(tf, 3),
                         source_ink=round(src, 3) if src is not None else "",
                         ink_added=round(added, 3), incorrect=int(flag)))
        if len(rows) % 200 == 0:
            print(f"  scanned {len(rows)} ...", flush=True)
    rows.sort(key=lambda d: -d["transformed_ink"])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "kisumu_transform_qc.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    nbad = sum(r["incorrect"] for r in rows)
    print(f"\nscanned {len(rows)} Kisumu drawings")
    print(f"FLAGGED INCORRECT (corrupted transform): {nbad}")
    print(f"wrote {out}")
    print("\nworst 15:")
    for r in rows[:15]:
        print(f"  {r['filename']:42s} tf={r['transformed_ink']:.2f} "
              f"src={r['source_ink']} added={r['ink_added']:.2f} "
              f"{'INCORRECT' if r['incorrect'] else ''}")


if __name__ == "__main__":
    main()
