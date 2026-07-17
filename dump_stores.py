#!/usr/bin/env python3
"""Decode the vislearnlabpy CLIP embedding stores (.docs) to a plain stores.npz.

The .docs files are a stream of [4-byte big-endian length][gzip protobuf DocProto]
records (one CLIPImageEmbedding per record: a source-image `url` + a 512-d CLIP
embedding). docarray v2 + protobuf are only available in a side venv, so this
script runs THERE and writes a dependency-free stores.npz that build_data.py
(base env) consumes to attach the exact source `url` to each emb_df row.

    <venv>/bin/python dump_stores.py \
        /Users/brialong/Documents/GitHub/across-site-draw/data/embeddings \
        stores.npz
"""
import sys
import struct
import gzip
from typing import Optional

import numpy as np
from docarray import BaseDoc, DocList  # noqa: F401  (DocList import validates env)
from docarray.typing import NdArray


class E(BaseDoc):
    url: Optional[str] = None
    embedding: Optional[NdArray] = None
    text: Optional[str] = None


STORES = ["sanjose_store.docs", "beijing_store.docs",
          "newdelhi_store.docs", "kisumu_store.docs"]


def records(path):
    data = open(path, "rb").read()
    off = 0
    while off < len(data) - 6:
        ln = struct.unpack(">I", data[off:off + 4])[0]
        blob = data[off + 4:off + 4 + ln]
        if blob[:2] == b"\x1f\x8b":
            yield blob
            off += 4 + ln
        else:
            off += 1


def main():
    emb_dir = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "stores.npz"
    urls, embs, stores = [], [], []
    for s in STORES:
        path = f"{emb_dir}/{s}"
        n = 0
        for blob in records(path):
            try:
                d = E.from_bytes(blob, protocol="protobuf", compress="gzip")
            except Exception:
                continue
            if d.embedding is None or d.url is None:
                continue
            urls.append(str(d.url))
            embs.append(np.asarray(d.embedding, dtype=np.float16))
            stores.append(s.replace("_store.docs", ""))
            n += 1
        print(f"{s}: {n} embeddings")
    embs = np.stack(embs).astype(np.float16)
    np.savez_compressed(out, urls=np.array(urls), embs=embs,
                        stores=np.array(stores))
    print(f"wrote {out}: {embs.shape[0]} embeddings x {embs.shape[1]} dims")


if __name__ == "__main__":
    main()
