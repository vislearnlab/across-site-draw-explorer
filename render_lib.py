#!/usr/bin/env python3
"""
Render children's line drawings from raw stroke data (SVG path strings) to

  * a high-resolution PNG (rasterised, round-capped, flattened on white), and
  * a compact normalised SVG path list for crisp in-browser vector display.

Two stroke sources feed this, both giving one SVG `d` string per stroke:

  * devphotodraw  (San Jose CDM + Beijing THU)  -> all_strokes.csv `svg` column
                  (a full <path d="..."/> element; we pull out the `d`)
  * india_run_v1  (New Delhi)                   -> MongoDB stroke docs `svg` field
                  (a bare `d` string)

The tablet apps export strokes as polylines (M + relative l/h/v segments); we
also handle C/Q/S/T beziers defensively. Coordinates are in the app's own canvas
(~0..800, occasionally slightly negative); we fit each drawing to its own content
bounding box so every render is centred and tight regardless of source canvas.
"""
import re
import io
import math
from PIL import Image, ImageDraw

_TOKEN = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|(-?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def _d_from_svg(svg):
    """Accept either a full <path .../> element or a bare `d` string; return `d`."""
    if "<path" in svg or 'd="' in svg:
        m = re.search(r'd="([^"]+)"', svg)
        return m.group(1) if m else ""
    return svg.strip()


def parse_path(svg):
    """Parse an SVG path into a list of polylines (each a list of (x, y) points).

    Curves are flattened to short line segments so downstream code only ever
    deals with polylines. Returns [] for empty/degenerate input."""
    d = _d_from_svg(svg)
    if not d:
        return []
    toks = _TOKEN.findall(d)
    nums, cmds, i = [], [], 0
    # Flatten into an ordered command stream
    stream = []
    for cmd, num in toks:
        if cmd:
            stream.append(("c", cmd))
        else:
            stream.append(("n", float(num)))

    polylines = []
    cur = []
    x = y = 0.0
    sx = sy = 0.0            # subpath start (for Z)
    op = None
    pos = 0

    def nextnums(k):
        nonlocal pos
        vals = []
        while len(vals) < k and pos < len(stream) and stream[pos][0] == "n":
            vals.append(stream[pos][1]); pos += 1
        return vals

    while pos < len(stream):
        kind, val = stream[pos]
        if kind == "c":
            op = val; pos += 1
        # Implicit repeat: keep the previous op if numbers keep coming
        if op is None:
            pos += 1; continue

        o = op
        if o in "Mm":
            v = nextnums(2)
            if len(v) < 2: break
            if o == "m":
                x += v[0]; y += v[1]
            else:
                x, y = v[0], v[1]
            if cur:
                polylines.append(cur)
            cur = [(x, y)]
            sx, sy = x, y
            op = "l" if o == "m" else "L"   # subsequent pairs are implicit linetos
        elif o in "Ll":
            v = nextnums(2)
            if len(v) < 2: break
            if o == "l":
                x += v[0]; y += v[1]
            else:
                x, y = v[0], v[1]
            cur.append((x, y))
        elif o in "Hh":
            v = nextnums(1)
            if not v: break
            x = x + v[0] if o == "h" else v[0]
            cur.append((x, y))
        elif o in "Vv":
            v = nextnums(1)
            if not v: break
            y = y + v[0] if o == "v" else v[0]
            cur.append((x, y))
        elif o in "CcSsQqTt":
            # Bezier: read control pts + endpoint, flatten to segments.
            n = {"C": 6, "S": 4, "Q": 4, "T": 2}[o.upper()]
            v = nextnums(n)
            if len(v) < n: break
            pts = []
            if o.islower():
                acc = []
                for j in range(0, n, 2):
                    acc.append((x + v[j], y + v[j + 1]))
                pts = acc
            else:
                for j in range(0, n, 2):
                    pts.append((v[j], v[j + 1]))
            p0 = (x, y)
            if o.upper() == "C":
                ctrl = [p0, pts[0], pts[1], pts[2]]
            elif o.upper() == "S":
                ctrl = [p0, p0, pts[0], pts[1]]
            elif o.upper() == "Q":
                ctrl = [p0, pts[0], pts[0], pts[1]]
            else:  # T
                ctrl = [p0, p0, p0, pts[0]]
            for t in (k / 8.0 for k in range(1, 9)):
                mt = 1 - t
                bx = (mt**3 * ctrl[0][0] + 3 * mt**2 * t * ctrl[1][0]
                      + 3 * mt * t**2 * ctrl[2][0] + t**3 * ctrl[3][0])
                by = (mt**3 * ctrl[0][1] + 3 * mt**2 * t * ctrl[1][1]
                      + 3 * mt * t**2 * ctrl[2][1] + t**3 * ctrl[3][1])
                cur.append((bx, by))
            x, y = ctrl[3]
        elif o in "Aa":
            # Arc: approximate by a straight segment to the endpoint (rare here).
            v = nextnums(7)
            if len(v) < 7: break
            x = x + v[5] if o == "a" else v[5]
            y = y + v[6] if o == "a" else v[6]
            cur.append((x, y))
        elif o in "Zz":
            if cur:
                cur.append((sx, sy))
            x, y = sx, sy
            pos += 1
            op = None
        else:
            pos += 1
    if cur:
        polylines.append(cur)
    # Drop empty/degenerate polylines but keep single-point dots (rendered as blobs)
    return [pl for pl in polylines if pl]


def bbox(polylines):
    xs = [p[0] for pl in polylines for p in pl]
    ys = [p[1] for pl in polylines for p in pl]
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _fit(polylines, size, pad):
    """Scale/translate polylines to fit a size×size box (uniform aspect), padded."""
    bb = bbox(polylines)
    if bb is None:
        return None, None
    x0, y0, x1, y1 = bb
    w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    inner = size - 2 * pad
    s = inner / max(w, h)
    ox = pad + (inner - w * s) / 2
    oy = pad + (inner - h * s) / 2

    def tf(p):
        return (ox + (p[0] - x0) * s, oy + (p[1] - y0) * s)
    return [[tf(p) for p in pl] for pl in polylines], s


def render_png(strokes, size=400, line_width=None, pad=None, supersample=2,
               transparent=True):
    """Rasterise a list of stroke SVG strings to a size×size PNG (bytes).

    By default the background is transparent and the ink is black — so drawings can
    be tinted to any color and overlap without occluding one another. Round caps/joins
    via a dot at every vertex plus thick segments; supersampled then downsampled for
    smooth edges. Returns PNG bytes, or None if nothing to draw."""
    polylines = []
    for s in strokes:
        polylines.extend(parse_path(s))
    if not polylines:
        return None
    if pad is None:
        pad = int(size * 0.06)
    S = size * supersample
    lw = (line_width if line_width is not None else max(2.0, size / 90.0)) * supersample
    fitted, _ = _fit(polylines, S, pad * supersample)
    if fitted is None:
        return None
    if transparent:
        img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    else:
        img = Image.new("RGB", (S, S), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    ink = (0, 0, 0, 255) if transparent else (0, 0, 0)
    r = lw / 2.0
    for pl in fitted:
        if len(pl) == 1:
            x, y = pl[0]
            dr.ellipse([x - r, y - r, x + r, y + r], fill=ink)
            continue
        dr.line(pl, fill=ink, width=int(round(lw)), joint="curve")
        for (x, y) in pl:                     # round caps at every vertex
            dr.ellipse([x - r, y - r, x + r, y + r], fill=ink)
    img = img.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def normalized_svg(strokes, viewbox=100, ndigits=1):
    """Return (paths, stroke_count) where paths is a list of `d` strings scaled to
    a 0..viewbox square (uniform aspect, centred) for crisp in-browser rendering."""
    polylines = []
    for s in strokes:
        polylines.extend(parse_path(s))
    if not polylines:
        return [], 0
    fitted, _ = _fit(polylines, viewbox, viewbox * 0.04)
    out = []
    for pl in fitted:
        pts = [f"{round(x, ndigits)},{round(y, ndigits)}" for (x, y) in pl]
        if len(pts) == 1:
            x, y = pl[0]
            out.append(f"M{pts[0]}l0.01,0")     # a dot
        else:
            out.append("M" + pts[0] + "L" + "L".join(pts[1:]))
    return out, len(polylines)
