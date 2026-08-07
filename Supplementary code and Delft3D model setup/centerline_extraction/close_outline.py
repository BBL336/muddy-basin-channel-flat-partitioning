import os
import numpy as np
from svgpathtools import svg2paths
import svgwrite

# =========================
# Configuration paths
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "svg_open")
OUT_DIR = os.path.join(SCRIPT_DIR, "svg_closed")
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE_N = 600


# =========================
# Utility functions
# =========================
def sample_path(path, n=SAMPLE_N):
    return np.array([
        [path.point(i / (n - 1)).real, path.point(i / (n - 1)).imag]
        for i in range(n)
    ])


def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def perpendicular(v):
    return np.array([-v[1], v[0]])


def fmt(p):
    return f"{p[0]},{p[1]}"


# =========================
# Main loop
# =========================
for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith(".svg"):
        continue

    print(f"\nProcessing: {fname}")

    paths, _ = svg2paths(os.path.join(INPUT_DIR, fname))
    if not paths:
        print("  Warning: no path found, skipped")
        continue

    # [1] Get the longest path and sample it
    path = max(paths, key=lambda p: p.length())
    pts = sample_path(path)

    centroid = pts.mean(axis=0)
    start, end = pts[0], pts[-1]

    # =========================
    # Core logic: midpoint outward extension
    # =========================

    # [2] Midpoint of the start and end points
    mid_point = (start + end) / 2

    # [3] Chord direction & normal
    chord_vec = end - start
    chord_dist = np.linalg.norm(chord_vec)
    n_out = normalize(perpendicular(chord_vec))

    # [4] Determine the outward direction
    if np.dot(n_out, mid_point - centroid) < 0:
        n_out = -n_out

    # [5] Extension distance (Euclidean distance x 10)
    L = chord_dist * 10
    if chord_dist == 0:
        L = 1000  # Fallback for extreme cases

    # Console output
    print(f"  Euclidean distance between endpoints = {chord_dist:.3f}")
    print(f"  Euclidean distance x 10 (L) = {L:.3f}")

    # [6] Extension point
    mid_ext = mid_point + n_out * L

    # =========================
    # Build the sealing path (forming a triangle)
    # =========================
    d = [f"M {fmt(start)}"]
    for p in pts[1:]:
        d.append(f"L {fmt(p)}")
    d.append(f"L {fmt(mid_ext)}")
    d.append("Z")

    # =========================
    # Recompute viewBox
    # =========================
    all_pts = np.vstack([pts, mid_ext])
    minx, miny = all_pts.min(axis=0)
    maxx, maxy = all_pts.max(axis=0)

    margin = 0.05 * max(maxx - minx, maxy - miny)
    minx -= margin
    miny -= margin
    width = (maxx - minx) + margin
    height = (maxy - miny) + margin

    out_svg = os.path.join(
        OUT_DIR,
        fname.replace(".svg", "_expanded.svg")
    )

    dwg = svgwrite.Drawing(
        out_svg,
        viewBox=f"{minx} {miny} {width} {height}",
        size=("100%", "100%")
    )

    dwg.add(
        dwg.path(
            d=" ".join(d),
            fill="none",
            stroke="#000000",
            stroke_width=50,
            stroke_linejoin="round",
            stroke_linecap="round"
        )
    )

    dwg.save()
    print("  Output saved:", out_svg)

print("\nAll done")
