"""
Standalone skeleton-difference visualization.

Unlike the debug SVG produced inside centerline.py (which draws the full
skeleton under the pruned skeleton, so the pruned layer always masks it),
this script draws the SET DIFFERENCE between the two skeletons so that
"removed" and "kept" pixels never overlap and both are clearly visible:

    - Cyan dots  = pixels present in the full skeleton but PRUNED OUT
                   (skel & ~robust_skel) -> the minor branches that were removed
    - Purple dots = pixels KEPT in the pruned skeleton (robust_skel)
    - Red line    = the selected main path (the principal axis)

The heavy lifting (rasterization, medial axis, pruning, shortest-path
selection) is reused from centerline.CenterlineExtractor; only the
drawing is replaced here.
"""

import os
import numpy as np
from svgpathtools import svg2paths
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from centerline import CenterlineExtractor

# =========================
# Configuration paths
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "svg_closed")
OUT_DIR = os.path.join(SCRIPT_DIR, "centerline_results")
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Reuse the same extractor parameters as run_centerline.py
# (output_skeleton=False: we draw our own visualization)
# =========================
extractor = CenterlineExtractor(
    input_dir=INPUT_DIR,
    output_dir=OUT_DIR,
    resolution=5000,
    smoothness=0.01,
    gauss_sigma=1.0,
    distance_threshold=2.0,
    spline_smoothness=0.01,
    chaikin_iterations=2,
    verbose=True,
    include_background=True,
    output_skeleton=False,
)


def parse_sealed_polygon(svg_path):
    """Read a sealed SVG (output of close_outline.py) and return the merged
    polygon, the inlet extension triangle, and the two inlet endpoints."""
    paths, _ = svg2paths(svg_path)
    polys = []
    for p in paths:
        pts = [complex(seg.start) for seg in p]
        pts.append(complex(p[-1].end))
        coords = [(pt.real, pt.imag) for pt in pts]
        if len(coords) >= 3:
            polys.append(Polygon(coords))

    if not polys:
        raise RuntimeError("No valid polygon features found in SVG")

    poly = unary_union(polys)

    # Triangle vertices follow the structure produced by close_outline.py:
    # [start, ..., end, extension_vertex, (closed)]
    coords = list(poly.exterior.coords)
    start_pt = coords[0]
    end_pt = coords[-3]
    mid_ext_pt = coords[-2]
    ext_triangle = Polygon([start_pt, end_pt, mid_ext_pt])

    return poly, ext_triangle, start_pt, end_pt


def create_skeleton_diff_svg(debug_info, poly, ext_triangle, entrance_pts,
                             output_path, distance_threshold):
    """Draw the skeleton as a difference set so pruned and kept pixels
    do not mask each other.

    Layers:
        - light gray fill : original polygon outline (context)
        - light green fill: inlet extension triangle
        - blue dashed line: inlet baseline
        - cyan dots        : PRUNED-OUT pixels  (skel & ~robust_skel)
        - purple dots      : KEPT pixels        (robust_skel)
        - red line         : selected main path
        - legend           : color key
    """
    skel = debug_info['skel']
    robust_skel = debug_info['robust_skel']
    best_path = debug_info['best_path']
    G = debug_info['G']
    minx = debug_info['minx']
    miny = debug_info['miny']
    maxx = debug_info['maxx']
    maxy = debug_info['maxy']
    scale = debug_info['scale']

    def img_to_world(x, y):
        return x / scale + minx, y / scale + miny

    w, h = maxx - minx, maxy - miny
    extent = max(w, h)
    margin = extent * 0.05
    view_box = f"{minx - margin} {miny - margin} {w + 2 * margin} {h + 2 * margin}"

    bg_stroke = max(1.0, extent / 1000.0)
    skel_stroke = max(3.0, extent / 300.0)
    entrance_stroke = max(2.0, extent / 500.0)
    dot_r = max(2.0, extent / 600.0)
    font_size = max(extent / 90.0, 10.0)

    # Set difference: pixels in the full skeleton but removed by pruning.
    # By construction this is DISJOINT from robust_skel, so neither layer
    # can mask the other.
    diff = skel & ~robust_skel

    diff_pts = np.argwhere(diff)        # rows of [y, x]
    kept_pts = np.argwhere(robust_skel)
    # NOTE: count True pixels with .sum(), NOT .size — .size returns the total
    # number of array elements (5000*5000), not the number of skeleton pixels.
    n_pruned = int(diff.sum())
    n_kept = int(robust_skel.sum())
    n_total = int(skel.sum())

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}">\n')
        f.write('<!-- Skeleton difference visualization -->\n')
        f.write('<!-- cyan = pruned-out pixels | purple = kept pixels | red = main path -->\n')

        # 1. Original polygon outline (context)
        f.write('<!-- Original polygon outline -->\n')
        outlines = (poly.geoms if isinstance(poly, MultiPolygon) else [poly])
        for geom in outlines:
            xs, ys = geom.exterior.xy
            f.write('<path d="')
            for i, (x, y) in enumerate(zip(xs, ys)):
                f.write(f'{"M" if i == 0 else "L"}{x},{y} ')
            f.write(f'Z" fill="#f5f5f5" stroke="#999999" stroke-width="{bg_stroke}"/>\n')

        # 2. Extension triangle (guide)
        if ext_triangle is not None:
            f.write('<!-- Extension triangle (guide) -->\n')
            xs, ys = ext_triangle.exterior.xy
            f.write('<path d="')
            for i, (x, y) in enumerate(zip(xs, ys)):
                f.write(f'{"M" if i == 0 else "L"}{x},{y} ')
            f.write(f'Z" fill="#d4f1d4" stroke="#00cc00" stroke-width="{bg_stroke}" opacity="0.6"/>\n')

        # 3. Inlet baseline
        if entrance_pts and len(entrance_pts) == 2:
            f.write('<!-- Inlet baseline -->\n')
            p1, p2 = entrance_pts
            f.write(f'<line x1="{p1[0]}" y1="{p1[1]}" x2="{p2[0]}" y2="{p2[1]}" ')
            f.write(f'stroke="#0066ff" stroke-width="{entrance_stroke}" '
                    f'stroke-dasharray="{entrance_stroke*2},{entrance_stroke}"/>\n')

        # 4. Pruned-out pixels (cyan) -- the removed minor branches
        f.write('<!-- Pruned-out skeleton pixels (cyan) -->\n')
        for (y, x) in diff_pts:
            wx, wy = img_to_world(x, y)
            f.write(f'<circle cx="{wx}" cy="{wy}" r="{dot_r}" fill="#40d9ff" stroke="none"/>\n')

        # 5. Kept skeleton pixels (purple)
        f.write('<!-- Kept skeleton pixels (purple) -->\n')
        for (y, x) in kept_pts:
            wx, wy = img_to_world(x, y)
            f.write(f'<circle cx="{wx}" cy="{wy}" r="{dot_r}" fill="#7722dd" stroke="none"/>\n')

        # 6. Main path (red line)
        if best_path and len(best_path) > 1:
            f.write('<!-- Main path (red line) -->\n')
            world_main = [img_to_world(*G.nodes[node]['pos']) for node in best_path]
            f.write('<polyline points="')
            for x, y in world_main:
                f.write(f'{x},{y} ')
            f.write(f'" fill="none" stroke="#ff3300" stroke-width="{skel_stroke}" '
                    f'stroke-linecap="round" stroke-linejoin="round"/>\n')

        # 7. Legend
        f.write('<!-- Legend -->\n')
        lx = minx
        ly = miny
        lh = font_size * 1.6
        box_w = extent * 0.34
        box_h = lh * 4.6
        f.write(f'<rect x="{lx}" y="{ly}" width="{box_w}" height="{box_h}" '
                f'fill="white" stroke="#888888" stroke-width="{bg_stroke}" opacity="0.9"/>\n')

        legend_rows = [
            ("#40d9ff", f"Pruned-out (removed): {n_pruned} px"),
            ("#7722dd", f"Kept (after pruning): {n_kept} px"),
            ("#ff3300", "Main path (principal axis)"),
            ("#0066ff", f"Inlet baseline (thr={distance_threshold})"),
        ]
        for i, (color, text) in enumerate(legend_rows):
            cy = ly + lh * (i + 1)
            cx = lx + lh * 0.6
            f.write(f'<circle cx="{cx}" cy="{cy}" r="{dot_r}" fill="{color}" stroke="none"/>\n')
            f.write(f'<text x="{cx + lh}" y="{cy + font_size*0.35}" '
                    f'font-size="{font_size}" font-family="sans-serif" fill="#222222">{text}</text>\n')

        f.write('</svg>\n')

    return n_kept, n_pruned, n_total


# =========================
# Main loop
# =========================
svg_files = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".svg"))
if not svg_files:
    print(f"No SVG files found in {INPUT_DIR}")
else:
    print("=" * 60)
    print("Skeleton difference visualization")
    print("=" * 60)
    print(f"Input : {INPUT_DIR}")
    print(f"Output: {OUT_DIR}")
    print(f"distance_threshold = {extractor.distance_threshold}")
    print("=" * 60)

    for fname in svg_files:
        svg_path = os.path.join(INPUT_DIR, fname)
        base_name = os.path.splitext(fname)[0]
        out_path = os.path.join(OUT_DIR, f"{base_name}_skeleton_diff.svg")
        print(f"\nProcessing: {fname}")

        try:
            poly, ext_triangle, start_pt, end_pt = parse_sealed_polygon(svg_path)
            smoothed_poly = extractor.smooth_boundary(poly, extractor.smoothness)

            # Reuse the extractor: rasterize -> medial axis -> prune -> main path.
            # debug_info already holds both skel and robust_skel.
            centerline, debug_info = extractor.calculate_centerline(poly, smoothed_poly)
            if centerline is None or debug_info is None:
                print("  [SKIP] centerline extraction returned None")
                continue

            n_kept, n_pruned, n_total = create_skeleton_diff_svg(
                debug_info, poly, ext_triangle, (start_pt, end_pt),
                out_path, extractor.distance_threshold
            )

            pct = (n_pruned / n_total * 100.0) if n_total else 0.0
            print(f"  skeleton pixels   : {n_total}")
            print(f"  kept (pruned set) : {n_kept} ({100 - pct:.1f}%)")
            print(f"  pruned-out        : {n_pruned} ({pct:.1f}%)")
            print(f"  -> {os.path.basename(out_path)}")

        except Exception as e:
            print(f"  [FAIL] {e}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)
