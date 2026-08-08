# Tidal-basin centerline extraction

This folder contains the complete workflow for converting nine open tidal-basin outlines into closed polygons, extracting each basin's principal centerline, and calculating the shape index:

\[
S = \mathit{L}^{2} / \mathit{A}
\]

Here, \(\mathit{L}\) and \(\mathit{A}\) are the effective basin length and area after excluding the artificial inlet-extension triangle added during boundary sealing.

## Included files

```text
centerline_extraction/
|- close_outline.py       Stage 1: seal open inlet outlines
|- centerline.py          CenterlineExtractor implementation
|- run_centerline.py      Stage 2: batch centerline extraction and index calculation
|- skeleton_debug.py      Optional skeleton-difference diagnostic
|- svg_open/              Nine input SVG files with open inlets
|- svg_closed/            Nine sealed SVG files used by Stage 2
|- centerline_results/    Generated centerline SVGs, debug SVGs, and CSV summary
`- README.md
```

Standard skeleton diagnostics are generated directly by `centerline.py` when `output_skeleton=True`.

## Dependencies

General use requires Python 3.8 or later plus:

```text
numpy
scipy
scikit-image
svgpathtools
svgwrite
shapely
networkx
pandas
```

Install them with:

```powershell
pip install numpy scipy scikit-image svgpathtools svgwrite shapely networkx pandas
```

For strict reproduction of the `this_paper` preset, the workflow was verified with Python 3.13 and `scikit-image==0.26.0`. The medial-axis implementation can differ between scikit-image releases.

## Run from this folder

Open PowerShell in `centerline_extraction` and use one of the following workflows.

### Reuse the included sealed polygons

This is the normal route for reproducing the centerlines and shape indices:

```powershell
python run_centerline.py
```

It reads `svg_closed/` and writes into `centerline_results/`.

### Regenerate sealed polygons first

Run this only after changing files in `svg_open/`:

```powershell
python close_outline.py
python run_centerline.py
```

`close_outline.py` selects the longest path in each input SVG, samples it at `SAMPLE_N = 600` points, and closes its inlet using a triangle. The extension vertex is placed along the outward normal at ten times the inlet-endpoint distance. It overwrites same-named `*_expanded.svg` files in `svg_closed/`.

## Outputs

Each run of `run_centerline.py` overwrites the corresponding files in `centerline_results/`:

- `*_expanded_centerline.svg`: original polygon outline (when `include_background=True`), inlet baseline, extracted centerline, and centerline-inlet intersection marker.

- `*_expanded_skeleton_debug.svg`: written only when `output_skeleton=True`; contains the gray original outline, green extension triangle, dashed blue inlet baseline, cyan full skeleton, purple pruned skeleton, and red selected main path.

- `centerline_summary.csv`: one row per successfully processed SVG, with `bay_length`, `bay_area`, `total_length`, `total_area`, `extension_length`, `shape_index`, and the centerline SVG filename.

### Optional `skeleton_debug.py`

`skeleton_debug.py` is not required to reproduce the centerlines, shape indices, or this paper's figures. It is a diagnostic-only script for checking whether the branch-pruning threshold removes appropriate skeleton branches.

Run it only when troubleshooting or tuning parameters:

```powershell
python skeleton_debug.py
```

It reads `svg_closed/` and writes `*_expanded_skeleton_diff.svg` files to `centerline_results/`. In these files, cyan pixels were pruned out, purple pixels were retained after pruning, and the red line is the selected main path. Unlike the standard skeleton-debug SVG, the cyan and purple pixel sets do not overlap, so small removed branches are easier to inspect.

This script does not change `centerline_summary.csv` or the standard `*_expanded_centerline.svg` files. Its parameters are currently independent of `run_centerline.py` (it uses `resolution=5000` and has no `this_paper` seed map), so use it only as a visual diagnostic unless its settings are deliberately synchronized with the selected run mode.

## Processing sequence

1. Simplify the polygon boundary with a Douglas-Peucker tolerance proportional to its boundary length.
2. Rasterize the polygon at `resolution x resolution` pixels.
3. Gaussian-smooth the raster, then calculate its medial axis and distance field.
4. Remove skeleton pixels closer than `distance_threshold` to the boundary, unless that would remove too much of the skeleton.
5. Convert the remaining skeleton to an 8-connected weighted graph.
6. Select the longest shortest path between skeleton endpoints as the principal path.
7. Snap external points to the smoothed boundary, simplify the path, apply B-spline and Chaikin smoothing, then remove remaining external points.
8. Deduct the part of the centerline and the area inside the inlet-extension triangle, then calculate \(S = \mathit{L}^{2} / \mathit{A}\).

## Run modes

Set `RUN_MODE` near the top of `run_centerline.py`.

| Mode | Random-tie behavior in `medial_axis` | Resolution |
| :-- | :-- | :-- |
| `"unfixed"` | No seed is passed. Equal-distance ties can lead to different centerlines in different runs. | `RESOLUTION` set by the user |
| `"fixed"` | The integer `FIXED_RANDOM_SEED` is used for all basins. | `RESOLUTION` set by the user |
| `"this_paper"` | Uses this paper's fixed seed for each individual basin. | `THIS_PAPER_RESOLUTION` (3000) |

Examples:

```python
# Default: reproduce this paper's included result
RUN_MODE = "this_paper"

# One user-selected seed for all basins
RUN_MODE = "fixed"
RESOLUTION = 5000
FIXED_RANDOM_SEED = 42

# Deliberately leave medial-axis tie breaking unfixed
RUN_MODE = "unfixed"
RESOLUTION = 5000
```

### What is `FIXED_RANDOM_SEED`?

`FIXED_RANDOM_SEED` is an integer identifier for a repeatable random-number sequence. It is not a count, iteration number, resolution, or number of skeleton points. For example, `42` and `100` select two different but individually repeatable tie-breaking sequences.

The seed is used only when `RUN_MODE = "fixed"`. `medial_axis` uses it only when two or more candidate skeleton pixels are equally valid. Therefore, changing the seed can produce a slightly different centerline and shape index, but using the same seed with the same input files, parameters, and software version gives the same result every time.

Use any non-negative integer, for example:

```python
RUN_MODE = "fixed"
RESOLUTION = 5000
FIXED_RANDOM_SEED = 42  # 0, 42, 100, and 2026 are all valid choices
```

Negative values are invalid. To run without a fixed seed, use `RUN_MODE = "unfixed"`; do not use a negative seed. The `this_paper` mode does not use `FIXED_RANDOM_SEED`, because it applies the paper's per-basin seed map instead.

For the included basin outlines, repeated tests with the `unfixed`, `fixed`, and `this_paper` modes produced only small differences in the calculated shape indices. However, exact centerline geometry and unrounded values can still differ because `medial_axis` resolves equal-distance pixels differently.

Use `RUN_MODE = "this_paper"` when exact reproduction of this paper is required.

## Exact settings for this paper

To reproduce this paper's included centerlines and two-decimal shape indices, do not change:

```python
RUN_MODE = "this_paper"
THIS_PAPER_RESOLUTION = 3000
```

The `this_paper` preset uses per-basin seeds XS=2, PB=1, LY=2, and seed 0 for SM, YQ, SC, SS, FQ, and XH. It also uses these parameters from `run_centerline.py`:

```text
smoothness = 0.01
gauss_sigma = 1.0
distance_threshold = 2.0
spline_smoothness = 0.01
chaikin_iterations = 2
include_background = True
output_skeleton = True
```

With the included files in `svg_closed/`, the resulting shape indices are:

| Basin | XS | SM | PB | YQ | SC | SS | LY | FQ | XH |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| \(S\) (two decimals) | 8.40 | 2.51 | 6.64 | 3.68 | 15.90 | 3.08 | 5.04 | 1.81 | 2.99 |

## Parameters in `CenterlineExtractor`

| Parameter | Default | Effect |
| :-- | --: | :-- |
| `resolution` | 5000 | Raster resolution. Keep it fixed when comparing results. |
| `smoothness` | 0.01 | Boundary-simplification tolerance relative to boundary length. |
| `gauss_sigma` | 1.0 | Gaussian smoothing before skeletonization. |
| `distance_threshold` | 2.0 | Prunes skeleton pixels close to the boundary. |
| `spline_smoothness` | 0.01 | B-spline smoothing factor. |
| `chaikin_iterations` | 2 | Number of Chaikin smoothing iterations. |
| `verbose` | `True` | Prints processing diagnostics to the console. |
| `include_background` | `True` | Includes the original polygon outline in the centerline SVG. |
| `output_skeleton` | `False` | Exports skeleton-debug SVGs. The runner sets it to `True`. |
| `random_seed` | `None` | Optional seed for medial-axis tie breaking. |
| `random_seed_by_filename` | `None` | Per-SVG seed map; overrides `random_seed` for matching filenames. |

The parameters most likely to alter the extracted path are `distance_threshold`, `resolution`, `gauss_sigma`, `smoothness`, and the random-seed configuration. Change one at a time and inspect the debug SVGs before using modified results.

## Basin abbreviations

| Abbreviation | Basin name （Chinese name） | File stem |
| :-- | :-- | :-- |
| XS | Xiangshan Gang (象山港) | `Xiangshangang` |
| SM | Sanmen Wan (三门湾) | `Sanmenwan` |
| PB | Pubagang (浦坝港) | `Pubagang` |
| YQ | Yueqing Wan (乐清湾) | `Yueqingwan` |
| SC | Shacheng Gang (沙埕港) | `Shachenggang` |
| SS | Sansha Wan (三沙湾) | `Sanshawan` |
| LY | Luoyuan Wan (罗源湾) | `Luoyuanwan` |
| FQ | Fuqing Wan (福清湾) | `Fuqingwan` |
| XH | Xinghua Wan (兴化湾) | `Xinghuawan` |

## References

- Blum, H. (1967). A transformation for extracting new descriptors of shape. In W. Wathen-Dunn (Ed.), *Models for the Perception of Speech and Visual Form* (pp. 362-380). MIT Press.

- de Boor, C. (1978). *A Practical Guide to Splines* (Applied Mathematical Sciences, Vol. 27). Springer-Verlag.

- Dijkstra, E. W. (1959). A note on two problems in connexion with graphs. *Numerische Mathematik, 1*(1), 269-271.
