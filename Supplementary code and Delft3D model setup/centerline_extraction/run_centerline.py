import os
# Import the improved extractor class
from centerline import CenterlineExtractor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(SCRIPT_DIR, "svg_closed")
output_folder = os.path.join(SCRIPT_DIR, "centerline_results")

# ================= User-selectable settings =================
# Choose one of: "this_paper" (exact paper reproduction), "fixed", "unfixed".
RUN_MODE = "this_paper"

# Used in "fixed" and "unfixed" modes. Any positive integer is allowed.
RESOLUTION = 5000

# Used only in "fixed" mode. Change to any integer to select another
# reproducible medial-axis tie-breaking sequence.
FIXED_RANDOM_SEED = 0

# Paper fixed configuration. Do not alter these values if the goal is to
# reproduce the published results exactly.
THIS_PAPER_RESOLUTION = 3000
THIS_PAPER_RANDOM_SEEDS = {
    "Xiangshangang_expanded.svg": 2,
    "Sanmenwan_expanded.svg": 0,
    "Pubagang_expanded.svg": 1,
    "Yueqingwan_expanded.svg": 0,
    "Shachenggang_expanded.svg": 0,
    "Sanshawan_expanded.svg": 0,
    "Luoyuanwan_expanded.svg": 2,
    "Fuqingwan_expanded.svg": 0,
    "Xinghuawan_expanded.svg": 0,
}

if RUN_MODE == "this_paper":
    resolution = THIS_PAPER_RESOLUTION
    random_seed = 0
    random_seed_by_filename = THIS_PAPER_RANDOM_SEEDS
elif RUN_MODE == "fixed":
    resolution = RESOLUTION
    random_seed = FIXED_RANDOM_SEED
    random_seed_by_filename = None
elif RUN_MODE == "unfixed":
    resolution = RESOLUTION
    random_seed = None
    random_seed_by_filename = None
else:
    raise ValueError("RUN_MODE must be 'this_paper', 'fixed', or 'unfixed'.")

# Create the extractor instance.
# Set output_skeleton=True to generate skeleton visualization SVGs for debugging
# The skeleton SVG shows three layers:
# - Light cyan dots: Full skeleton (all medial axis pixels)
# - Dark purple dots: Pruned skeleton (after distance-based filtering)
# - Red line: Main path (extracted centerline in pixel space)
extractor = CenterlineExtractor(
    input_dir=input_folder,
    output_dir=output_folder,
    resolution=resolution,
    smoothness=0.01,
    gauss_sigma=1.0,
    distance_threshold=2.0,
    spline_smoothness=0.01,
    chaikin_iterations=2,
    verbose=True,
    include_background=True,
    output_skeleton=True,  # Set to True to output skeleton debug SVGs (*_skeleton_debug.svg)
    random_seed=random_seed,
    random_seed_by_filename=random_seed_by_filename,
)

# Batch processing
results = extractor.process_batch()
