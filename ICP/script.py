# %% [markdown]
# # Strava Map Art — Shape-Priority ICP with Grid Search (v4)
# 
# Converts hand-drawn sketches into GPS routes on real streets using **Iterative Closest Point (ICP)** alignment with shape-priority scoring, systematic grid search, and multi-scale placement.
# 
# ## What's New over ICP v3
# 
# ### Shape Fidelity Improvements (v4)
# - **Systematic grid search** — instead of only random node sampling, v4 covers the entire map with a grid of candidate positions. This ensures the algorithm finds the best neighborhood for the sketch, especially important for short routes where the optimal placement region is small.
# - **Multi-scale search** — tests multiple scale factors (auto-selected based on route distance) to find placements where street intersections best match the sketch's proportions. Short routes try more scales.
# - **Shape-priority scoring** — blends distance-based and shape-based scoring with an adaptive weight. Short routes (< 5 km) heavily prioritize contour fidelity; long routes balance distance and shape.
# - **Adaptive shape penalty** — the penalty annealing floor scales inversely with route distance, preventing short-route shapes from being "relaxed away" during refinement.
# - **Angle-constrained matching** — after ICP convergence, each matched point is re-evaluated against nearby alternatives that better preserve edge angles with its sketch neighbors.
# - **More rotations** — 24 rotation samples (every 15°) instead of 12, catching orientations that v3 missed.
# 
# ### Inherited from v3
# - Skeleton-based extraction for line drawings (auto-detect)
# - Morphological closing and max-parts filtering for contour mode
# - Density-aware ICP scoring (global search penalizes dense areas)
# - Uniqueness constraint at final matching only
# - Annealed shape penalty
# - Bounding-box guard
# - Route-aware post-matching (re-matches excessive street detours)
# - Open-curve detection
# - Selective edge deduplication (even multiplicity)
# - Street-distance MST bridging (top-3 candidates)
# - Geo-fenced Eulerization with progressive distance penalty
# - Proactive odd-degree reduction by edge doubling
# 
# ## Pipeline
# 1. **Image → Sketch Points**: Auto-detect line drawings (skeleton) vs filled shapes (contour), simplify, detect open/closed
# 2. **Fetch City Map**: Download street network via OpenStreetMap
# 3. **Density Weights**: Grid-based inverse-density weights for candidate scoring
# 4. **ICP Alignment (v4)**: Grid + random search × multi-scale × shape-priority scoring + angle-constrained matching
# 5. **Route-Aware Refinement**: Re-match edges with excessive street detours
# 6. **Route on Streets**: Shortest paths → Art Graph → Street-distance MST → Geo-fenced Eulerization
# 7. **GPX Export**: Continuous traversable route file

# %% [markdown]
# ## 1. Setup and Imports

# %%
import os
import cv2
import numpy as np
from collections import Counter
from skimage.measure import approximate_polygon, find_contours
from skimage.morphology import skeletonize
from scipy.spatial import KDTree, ConvexHull
from scipy.optimize import linear_sum_assignment
import heapq
import networkx as nx
import osmnx as ox
import gpxpy
import gpxpy.gpx
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

# %% [markdown]
# ## 2. Configuration
# 
# Key changes in v4:
# - **Grid search parameters** — `ICP_GRID_SEARCH`, `ICP_GRID_STEP_FACTOR` control systematic map coverage
# - **Multi-scale parameters** — `ICP_SCALE_FACTORS` (auto-selected if None)
# - **Shape priority** — `SHAPE_PRIORITY` auto-scales with route distance; `SHAPE_CONSTRAINED_K` and `SHAPE_ANGLE_WEIGHT` for angle-constrained matching
# - **More search coverage** — `ICP_RANDOM_SAMPLES=300`, `ICP_ROTATION_SAMPLES=24`
# - **Adaptive anneal** — `PENALTY_ANNEAL_END` auto-adjusted for short routes inside the ICP function

# %%
# --- Location & Route ---
CITY_NAME = "Manhattan, New York, USA"
ROUTE_DISTANCE_KM = 10
ROUTE_DISTANCE_METERS = ROUTE_DISTANCE_KM * 1000

# --- Distance Correction Loop (NEW) ---
# Replaces the old static ROUTE_DISTANCE_CORRECTION constant. The correction
# factor now starts at 0.85 and is iteratively adjusted post-hoc based on the
# actual routed distance, until it lies within DISTANCE_CORRECTION_TOLERANCE
# of the target (or DISTANCE_CORRECTION_MAX_ITERS retries are exhausted).
DISTANCE_CORRECTION_INITIAL = 0.85
DISTANCE_CORRECTION_MAX_ITERS = 2
DISTANCE_CORRECTION_TOLERANCE = 0.10

# --- Image ---
IMAGE_FOLDER = "../images"
OUTPUT_FOLDER = "../outputs/ICP_v4"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- ICP Parameters (v4: more coverage) ---
ICP_RANDOM_SAMPLES = 300         # Random starting nodes (was ICP_SAMPLES=200 in v3)
ICP_ROTATION_SAMPLES = 24        # Rotations per position (was 12 in v3)
ICP_SAMPLE_ITERATIONS = 10       # Quick ICP iters per candidate (was 15 in v3; more candidates compensate)
ICP_REFINE_ITERATIONS = 100      # Refinement iters on best (was 80 in v3)
ICP_CONVERGENCE_TOL = 1e-6

# --- Grid Search (NEW in v4) ---
ICP_GRID_SEARCH = True           # Enable systematic grid search over the map
ICP_GRID_STEP_FACTOR = 0.6      # Grid step = sketch_diameter × this factor (smaller = denser grid)

# --- Multi-Scale Search (NEW in v4) ---
ICP_SCALE_FACTORS = None         # None = auto-select based on route distance
                                 # Short routes: [0.75, 0.85, 1.0, 1.15]
                                 # Long routes: [0.9, 1.0]

# --- Shape Priority (NEW in v4) ---
SHAPE_PRIORITY = None            # None = auto (short routes get higher shape priority)
                                 # 0.0 = distance only, 1.0 = shape only
SHAPE_CONSTRAINED_K = 5          # k nearest candidates for angle-constrained matching
SHAPE_ANGLE_WEIGHT = 10.0        # Weight for angular distortion in constrained matching

# --- Shape Preservation ---
SHAPE_PENALTY_WEIGHT = 2.0
SHAPE_ALPHA = 20.0
SHAPE_BETA = 0.3

# --- Density Normalization ---
DENSITY_GRID_SIZE = 50.0

# --- Penalty Annealing ---
PENALTY_ANNEAL_START = 4.0
PENALTY_ANNEAL_END = 0.5         # Base floor; auto-raised for short routes in find_best_fit_v4

# --- Open-Curve Detection ---
OPEN_CURVE_THRESHOLD = 0.15

# --- Route-Aware Post-Matching ---
DETOUR_RATIO_THRESHOLD = 2.0
DETOUR_K_CANDIDATES = 10

# --- Geo-Fenced Eulerization ---
GEO_FENCE_OUTSIDE_PENALTY = 10.0
MAX_ODD_NODES_FOR_CUSTOM_EULER = 100

# --- Skeleton Extraction (from v3) ---
SKELETON_FILL_THRESHOLD = 0.15
SKELETON_TOLERANCE = 5.0
SKELETON_MAX_PARTS = 50
MIN_CONTOUR_LEN = 100

# --- Contour Cleanup (from v3) ---
MORPH_CLOSE_SIZE = 5
MAX_CONTOUR_PARTS = 10

# --- Image Output ---
MAX_IMAGE_PIXELS = 8000

# --- Reproducibility ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def _cap_figure_dpi(target_dpi=200):
    """Cap the current figure's DPI so no rendered dimension exceeds MAX_IMAGE_PIXELS."""
    fig = plt.gcf()
    w, h = fig.get_size_inches()
    max_dim = max(w, h)
    if max_dim > 0:
        capped_dpi = min(target_dpi, MAX_IMAGE_PIXELS / max_dim)
    else:
        capped_dpi = target_dpi
    fig.set_dpi(int(capped_dpi))
    return int(capped_dpi)


def _safe_savefig(path, target_dpi=200, **kwargs):
    """Save current figure, capping DPI so no dimension exceeds MAX_IMAGE_PIXELS."""
    dpi = _cap_figure_dpi(target_dpi)
    plt.gcf().savefig(path, dpi=dpi, **kwargs)

# %% [markdown]
# ## 3. Image Processing — Skeleton + Contour Dual-Mode Extraction
# 
# Extracts sketch topology from images using two modes:
# 
# ### Skeleton Mode (line drawings)
# For images with low foreground fill ratio (< 15%), like stick figures:
# 1. Threshold → binary image
# 2. `skeletonize()` → 1-pixel-wide medial axis
# 3. Classify pixels: endpoints (1 neighbor), junctions (3+ neighbors), path pixels (2 neighbors)
# 4. Trace paths between key points along 8-connected skeleton
# 5. Detect pure cycles (e.g., head circles with no junctions)
# 6. Simplify each path with `approximate_polygon(tolerance=5.0)`
# 
# This correctly captures branching topology (body→arms, body→legs) that contour extraction misses.
# 
# ### Contour Mode (filled shapes)
# For images with higher fill ratio, like cat, face, pi:
# 1. Threshold → binary with morphological closing (fills gaps, merges fragments)
# 2. `find_contours()` → boundary traces
# 3. Filter by `min_contour_len` (100) and keep top N by arc length (`max_parts=10`)
# 4. Simplify with `approximate_polygon(tolerance=20.0)`
# 5. Auto-detect open vs closed curves
# 
# Both modes return the same format: `(P_sketch, E_sketch, polygon_parts, is_closed)`.

# %%
def _get_skeleton_graph(img_binary, tolerance=5.0):
    """
    Extract sketch topology from a line drawing using skeletonization.

    Thins the binary image to a 1-pixel skeleton, then traces paths between
    junction/endpoint pixels to produce a graph of simplified polylines.

    Handles:
    - Branching structures (e.g., stickman body → arms/legs)
    - Pure cycles with no junctions (e.g., a head circle)
    - Multiple disconnected components

    Returns same format as contour extraction:
        P_sketch, E_sketch, polygon_parts, is_closed
    """
    skeleton = skeletonize(img_binary > 0)
    ys, xs = np.nonzero(skeleton)
    if len(ys) == 0:
        raise ValueError("Skeleton is empty after skeletonization.")

    # Build pixel coordinate set and adjacency
    pixel_set = set(zip(ys.tolist(), xs.tolist()))
    neighbors_8 = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1)]

    def get_neighbors(p):
        r, c = p
        return [(r + dr, c + dc) for dr, dc in neighbors_8 if (r + dr, c + dc) in pixel_set]

    # Classify pixels
    pixel_nbrs = {p: get_neighbors(p) for p in pixel_set}
    key_points = set()  # endpoints and junctions
    for p, nbrs in pixel_nbrs.items():
        deg = len(nbrs)
        if deg != 2:
            key_points.add(p)  # endpoint (1) or junction (3+)

    # Trace a path from start_pixel in direction of next_pixel until hitting
    # another key point or revisiting start (cycle)
    def trace_path(start, next_px, visited_edges):
        path = [start, next_px]
        edge = (min(start, next_px), max(start, next_px))
        visited_edges.add(edge)
        current = next_px

        while current not in key_points:
            nbrs = pixel_nbrs[current]
            moved = False
            for nb in nbrs:
                e = (min(current, nb), max(current, nb))
                if e not in visited_edges:
                    visited_edges.add(e)
                    path.append(nb)
                    current = nb
                    moved = True
                    break
            if not moved:
                break  # dead end or all edges visited

        return path

    # Trace all paths originating from key points
    visited_edges = set()
    raw_paths = []

    for kp in key_points:
        for nb in pixel_nbrs[kp]:
            edge = (min(kp, nb), max(kp, nb))
            if edge not in visited_edges:
                path = trace_path(kp, nb, visited_edges)
                if len(path) >= 2:
                    raw_paths.append(path)

    # Detect pure cycles: connected components of degree-2 pixels with no key points
    # These are loops like the head circle of a stickman
    visited_cycle_pixels = set()
    for p in pixel_set:
        if p in key_points or p in visited_cycle_pixels:
            continue
        # Check if this pixel's edges are all visited
        all_visited = all(
            (min(p, nb), max(p, nb)) in visited_edges for nb in pixel_nbrs[p]
        )
        if all_visited:
            visited_cycle_pixels.add(p)
            continue

        # Try to trace a cycle from this degree-2 pixel
        if len(pixel_nbrs[p]) != 2:
            continue

        cycle = [p]
        visited_cycle_pixels.add(p)
        current = p
        # Pick first unvisited neighbor direction
        first_nb = None
        for nb in pixel_nbrs[current]:
            e = (min(current, nb), max(current, nb))
            if e not in visited_edges:
                first_nb = nb
                break
        if first_nb is None:
            continue

        visited_edges.add((min(current, first_nb), max(current, first_nb)))
        cycle.append(first_nb)
        visited_cycle_pixels.add(first_nb)
        current = first_nb

        while current != p:
            nbrs = pixel_nbrs[current]
            moved = False
            for nb in nbrs:
                e = (min(current, nb), max(current, nb))
                if e not in visited_edges:
                    visited_edges.add(e)
                    cycle.append(nb)
                    visited_cycle_pixels.add(nb)
                    current = nb
                    moved = True
                    break
            if not moved:
                break

        if len(cycle) >= 3 and current == p:
            raw_paths.append(cycle)  # closed cycle — last point == first point conceptually

    if not raw_paths:
        raise ValueError("No paths found in skeleton.")

    # Convert pixel paths to simplified point arrays
    all_points, all_edges, polygon_parts, is_closed_list = [], [], [], []
    current_idx = 0

    for path in raw_paths:
        # Convert (row, col) to (x, y) = (col, row)
        coords = np.array([(c, r) for r, c in path], dtype=float)

        # Check if this is a closed cycle
        is_cycle = (path[0] == path[-1]) or (
            len(path) > 2 and path[-1] in [nb for nb in pixel_nbrs.get(path[0], [])]
            and path[0] not in key_points and path[-1] not in key_points
        )
        # Also check the pure cycle case: trace returned to start
        if len(path) > 3 and np.linalg.norm(coords[0] - coords[-1]) <= np.sqrt(2):
            is_cycle = True

        # Simplify
        simplified = approximate_polygon(coords, tolerance=tolerance)
        if len(simplified) < 2:
            continue

        # Remove duplicate last point if cycle
        if is_cycle and np.array_equal(simplified[0], simplified[-1]):
            simplified = simplified[:-1]
        if len(simplified) < 2:
            continue

        n = len(simplified)
        all_points.append(simplified)

        if is_cycle:
            edges = np.array([[current_idx + i, current_idx + (i + 1) % n] for i in range(n)])
        else:
            edges = np.array([[current_idx + i, current_idx + i + 1] for i in range(n - 1)])

        all_edges.append(edges)
        polygon_parts.append(np.arange(current_idx, current_idx + n))
        is_closed_list.append(is_cycle)
        current_idx += n

    if not all_points:
        raise ValueError("No valid paths after simplification.")

    return np.vstack(all_points), np.vstack(all_edges), polygon_parts, is_closed_list


def _extract_contours(img_thresh, tolerance, min_contour_len, open_curve_threshold,
                      close_size, max_parts):
    """
    Contour-mode extraction helper. Runs morphological closing, finds contours,
    filters, simplifies, and returns (P, E, parts, is_closed).
    Skips morphological closing when few contours exist (preserves sharp corners).
    """
    # First pass: find contours WITHOUT closing to check complexity
    raw_contours = find_contours(img_thresh, level=0.8)
    significant_raw = [c for c in raw_contours if len(c) > min_contour_len]

    # Only apply morphological closing if there are many contours (complex image)
    # For simple shapes (square, triangle), closing rounds corners unnecessarily
    if close_size > 0 and len(significant_raw) > 3:
        kernel = np.ones((close_size, close_size), np.uint8)
        img_closed = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
        contours = find_contours(img_closed, level=0.8)
        significant_contours = [c for c in contours if len(c) > min_contour_len]
        if not significant_contours:
            significant_contours = significant_raw
        else:
            print(f"  Applied morphological closing (kernel={close_size})")
    else:
        significant_contours = significant_raw

    if not significant_contours:
        raise ValueError("No significant contours found.")

    # Sort by arc length (descending) and keep only top max_parts
    def arc_length(contour):
        return sum(np.linalg.norm(contour[i+1] - contour[i])
                   for i in range(len(contour) - 1))

    significant_contours.sort(key=arc_length, reverse=True)
    if max_parts > 0 and len(significant_contours) > max_parts:
        print(f"  Keeping top {max_parts} of {len(significant_contours)} contours by arc length")
        significant_contours = significant_contours[:max_parts]

    all_points, all_edges, polygon_parts, is_closed = [], [], [], []
    current_idx = 0

    for contour in significant_contours:
        polygon = approximate_polygon(contour, tolerance=tolerance)
        if np.array_equal(polygon[0], polygon[-1]):
            polygon = polygon[:-1]

        n = len(polygon)
        if n < 2:
            continue
        points = polygon[:, ::-1]  # (row, col) → (x, y)

        perimeter = sum(np.linalg.norm(points[(i+1) % n] - points[i]) for i in range(n))
        gap = np.linalg.norm(points[-1] - points[0])
        closed = (gap / max(perimeter, 1e-12)) <= open_curve_threshold

        all_points.append(points)

        if closed:
            edges = np.array([[current_idx + i, current_idx + (i + 1) % n] for i in range(n)])
        else:
            edges = np.array([[current_idx + i, current_idx + i + 1] for i in range(n - 1)])

        all_edges.append(edges)
        polygon_parts.append(np.arange(current_idx, current_idx + n))
        is_closed.append(closed)
        current_idx += n

    if not all_points:
        raise ValueError("No valid contours after processing.")

    return np.vstack(all_points), np.vstack(all_edges), polygon_parts, is_closed


def get_sketch_points_and_edges(image_path, tolerance=20.0, min_contour_len=100,
                                open_curve_threshold=0.15,
                                skeleton_fill_threshold=0.15,
                                skeleton_tolerance=5.0,
                                skeleton_max_parts=50,
                                close_size=5, max_parts=10):
    """
    Processes an image to extract simplified sketch topology.

    Auto-detects extraction mode:
    - Skeleton mode (line drawings): fill_ratio < skeleton_fill_threshold
    - Falls back to contour mode if skeleton produces > skeleton_max_parts
    - Contour mode (filled shapes): fill_ratio >= skeleton_fill_threshold

    Returns:
        P_sketch, E_sketch, polygon_parts, is_closed, mode
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")

    img = cv2.imread(image_path, 0)
    if img is None:
        raise ValueError("Could not read the image.")

    _, img_thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # Auto-detect mode based on foreground fill ratio
    fill_ratio = np.sum(img_thresh > 0) / img_thresh.size
    print(f"  Fill ratio: {fill_ratio:.3f}", end="")

    if fill_ratio < skeleton_fill_threshold:
        # --- TRY SKELETON MODE ---
        print(f" < {skeleton_fill_threshold} → trying skeleton mode...")
        try:
            P, E, parts, closed = _get_skeleton_graph(img_thresh, tolerance=skeleton_tolerance)
            if len(parts) <= skeleton_max_parts:
                print(f"  Skeleton OK: {len(parts)} parts, {len(P)} points")
                return P, E, parts, closed, 'skeleton'
            else:
                print(f"  Skeleton produced {len(parts)} parts (> {skeleton_max_parts}) → falling back to contour mode")
        except ValueError as e:
            print(f"  Skeleton failed ({e}) → falling back to contour mode")

    else:
        print(f" >= {skeleton_fill_threshold} → contour mode")

    # --- CONTOUR MODE ---
    P, E, parts, closed = _extract_contours(
        img_thresh, tolerance, min_contour_len, open_curve_threshold,
        close_size, max_parts)
    return P, E, parts, closed, 'contour'


# --- Quick visualization ---
if os.path.exists(IMAGE_FOLDER):
    for image_file in sorted(os.listdir(IMAGE_FOLDER)):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(IMAGE_FOLDER, image_file)
            try:
                P, E, parts, closed, mode = get_sketch_points_and_edges(
                    image_path,
                    open_curve_threshold=OPEN_CURVE_THRESHOLD,
                    skeleton_fill_threshold=SKELETON_FILL_THRESHOLD,
                    skeleton_tolerance=SKELETON_TOLERANCE,
                    skeleton_max_parts=SKELETON_MAX_PARTS,
                    min_contour_len=MIN_CONTOUR_LEN,
                    close_size=MORPH_CLOSE_SIZE,
                    max_parts=MAX_CONTOUR_PARTS)
                plt.figure(figsize=(5, 5))
                plt.imshow(cv2.imread(image_path, 0), cmap='gray')
                for i, idx in enumerate(parts):
                    pts = P[idx]
                    status = "closed" if closed[i] else "open"
                    plt.plot(pts[:, 0], pts[:, 1], '-o', markersize=3,
                             label=f'Part {i+1} ({status})')
                plt.title(f"{image_file} [{mode}]: {len(P)} pts, {len(parts)} parts")
                plt.legend(fontsize=8)
                plt.show()
            except (FileNotFoundError, ValueError) as e:
                print(f"  Skip {image_file}: {e}")

# %% [markdown]
# ## 4. Density Weights for Scoring
# 
# Compute per-node inverse-density weights. These are used **only for scoring** during the global search — NOT inside the SVD transform. This prevents ICP from preferring dense areas without corrupting the rotation/translation math.

# %%
def compute_density_weights(Q_city, grid_size=50.0):
    """
    Compute per-node inverse-density weights using a spatial grid.
    
    Nodes in dense areas (many nodes per grid cell) get LOW weight.
    Nodes in sparse areas (few nodes per grid cell) get HIGH weight.
    
    Used ONLY for scoring candidates — NOT in the ICP SVD step.
    
    Args:
        Q_city: (m, 2) array of projected city node coordinates.
        grid_size: Size of each grid cell in meters.
        
    Returns:
        weights: (m,) array of normalized weights summing to 1.
    """
    city_min = Q_city.min(axis=0)
    city_max = Q_city.max(axis=0)
    grid_dims = np.ceil((city_max - city_min) / grid_size).astype(int)
    
    # Assign each node to a grid cell
    grid_indices = ((Q_city - city_min) / grid_size).astype(int)
    grid_indices = np.clip(grid_indices, 0, grid_dims - 1)
    
    # Count how many nodes fall in each cell
    cell_keys = [tuple(g) for g in grid_indices]
    cell_counts = Counter(cell_keys)
    
    # Weight = 1 / count (inverse density)
    weights = np.array([1.0 / cell_counts[k] for k in cell_keys])
    weights /= weights.sum()  # normalize to sum to 1
    
    # Stats
    max_density = max(cell_counts.values())
    min_density = min(cell_counts.values())
    n_cells = len(cell_counts)
    print(f"  Density grid: {grid_dims[0]}x{grid_dims[1]} cells, {n_cells} occupied")
    print(f"  Nodes per cell: min={min_density}, max={max_density}, "
          f"median={np.median(list(cell_counts.values())):.0f}")
    print(f"  Weight range: {weights.min():.6f} - {weights.max():.6f}")
    
    return weights

# %% [markdown]
# ## 5. Shape-Priority ICP Alignment (v4)
# 
# The core algorithm, redesigned for short-route shape fidelity.
# 
# ### Grid + Random search (NEW)
# Instead of only sampling random city nodes, v4 systematically covers the map with a grid of candidate positions (spacing = sketch diameter × 0.6). This ensures every neighborhood is tested — critical for short routes where the "good" placement region is small relative to the city.
# 
# ### Multi-scale search (NEW)
# Instead of fixing the scale from route distance alone, v4 tests multiple scale factors. Street intersections don't align perfectly at any single scale, so trying 3–4 sizes finds placements where nodes naturally fall on the sketch's contour. Short routes (< 5 km) search [0.75, 0.85, 1.0, 1.15]; long routes use [0.9, 1.0].
# 
# ### Shape-priority scoring (NEW)
# Scoring blends distance-based and shape-based components with an adaptive `shape_priority` weight. For short routes, shape preservation dominates (priority ~0.7); for long routes, distance balances equally (priority ~0.3). The shape score includes both angular distortion and edge-length proportionality (variance of length ratios).
# 
# ### Angle-constrained final matching (NEW)
# After ICP converges and uniqueness is enforced, each matched point is re-evaluated: for k nearest city-node alternatives, the one that best preserves edge angles with sketch neighbors is chosen (weighted by `SHAPE_ANGLE_WEIGHT`).
# 
# ### Adaptive penalty annealing (NEW)
# The penalty anneal floor auto-scales inversely with route distance: short routes keep a higher minimum shape penalty (floor ≈ 1.5), preventing the refinement phase from "relaxing" the shape.
# 
# ### Inherited from v3
# - Standard unweighted ICP transforms (Procrustes SVD)
# - Density-weighted scoring (NOT in SVD)
# - Uniqueness only at final matching
# - Bounding-box guard

# %%
def resample_sketch_edges(P_sketch, E_sketch, max_spacing_ratio=0.05):
    """
    Add intermediate points along sketch edges that are longer than
    max_spacing_ratio × sketch_diameter. This gives ICP more constraints
    along straight edges, preventing ballooning/pinching between corners.

    Returns:
        P_new: (n_new, 2) resampled points
        E_new: (k_new, 2) resampled edges
        original_indices: (n_original,) indices into P_new for original points
    """
    P_sketch = np.asarray(P_sketch, dtype=float)
    E_sketch = np.asarray(E_sketch)
    if len(P_sketch) == 0:
        return P_sketch.copy(), E_sketch.copy(), np.arange(len(P_sketch))

    diameter = float(np.max(np.ptp(P_sketch, axis=0))) if len(P_sketch) > 1 else 0.0
    threshold = max_spacing_ratio * diameter
    if threshold < 1e-6:
        return P_sketch.copy(), E_sketch.copy(), np.arange(len(P_sketch))

    new_points = [pt for pt in P_sketch]
    original_indices = list(range(len(P_sketch)))
    new_edges = []

    for p1, p2 in E_sketch:
        p1 = int(p1)
        p2 = int(p2)
        dist = float(np.linalg.norm(P_sketch[p2] - P_sketch[p1]))
        if dist > threshold:
            n_segments = int(np.ceil(dist / threshold))
            interp = np.linspace(P_sketch[p1], P_sketch[p2], n_segments + 1)
            start_idx = len(new_points)
            for pt in interp[1:-1]:
                new_points.append(pt)
            indices_chain = (
                [p1]
                + list(range(start_idx, start_idx + n_segments - 1))
                + [p2]
            )
            for i in range(len(indices_chain) - 1):
                new_edges.append([indices_chain[i], indices_chain[i + 1]])
        else:
            new_edges.append([p1, p2])

    return np.array(new_points), np.array(new_edges), np.array(original_indices)


def find_best_fit_v4(P_sketch, E_sketch, Q_city, route_distance_meters,
                     density_weights,
                     num_iterations=100, num_random_samples=300,
                     sample_iterations=10, num_rotation_samples=24,
                     convergence_tol=1e-6,
                     penalty_weight=2.0, alpha=20.0, beta=0.3,
                     distance_correction=0.85,
                     penalty_anneal_start=4.0, penalty_anneal_end=0.5,
                     grid_search=True, grid_step_factor=0.6,
                     scale_factors=None, shape_priority=None,
                     shape_constrained_k=5, shape_angle_weight=10.0,
                     G_proj=None, node_ids=None):
    """
    Shape-priority ICP v4: grid+random search, multi-scale, shape-first scoring,
    angle-constrained matching.

    Key improvements over v3's find_best_fit_v2:
    - Systematic grid search covers every map neighborhood
    - Multi-scale search finds best physical size for the sketch
    - Shape-priority scoring weights contour fidelity adaptively
    - Angle-constrained final matching preserves edge angles
    - Adaptive penalty anneal floor for short routes

    Args:
        P_sketch: (n, 2) sketch node coordinates.
        E_sketch: (k, 2) edge index pairs.
        Q_city: (m, 2) city intersection coordinates (projected).
        route_distance_meters: Target total route length.
        density_weights: (m,) per-city-node inverse-density weights.
        num_iterations: Refinement iterations on the best candidate.
        num_random_samples: Random starting positions for search.
        sample_iterations: Quick ICP iterations per candidate.
        num_rotation_samples: Rotations tested per position.
        convergence_tol: Early-stop threshold (relative error change).
        penalty_weight: Base shape penalty weight.
        alpha: Angular distortion weight in shape penalty.
        beta: Length distortion weight in shape penalty.
        distance_correction: Scale correction factor.
        penalty_anneal_start: Shape penalty weight at start of refinement.
        penalty_anneal_end: Shape penalty weight base floor.
        grid_search: If True, add systematic grid positions.
        grid_step_factor: Grid step = sketch_diameter × this factor.
        scale_factors: List of scale multipliers to try, or None for auto.
        shape_priority: Shape vs distance blend (0-1), or None for auto.
        shape_constrained_k: k nearest alternatives for angle-constrained matching.
        shape_angle_weight: Weight for angular distortion in post-matching.
        G_proj: Optional projected city graph (NetworkX) for street-distance
            queries during top-N candidate re-scoring and topology-aware
            refinement. If None, those steps are skipped.
        node_ids: Optional (m,) array of OSM node IDs aligned with Q_city.
            Required alongside G_proj.

    Returns:
        R: (2, 2) rotation matrix.
        t: (2,) translation vector.
        s: Scale factor applied to sketch.
        matched_indices: Indices of matched points in Q_city, restricted to
            the original (un-resampled) sketch points.
    """
    # Resample sketch edges for denser ICP constraints. The closure functions
    # below operate on the resampled P_sketch/E_sketch; we map matches back to
    # the original points before returning.
    P_sketch_orig = np.asarray(P_sketch)
    E_sketch_orig = np.asarray(E_sketch)
    P_sketch, E_sketch, original_point_indices = resample_sketch_edges(
        P_sketch_orig, E_sketch_orig)
    if len(P_sketch) > len(P_sketch_orig):
        print(f"  Resampled: {len(P_sketch_orig)} -> {len(P_sketch)} points, "
              f"{len(E_sketch_orig)} -> {len(E_sketch)} edges")

    n_sketch = len(P_sketch)
    route_km = route_distance_meters / 1000.0

    # --- Auto-tune parameters based on route distance ---
    if shape_priority is None:
        shape_priority = float(np.clip(1.0 - route_km / 15.0, 0.3, 0.8))

    if scale_factors is None:
        if route_km < 5:
            scale_factors = [0.75, 0.85, 1.0, 1.15]
        elif route_km < 10:
            scale_factors = [0.85, 1.0, 1.1]
        else:
            scale_factors = [0.9, 1.0]

    # Adaptive penalty floor: short routes keep higher shape penalty
    effective_anneal_end = max(penalty_anneal_end, 2.0 - route_km / 10.0)

    # Shape multiplier: higher for short routes to boost shape penalty in scoring
    shape_multiplier = float(np.clip(4.0 - route_km / 5.0, 1.0, 4.0))

    def shape_preservation_penalty(indices, R, t, s):
        """Penalizes angular and length distortion between aligned sketch edges
        and the corresponding matched city-point edges."""
        P_aligned = (R @ (P_sketch * s).T).T + t
        penalty = 0.0
        for p1, p2 in E_sketch:
            v_sketch = P_aligned[p2] - P_aligned[p1]
            v_city = Q_city[indices[p2]] - Q_city[indices[p1]]
            norm_s = np.linalg.norm(v_sketch)
            norm_c = np.linalg.norm(v_city)
            if norm_s == 0 or norm_c == 0:
                continue
            cos_theta = np.clip(np.dot(v_sketch, v_city) / (norm_s * norm_c), -1.0, 1.0)
            penalty += alpha * np.arccos(cos_theta) + beta * abs(norm_s - norm_c)
        return penalty

    def shape_fidelity_score(indices, R, t, s):
        """
        Measures how well matched points preserve the sketch's angular structure
        and proportional edge lengths.
        Returns: mean angular error (radians) + edge-length ratio variance.
        Lower = better shape preservation.
        """
        P_aligned = (R @ (P_sketch * s).T).T + t
        angle_errors = []
        length_ratios = []

        for p1, p2 in E_sketch:
            v_sketch = P_aligned[p2] - P_aligned[p1]
            v_city = Q_city[indices[p2]] - Q_city[indices[p1]]
            norm_s = np.linalg.norm(v_sketch)
            norm_c = np.linalg.norm(v_city)
            if norm_s < 1e-6 or norm_c < 1e-6:
                continue
            cos_theta = np.clip(np.dot(v_sketch, v_city) / (norm_s * norm_c), -1.0, 1.0)
            angle_errors.append(np.arccos(cos_theta))
            length_ratios.append(norm_c / norm_s)

        if not angle_errors:
            return 0.0

        mean_angle_error = np.mean(angle_errors)
        ratio_variance = np.var(length_ratios) if len(length_ratios) > 1 else 0.0
        return mean_angle_error + ratio_variance

    def enforce_uniqueness(P_transformed, indices, city_kdtree):
        """At final matching: ensure each city node is used at most once.
        Uses Hungarian algorithm for optimal one-to-one assignment."""
        n = len(P_transformed)
        if n == 0:
            return indices.copy()

        k = min(30, len(Q_city))  # consider top-30 nearest neighbors
        dists, knn = city_kdtree.query(P_transformed, k=k)
        # Ensure 2D shape even when k == 1
        if knn.ndim == 1:
            knn = knn[:, None]
            dists = dists[:, None]

        # Collect all candidate city nodes
        all_candidates = set()
        for i in range(n):
            for j in range(k):
                all_candidates.add(int(knn[i, j]))
        all_candidates = sorted(all_candidates)
        cand_to_col = {c: j for j, c in enumerate(all_candidates)}

        # Build cost matrix: n_sketch × n_candidates
        cost_matrix = np.full((n, len(all_candidates)), 1e12)
        for i in range(n):
            for j in range(k):
                city_idx = int(knn[i, j])
                col = cand_to_col[city_idx]
                cost_matrix[i, col] = dists[i, j]

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        unique_indices = indices.copy()
        for r, c in zip(row_ind, col_ind):
            unique_indices[r] = all_candidates[c]
        return unique_indices

    def angle_constrained_refinement(P_transformed, indices, city_kdtree,
                                     k=5, angle_weight=10.0):
        """
        Post-ICP refinement: for each sketch point, check if a nearby city node
        would better preserve the edge angles with its sketch neighbors.
        Uses a combined score of distance + angle_weight × angular_distortion.
        """
        refined = indices.copy()
        used = set(int(c) for c in refined)
        adj = {i: [] for i in range(n_sketch)}
        for p1, p2 in E_sketch:
            adj[int(p1)].append(int(p2))
            adj[int(p2)].append(int(p1))

        improvements = 0
        for i in range(n_sketch):
            if not adj[i]:
                continue

            current_idx = int(refined[i])
            current_pos = Q_city[current_idx]

            def angular_cost(city_idx):
                pos = Q_city[city_idx]
                cost = 0.0
                for nb in adj[i]:
                    v_sketch = P_transformed[nb] - P_transformed[i]
                    v_city = Q_city[refined[nb]] - pos
                    ns = np.linalg.norm(v_sketch)
                    nc = np.linalg.norm(v_city)
                    if ns < 1e-6 or nc < 1e-6:
                        continue
                    cos_t = np.clip(np.dot(v_sketch, v_city) / (ns * nc), -1.0, 1.0)
                    cost += np.arccos(cos_t)
                return cost

            current_cost = angular_cost(current_idx)
            current_dist = np.linalg.norm(P_transformed[i] - current_pos)
            current_total = current_dist + angle_weight * current_cost

            _, candidates = city_kdtree.query(P_transformed[i], k=k)
            best_total = current_total
            best_idx = current_idx

            for cand in np.atleast_1d(candidates):
                cand = int(cand)
                if cand == current_idx:
                    continue
                if cand in used and cand != current_idx:
                    continue  # already claimed by another sketch point
                cand_dist = np.linalg.norm(P_transformed[i] - Q_city[cand])
                cand_cost = angular_cost(cand)
                cand_total = cand_dist + angle_weight * cand_cost
                if cand_total < best_total:
                    best_total = cand_total
                    best_idx = cand

            if best_idx != current_idx:
                used.discard(current_idx)
                used.add(best_idx)
                refined[i] = best_idx
                improvements += 1

        return refined, improvements

    # --- Compute base scale ---
    total_pixel_length = sum(np.linalg.norm(P_sketch[p1] - P_sketch[p2])
                             for p1, p2 in E_sketch)
    if total_pixel_length == 0:
        raise ValueError("Sketch has zero total edge length.")
    base_s = (route_distance_meters * distance_correction) / total_pixel_length

    # --- KDTree ---
    city_kdtree = KDTree(Q_city)

    # --- City bounding box (for guard) ---
    city_min = Q_city.min(axis=0)
    city_max = Q_city.max(axis=0)
    city_margin = 0.05 * (city_max - city_min)

    # =====================================================================
    # GENERATE CANDIDATE STARTING POSITIONS
    # Grid-based (systematic) + random node sampling
    # =====================================================================
    starting_positions = []

    # Random node samples
    random_indices = np.random.choice(
        len(Q_city), size=min(num_random_samples, len(Q_city)), replace=False)
    for idx in random_indices:
        starting_positions.append(Q_city[idx])

    # Grid-based search: cover the map systematically
    n_grid = 0
    if grid_search:
        P_scaled_base = P_sketch * base_s
        sketch_extent = P_scaled_base.max(axis=0) - P_scaled_base.min(axis=0)
        grid_step = max(sketch_extent.max() * grid_step_factor, 100)
        x_range = np.arange(city_min[0], city_max[0] + grid_step, grid_step)
        y_range = np.arange(city_min[1], city_max[1] + grid_step, grid_step)
        for gx in x_range:
            for gy in y_range:
                starting_positions.append(np.array([gx, gy]))
        n_grid = len(x_range) * len(y_range)
        print(f"  Grid search: {len(x_range)}×{len(y_range)} = {n_grid} grid points "
              f"(step={grid_step:.0f}m)")

    n_positions = len(starting_positions)
    total_candidates = n_positions * num_rotation_samples * len(scale_factors)
    print(f"Shape-priority search: {n_positions} positions "
          f"({num_random_samples} random + {n_grid} grid) "
          f"× {num_rotation_samples} rotations × {len(scale_factors)} scales "
          f"= {total_candidates} candidates")
    print(f"  Shape priority: {shape_priority:.2f} | Shape multiplier: {shape_multiplier:.1f} | "
          f"Route: {route_km:.1f} km | Scales: {scale_factors} | "
          f"Anneal floor: {effective_anneal_end:.2f}")

    # =====================================================================
    # GLOBAL SEARCH PHASE: Grid + Random × Multi-Scale × Shape-Priority Scoring
    # =====================================================================
    best_error = float('inf')
    best_R, best_t, best_s = np.identity(2), np.zeros(2), base_s
    report_every = max(1, n_positions // 5)

    # Top-K candidate tracking (max-heap of size <= top_k_candidates).
    # Stored as (-total_error, counter, R, t, s) so heap[0] is the worst-of-best.
    top_k_candidates = 10
    candidate_heap = []
    candidate_counter = 0

    for sf in scale_factors:
        s = base_s * sf
        P_scaled = P_sketch * s
        c_p = np.mean(P_scaled, axis=0)
        P_centered = P_scaled - c_p
        sketch_diam = (float(np.max(np.ptp(P_scaled, axis=0)))
                       if len(P_scaled) > 1 else 1.0)
        euler_penalty_weight = 0.5  # tunable

        for i, start_pos in enumerate(starting_positions):
            if (i + 1) % report_every == 0:
                print(f"  [scale {sf:.2f}] [{i+1}/{n_positions}] best error: {best_error:.1f}")

            for angle in np.linspace(0, 2 * np.pi, num_rotation_samples, endpoint=False):
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                R_sample = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                t_sample = start_pos - R_sample @ c_p

                # Quick ICP iterations (standard unweighted)
                prev_error = float('inf')
                for _ in range(sample_iterations):
                    P_transformed = (R_sample @ P_scaled.T).T + t_sample
                    distances, indices = city_kdtree.query(P_transformed)
                    Q_matched = Q_city[indices]
                    c_q = np.mean(Q_matched, axis=0)
                    Q_c = Q_matched - c_q

                    W = Q_c.T @ P_centered
                    U, _, Vt = np.linalg.svd(W)
                    R_new = U @ Vt
                    if np.linalg.det(R_new) < 0:
                        Vt[1, :] *= -1
                        R_new = U @ Vt

                    t_new = c_q - R_new @ c_p
                    R_sample, t_sample = R_new, t_new

                    cur_error = np.sum(distances ** 2)
                    if abs(prev_error - cur_error) / max(prev_error, 1e-12) < convergence_tol:
                        break
                    prev_error = cur_error

                # --- SHAPE-PRIORITY SCORING ---
                P_final = (R_sample @ P_scaled.T).T + t_sample
                distances, indices = city_kdtree.query(P_final)

                # Distance component (density-weighted)
                n_city = len(Q_city)
                density_penalty = np.sum(
                    distances ** 2 / (density_weights[indices] * n_city + 1e-12))

                # Shape components
                penalty = shape_preservation_penalty(indices, R_sample, t_sample, s)
                fidelity = shape_fidelity_score(indices, R_sample, t_sample, s)

                # Estimate Eulerization overhead from odd-degree count of the
                # matched-node sub-multigraph induced by the sketch edges.
                degree_count = Counter()
                for p1, p2 in E_sketch:
                    degree_count[int(indices[int(p1)])] += 1
                    degree_count[int(indices[int(p2)])] += 1
                n_odd = sum(1 for d in degree_count.values() if d % 2 == 1)
                # Each odd-degree pair needs ~1 backtrack edge; estimate avg
                # backtrack length as sketch_diameter * 0.3.
                euler_overhead_estimate = (n_odd / 2) * sketch_diam * 0.3

                # Combined score: shape_multiplier boosts penalty for short routes
                # fidelity captures proportionality (edge-length ratio variance)
                effective_pw = penalty_weight * shape_multiplier
                total_error = (density_penalty
                               + effective_pw * penalty
                               + shape_priority * 500 * fidelity
                               + euler_penalty_weight * euler_overhead_estimate)

                if total_error < best_error:
                    best_error = total_error
                    best_R, best_t, best_s = R_sample.copy(), t_sample.copy(), s

                # Maintain top-K candidates for street-distance re-scoring.
                candidate_counter += 1
                entry = (-total_error, candidate_counter,
                         R_sample.copy(), t_sample.copy(), s)
                if len(candidate_heap) < top_k_candidates:
                    heapq.heappush(candidate_heap, entry)
                elif -total_error > candidate_heap[0][0]:
                    # New error is smaller than worst-of-best: replace.
                    heapq.heapreplace(candidate_heap, entry)

    print(f"Search complete (best error: {best_error:.1f}). "
          f"Refining with adaptive shape penalty...")

    # =====================================================================
    # STREET-DISTANCE RE-SCORING for top candidates
    # =====================================================================
    if (G_proj is not None and node_ids is not None and len(candidate_heap) > 0
            and route_distance_meters > 0):
        # Sort ascending by total_error for stable iteration order.
        top_candidates = sorted(candidate_heap, key=lambda x: -x[0])
        print(f"  Re-scoring top {len(top_candidates)} candidates with street distances...")
        best_street_score = float('inf')
        sb_R, sb_t, sb_s = best_R, best_t, best_s
        for rank, (neg_err, _cnt, R_cand, t_cand, s_cand) in enumerate(top_candidates):
            err = -neg_err
            P_cand = (R_cand @ (P_sketch * s_cand).T).T + t_cand
            _, cand_indices = city_kdtree.query(P_cand)
            total_street_dist = 0.0
            for p1, p2 in E_sketch:
                u_idx = int(cand_indices[int(p1)])
                v_idx = int(cand_indices[int(p2)])
                u_id = node_ids[u_idx]
                v_id = node_ids[v_idx]
                try:
                    sd = nx.shortest_path_length(G_proj, u_id, v_id, weight='length')
                except nx.NetworkXNoPath:
                    euclid = np.linalg.norm(Q_city[u_idx] - Q_city[v_idx])
                    sd = euclid * 3  # heavy penalty
                total_street_dist += sd
            street_deviation = abs(total_street_dist - route_distance_meters) / route_distance_meters
            combined = err * 0.5 + street_deviation * route_distance_meters * 0.5
            if combined < best_street_score:
                best_street_score = combined
                sb_R, sb_t, sb_s = R_cand, t_cand, s_cand
        if (sb_R is not best_R or sb_t is not best_t or sb_s != best_s):
            print(f"  Street-distance rescoring promoted a different candidate "
                  f"(scale {sb_s:.4f}).")
        best_R, best_t, best_s = sb_R, sb_t, sb_s

    # =====================================================================
    # FINE-GRAINED SCALE SWEEP around the chosen candidate
    # =====================================================================
    print(f"  Fine-tuning scale around {best_s:.4f}...")
    fine_scales = np.linspace(best_s * 0.90, best_s * 1.10, 11)  # ±10% in 2% steps
    fine_best_error = best_error
    fine_best_R, fine_best_t, fine_best_s = best_R, best_t, best_s

    # Centroid of the sketch at best_s, used to adjust translation when
    # changing scale.
    P_best_scaled = P_sketch * best_s
    c_p_best = np.mean(P_best_scaled, axis=0)

    for fs in fine_scales:
        P_fine = P_sketch * fs
        c_p_fine = np.mean(P_fine, axis=0)
        P_centered_fine = P_fine - c_p_fine

        # Use best_R and best_t as starting point, adjusted for new scale.
        R_fine = best_R.copy()
        t_fine = best_t + best_R @ (c_p_best - c_p_fine)

        # Run a few ICP iterations.
        for _ in range(sample_iterations):
            P_transformed = (R_fine @ P_fine.T).T + t_fine
            distances, indices = city_kdtree.query(P_transformed)
            Q_matched = Q_city[indices]
            c_q = np.mean(Q_matched, axis=0)
            Q_c = Q_matched - c_q
            W = Q_c.T @ P_centered_fine
            U, _, Vt = np.linalg.svd(W)
            R_new = U @ Vt
            if np.linalg.det(R_new) < 0:
                Vt[1, :] *= -1
                R_new = U @ Vt
            t_new = c_q - R_new @ c_p_fine
            R_fine, t_fine = R_new, t_new

        # Score (same composition as global-search scoring, minus euler estimate
        # which is comparatively expensive and not needed for fine-tuning).
        P_final_fine = (R_fine @ P_fine.T).T + t_fine
        distances, indices = city_kdtree.query(P_final_fine)
        n_city = len(Q_city)
        dp = np.sum(distances ** 2 / (density_weights[indices] * n_city + 1e-12))
        pen = shape_preservation_penalty(indices, R_fine, t_fine, fs)
        fid = shape_fidelity_score(indices, R_fine, t_fine, fs)
        effective_pw = penalty_weight * shape_multiplier
        err = dp + effective_pw * pen + shape_priority * 500 * fid

        if err < fine_best_error:
            fine_best_error = err
            fine_best_R, fine_best_t, fine_best_s = R_fine, t_fine, fs

    if fine_best_s != best_s:
        print(f"  Scale refined: {best_s:.4f} -> {fine_best_s:.4f}")
    best_R, best_t, best_s = fine_best_R, fine_best_t, fine_best_s
    best_error = fine_best_error

    # =====================================================================
    # LOCAL REFINEMENT PHASE
    # Standard unweighted ICP with adaptive annealed shape penalty
    # =====================================================================
    R, t, s = best_R, best_t, best_s
    P_scaled = P_sketch * s
    c_p = np.mean(P_scaled, axis=0)
    P_centered = P_scaled - c_p
    prev_error = float('inf')

    for it in range(num_iterations):
        progress = it / max(num_iterations - 1, 1)
        current_penalty_w = (penalty_anneal_start
                             + (effective_anneal_end - penalty_anneal_start) * progress)

        P_transformed = (R @ P_scaled.T).T + t
        distances, indices = city_kdtree.query(P_transformed)
        Q_matched = Q_city[indices]

        c_q = np.mean(Q_matched, axis=0)
        Q_c = Q_matched - c_q
        W = Q_c.T @ P_centered
        U, _, Vt = np.linalg.svd(W)
        R_new = U @ Vt
        if np.linalg.det(R_new) < 0:
            Vt[1, :] *= -1
            R_new = U @ Vt

        t_new = c_q - R_new @ c_p

        # Bounding-box guard
        P_candidate = (R_new @ P_scaled.T).T + t_new
        centroid_candidate = np.mean(P_candidate, axis=0)
        if not (np.all(centroid_candidate >= city_min - city_margin) and
                np.all(centroid_candidate <= city_max + city_margin)):
            continue

        R, t = R_new, t_new

        dist_error = np.sum(distances ** 2)
        penalty = shape_preservation_penalty(indices, R, t, s)

        # Topology-aware penalty: discourages matches whose street distance is
        # much larger than their euclidean distance. Expensive, so only every
        # 10 iterations.
        topo_penalty = 0.0
        if G_proj is not None and node_ids is not None and it % 10 == 0:
            for p1, p2 in E_sketch:
                u_idx = int(indices[int(p1)])
                v_idx = int(indices[int(p2)])
                u_id = node_ids[u_idx]
                v_id = node_ids[v_idx]
                euclid = np.linalg.norm(Q_city[u_idx] - Q_city[v_idx])
                if euclid < 1e-6:
                    continue
                try:
                    street_d = nx.shortest_path_length(
                        G_proj, u_id, v_id, weight='length')
                    if street_d > 3 * euclid:
                        topo_penalty += (street_d - euclid)
                except nx.NetworkXNoPath:
                    topo_penalty += euclid * 10  # heavy penalty

        cur_error = dist_error + current_penalty_w * penalty + 0.1 * topo_penalty

        if abs(prev_error - cur_error) / max(prev_error, 1e-12) < convergence_tol:
            print(f"  Converged at iteration {it+1} (penalty_w={current_penalty_w:.2f})")
            break
        prev_error = cur_error

    # =====================================================================
    # FINAL MATCHING: uniqueness + angle-constrained refinement
    # =====================================================================
    P_final = (R @ P_scaled.T).T + t
    _, raw_indices = city_kdtree.query(P_final)
    final_indices = enforce_uniqueness(P_final, raw_indices, city_kdtree)

    # Angle-constrained refinement (NEW in v4)
    final_indices, n_angle_refined = angle_constrained_refinement(
        P_final, final_indices, city_kdtree,
        k=shape_constrained_k, angle_weight=shape_angle_weight)

    n_unique = np.unique(final_indices).size
    n_collisions = n_sketch - n_unique
    print(f"  Final: {n_unique}/{n_sketch} unique city nodes matched"
          + (f" ({n_collisions} reassigned)" if n_collisions > 0 else ""))
    if n_angle_refined > 0:
        print(f"  Angle-constrained refinement improved {n_angle_refined} matches")

    # Map back to original (un-resampled) sketch points only.
    final_indices_original = final_indices[original_point_indices]
    return R, t, s, final_indices_original

# %% [markdown]
# ## 6. Route-Aware Post-Matching Refinement
# 
# After ICP alignment, some sketch edges may map to city node pairs that are geometrically close but far apart on the street network (e.g., on opposite sides of a park, across a river). These create long detour paths that appear as trailing segments.
# 
# This step detects such edges (where `street_distance > 2× euclidean_distance`) and re-matches their endpoints to nearby city nodes that have shorter street connections.

# %%
def refine_matches_for_routability(G_proj, E_sketch, matched_node_ids, Q_city,
                                   node_ids, city_kdtree, P_final,
                                   detour_ratio=2.0, k_candidates=10):
    """
    Re-matches sketch edge endpoints that have excessive street detours.

    For each sketch edge (u, v): if the shortest street path between the
    matched city nodes is > detour_ratio × their Euclidean distance,
    search nearby city nodes for a pair with a shorter street connection.

    Args:
        G_proj: Projected city graph (for shortest path queries).
        E_sketch: (k, 2) array of edge index pairs.
        matched_node_ids: (n,) array of OSM node IDs matched to sketch points.
        Q_city: (m, 2) array of city node coordinates (projected).
        node_ids: (m,) array of OSM node IDs for Q_city.
        city_kdtree: KDTree of Q_city for fast neighbor queries.
        P_final: (n, 2) array of aligned sketch point positions (projected).
        detour_ratio: Threshold ratio; re-match if street_dist > this × euclidean_dist.
        k_candidates: Number of nearest city nodes to consider per endpoint.

    Returns:
        refined_node_ids: (n,) array of refined matched OSM node IDs.
        n_refined: Number of edges that were re-matched.
    """
    refined = matched_node_ids.copy()
    n_refined = 0
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # Build adjacency: which sketch edges touch each sketch point.
    point_edges = {}
    for edge_idx, (p1, p2) in enumerate(E_sketch):
        point_edges.setdefault(int(p1), []).append(edge_idx)
        point_edges.setdefault(int(p2), []).append(edge_idx)

    # Precompute candidate city nodes for each sketch point.
    n_sketch = len(P_final)
    all_candidates = {}
    for i in range(n_sketch):
        _, cands = city_kdtree.query(P_final[i], k=k_candidates)
        all_candidates[i] = np.atleast_1d(cands)

    # Precompute street distances between OSM node pairs (cache).
    street_dist_cache = {}

    def cached_street_dist(u_id, v_id):
        if u_id == v_id:
            return 0.0
        key = (u_id, v_id) if u_id < v_id else (v_id, u_id)
        if key not in street_dist_cache:
            try:
                street_dist_cache[key] = nx.shortest_path_length(
                    G_proj, u_id, v_id, weight='length')
            except nx.NetworkXNoPath:
                street_dist_cache[key] = float('inf')
        return street_dist_cache[key]

    def adjacent_cost(point, candidate_id, partner_point=None):
        """Sum street distances for all edges incident to `point` in the
        current refined matching, treating `candidate_id` as the (hypothetical)
        match for `point`. The edge to `partner_point` is excluded — it is
        added separately by the caller."""
        total = 0.0
        for adj_edge_idx in point_edges.get(point, []):
            ep1, ep2 = E_sketch[adj_edge_idx]
            other_point = int(ep2) if int(ep1) == point else int(ep1)
            if other_point == partner_point:
                continue
            other_id = refined[other_point]
            total += cached_street_dist(candidate_id, other_id)
        return total

    for p1, p2 in E_sketch:
        p1 = int(p1)
        p2 = int(p2)
        u_id = refined[p1]
        v_id = refined[p2]

        if u_id == v_id:
            continue

        u_idx = node_id_to_idx.get(u_id)
        v_idx = node_id_to_idx.get(v_id)
        if u_idx is None or v_idx is None:
            continue

        euclid_dist = np.linalg.norm(Q_city[u_idx] - Q_city[v_idx])
        if euclid_dist < 1e-6:
            continue

        street_dist = cached_street_dist(u_id, v_id)
        if street_dist == float('inf'):
            continue

        if street_dist <= detour_ratio * euclid_dist:
            continue  # acceptable detour

        u_candidates = all_candidates[p1]
        v_candidates = all_candidates[p2]

        # Baseline: cost of the current matching, summed over the current
        # edge plus all other edges incident to p1 and p2.
        current_total = (street_dist
                         + adjacent_cost(p1, u_id, partner_point=p2)
                         + adjacent_cost(p2, v_id, partner_point=p1))
        best_total = current_total
        best_u_id, best_v_id = u_id, v_id

        for u_cand_idx in u_candidates:
            u_cand_idx = int(u_cand_idx)
            u_cand_id = node_ids[u_cand_idx]
            cand_dist_u = np.linalg.norm(Q_city[u_cand_idx] - P_final[p1])
            if cand_dist_u > euclid_dist:
                continue

            u_adj = adjacent_cost(p1, u_cand_id, partner_point=p2)
            if u_adj == float('inf'):
                continue

            for v_cand_idx in v_candidates:
                v_cand_idx = int(v_cand_idx)
                v_cand_id = node_ids[v_cand_idx]
                if u_cand_id == v_cand_id:
                    continue

                cand_dist_v = np.linalg.norm(Q_city[v_cand_idx] - P_final[p2])
                if cand_dist_v > euclid_dist:
                    continue

                edge_cost = cached_street_dist(u_cand_id, v_cand_id)
                if edge_cost == float('inf'):
                    continue

                v_adj = adjacent_cost(p2, v_cand_id, partner_point=p1)
                if v_adj == float('inf'):
                    continue

                cand_total = edge_cost + u_adj + v_adj
                if cand_total < best_total:
                    best_total = cand_total
                    best_u_id, best_v_id = u_cand_id, v_cand_id

        if best_u_id != u_id or best_v_id != v_id:
            refined[p1] = best_u_id
            refined[p2] = best_v_id
            n_refined += 1

    return refined, n_refined

# %% [markdown]
# ## 7. Improved Route Generation & GPX Export
# 
# Builds an "art graph" from shortest paths along sketch edges, connects disconnected components, makes it Eulerian, and exports a continuous GPX route.
# 
# **v2 Improvements:**
# - **Selective edge deduplication** — allows up to 2 copies of each edge (even multiplicity reduces odd-degree nodes), collapses 3+ copies to 2
# - **Street-distance MST bridging** — evaluates top-3 nearest Euclidean pairs per component, picks shortest street path
# - **Open-curve support** — uses `is_closed` flag; open curves don't get forced closure edges
# - **Geo-fenced Eulerization** — custom min-weight matching penalizes paths leaving the sketch convex hull, keeping backtrack inside the drawing
# - **Diagnostic overlay** — separate colors for art edges vs Eulerization-added edges

# %%
def reduce_odd_degree_by_doubling(G_art, G_proj):
    """
    Greedily double edges where BOTH endpoints are odd-degree.
    Each doubling flips both endpoints from odd to even, reducing the number
    of odd-degree nodes without adding new routing beyond existing art edges.
    """
    total_doubled = 0
    changed = True
    while changed:
        changed = False
        odd_nodes = set(n for n in G_art.nodes if G_art.degree(n) % 2 == 1)
        candidates = []
        seen = set()
        for u in odd_nodes:
            for v in G_art.neighbors(u):
                if v in odd_nodes and v != u:
                    ek = (min(u, v), max(u, v))
                    if ek not in seen:
                        seen.add(ek)
                        edge_data = G_proj.get_edge_data(u, v)
                        length = 0
                        if edge_data:
                            first_key = next(iter(edge_data))
                            length = edge_data[first_key].get('length', 0)
                        candidates.append((u, v, length, edge_data))
        candidates.sort(key=lambda x: x[2])
        for u, v, length, edge_data in candidates:
            if G_art.degree(u) % 2 == 1 and G_art.degree(v) % 2 == 1:
                if edge_data:
                    first_key = next(iter(edge_data))
                    G_art.add_edge(u, v, **edge_data[first_key])
                    total_doubled += 1
                    changed = True
    return total_doubled


def eulerize_bounded(G_art, G_proj, art_edge_set=None,
                     outside_penalty=5.0, max_odd_nodes=100):
    """
    Geo-fenced Eulerization with progressive distance penalty.
    Pairs odd-degree nodes using min-weight matching, penalizing paths
    that leave the sketch footprint and penalizing long pairings.

    If `art_edge_set` is provided, paths that reuse art edges receive a
    multiplicative discount, encouraging the matcher to backtrack along the
    sketch itself rather than introduce new strokes.
    """
    odd_nodes = [n for n in G_art.nodes if G_art.degree(n) % 2 == 1]
    print(f"  Odd-degree nodes: {len(odd_nodes)}")

    if len(odd_nodes) == 0:
        return G_art.copy(), set()

    if len(odd_nodes) > max_odd_nodes:
        print(f"  Too many odd nodes ({len(odd_nodes)}), falling back to nx.eulerize()")
        G_eul = nx.eulerize(G_art)
        added = set()
        for u, v, k in G_eul.edges(keys=True):
            if not G_art.has_edge(u, v):
                added.add((u, v))
        return G_eul, added

    art_coords = np.array([[G_proj.nodes[n]['x'], G_proj.nodes[n]['y']]
                           for n in G_art.nodes if n in G_proj.nodes])

    sketch_diameter = np.max(np.ptp(art_coords, axis=0)) if len(art_coords) > 1 else 1.0

    if len(art_coords) >= 3:
        try:
            hull = ConvexHull(art_coords)
            hull_path = MplPath(art_coords[hull.vertices])
        except Exception:
            hull_path = None
    else:
        hull_path = None

    odd_graph = nx.Graph()
    odd_paths = {}

    for i, u in enumerate(odd_nodes):
        for j, v in enumerate(odd_nodes):
            if j <= i:
                continue
            try:
                path = nx.shortest_path(G_proj, u, v, weight='length')
                path_len = sum(
                    G_proj[path[k]][path[k+1]][0].get('length', 0)
                    for k in range(len(path) - 1)
                )
            except (nx.NetworkXNoPath, KeyError):
                continue

            weight = path_len
            if hull_path is not None and len(path) > 2:
                mid_coords = np.array([[G_proj.nodes[n]['x'], G_proj.nodes[n]['y']]
                                       for n in path[1:-1]])
                inside = hull_path.contains_points(mid_coords)
                frac_outside = 1.0 - (np.sum(inside) / len(inside))
                weight = path_len * (1.0 + outside_penalty * frac_outside)

            # Progressive penalty: longer pairings relative to sketch size cost more
            if sketch_diameter > 0:
                relative_length = path_len / sketch_diameter
                weight *= (1.0 + relative_length)

            # Shape-aware discount: paths that reuse art edges are cheaper
            # (each overlapping edge gives a 30% multiplicative discount).
            if art_edge_set is not None and len(path) > 1:
                art_overlap_count = sum(
                    1 for k in range(len(path) - 1)
                    if (min(path[k], path[k + 1]),
                        max(path[k], path[k + 1])) in art_edge_set
                )
                if art_overlap_count > 0:
                    overlap_discount = 0.7 ** art_overlap_count
                    weight *= overlap_discount

            odd_graph.add_edge(u, v, weight=weight, path=path)
            odd_paths[(u, v)] = path
            odd_paths[(v, u)] = list(reversed(path))

    matching = nx.min_weight_matching(odd_graph, weight='weight')

    G_eulerian = G_art.copy()
    added_edges = set()

    for u, v in matching:
        path = odd_paths.get((u, v))
        if path is None:
            continue
        for k in range(len(path) - 1):
            n1, n2 = path[k], path[k+1]
            if not G_eulerian.has_node(n1):
                G_eulerian.add_node(n1, **G_proj.nodes[n1])
            if not G_eulerian.has_node(n2):
                G_eulerian.add_node(n2, **G_proj.nodes[n2])
            edge_data = G_proj.get_edge_data(n1, n2)
            if edge_data:
                first_key = next(iter(edge_data))
                G_eulerian.add_edge(n1, n2, **edge_data[first_key])
            added_edges.add((min(n1, n2), max(n1, n2)))

    remaining_odd = [n for n in G_eulerian.nodes if G_eulerian.degree(n) % 2 == 1]
    if remaining_odd:
        print(f"  Custom matching left {len(remaining_odd)} odd nodes, falling back to nx.eulerize()")
        G_eulerian = nx.eulerize(G_art)
        added_edges = set()
        for u, v, k in G_eulerian.edges(keys=True):
            if not G_art.has_edge(u, v):
                added_edges.add((min(u, v), max(u, v)))

    print(f"  Eulerization added {len(added_edges)} edge segments")
    return G_eulerian, added_edges


def create_route_and_gpx(G_proj, G_unproj, matched_node_ids, polygon_parts,
                         is_closed, output_filename="route.gpx",
                         outside_penalty=5.0, max_odd_nodes=100):
    """
    Builds an Eulerian art-graph from matched nodes and exports a GPX file.
    v2: selective dedup, street-distance MST, open-curve support,
    proactive odd-degree reduction, geo-fenced Eulerization, zoomed diagnostic overlay.
    """
    if not polygon_parts:
        print("No polygon parts to process.")
        return None

    G_art = nx.MultiGraph()
    G_art.graph = G_proj.graph.copy()
    edge_counts = Counter()
    start_node = None

    def add_path_to_art(path):
        for node_id in path:
            if not G_art.has_node(node_id):
                G_art.add_node(node_id, **G_proj.nodes[node_id])
        for j in range(len(path) - 1):
            n1, n2 = path[j], path[j + 1]
            edge_key = (min(n1, n2), max(n1, n2))
            if edge_counts[edge_key] < 2:
                edge_counts[edge_key] += 1
                edge_data = G_proj.get_edge_data(n1, n2)
                if edge_data:
                    first_key = next(iter(edge_data))
                    G_art.add_edge(n1, n2, **edge_data[first_key])

    # 1. Shortest paths for each polygon part (respecting open/closed)
    part_routes = []
    for part_idx_num, part_indices in enumerate(polygon_parts):
        part_nodes = matched_node_ids[part_indices]
        if start_node is None and len(part_nodes) > 0:
            start_node = part_nodes[0]

        closed = is_closed[part_idx_num] if part_idx_num < len(is_closed) else True
        n_nodes = len(part_nodes)
        route_segment = []
        edge_range = n_nodes if closed else (n_nodes - 1)

        for i in range(edge_range):
            u = part_nodes[i]
            v = part_nodes[(i + 1) % n_nodes]
            try:
                path = nx.shortest_path(G_proj, u, v, weight='length')
                route_segment.extend(path[:-1])
                add_path_to_art(path)
            except nx.NetworkXNoPath:
                print(f"  Warning: No path between {u} and {v}")

        if route_segment:
            part_routes.append(list(dict.fromkeys(route_segment)))

    if not part_routes or start_node is None:
        print("Could not generate any route segments.")
        return None

    # 2. Connect multiple parts via street-distance MST (top-3 candidates)
    if len(part_routes) > 1:
        component_graph = nx.Graph()
        for ci, cj in combinations(range(len(part_routes)), 2):
            nodes_i = list(set(part_routes[ci]))
            nodes_j = list(set(part_routes[cj]))
            coords_i = np.array([[G_proj.nodes[n]['x'], G_proj.nodes[n]['y']]
                                 for n in nodes_i])
            coords_j = np.array([[G_proj.nodes[n]['x'], G_proj.nodes[n]['y']]
                                 for n in nodes_j])

            kdt = KDTree(coords_i)
            dists, idxs = kdt.query(coords_j, k=1)
            sorted_j = np.argsort(dists.ravel())[:3]

            best_street_len = float('inf')
            best_path = None

            for rank in sorted_j:
                j_idx = rank
                i_idx = idxs.ravel()[rank]
                pair = (nodes_i[i_idx], nodes_j[j_idx])
                try:
                    path = nx.shortest_path(G_proj, source=pair[0],
                                            target=pair[1], weight='length')
                    path_len = sum(G_proj[u][v][0]['length']
                                   for u, v in zip(path[:-1], path[1:]))
                    if path_len < best_street_len:
                        best_street_len = path_len
                        best_path = path
                except (nx.NetworkXNoPath, KeyError):
                    continue

            if best_path is not None:
                component_graph.add_edge(ci, cj, weight=best_street_len,
                                         path=best_path)

        if component_graph.edges and nx.is_connected(component_graph):
            mst = nx.minimum_spanning_tree(component_graph)
            for _, _, data in mst.edges(data=True):
                add_path_to_art(data['path'])

    # 3. Force connectivity with KDTree (fallback)
    if not nx.is_connected(G_art):
        print("  Forcing connectivity via KDTree nearest-pair...")
        components = list(nx.connected_components(G_art))
        main_nodes = list(components[0])
        main_coords = np.array([[G_proj.nodes[n]['x'], G_proj.nodes[n]['y']]
                                for n in main_nodes])
        main_kdt = KDTree(main_coords)

        for comp in components[1:]:
            comp_nodes = list(comp)
            comp_coords = np.array([[G_proj.nodes[n]['x'], G_proj.nodes[n]['y']]
                                    for n in comp_nodes])
            dists, idxs = main_kdt.query(comp_coords, k=1)
            best_comp = np.argmin(dists)
            best_main = idxs[best_comp]
            try:
                path = nx.shortest_path(G_proj, main_nodes[best_main],
                                        comp_nodes[best_comp], weight='length')
                add_path_to_art(path)
            except nx.NetworkXNoPath:
                print(f"  Warning: Could not connect component of size {len(comp)}")

    # 3.5 Proactive odd-degree reduction: double edges where both ends are odd
    if G_art.nodes and nx.is_connected(G_art):
        n_doubled = reduce_odd_degree_by_doubling(G_art, G_proj)
        if n_doubled:
            print(f"  Proactive doubling: {n_doubled} edges doubled to reduce odd-degree nodes")

    # 4. Geo-fenced Eulerization
    if not G_art.nodes or not nx.is_connected(G_art):
        print("Art graph is empty or disconnected. Cannot create route.")
        return None

    art_edge_set = set()
    for u, v, _ in G_art.edges(keys=True):
        art_edge_set.add((min(u, v), max(u, v)))

    G_eulerian, euler_added_edges = eulerize_bounded(
        G_art, G_proj, art_edge_set=art_edge_set,
        outside_penalty=outside_penalty,
        max_odd_nodes=max_odd_nodes)

    circuit = list(nx.eulerian_circuit(G_eulerian, source=start_node))
    route_nodes = [start_node] + [v for _, v in circuit]

    # 5. GPX export
    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(track)
    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)

    for node_id in route_nodes:
        data = G_unproj.nodes[node_id]
        segment.points.append(gpxpy.gpx.GPXTrackPoint(
            latitude=data['y'], longitude=data['x']))

    with open(output_filename, 'w') as f:
        f.write(gpx.to_xml())

    # Compute stats
    total_km = sum(
        G_proj[u][v][0].get('length', 0) if G_proj.has_edge(u, v) else 0
        for u, v in circuit
    ) / 1000
    euler_km = sum(
        G_proj[u][v][0].get('length', 0) if G_proj.has_edge(u, v) else 0
        for u, v in circuit
        if (min(u, v), max(u, v)) in euler_added_edges
    ) / 1000

    print(f"  GPX saved: {output_filename}")
    print(f"  Route: {len(route_nodes)} points, ~{total_km:.1f} km total")
    print(f"  Art edges: {len(art_edge_set)}, Eulerization added: {len(euler_added_edges)} "
          f"(~{euler_km:.1f} km)")

    # 6. Diagnostic overlay zoomed to route bounding box
    try:
        fig, ax = ox.plot_graph(G_proj, node_size=0, edge_color='#333333',
                                edge_linewidth=0.5, show=False, close=False,
                                bgcolor='k')

        for u, v, _ in G_art.edges(keys=True):
            x_coords = [G_proj.nodes[u]['x'], G_proj.nodes[v]['x']]
            y_coords = [G_proj.nodes[u]['y'], G_proj.nodes[v]['y']]
            ax.plot(x_coords, y_coords, color='#00ff00', linewidth=2, alpha=0.7)

        for u, v in euler_added_edges:
            if u in G_proj.nodes and v in G_proj.nodes:
                x_coords = [G_proj.nodes[u]['x'], G_proj.nodes[v]['x']]
                y_coords = [G_proj.nodes[u]['y'], G_proj.nodes[v]['y']]
                ax.plot(x_coords, y_coords, color='red', linewidth=2.5, alpha=0.9)

        # Zoom to route bounding box
        all_route_nodes = set(G_eulerian.nodes)
        route_xs = [G_proj.nodes[n]['x'] for n in all_route_nodes if n in G_proj.nodes]
        route_ys = [G_proj.nodes[n]['y'] for n in all_route_nodes if n in G_proj.nodes]
        if route_xs and route_ys:
            x_min, x_max = min(route_xs), max(route_xs)
            y_min, y_max = min(route_ys), max(route_ys)
            x_pad = max((x_max - x_min) * 0.15, 50)
            y_pad = max((y_max - y_min) * 0.15, 50)
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)

        plt.title(f"Route: {os.path.basename(output_filename)} "
                  f"({total_km:.1f} km, green=art, red=euler)")
        route_img_path = output_filename.replace('.gpx', '_route.png')
        _safe_savefig(route_img_path, bbox_inches='tight')
        print(f"  Route image saved: {route_img_path}")
        plt.show()
    except Exception as e:
        print(f"  Could not plot: {e}")

    return total_km


# %% [markdown]
# ## 8. Run — Fetch City Data, Shape-Priority Align All Sketches, Generate GPX

# %%
# --- Fetch city data once ---
print(f"Fetching street network for {CITY_NAME}...")
try:
    G_city = ox.graph_from_place(CITY_NAME, network_type='all')
    G_city_proj = ox.project_graph(G_city)

    nodes_proj = G_city_proj.nodes(data=True)
    Q_city = np.array([[data['x'], data['y']] for _, data in nodes_proj])
    node_ids = np.array([nid for nid, _ in G_city_proj.nodes(data=True)])
    print(f"  Loaded {len(Q_city)} intersections, {G_city_proj.number_of_edges()} edges")

    # Compute density weights
    print("Computing density weights...")
    density_weights = compute_density_weights(Q_city, grid_size=DENSITY_GRID_SIZE)

    # Build KDTree once for route-aware refinement
    city_kdtree = KDTree(Q_city)

    city_loaded = True
except Exception as e:
    print(f"  Failed: {e}")
    city_loaded = False

# --- Process each image ---
if city_loaded and os.path.exists(IMAGE_FOLDER):
    for image_file in sorted(os.listdir(IMAGE_FOLDER)):
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {image_file} (ICP v4 — shape-priority)")
        print(f"{'='*60}")

        image_path = os.path.join(IMAGE_FOLDER, image_file)
        try:
            P_sketch, E_sketch, polygon_parts, is_closed, extraction_mode = \
                get_sketch_points_and_edges(
                    image_path,
                    open_curve_threshold=OPEN_CURVE_THRESHOLD,
                    skeleton_fill_threshold=SKELETON_FILL_THRESHOLD,
                    skeleton_tolerance=SKELETON_TOLERANCE,
                    min_contour_len=MIN_CONTOUR_LEN,
                    close_size=MORPH_CLOSE_SIZE,
                    max_parts=MAX_CONTOUR_PARTS)
            n_open = sum(1 for c in is_closed if not c)
            n_closed = sum(1 for c in is_closed if c)
            print(f"  Mode: {extraction_mode} | Parts: {n_closed} closed, {n_open} open | Points: {len(P_sketch)}")
        except (FileNotFoundError, ValueError) as e:
            print(f"  Skipping: {e}")
            continue

        # =====================================================================
        # Post-hoc distance correction loop: run alignment + routing, measure
        # the actual routed distance, and adjust the scale-correction factor
        # until the route is within DISTANCE_CORRECTION_TOLERANCE of the target
        # (or DISTANCE_CORRECTION_MAX_ITERS retries are exhausted).
        # =====================================================================
        distance_correction = DISTANCE_CORRECTION_INITIAL
        target_km = ROUTE_DISTANCE_KM
        total_km = None
        output_path = os.path.join(
            OUTPUT_FOLDER, f"v4_route_{os.path.splitext(image_file)[0]}.gpx")

        for correction_iter in range(DISTANCE_CORRECTION_MAX_ITERS + 1):
            if correction_iter > 0:
                if total_km is None:
                    print(f"  Distance correction iter {correction_iter}: previous "
                          f"iteration produced no route, aborting correction loop.")
                    break
                actual_km = total_km
                if abs(actual_km - target_km) / target_km <= DISTANCE_CORRECTION_TOLERANCE:
                    print(f"  Distance within tolerance "
                          f"({actual_km:.1f} km vs {target_km:.1f} km target)")
                    break
                correction_factor = target_km / actual_km
                distance_correction *= correction_factor
                print(f"  Distance correction iter {correction_iter}: "
                      f"{actual_km:.1f} km -> target {target_km:.1f} km, "
                      f"new correction={distance_correction:.3f}")

            # v4 shape-priority ICP alignment
            R, t, s, matched_indices = find_best_fit_v4(
                P_sketch, E_sketch, Q_city, ROUTE_DISTANCE_METERS,
                density_weights=density_weights,
                num_iterations=ICP_REFINE_ITERATIONS,
                num_random_samples=ICP_RANDOM_SAMPLES,
                sample_iterations=ICP_SAMPLE_ITERATIONS,
                num_rotation_samples=ICP_ROTATION_SAMPLES,
                convergence_tol=ICP_CONVERGENCE_TOL,
                penalty_weight=SHAPE_PENALTY_WEIGHT,
                alpha=SHAPE_ALPHA,
                beta=SHAPE_BETA,
                distance_correction=distance_correction,
                penalty_anneal_start=PENALTY_ANNEAL_START,
                penalty_anneal_end=PENALTY_ANNEAL_END,
                grid_search=ICP_GRID_SEARCH,
                grid_step_factor=ICP_GRID_STEP_FACTOR,
                scale_factors=ICP_SCALE_FACTORS,
                shape_priority=SHAPE_PRIORITY,
                shape_constrained_k=SHAPE_CONSTRAINED_K,
                shape_angle_weight=SHAPE_ANGLE_WEIGHT,
                G_proj=G_city_proj,
                node_ids=node_ids,
            )

            matched_node_ids = node_ids[matched_indices]
            P_final = (R @ (P_sketch * s).T).T + t

            # Route-aware post-matching refinement
            print("Refining matches for routability...")
            matched_node_ids, n_refined = refine_matches_for_routability(
                G_city_proj, E_sketch, matched_node_ids, Q_city, node_ids,
                city_kdtree, P_final,
                detour_ratio=DETOUR_RATIO_THRESHOLD,
                k_candidates=DETOUR_K_CANDIDATES,
            )
            print(f"  Re-matched {n_refined} edges with excessive detours")

            # Plot alignment — zoomed to sketch bounding box
            fig, ax = ox.plot_graph(G_city_proj, show=False, close=False,
                                    node_size=5, edge_color='gray', bgcolor='w')
            for i, part_idx in enumerate(polygon_parts):
                pts = P_final[part_idx]
                status = "closed" if is_closed[i] else "open"
                ax.plot(pts[:, 0], pts[:, 1], '-o', markersize=4,
                        label=f'Part {i+1} ({status})')
            # Zoom to aligned sketch bounding box
            all_pts = np.vstack([P_final[pi] for pi in polygon_parts])
            x_min, y_min = all_pts.min(axis=0)
            x_max, y_max = all_pts.max(axis=0)
            x_pad = max((x_max - x_min) * 0.15, 50)
            y_pad = max((y_max - y_min) * 0.15, 50)
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            plt.title(f"ICP v4 [{extraction_mode}] '{image_file}' on {CITY_NAME}")
            plt.legend()
            align_img_path = os.path.join(
                OUTPUT_FOLDER, f"v4_align_{os.path.splitext(image_file)[0]}.png")
            _safe_savefig(align_img_path, bbox_inches='tight')
            print(f"  Alignment image saved: {align_img_path}")
            plt.show()

            # Generate route & GPX
            total_km = create_route_and_gpx(
                G_city_proj, G_city, matched_node_ids, polygon_parts, is_closed,
                output_path, outside_penalty=GEO_FENCE_OUTSIDE_PENALTY,
                max_odd_nodes=MAX_ODD_NODES_FOR_CUSTOM_EULER)

        if total_km is not None:
            print(f"  Final route: {total_km:.1f} km vs {target_km:.1f} km target "
                  f"(correction={distance_correction:.3f})")

elif not city_loaded:
    print("City data failed to load.")

else:
    print(f"Image folder '{IMAGE_FOLDER}' not found.")


