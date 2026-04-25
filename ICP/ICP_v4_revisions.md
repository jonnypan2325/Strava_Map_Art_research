# Strava Map Art — ICP Alignment

Converts hand-drawn sketches into GPS routes on real streets. You draw a shape, pick a city, set a target distance, and the algorithm finds a neighborhood where street intersections closely match your sketch — then generates a GPX file you can follow on Strava.

## How It Works

The core idea is **Iterative Closest Point (ICP)** alignment: treat your sketch as a set of 2D points and the city's street intersections as another set, then find the rotation, translation, and scale that best overlays one onto the other.

### Pipeline

```
Image → Extract Sketch → Fetch Street Map → ICP Alignment → Route on Streets → GPX
```

1. **Image extraction** — Reads your sketch image and auto-detects whether it's a line drawing (stick figure, text) or a filled shape (cat, face). Line drawings are skeletonized to extract branching topology; filled shapes use contour extraction with morphological cleanup.

2. **Street network** — Downloads the OpenStreetMap graph for the target city/address and projects it to metric coordinates.

3. **ICP alignment** — The expensive part. Tries many candidate placements (position × rotation × scale) across the map, runs a few ICP iterations at each one, and scores the result. The best candidate is then refined with more iterations.

4. **Route generation** — Connects matched intersections via shortest street paths, makes the graph Eulerian (so it can be traversed in one continuous route), and exports a GPX file.

## Versions

### v3 — Density-Aware ICP (`ICP_map_art_v3.ipynb`)

The baseline. Works well for longer routes (8+ km) on dense grids like Manhattan.

**Search strategy:** 200 random starting nodes × 12 rotations × 1 scale = 2,400 candidates.

**Key features:**
- Skeleton + contour dual-mode image extraction
- Density-weighted scoring (prefers sparse neighborhoods to avoid dense grid collapse)
- Annealed shape penalty (starts high → decays to 0.5)
- Route-aware post-matching (re-matches edges with excessive street detours)
- Geo-fenced Eulerization (keeps backtrack paths inside the sketch footprint)

**Limitation:** With only 2,400 candidates sampling random nodes, shorter routes often land in a suboptimal neighborhood. The single scale factor means the algorithm can't adjust when the "ideal" physical size doesn't perfectly match the distance-derived scale. The penalty anneal decays to 0.5 for all routes — fine for long routes with many points, but short routes lose their shape during refinement.

### v4 — Shape-Priority ICP (`ICP_map_art_v4.ipynb`)

Redesigned for shape fidelity, especially on short routes (2–6 km).

**Search strategy:** (300 random + grid) × 24 rotations × 2–4 scales = 50,000–200,000+ candidates.

**What changed and why:**

#### 1. Systematic Grid Search

**Problem:** v3 picks 200 random intersections as starting positions. For a 3 km route, the sketch footprint is roughly 500m across — a tiny fraction of a city. Random sampling frequently misses the best pockets entirely.

**Solution:** Overlay a grid across the entire city with spacing = 60% of the sketch diameter. Every neighborhood gets tested. The grid positions are combined with random node samples (300) for coverage of both systematic and node-aligned starting points.

#### 2. Multi-Scale Search

**Problem:** v3 computes one scale factor from `route_distance / total_sketch_edge_length`. But streets don't form a perfect grid — the actual optimal physical size might be 15–25% smaller or larger to land sketch vertices on real intersections.

**Solution:** Test multiple scale factors automatically:
- Short routes (< 5 km): `[0.75, 0.85, 1.0, 1.15]` — 4 sizes
- Medium routes (5–10 km): `[0.85, 1.0, 1.1]` — 3 sizes
- Long routes (> 10 km): `[0.9, 1.0]` — 2 sizes

Short routes get more scales because relative spacing matters more (a 100m shift is 5% of a 2 km route but 1% of a 10 km route).

#### 3. Shape-Priority Scoring

**Problem:** v3's scoring is `density_penalty + penalty_weight × shape_penalty`. For short routes with few sketch points, the density penalty dominates and the algorithm picks placements where points are *close to nodes* rather than where the *shape is preserved*.

**Solution:** Add a **shape fidelity score** that measures two things:
- **Angular distortion** — mean angle error between sketch edges and their matched city-node edges
- **Proportionality** — variance of edge-length ratios (all edges should scale uniformly)

The combined score uses an adaptive `shape_priority` weight (auto-computed from route distance):
- Short routes (< 5 km): shape priority ~0.7 — contour shape dominates
- Long routes (> 10 km): shape priority ~0.3 — distance matters more

A `shape_multiplier` (1–4×) further boosts the shape penalty weight for short routes.

#### 4. Adaptive Penalty Annealing

**Problem:** v3 anneals the shape penalty from 4.0 → 0.5 during refinement. For long routes this is fine — many points constrain the shape even at low penalty. For short routes with ~10–20 points, decaying to 0.5 lets the ICP refinement drift points off the contour to minimize raw distance.

**Solution:** The anneal floor auto-scales inversely with route distance:
```
effective_floor = max(0.5, 2.0 - route_km / 10)
```
- 3 km route → floor = 1.7 (strong shape constraint throughout)
- 8 km route → floor = 1.2 (moderate)
- 15 km route → floor = 0.5 (same as v3)

#### 5. Angle-Constrained Matching

**Problem:** After ICP converges, the nearest city node to each sketch point isn't always the best match *for the shape*. A node 30m away might preserve edge angles much better than the nearest node 10m away.

**Solution:** After uniqueness enforcement, each matched point is re-evaluated against `k=5` nearby alternatives. For each candidate, compute the angular cost with all sketch-edge neighbors, and pick the one minimizing `distance + angle_weight × angular_error`. This catches cases where a slightly more distant node produces dramatically better contour angles.

#### 6. More Rotations

**Problem:** v3 tests 12 rotations (every 30°). Some shapes (e.g., a horse profile) have optimal orientations between the 30° samples.

**Solution:** 24 rotations (every 15°) — doubles angular resolution at minimal cost per candidate.

## Demo Notebooks

- **`demo.ipynb`** — Simple 4-cell workflow: set image + city + distance, run all cells, get a GPX file.
- **`demo copy.ipynb`** — Copy for parallel experiments with different settings.

Both load functions from the v4 notebook via `exec()` so you only need to maintain one copy of the algorithm.

## Usage

```python
# In demo.ipynb, edit cell 1:
IMAGE_PATH = "../images/horse2.jpg"
CITY_NAME  = "93 Cornell, Irvine, California, USA"
ROUTE_DISTANCE_KM = 4
```

Then run all cells. Output GPX + diagnostic images go to `../outputs/demo/`.

## Parameters Quick Reference

| Parameter | v3 Default | v4 Default | Effect |
|-----------|-----------|-----------|--------|
| `ICP_RANDOM_SAMPLES` | 200 | 300 | Random starting positions |
| `ICP_ROTATION_SAMPLES` | 12 | 24 | Rotations per position |
| `ICP_GRID_SEARCH` | — | `True` | Systematic grid coverage |
| `ICP_GRID_STEP_FACTOR` | — | 0.6 | Grid density (smaller = finer) |
| `ICP_SCALE_FACTORS` | — | auto | Scale multipliers to test |
| `SHAPE_PRIORITY` | — | auto | Shape vs distance blend (0–1) |
| `SHAPE_CONSTRAINED_K` | — | 5 | Angle-matching candidates |
| `PENALTY_ANNEAL_END` | 0.5 | adaptive | Min shape penalty in refinement |
| `ICP_REFINE_ITERATIONS` | 80 | 100 | More refinement steps |

## Requirements

See `../requirements.txt`. Core dependencies: `osmnx`, `networkx`, `opencv-python`, `scikit-image`, `scipy`, `gpxpy`, `matplotlib`.
