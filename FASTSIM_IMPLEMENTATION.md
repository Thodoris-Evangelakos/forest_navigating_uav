# FastSim In-Memory World Generation & Efficient Simulation

## Summary

Implemented high-performance in-memory world generation for fastsim training that reuses the existing `worldgen` layout/pattern logic without SDF export overhead. Added efficient lidar raycasting and spatial indexing for fast collision/sensor queries during training episodes.

## Key Features

### 1. In-Memory World Generation
- **API**: `generate_positions_from_config(config_file, seed)` in `worldgen/forest_worldgen/generate_world.py`
- **Behavior**: Generates tree positions using existing worldgen layouts (single/zones/mixture_field) and distributions (csr/regular/clustered/scale_dependent) with deterministic seeding
- **Integration**: Called during `ForestNavEnv.reset()` to sample fresh obstacle configurations per episode
- **CLI Compatibility**: Original `main()` CLI workflow preserved for Gazebo SDF export

### 2. Per-Tree Radius Sampling
- **Location**: `ForestNavEnv._sample_forest()` in `forest_nav_env.py`
- **Distribution**: Truncated normal with configurable mean/std/min/max via `SimParams`
- **Default**: mean=0.25m, std=0.05m, clipped to [0.10, 0.60]m
- **Output**: `(N, 3)` array with `(x, y, radius)` per tree

### 3. Lidar Raycasting
- **Algorithm**: Vectorized 2D circle-ray intersection (analytical solution)
- **Optimization**: Precomputed beam geometry + per-tree angular narrowing
- **Performance**: ~560 μs/scan (180 beams, 50 trees) = 1776 scans/sec

### 4. Spatial Grid Acceleration
- **Structure**: Uniform grid hash with cell size = `lidar_range_max / 2.0`
- **Usage**: Both lidar and safety shield query only nearby trees per step
- **Build Cost**: Amortized once per reset (trees inserted into overlapping cells)
- **Query Performance**: Filters 50-tree world to ~41 trees for shield (12m radius), ~50 for lidar (30m radius at center)

### 5. Safety Shield Integration
- **Update**: Shield now queries spatial grid instead of iterating all trees
- **Lookahead**: Queries within `max_step_velocity * dt + r_safe + margin`
- **Performance**: Negligible overhead with grid filtering

## Performance Metrics

### Typical Training Config (50 trees, 180 beams, 30m range)
- **Full env.step()**: 0.91 ms/step = 1097 steps/sec
- **Lidar scan alone**: 563 μs/scan = 1776 scans/sec
- **Grid build (reset)**: ~1-2 ms for 50 trees
- **Grid cells**: 14 cells for 50m**2 world with 15m cell size

### Dense Scenario (200 trees, 80m**2 world)
- **Lidar scan**: 900 μs/scan = 1110 scans/sec
- **Grid cells**: 36 cells with 15m cell size
- **Query filtering**: Corner position sees 84/200 trees, center sees 153/200

## Configuration Parameters

Added to `SimParams` in `forest_nav_env.py`:

```python
# World generation
worldgen_config_relpath: str = "configs/worldgen/worldgen_run.yaml"
worldgen_seed_offset: int = 0

# Tree geometry
tree_radius_mean: float = 0.25
tree_radius_std: float = 0.05
tree_radius_min: float = 0.10
tree_radius_max: float = 0.60

# Start/goal sampling
start_goal_clearance: float = 1.0
min_start_goal_distance: float = 8.0
spawn_max_attempts: int = 500
```

## Files Modified

1. **worldgen/forest_worldgen/generate_world.py**
   - Added `generate_positions_from_config()` pure function
   - Added `_local_random_seed()` context manager for determinism
   - Refactored `main()` to use new API while preserving CLI behavior

2. **src/fastsim_forest_nav/fastsim_forest_nav/envs/forest_nav_env.py**
   - Extended `SimParams` with worldgen + radius + spawn controls
   - Implemented `_sample_forest()` with worldgen integration
   - Implemented `_lidar_scan()` with vectorized raycasting
   - Added spatial grid: `_build_tree_grid()` and `_query_nearby_trees()`
   - Updated `_apply_shield()` to use grid queries
   - Updated `_sample_start_pose()` and `_sample_goal_pose()` with clearance checks
   - Exposed `worldgen_seed` and `tree_count` in info dict

## Testing & Validation

### Geometry Correctness
- ✅ Tree at (10, 0) with r=1 → forward beam reads 9.0
- ✅ Tree behind UAV does not affect forward beam
- ✅ UAV inside tree trunk → all beams return 0.0

### Determinism
- ✅ Same env seed → identical worldgen_seed and tree positions across resets
- ✅ Different env seed → distinct tree configurations

### Performance
- ✅ <1ms per env.step() with full lidar + shield + worldgen
- ✅ >1000 steps/sec training throughput

## Usage Example

```python
from fastsim_forest_nav.envs.forest_nav_env import SimParams, ForestNavEnv

# Configure environment
params = SimParams(
    lidar_num_beams=180,
    lidar_range_max=30.0,
    worldgen_config_relpath="configs/worldgen/worldgen_run.yaml",
    tree_radius_mean=0.25,
)

# Create and run
env = ForestNavEnv(params)
obs, info = env.reset(seed=42)
print(f"Trees: {info['tree_count']}, Seed: {info['worldgen_seed']}")

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    if term or trunc:
        obs, info = env.reset()
```

## Future Optimizations

1. **Numba JIT**: Compile lidar raycasting core with `@njit` for ~2-5× speedup
2. **Parallel Lidar**: Split beams across workers for multi-tree raycasting
3. **Adaptive Grid**: Dynamically adjust cell size based on tree density
4. **Curriculum**: Progressive world difficulty via config switching during training
5. **Lidar Noise**: Add sensor noise model for sim-to-real transfer
