# UAV Spawn Scripts

This directory contains scripts for spawning UAVs at calculated edge locations in a generated forest world.

## Scripts

### 1. `run_world_with_uav.sh` ⭐ **Main Orchestrator**

Complete end-to-end workflow: generate world → calculate spawn points → spawn UAV.

**Usage:**
```bash
./scripts/run_world_with_uav.sh [config_file] [--seed SEED] [--margin MARGIN] [--height HEIGHT]
```

**Arguments:**
- `config_file` (optional): Path to worldgen config (default: `configs/worldgen/worldgen_run.yaml`)
- `--seed SEED`: Random seed for world generation (reproducible)
- `--margin MARGIN`: Distance inset from world edge in meters (default: `1.0`)
- `--height HEIGHT`: Spawn height above ground in meters (default: `2.0`)

**Example:**
```bash
./scripts/run_world_with_uav.sh --seed 42 --margin 1.5 --height 3.0
```

**What it does:**
1. Calls `gen_world.sh` to generate the randomized forest
2. Calls `spawn_uav.sh` to calculate edge-based spawn points
3. Uses `ros2 run ros_gz_sim create` to spawn the UAV at the first calculated location
4. Saves spawn metadata to `worldgen/outputs/latest/spawn_points.json`

---

### 2. `spawn_uav.sh`

Shell wrapper for spawn point calculation. Used by `run_world_with_uav.sh` but can be called standalone.

**Usage:**
```bash
./scripts/spawn_uav.sh [config_file] [--seed SEED] [--margin MARGIN] [--height HEIGHT] [--count COUNT] [--dist DIST]
```

**Arguments:**
- `config_file` (optional): Path to worldgen config
- `--seed SEED`: Random seed for spawn point distribution
- `--margin MARGIN`: Edge inset (default: `1.0`)
- `--height HEIGHT`: Spawn height (default: `2.0`)
- `--count COUNT`: Number of spawn points to generate (default: `4`)
- `--dist DIST`: Distribution strategy: `uniform` or `random` (default: `uniform`)

**Output:**
- Prints spawn points to console
- Saves metadata JSON to `worldgen/outputs/latest/spawn_points.json`

---

### 3. `spawn_utils.py` (Python Module)

Core logic for calculating spawn points. Used by `spawn_uav.sh` but can also be imported/used programmatically.

**CLI Usage:**
```bash
python3 -m worldgen.forest_worldgen.spawn_utils <config_file> [options]
```

**Python Usage:**
```python
from worldgen.forest_worldgen.spawn_utils import generate_spawn_points, calculate_edge_spawns

# Get spawn points for a world config
spawn_meta = generate_spawn_points(
    config_file="configs/worldgen/worldgen_run.yaml",
    margin=1.0,
    z_height=2.0,
    count=4,
    distribution='uniform',
    seed=42
)

# Access results
for i, pt in enumerate(spawn_meta['spawn_points']):
    print(f"Spawn {i}: ({pt['x']:.2f}, {pt['y']:.2f}, {pt['z']:.2f})")

# Or calculate raw positions (no config needed)
spawns = calculate_edge_spawns(
    area_size=50,
    margin=1.0,
    count=4,
    distribution='uniform',
    z_height=2.0
)
```

---

## World Coordinate System

The world is a square spanning `[-K/2, K/2]²` where `K` is the `area_size` from config:

- **World config**: `area_size: 50` → world spans `[-25, 25]` in both x and y
- **Edges with margin=1.0**: span from `-24` to `24` in both x and y
- **Spawn distribution**: 
  - `uniform`: 4 spawns are evenly placed (1 per edge)
  - `random`: spawns randomly scattered along perimeter

---

## Spawn Points JSON Format

After running `spawn_uav.sh` or `run_world_with_uav.sh`, metadata is saved to:
```
worldgen/outputs/latest/spawn_points.json
```

**Example:**
```json
{
  "spawn_points": [
    {"x": -24.0, "y": 24.0, "z": 2.0},
    {"x": 0.0, "y": 24.0, "z": 2.0},
    {"x": 24.0, "y": 24.0, "z": 2.0}
  ],
  "area_size": 50,
  "margin": 1.0,
  "z_height": 2.0,
  "count": 3,
  "distribution": "uniform",
  "seed": null,
  "timestamp": "2026-02-09T10:30:45.123456"
}
```

---

## Examples

### Quick Test: Generate world and spawn UAV with default parameters
```bash
./scripts/run_world_with_uav.sh
```

### Reproducible Run: Use a fixed seed
```bash
./scripts/run_world_with_uav.sh --seed 12345
```

### Higher Spawn: Place UAV higher above ground
```bash
./scripts/run_world_with_uav.sh --height 5.0
```

### Closer to Edge: Smaller margin
```bash
./scripts/run_world_with_uav.sh --margin 0.5
```

### Calculate Spawns Only (without full orchestration)
```bash
./scripts/spawn_uav.sh --count 8 --dist random --seed 999
```

---

## Notes

- **Dependencies**: Python 3, PyYAML, ros2, Gazebo
- **UAV Model**: Uses `models/drones/uav_simple/model.sdf` (with GPU lidar sensor)
- **World Name**: Defaults to `randomized_world` (can be changed in `configs/worldgen/world.default.yaml`)
- **Margin**: Ensures UAVs don't spawn outside world boundaries or in trees
- **Height**: Default `2.0` m is safe for most scenarios; adjust based on tree heights in your config

---

## Troubleshooting

**Error: "Config file not found"**
- Ensure the config file path is correct relative to the script location
- Use absolute paths if in doubt

**Error: "Spawn points file not found"**
- Ensure `spawn_uav.sh` completed successfully before trying to spawn
- Check `worldgen/outputs/latest/spawn_points.json` exists

**UAV spawns but doesn't appear in Gazebo**
- Ensure the world is already running in Gazebo
- Check the world name matches (should be `randomized_world`)
- Verify the UAV model file exists: `models/drones/uav_simple/model.sdf`

---

## Implementation Details

- **Edge Calculation**: Spawn points are placed `margin` meters inset from world boundaries
- **Uniform Distribution**: Points evenly spaced along 4 edges (N, S, E, W)
- **Random Distribution**: Points randomly scattered along perimeter within margin band
- **Height**: All spawns use the same `z_height` (default 2.0 m)
- **Seedable**: All RNG operations respect the optional `--seed` flag for reproducibility
