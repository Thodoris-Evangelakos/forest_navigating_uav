# Forest World Generation & UAV Scripts

This directory contains scripts for generating forest worlds and spawning UAVs.

## Script Organization

Scripts are organized into two categories:

### **Generation** (World Creation)
- `generate_world.sh` - Generate world only (saves to `outputs/runs/`)
- `generate_world_and_run.sh` - Generate world + launch Gazebo

### **UAV** (UAV Management)
- `spawn_uav.sh` - Spawn UAV in existing world (requires Gazebo running)
- `generate_world_with_uav.sh` - Generate world + launch Gazebo + spawn UAV

---

## Scripts

### 1. `generate_world.sh` - World Generation Only

Generate a randomized forest world without launching Gazebo.

**Usage:**
```bash
./scripts/generate_world.sh [config_file] [--seed SEED]
```

**Arguments:**
- `config_file` (optional): Path to worldgen config (default: `configs/worldgen/worldgen_run.yaml`)
- `--seed SEED`: Random seed for reproducible generation

**Output:**
- World files saved to: `worldgen/outputs/runs/YYYY-MM-DD_HHMMSS_[seed####|random]/`
  - `world.sdf` - Gazebo world file
  - `meta.json` - World metadata
  - `preview.png` - Top-down visualization

**Example:**
```bash
./scripts/generate_world.sh --seed 42
```

---

### 2. `generate_world_and_run.sh` - Generate + Visualize

Generate a world and immediately launch it in Gazebo for visualization.

**Usage:**
```bash
./scripts/generate_world_and_run.sh [config_file] [--seed SEED]
```

**Arguments:**
- Same as `generate_world.sh`

**Example:**
```bash
./scripts/generate_world_and_run.sh --seed 123
```

---

### 3. `spawn_uav.sh` - Add UAV to Existing World

Spawn a UAV at an edge location in a world that's already running in Gazebo.

**Usage:**
```bash
./scripts/spawn_uav.sh <world_sdf_path> [--index INDEX] [--margin MARGIN] [--height HEIGHT]
```

**Arguments:**
- `world_sdf_path` (required): Path to world.sdf file
- `--index INDEX`: Corner position 0-3 (0=NW, 1=SW, 2=SE, 3=NE, default: 0)
- `--margin MARGIN`: Distance from world edge in meters (default: 1.0)
- `--height HEIGHT`: Altitude above ground in meters (default: 2.0)

**Example:**
```bash
# First, start Gazebo with a world:
gz sim worldgen/outputs/runs/2026-02-11_120000_seed0042/world.sdf &

# Then spawn UAV:
./scripts/spawn_uav.sh worldgen/outputs/runs/2026-02-11_120000_seed0042/world.sdf --index 0
```

**Note:** Requires Gazebo to be already running with the specified world loaded.

---

### 4. `generate_world_with_uav.sh` ⭐ **Complete Workflow**

Generate world, launch Gazebo, and spawn UAV - all in one step.

**Usage:**
```bash
./scripts/generate_world_with_uav.sh [config_file] [--seed SEED] [--index INDEX] [--margin MARGIN] [--height HEIGHT]
```

**Arguments:**
- `config_file` (optional): Path to worldgen config (default: `configs/worldgen/worldgen_run.yaml`)
- `--seed SEED`: Random seed for world generation
- `--index INDEX`: UAV spawn corner 0-3 (default: 0)
- `--margin MARGIN`: Edge distance in meters (default: 1.0)
- `--height HEIGHT`: Spawn altitude in meters (default: 2.0)

**Example:**
```bash
./scripts/generate_world_with_uav.sh --seed 42 --index 2 --height 3.0
```

**What it does:**
1. Generates randomized forest world
2. Launches Gazebo with the world
3. Spawns UAV at specified edge position
4. Keeps Gazebo running for interaction

---

## World Coordinate System

The world is a square spanning `[-K/2, K/2]²` where `K` is the `area_size` from config:

- **World config**: `area_size: 50` → world spans `[-25, 25]` in both x and y
- **Edges with margin=1.0**: spawn at `-24` to `24` in both x and y
- **Corner indices**: 
  - `0` = NW (Northwest): `(-edge, edge)`
  - `1` = SW (Southwest): `(-edge, -edge)`
  - `2` = SE (Southeast): `(edge, -edge)`
  - `3` = NE (Northeast): `(edge, edge)`

---

## Common Workflows

### 1. Generate and visualize a world (no UAV)
```bash
./scripts/generate_world_and_run.sh --seed 42
```

### 2. Generate world with UAV at northwest corner
```bash
./scripts/generate_world_with_uav.sh --seed 42 --index 0
```

### 3. Add UAV to existing world
```bash
# First, start Gazebo with a previously generated world
gz sim worldgen/outputs/runs/2026-02-11_120000_seed0042/world.sdf &

# Then spawn UAV at southeast corner
./scripts/spawn_uav.sh worldgen/outputs/runs/2026-02-11_120000_seed0042/world.sdf --index 2
```

### 4. Just generate a world (for later use)
```bash
./scripts/generate_world.sh --seed 123
# Output saved to: worldgen/outputs/runs/2026-02-11_HHMMSS_seed0123/
```

---

## Output Directory Structure

All generated worlds are saved to timestamped directories:

```
worldgen/outputs/runs/
├── 2026-02-11_120000_seed0042/
│   ├── world.sdf       # Gazebo world file
│   ├── meta.json       # World metadata
│   └── preview.png     # Top-down visualization
├── 2026-02-11_123456_random/
│   ├── world.sdf
│   ├── meta.json
│   └── preview.png
└── ...
```

- With seed: `YYYY-MM-DD_HHMMSS_seed####`
- Without seed: `YYYY-MM-DD_HHMMSS_random`

---

## Notes

- **Dependencies**: Python 3, PyYAML, matplotlib, ros2, Gazebo
- **UAV Model**: Uses `models/drones/uav_simple/model.sdf` (with GPU lidar sensor)
- **World Name**: Defaults to `randomized_world` (can be changed in `configs/worldgen/world.default.yaml`)
- **Margin**: Ensures UAVs don't spawn outside world boundaries or in trees
- **Height**: Default `2.0` m is safe for most scenarios; adjust based on tree heights

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
