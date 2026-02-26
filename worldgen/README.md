# Forest World Generator

A modular world generation system for creating randomized forest environments with configurable spatial distributions and layouts for Gazebo simulation.

## Structure

```
worldgen/
├── forest_worldgen/          # Core generation package
│   ├── patterns/             # Spatial distribution patterns
│   │   ├── csr.py           # Complete Spatial Randomness
│   │   ├── regular.py       # Regular/Poisson-disc distribution
│   │   └── clustered.py     # Thomas-like cluster process
│   ├── layouts/              # Spatial layout strategies
│   │   ├── single_zone.py   # Uniform single distribution
│   │   ├── mixed_zones.py   # Multiple rectangular regions
│   │   └── mixed_field.py   # Smooth distribution blending
│   ├── export/               # Output generation
│   │   ├── export_sdf.py    # SDF world file export
│   │   ├── export_meta.py   # JSON metadata export
│   │   └── preview_topdown.py  # PNG preview generation
│   ├── config.py             # Configuration utilities
│   ├── generate_world.py     # Main generation script
│   └── cli.py                # Command-line interface
│
├── templates/                # SDF templates
│   ├── world_base.sdf       # Base world template
│   └── include.sdf          # Object include template
│
└── outputs/                  # Generated outputs (gitignored)
    ├── latest/              # Most recent generation
    │   ├── world.sdf
    │   ├── meta.json
    │   └── preview.png
    └── runs/                # Timestamped runs with seeds
        └── 2026-02-08_1859_seed0999/
            ├── world.sdf
            ├── meta.json
            └── preview.png
```

## Usage

### Basic Generation

Generate a world using the default configuration:
```bash
./scripts/worldgen/generate_world.sh
```

### With Seed (Reproducible)

Generate with a specific random seed for reproducibility:
```bash
./scripts/worldgen/generate_world.sh configs/worldgen/worldgen_run.yaml --seed 42
```

### Test in Gazebo

Generate and immediately launch in Gazebo:
```bash
./scripts/worldgen/generate_world_and_run.sh
```

With seed:
```bash
./scripts/worldgen/generate_world_and_run.sh configs/worldgen/worldgen_run.yaml --seed 123
```

### Python Module

Run directly as a Python module:
```bash
python3 -m worldgen.forest_worldgen.generate_world configs/worldgen/worldgen_run.yaml --seed 42
```

## Configuration

### World Configuration (`world.default.yaml`)

Defines physics, lighting, and basic generation parameters:
- Area size (K*K meters)
- Object count
- Min distance between objects
- Tree height range
- Model URIs

### Layout Configuration

**Zones** (`mixed_zones.yaml`): Hard-edged rectangular regions with different distributions
```yaml
layout:
  type: "zones"
  zones:
    - name: "west_clustered"
      region: {x_min: -25, x_max: 0, y_min: -25, y_max: 25}
      count: 80
      distribution_ref: "clustered.yaml"
```

**Mixture Field** (`mixed_field.yaml`): Smooth blending between distributions
```yaml
layout:
  type: "mixture_field"
  total_count: 150
  field:
    kind: "sigmoid_x"
    center_x: 0.0
    width: 8.0
```

### Run Configuration (`worldgen_run.yaml`)

The run config can now optionally enable stochastic sampling of layout and distribution references at each generation call (useful for RL resets).

```yaml
include:
  world: "configs/worldgen/world.default.yaml"
  layout: "configs/worldgen/layouts/mixed_field.yaml"  # fallback when stochastic is disabled

stochastic:
  enabled: false  # set true to enable stochastic layout/distribution sampling
  layout_choices:
    - "configs/worldgen/layouts/mixed_field.yaml"
  distribution_choices:
    - "csr.yaml"
    - "regular.yaml"
    - "clustered.yaml"
    - "scale_dependent.yaml"
  distribution_mode: "per_entry"
```

`distribution_mode` behavior:
- `global`: one distribution is sampled and applied everywhere in the selected layout
- `per_entry` (aliases: `per_component`, `per_zone`): each `distribution_ref` entry in the selected layout is sampled independently

When `--seed` is provided, stochastic choices are deterministic for that seed.

### Distribution Patterns

**CSR**: Uniform random (Complete Spatial Randomness)
**Regular**: Poisson-disc inhibitory process
**Clustered**: Thomas-like cluster process with parent points and Gaussian scatter

## Outputs

Each generation creates:
1. **world.sdf** - Gazebo world file
2. **meta.json** - Metadata (positions, statistics, config info)
3. **preview.png** - Top-down visualization (requires matplotlib)

Outputs go to `worldgen/outputs/latest/` and optionally to timestamped run directories when using `--seed`.

## Dependencies

- Python 3.x
- PyYAML
- matplotlib (optional, for preview generation)

## Examples

```bash
# Default mixed zones layout
./scripts/worldgen/generate_world.sh

# Reproducible generation
./scripts/worldgen/generate_world.sh configs/worldgen/worldgen_run.yaml --seed 42

# Generate and test
./scripts/worldgen/generate_world_and_run.sh configs/worldgen/worldgen_run.yaml --seed 999
```
