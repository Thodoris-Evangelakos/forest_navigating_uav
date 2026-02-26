"""
Mixture field layout - smooth blending between two distributions
"""

import random
import math
from ..config import load_distribution
from ..patterns.csr import _point_in_area, _check_min_distance, ProximityGrid
from ..patterns.clustered import _scatter_gaussian, _SCATTER_FNS


def _sigmoid(x, center, width):
    """Smooth sigmoid weight: returns value in [0, 1]"""
    t = (x - center) / max(width, 1e-6)
    return 1.0 / (1.0 + math.exp(-t))


def _clamp01(v):
    if v <= 0.0:
        return 0.0
    if v >= 1.0:
        return 1.0
    return v


def _field_value(kind, x, y, K, cfg):
    if kind == 'sigmoid_x':
        v = _sigmoid(x, cfg.get('center_x', 0.0), cfg.get('width', 1.0))
    elif kind == 'sigmoid_y':
        v = _sigmoid(y, cfg.get('center_y', 0.0), cfg.get('width', 1.0))
    elif kind == 'radial':
        cx = cfg.get('center_x', 0.0)
        cy = cfg.get('center_y', 0.0)
        radius = cfg.get('radius', K * 0.25)
        width = cfg.get('width', max(radius * 0.25, 1e-6))
        dist = math.hypot(x - cx, y - cy)
        v = _sigmoid(radius - dist, 0.0, width)
    elif kind == 'linear_x':
        x_min = cfg.get('x_min', -K / 2)
        x_max = cfg.get('x_max', K / 2)
        denom = max(x_max - x_min, 1e-6)
        v = (x - x_min) / denom
    elif kind == 'linear_y':
        y_min = cfg.get('y_min', -K / 2)
        y_max = cfg.get('y_max', K / 2)
        denom = max(y_max - y_min, 1e-6)
        v = (y - y_min) / denom
    else:
        v = 1.0

    if cfg.get('invert'):
        v = 1.0 - v

    v = v * cfg.get('scale', 1.0) + cfg.get('bias', 0.0)
    return _clamp01(v)


def _build_candidate_sampler(dist_type, params, K):
    """Build a candidate point sampler for mixture-field placement.

    For ``regular`` we still return uniform candidates - the hard min-distance
    constraint is already enforced by the outer placement loop in
    ``generate_mixed_field``.  The spatial inhibition in the standalone
    ``sample_regular`` (Bridson Poisson-disc) is used when the regular
    pattern owns the whole zone / world; inside a mixture field the outer
    rejection loop handles it.
    """
    if dist_type in ('csr', 'regular'):
        return lambda: _point_in_area(K)

    if dist_type in ('clustered', 'scale_dependent'):
        cluster_count = max(1, int(params.get('cluster_count', 5)))
        cluster_radius = float(params.get('cluster_radius', 3.0))
        scatter_shape = params.get('scatter_shape', 'gaussian')
        allow_overlap = bool(params.get('allow_cluster_overlap', False))
        min_parent_dist = (0.0 if allow_overlap
                           else float(params.get('min_parent_distance',
                                                  cluster_radius * 0.5)))

        scatter_fn = _SCATTER_FNS.get(scatter_shape, _scatter_gaussian)

        parents = []
        for _ in range(cluster_count):
            for _ in range(60):
                cx, cy = _point_in_area(K)
                if min_parent_dist <= 0 or _check_min_distance(cx, cy, parents, min_parent_dist):
                    parents.append((cx, cy))
                    break
            else:
                parents.append(_point_in_area(K))

        def _sample_clustered():
            if parents:
                cx, cy = random.choice(parents)
                x, y = scatter_fn(cx, cy, cluster_radius)
                x = max(-K / 2, min(K / 2, x))
                y = max(-K / 2, min(K / 2, y))
                return x, y
            return _point_in_area(K)

        return _sample_clustered

    raise ValueError(f"Unknown distribution type: {dist_type}")


def _weighted_choice(weights, total_weight):
    r = random.random() * total_weight
    acc = 0.0
    for idx, w in enumerate(weights):
        acc += w
        if r <= acc:
            return idx
    return len(weights) - 1


def generate_mixed_field(layout_config, world_config, project_root):
    """
    Blends multiple distributions across the world using a spatial weight field
    
    Returns list of (x, y) positions
    """
    # World-level params always come from world.default.yaml (single source of truth)
    K = world_config['generation']['area_size']
    min_distance = world_config['generation']['min_distance']
    total_count = world_config['generation']['object_count']
    components = layout_config['layout'].get('components', [])
    field_cfg = layout_config['layout'].get('field', {}) or {}
    max_attempts = int(layout_config['layout'].get('max_attempts', 200))

    dists = []
    for comp in components:
        dc = load_distribution(comp['distribution_ref'], project_root)
        dist_type = dc['distribution']['type']
        dist_params = dc['distribution'].get('params', {}) or {}
        dists.append({
            'name': comp['name'],
            'type': dist_type,
            'params': dist_params,
            'weight': float(comp.get('weight', 1.0)),
            'field': comp.get('field'),
            'sampler': _build_candidate_sampler(dist_type, dist_params, K),
        })

    if not dists:
        return []

    name_to_index = {d['name']: i for i, d in enumerate(dists)}
    component_fields = {}
    for fc in field_cfg.get('components', []) or []:
        ref = fc.get('component')
        if isinstance(ref, int) and 0 <= ref < len(dists):
            component_fields[ref] = fc
        elif ref in name_to_index:
            component_fields[name_to_index[ref]] = fc

    def _component_field_weight(i, x, y):
        if dists[i].get('field') is not None:
            fc = dists[i]['field']
            return _field_value(fc.get('kind', 'constant'), x, y, K, fc)
        if i in component_fields:
            fc = component_fields[i]
            return _field_value(fc.get('kind', 'constant'), x, y, K, fc)
        if 'kind' in field_cfg:
            base = _field_value(field_cfg.get('kind', 'constant'), x, y, K, field_cfg)
            if len(dists) == 1:
                return 1.0
            if i == 0:
                return base
            return _clamp01((1.0 - base) / max(len(dists) - 1, 1))
        return 1.0

    base_weights = [max(0.0, d['weight']) for d in dists]
    total_weight = sum(base_weights)
    if total_weight <= 0.0:
        base_weights = [1.0] * len(dists)
        total_weight = float(len(dists))

    all_positions = []
    counts = [0] * len(dists)
    warned = False
    grid = ProximityGrid(min_distance) if min_distance > 0 else None
    forced_relaxations = 0

    for placed_count in range(total_count):
        progress = (placed_count / max(total_count, 1))
        if progress >= 0.95:
            attempt_budget = min(max_attempts, 20)
        elif progress >= 0.90:
            attempt_budget = min(max_attempts, 30)
        elif progress >= 0.80:
            attempt_budget = min(max_attempts, 60)
        else:
            attempt_budget = max_attempts

        if forced_relaxations >= 8:
            attempt_budget = min(attempt_budget, 25)

        placed = False
        for _ in range(attempt_budget):
            chosen = _weighted_choice(base_weights, total_weight)
            x, y = dists[chosen]['sampler']()
            if random.random() > _component_field_weight(chosen, x, y):
                continue
            if grid is not None and not grid.check(x, y, min_distance):
                continue
            all_positions.append((x, y))
            if grid is not None:
                grid.insert(x, y)
            counts[chosen] += 1
            placed = True
            forced_relaxations = max(0, forced_relaxations - 1)
            break
        if not placed:
            chosen = _weighted_choice(base_weights, total_weight)
            x, y = dists[chosen]['sampler']()
            all_positions.append((x, y))
            if grid is not None:
                grid.insert(x, y)
            counts[chosen] += 1
            forced_relaxations += 1
            if not warned:
                print("Warning: relaxed placement constraints in mixture_field")
                warned = True

    summary = ", ".join(f"{counts[i]} '{d['name']}'" for i, d in enumerate(dists))
    print(f"  mixture_field: {summary}")
    return all_positions
