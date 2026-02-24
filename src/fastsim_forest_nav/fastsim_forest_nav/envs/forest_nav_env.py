#from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Tuple
from pathlib import Path
import importlib
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env

@dataclass
class SimParams:
    """Parameters for the simulation - all values must be provided from config.
    No defaults here to ensure single source of truth in sac.yaml.
    """
    # simulation / physics
    dt: float
    lidar_num_beams: int
    lidar_range_max: float
    v_max: float
    wz_max: float
    vz_max: float
    r_safe: float
    episode_seconds: float
    goal_tolerance: float
    world_radius: float
    collision_threshold: float

    # vertical tracking / observation scaling
    default_z_target: float
    z_error_scale: float

    # reward shaping
    reward_progress_scale: float
    reward_speed_scale: float
    reward_step_penalty: float
    reward_proximity_scale: float
    reward_shield_penalty: float
    reward_collision_penalty: float
    reward_success_bonus: float
    reward_truncation_penalty: float
    reward_yaw_rate_scale: float
    reward_stall_penalty: float
    progress_stall_threshold: float
    yaw_penalty_speed_gate: float

    # safety shield
    shield_floor_z_min: float
    shield_yaw_damping: float
    shield_lookahead_margin: float

    # world generation for fastsim (pure in-memory)
    worldgen_config_relpath: str
    worldgen_seed_offset: int
    tree_radius_mean: float
    tree_radius_std: float
    tree_radius_min: float
    tree_radius_max: float

    # start / goal sampling
    start_goal_clearance: float
    min_start_goal_distance: float
    spawn_max_attempts: int

class ForestNavEnv(gym.Env):
    """ Environment as expected by SBR3 with a continous action space of (v, wz, vz)
    and an observation space of (lidar ranges..., cos(theta_goal), sin(theta_goal), forward_speed, yaw_rate, height_error)

    Args:
        gym (Env): parent gym class, inheriting from that
    """
    def __init__(self, params: SimParams, render_mode: Optional[str] = None):
        super().__init__()
        self.p = params
        self.render_mode = render_mode

        # Observation space: [lidar_ranges normalized...,
        # cos(theta_goal), sin(theta_goal), forward_speed norm, yaw_rate norm, height error normalized =
        # clip((z_target -z) / z_scale, -1, 1)]
        # I can change forward speed to vx, vy normalized (+1 box size)
        obs_dim = self.p.lidar_num_beams + 6
        self.observation_space = spaces.Box(
            low = -1.0, high = 1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Action space: a[0] = forward speed command, a[1] = yaw rate command, a[2] = vertical speed command
        # all 3 normalized in their respective v max. E.g. v = a[0] * v_max
        # SAC in SB3 is built for continuous Box actions
        self.action_space = spaces.Box(
            low = -1.0, high = 1.0, shape = (3,), dtype = np.float32
        )

        self._t = 0.0
        self._step_count = 0

        # Cached lidar scan to avoid recomputation
        self._cached_lidar: np.ndarray | None = None
        self._cached_min_range: float = float('inf')

        # State example
        self.pos = np.zeros(3, dtype=np.float32)  # x, y, z. Will probably rip it from the sim/gz directly for now
        self.yaw = np.float32(0.0)
        self.v = np.float32(0.0)
        self.wz = np.float32(0.0)
        self.vz = np.float32(0.0)

        self.goal = np.zeros(3, dtype=np.float32)  # x, y, z
        self.z_target = np.float32(0.0) # maintaing this height target for now

        self.trees = None # list/array of cylinders (x, y, radius)
        self._prev_dist: Optional[float] = None # last distance, can use to calculate delta
        self._world_half_extent = float(self.p.world_radius)
        self._last_worldgen_seed: Optional[int] = None

        # full 360 deg lidar beam geometry (relative to body x-axis)
        self._beam_angle_start = -np.pi
        self._beam_angle_step = (2.0 * np.pi) / float(self.p.lidar_num_beams)
        rel_angles = self._beam_angle_start + self._beam_angle_step * np.arange(
            self.p.lidar_num_beams,
            dtype=np.float64,
        )
        self._beam_rel_cos = np.cos(rel_angles)
        self._beam_rel_sin = np.sin(rel_angles)

        # spatial grid for fast nearest-tree queries
        self._tree_grid: Optional[dict] = None
        self._tree_grid_cell_size = float(self.p.lidar_range_max) / 2.0
        self._worldgen_generate_positions_fn = None

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[4]

    def _worldgen_config_path(self) -> Path:
        return self._project_root() / self.p.worldgen_config_relpath

    def _build_tree_grid(self):
        """Build spatial hash grid for fast nearest-tree queries."""
        if self.trees is None or len(self.trees) == 0:
            self._tree_grid = {}
            return

        cell_size = self._tree_grid_cell_size
        grid = {}
        for idx, tree in enumerate(self.trees):
            x, y, r = float(tree[0]), float(tree[1]), float(tree[2])
            # insert into all cells overlapped by this tree's bounding box
            min_cell_x = int(np.floor((x - r) / cell_size))
            max_cell_x = int(np.floor((x + r) / cell_size))
            min_cell_y = int(np.floor((y - r) / cell_size))
            max_cell_y = int(np.floor((y + r) / cell_size))
            for cx in range(min_cell_x, max_cell_x + 1):
                for cy in range(min_cell_y, max_cell_y + 1):
                    key = (cx, cy)
                    if key not in grid:
                        grid[key] = []
                    grid[key].append(idx)
        self._tree_grid = grid

    def _query_nearby_trees(self, x: float, y: float, radius: float) -> np.ndarray:
        """Return indices of trees within 'radius' of (x, y) using spatial grid."""
        if self._tree_grid is None or len(self._tree_grid) == 0 or self.trees is None:
            return np.array([], dtype=np.int64)

        cell_size = self._tree_grid_cell_size
        min_cx = int(np.floor((x - radius) / cell_size))
        max_cx = int(np.floor((x + radius) / cell_size))
        min_cy = int(np.floor((y - radius) / cell_size))
        max_cy = int(np.floor((y + radius) / cell_size))

        candidates = set()
        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                if (cx, cy) in self._tree_grid:
                    candidates.update(self._tree_grid[(cx, cy)])

        if not candidates:
            return np.array([], dtype=np.int64)

        candidate_indices = np.array(list(candidates), dtype=np.int64)
        tree_subset = self.trees[candidate_indices]
        dx = tree_subset[:, 0] - x
        dy = tree_subset[:, 1] - y
        dist = np.sqrt(dx * dx + dy * dy)
        reach = dist + tree_subset[:, 2]
        in_range = reach >= 0.0  # always true but kept for clarity
        return candidate_indices[in_range]

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        
        # gym seeding contract
        super().reset(seed=seed)
        self._t = 0.0
        self._step_count = 0

        # Sample world, start, goal
        self.trees = self._sample_forest()
        self._build_tree_grid()
        self.pos, self.yaw = self._sample_start_pose()
        self.goal = self._sample_goal_pose()
        self.z_target = np.float32(self.p.default_z_target)

        # standing still
        self.v = np.float32(0.0)
        self.wz = np.float32(0.0)
        self.vz = np.float32(0.0)

        self._prev_dist = self._dist_to_goal()

        obs = self._get_obs()
        info = self._get_info(shield_active = 0, collision = 0, success = 0, shield_delta = 0.0)
        return obs, info
    
    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)

        # map normalized actions (more like suggestions) to actual commands
        cmd_v = float(action[0]) * self.p.v_max
        cmd_wz = float(action[1]) * self.p.wz_max
        cmd_vz = float(action[2]) * self.p.vz_max

        # safety shield clamps command (authoritative)
        safe_v, safe_wz, safe_vz, shield_active, shield_delta = self._apply_shield(
            cmd_v, cmd_wz, cmd_vz
        )

        # integrate simple kinematics (fastsim)
        self._integrate(safe_v, safe_wz, safe_vz)

        # sensor update
        lidar = self._lidar_scan()
        self._cached_lidar = lidar
        min_range = float(np.min(lidar))
        self._cached_min_range = min_range

        # reward and termination
        dist = self._dist_to_goal()
        prev_dist = self._prev_dist if self._prev_dist is not None else dist
        d_progress = prev_dist - dist
        self._prev_dist = dist

        collision = int(min_range < self.p.collision_threshold)
        success = int(dist < self.p.goal_tolerance)

        # Weights saved in the dataclass
        reward = 0.0
        reward += self.p.reward_progress_scale * d_progress # encourage progress towards goal
        reward += self.p.reward_speed_scale * (safe_v / self.p.v_max) # encourage faster speeds
        reward -= self.p.reward_step_penalty # small penalty for each step to encourage faster completion

        speed_norm = abs(float(safe_v)) / max(float(self.p.v_max), 1e-6)
        yaw_rate_norm = abs(float(safe_wz)) / max(float(self.p.wz_max), 1e-6)
        reward -= self.p.reward_yaw_rate_scale * speed_norm * yaw_rate_norm

        if d_progress < self.p.progress_stall_threshold and not success:
            stall_ratio = (self.p.progress_stall_threshold - d_progress) / max(
                self.p.progress_stall_threshold,
                1e-6,
            )
            stall_penalty = self.p.reward_stall_penalty * float(np.clip(stall_ratio, 0.0, 1.0))

            if speed_norm < self.p.yaw_penalty_speed_gate:
                low_speed_ratio = (self.p.yaw_penalty_speed_gate - speed_norm) / max(
                    self.p.yaw_penalty_speed_gate,
                    1e-6,
                )
                stall_penalty += 0.5 * self.p.reward_stall_penalty * float(
                    np.clip(low_speed_ratio, 0.0, 1.0)
                )

            reward -= stall_penalty

        if min_range < self.p.r_safe:
            reward -= self.p.reward_proximity_scale * (self.p.r_safe - min_range) / self.p.r_safe # penalty for getting too close to obstacles
        if shield_active:
            reward -= self.p.reward_shield_penalty
        if collision:
            reward -= self.p.reward_collision_penalty
        if success:
            reward += self.p.reward_success_bonus
        
        terminated = bool(collision or success)
        self._step_count += 1
        self._t += self.p.dt

        truncated = bool(self._t >= self.p.episode_seconds)
        if truncated and not terminated:
            reward -= self.p.reward_truncation_penalty

        obs = self._pack_obs(lidar, dist, safe_v, safe_wz) # purposefully not including vz
        info = self._get_info(
            shield_active = shield_active,
            collision = collision,
            success = success,
            shield_delta = shield_delta
        )
        return obs, float(reward), terminated, truncated, info
    
    def render(self):
        """ Returns a bunch of info for the time being until I build an actual visualizer (Gazebo?)

        Returns:
            string: string with time, position, yaw, goal, velocity
        """
        return f"t={self._t:2f} pos = {self.pos} yaw = {float(self.yaw):.2f} goal = {self.goal} v = {float(self.v):.2f}"
    
    def close(self):
        """ Any cleanup if needed, not really necessary for this simple env but good to have the structure in place for future extensions
        """
        pass

    ### HELPER FUNCTIONS ###

    def _pack_obs(self, lidar: np.ndarray, dist: float, v: np.float32, wz: np.float32) -> np.ndarray:
        # normalize lidar
        lidar_n = np.clip(lidar / self.p.lidar_range_max, 0.0, 1.0).astype(np.float32)

        # goal direction in body frame
        dx = float(self.goal[0] - self.pos[0])
        dy = float(self.goal[1] - self.pos[1])
        theta = np.arctan2(dy, dx) - float(self.yaw)
        c, s = np.cos(theta), np.sin(theta)

        dist_n = np.clip(dist / (2.0 * self._world_half_extent), 0.0, 1.0)
        v_n = np.clip(v / self.p.v_max, -1.0, 1.0)
        wz_n = np.clip(wz / self.p.wz_max, -1.0, 1.0)
        z_err = np.clip((float(self.z_target) - float(self.pos[2])) / self.p.z_error_scale, -1.0, 1.0)

        tail = np.array([c, s, dist_n, v_n, wz_n, z_err], dtype=np.float32)
        obs = np.concatenate([lidar_n, tail], axis=0).astype(np.float32)
        return obs
    
    def _get_obs(self):
        lidar = self._lidar_scan()
        dist = self._dist_to_goal()
        obs = self._pack_obs(lidar, dist, self.v, self.wz)
        return obs
    
    def _get_info (self, **kwargs) -> dict[str, Any]:
        # Use cached values from step() to avoid recomputation
        min_range = self._cached_min_range if self._cached_lidar is not None else float(np.min(self._lidar_scan()))
        info = {
            "dist_to_goal": float(self._dist_to_goal()),
            "min_range": min_range,
            "tree_count": int(0 if self.trees is None else len(self.trees)),
            "worldgen_seed": int(self._last_worldgen_seed) if self._last_worldgen_seed is not None else None,
        }
        info.update(kwargs)
        info["is_success"] = bool(info.get("success", False))
        return info
    
    def _dist_to_goal(self) -> float:
        # for now just use euclidean distance in xy plane ignoring z. I can add that in later if needed but it might not be super helpful
        return float(np.linalg.norm(self.goal[:2] - self.pos[:2]))
    
    def _apply_shield(self, v: float, wz: float, vz: float):
        """Velocity-barrier safety shield.

        For every tree, ensure the UAV cannot close more gap than available in one time-step
        If the commanded v would violate this constraint it is clamped to the largest safe value
        Yaw rate is passed through (only speed magnitude is regulated)

        Returns (safe_v, safe_wz, safe_vz, shield_active, shield_delta_norm)
        #NOTE:XXX This might result in the drone just getting stuck? 
        """
        safe_v = float(v)
        safe_wz = float(wz)
        safe_vz = float(vz)
        shield_active = 0

        # horizontal tree avoidance (query nearby trees only)
        if self.trees is not None and len(self.trees) > 0:
            vel_dir = np.array(
                [np.cos(float(self.yaw)), np.sin(float(self.yaw))],
                dtype=np.float64,
            )
            pos_xy = self.pos[:2].astype(np.float64)

            # max distance UAV can travel this step plus safety bubble
            lookahead = abs(safe_v) * self.p.dt + self.p.r_safe + self.p.shield_lookahead_margin
            nearby_idx = self._query_nearby_trees(float(self.pos[0]), float(self.pos[1]), lookahead)

            for idx in nearby_idx:
                tree = self.trees[idx]
                t_xy = tree[:2].astype(np.float64)
                t_r = float(tree[2])

                delta = t_xy - pos_xy  # vector from UAV to tree centre
                d = float(np.linalg.norm(delta))
                if d < 1e-6:
                    # degenerate: UAV on top of tree centre -> full stop
                    safe_v = 0.0
                    shield_active = 1
                    continue

                gap = d - t_r - self.p.r_safe  # remaining clearance
                c = float(np.dot(vel_dir, delta / d))  # cos(heading, tree dir)
                approach = safe_v * c  # > 0 when closing distance

                if gap <= 0.0:
                    # already inside safety bubble, block any further approach
                    if approach > 0.0:
                        safe_v = 0.0
                        shield_active = 1
                elif approach > 0.0:
                    # positive gap but approaching, cap approach speed
                    v_max_safe = (
                        gap / (self.p.dt * abs(c)) if abs(c) > 1e-6 else abs(safe_v)
                    )
                    if abs(safe_v) > v_max_safe:
                        safe_v = float(np.sign(safe_v)) * v_max_safe
                        shield_active = 1

        # vertical floor guard
        if float(self.pos[2]) <= self.p.shield_floor_z_min and safe_vz < 0.0:
            safe_vz = 0.0
            shield_active = 1

        if shield_active and abs(safe_wz) > 0.0 and abs(safe_v) > 1e-3:
            damping = float(np.clip(self.p.shield_yaw_damping, 0.0, 1.0))
            safe_wz = safe_wz * (1.0 - damping)

        # normalised intervention magnitude
        cmd_norm = abs(v) + abs(wz) + abs(vz)
        delta_norm = abs(v - safe_v) + abs(wz - safe_wz) + abs(vz - safe_vz)
        shield_delta = delta_norm / cmd_norm if cmd_norm > 1e-6 else 0.0

        return np.float32(safe_v), np.float32(safe_wz), np.float32(safe_vz), int(shield_active), float(shield_delta)

    def _integrate(self, v: np.float32, wz: np.float32, vz: np.float32):
        # simple kinematics
        self.yaw = np.float32(float(self.yaw) + wz * self.p.dt)
        self.pos[0] = np.float32(float(self.pos[0]) + v * np.cos(float(self.yaw)) * self.p.dt)
        self.pos[1] = np.float32(float(self.pos[1]) + v * np.sin(float(self.yaw)) * self.p.dt)
        self.pos[2] = np.float32(float(self.pos[2]) + vz * self.p.dt)

    def _lidar_scan(self) -> np.ndarray:
        # raycast against circular tree cross-sections in xy plane
        n_beams = self.p.lidar_num_beams
        max_range = float(self.p.lidar_range_max)
        ranges = np.full((n_beams,), max_range, dtype=np.float64)

        if self.trees is None or len(self.trees) == 0:
            return ranges.astype(np.float32)

        px = float(self.pos[0])
        py = float(self.pos[1])
        yaw = float(self.yaw)

        cos_yaw = float(np.cos(yaw))
        sin_yaw = float(np.sin(yaw))
        dir_x = cos_yaw * self._beam_rel_cos - sin_yaw * self._beam_rel_sin
        dir_y = sin_yaw * self._beam_rel_cos + cos_yaw * self._beam_rel_sin

        # query nearby trees only
        nearby_idx = self._query_nearby_trees(px, py, max_range)

        for idx in nearby_idx:
            tree = self.trees[idx]
            tx = float(tree[0])
            ty = float(tree[1])
            radius = float(tree[2])

            dx = tx - px
            dy = ty - py
            d2 = dx * dx + dy * dy
            r2 = radius * radius

            # If UAV center is inside a trunk, every beam collides at zero range
            if d2 <= r2:
                return np.zeros((n_beams,), dtype=np.float32)

            center_dist = float(np.sqrt(d2))
            if center_dist - radius > max_range:
                continue

            tree_rel = float(np.arctan2(dy, dx) - yaw)
            tree_rel = float(np.arctan2(np.sin(tree_rel), np.cos(tree_rel)))
            half_span = float(np.arcsin(min(0.999999, radius / center_dist)))

            center_float = (tree_rel - self._beam_angle_start) / self._beam_angle_step
            center_idx = int(np.round(center_float)) % n_beams
            half_beams = int(np.ceil(half_span / self._beam_angle_step))

            if half_beams >= n_beams // 2:
                candidate_idx = np.arange(n_beams, dtype=np.int64)
            else:
                offsets = np.arange(-half_beams, half_beams + 1, dtype=np.int64)
                candidate_idx = (center_idx + offsets) % n_beams

            cdx = dir_x[candidate_idx]
            cdy = dir_y[candidate_idx]

            # Ray-circle intersection from origin p + t * d, with |d|=1 and t>=0
            proj = dx * cdx + dy * cdy
            forward = proj > 0.0
            if not np.any(forward):
                continue

            perp2 = d2 - proj * proj
            hit = forward & (perp2 <= r2)
            if not np.any(hit):
                continue

            proj_hit = proj[hit]
            perp2_hit = perp2[hit]
            chord = np.sqrt(np.maximum(0.0, r2 - perp2_hit))
            t_near = np.maximum(0.0, proj_hit - chord)

            hit_idx = candidate_idx[hit]
            ranges[hit_idx] = np.minimum(ranges[hit_idx], t_near)

        np.clip(ranges, 0.0, max_range, out=ranges)
        return ranges.astype(np.float32)
    
    def _sample_forest(self):
        """ Samples a forest layout using the pure memory API of the worldgen code

        Raises:
            FileNotFoundError: Take a wild guess

        Returns:
            np.array: Array of shape (N, 3) with x, y coordinates and radius of each tree
        """
        # Try package import first; if this file is run directly, ensure project paths
        # are available so worldgen can still be imported.
        if self._worldgen_generate_positions_fn is None:
            module = None
            for module_name in (
                "worldgen.forest_worldgen.generate_world",
                "forest_worldgen.generate_world",
            ):
                try:
                    module = importlib.import_module(module_name)
                    break
                except ImportError:
                    continue

            if module is None:
                project_root = self._project_root()
                candidate_paths = (project_root, project_root / "worldgen")
                for path in candidate_paths:
                    path_str = str(path)
                    if path_str not in sys.path:
                        sys.path.insert(0, path_str)

                for module_name in (
                    "worldgen.forest_worldgen.generate_world",
                    "forest_worldgen.generate_world",
                ):
                    try:
                        module = importlib.import_module(module_name)
                        break
                    except ImportError:
                        continue

            if module is None:
                raise ImportError(
                    "Could not import world generator module. Tried "
                    "'worldgen.forest_worldgen.generate_world' and "
                    "'forest_worldgen.generate_world'."
                )

            self._worldgen_generate_positions_fn = getattr(module, "generate_positions_from_config")

        generate_positions_from_config = self._worldgen_generate_positions_fn

        config_path = self._worldgen_config_path()
        if not config_path.exists():
            raise FileNotFoundError(f"worldgen config not found: {config_path}")

        episode_seed = int(self.np_random.integers(0, np.iinfo(np.int32).max))
        episode_seed += int(self.p.worldgen_seed_offset)
        self._last_worldgen_seed = episode_seed

        positions_xy, world_config, _ = generate_positions_from_config(
            str(config_path),
            seed=episode_seed,
        )

        area_size = float(world_config['generation']['area_size'])
        self._world_half_extent = area_size / 2.0

        points_xy = np.asarray(positions_xy, dtype=np.float32)
        if points_xy.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        radii = self.np_random.normal(
            loc=self.p.tree_radius_mean,
            scale=max(self.p.tree_radius_std, 1e-6),
            size=(points_xy.shape[0],),
        ).astype(np.float32)
        radii = np.clip(radii, self.p.tree_radius_min, self.p.tree_radius_max)

        return np.column_stack([points_xy, radii]).astype(np.float32)

    def _point_clear_of_trees(self, x: float, y: float, clearance: float) -> bool:
        if self.trees is None or len(self.trees) == 0:
            return True
        dxy = self.trees[:, :2] - np.array([x, y], dtype=np.float32)
        d = np.linalg.norm(dxy, axis=1)
        needed = self.trees[:, 2] + clearance
        return bool(np.all(d >= needed))

    def _sample_free_xy(
        self,
        clearance: float,
        x_bounds: Optional[Tuple[float, float]] = None,
        y_bounds: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """ Sampling the world for a viable (clear of trees) start or goal position
            Obviously very junky approach, but should be fine for reasonable tree densities
            If it becomes an issue I can use some data structure or just setup the worldgen to output some pre-sampled free points

        Args:
            clearance (float): Minimum distance from any tree that the sampled point must be

        Returns:
            np.ndarray: Array of shape (2,) with x, y coordinates of the sampled point. If no valid point is found after max attempts
            returns (0, 0) which is likely to be in the middle of the world and hopefully not inside a tree (terrible idea I think, but saves us from looping forever)
            I should probably raise an exception
        """
        if x_bounds is None:
            x_bounds = (-self._world_half_extent, self._world_half_extent)
        if y_bounds is None:
            y_bounds = (-self._world_half_extent, self._world_half_extent)

        x_lo = float(max(-self._world_half_extent, min(x_bounds[0], x_bounds[1])))
        x_hi = float(min(self._world_half_extent, max(x_bounds[0], x_bounds[1])))
        y_lo = float(max(-self._world_half_extent, min(y_bounds[0], y_bounds[1])))
        y_hi = float(min(self._world_half_extent, max(y_bounds[0], y_bounds[1])))

        if x_lo >= x_hi or y_lo >= y_hi:
            return np.array([0.0, 0.0], dtype=np.float32)

        for _ in range(self.p.spawn_max_attempts):
            x = float(self.np_random.uniform(x_lo, x_hi))
            y = float(self.np_random.uniform(y_lo, y_hi))
            if self._point_clear_of_trees(x, y, clearance):
                return np.array([x, y], dtype=np.float32)

        return np.array([0.0, 0.0], dtype=np.float32)
    
    def _sample_start_pose(self):
        """ Sample a start pose (x, y, z, yaw) that is clear of trees and respects the clearance requirement

        Returns:
            np.array: Array of shape (3,) with x, y, z coordinates of the start position, and a separate float for yaw
            The z coordinate is set to the default_z_target for now, but I can change that later if I want to add some verticality to the start/goal sampling
        """
        half = float(self._world_half_extent)
        band_inner = 0.55 * half

        for _ in range(self.p.spawn_max_attempts):
            axis = int(self.np_random.integers(0, 2))
            side = -1.0 if bool(self.np_random.integers(0, 2)) else 1.0

            if axis == 0:
                x_bounds = (side * band_inner, side * half)
                y_bounds = (-half, half)
            else:
                x_bounds = (-half, half)
                y_bounds = (side * band_inner, side * half)

            start_xy = self._sample_free_xy(
                clearance=self.p.start_goal_clearance,
                x_bounds=x_bounds,
                y_bounds=y_bounds,
            )
            if not np.allclose(start_xy, 0.0):
                yaw = np.float32(self.np_random.uniform(-np.pi, np.pi))
                return np.array([start_xy[0], start_xy[1], self.p.default_z_target], dtype=np.float32), yaw

        start_xy = self._sample_free_xy(clearance=self.p.start_goal_clearance)
        yaw = np.float32(self.np_random.uniform(-np.pi, np.pi))
        return np.array([start_xy[0], start_xy[1], self.p.default_z_target], dtype=np.float32), yaw
    
    def _sample_goal_pose(self):
        """ Same idea as sample start pose, find a free (x, y) goal position

        Returns:
            np.array: Array of shape (3,) with x, y, z coordinates of the goal position
        """
        half = float(self._world_half_extent)
        band_inner = 0.55 * half

        start_xy = self.pos[:2].astype(np.float32)
        if abs(float(start_xy[0])) >= abs(float(start_xy[1])):
            axis = 0
            start_side = 1.0 if float(start_xy[0]) >= 0.0 else -1.0
        else:
            axis = 1
            start_side = 1.0 if float(start_xy[1]) >= 0.0 else -1.0

        goal_side = -start_side

        for _ in range(self.p.spawn_max_attempts):
            if axis == 0:
                x_bounds = (goal_side * band_inner, goal_side * half)
                y_bounds = (-half, half)
            else:
                x_bounds = (-half, half)
                y_bounds = (goal_side * band_inner, goal_side * half)

            goal_xy = self._sample_free_xy(
                clearance=self.p.start_goal_clearance,
                x_bounds=x_bounds,
                y_bounds=y_bounds,
            )
            if np.allclose(goal_xy, 0.0):
                continue

            if np.linalg.norm(goal_xy - self.pos[:2]) >= self.p.min_start_goal_distance:
                return np.array([goal_xy[0], goal_xy[1], self.p.default_z_target], dtype=np.float32)

        fallback = np.array([
            -np.sign(float(start_xy[0])) * band_inner if axis == 0 else 0.0,
            -np.sign(float(start_xy[1])) * band_inner if axis == 1 else 0.0,
        ], dtype=np.float32)
        return np.array([fallback[0], fallback[1], self.p.default_z_target], dtype=np.float32)
    
if __name__ == "__main__":
    # Minimal test params - for real usage, load from sac.yaml via utils.build_env_params
    params = SimParams(
        dt=0.1, lidar_num_beams=180, lidar_range_max=30.0, v_max=6.0, wz_max=2.5, vz_max=2.0,
        r_safe=0.6, episode_seconds=30.0, goal_tolerance=0.5, world_radius=20.0,
        collision_threshold=0.05, default_z_target=2.0, z_error_scale=5.0,
        reward_progress_scale=2.0, reward_speed_scale=0.02, reward_step_penalty=0.02,
        reward_proximity_scale=0.2, reward_shield_penalty=0.02, reward_collision_penalty=20.0,
        reward_success_bonus=5.0, reward_truncation_penalty=1.0, reward_yaw_rate_scale=0.03,
        reward_stall_penalty=0.05, progress_stall_threshold=0.02, yaw_penalty_speed_gate=0.25,
        shield_floor_z_min=0.05, shield_yaw_damping=0.35, shield_lookahead_margin=1.0,
        worldgen_config_relpath="configs/worldgen/worldgen_run.yaml", worldgen_seed_offset=0,
        tree_radius_mean=0.25, tree_radius_std=0.05, tree_radius_min=0.10, tree_radius_max=0.60,
        start_goal_clearance=1.0, min_start_goal_distance=8.0, spawn_max_attempts=500
    )
    env = ForestNavEnv(params)
    check_env(env, warn=True)