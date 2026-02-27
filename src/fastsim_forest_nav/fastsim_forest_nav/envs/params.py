"""Define simulation parameter dataclasses for navigation environments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SimParams:
    """Store configurable dynamics, sensing, reward, and sampling parameters."""

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
    default_z_target: float
    z_error_scale: float
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
    shield_floor_z_min: float
    shield_yaw_damping: float
    shield_lookahead_margin: float
    worldgen_config_relpath: str
    worldgen_seed_offset: int
    tree_radius_mean: float
    tree_radius_std: float
    tree_radius_min: float
    tree_radius_max: float
    start_goal_clearance: float
    min_start_goal_distance: float
    spawn_max_attempts: int
    worldgen_resample_every_n_episodes: int = field(default=1)
    worldgen_verbose: bool = field(default=False)
    start_goal_tree_exclusion_radius: float = field(default=3.0)
    shield_ceiling_z_max: float = field(default=0.0)
    action_mode: str = field(default="hybrid")
    accel_v_max: float = field(default=3.0)
    accel_wz_max: float = field(default=2.0)
    accel_vz_max: float = field(default=1.5)
    decel_v_max: float = field(default=0.0)
    decel_wz_max: float = field(default=0.0)
    decel_vz_max: float = field(default=0.0)
    drone_radius: float = field(default=0.0)
    reward_accel_clip_penalty: float = field(default=0.0)
    boundary_margin: float = field(default=1.0)
