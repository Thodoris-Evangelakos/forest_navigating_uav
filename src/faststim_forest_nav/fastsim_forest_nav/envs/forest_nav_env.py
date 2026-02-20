#from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env

@dataclass
class SimParams:
    dt: float = 0.1
    lidar_num_beams: int = 180
    lidar_range_max: float = 30.0
    v_max: float = 6.0
    wz_max: float = 2.5
    vz_max: float = 2.0
    r_safe: float = 0.6
    episode_seconds: float = 30.0
    goal_tolerance: float = 0.5
    world_radius: float = 20.0

    collision_threshold: float = 0.05

    # vertical tracking / observation scaling
    default_z_target: float = 2.0
    z_error_scale: float = 5.0

    # reward shaping
    reward_progress_scale: float = 2.0
    reward_speed_scale: float = 0.05
    reward_step_penalty: float = 0.01
    reward_proximity_scale: float = 0.2
    reward_shield_penalty: float = 0.02
    reward_collision_penalty: float = 5.0
    reward_success_bonus: float = 5.0
    reward_truncation_penalty: float = 0.0

    # safety shield
    shield_floor_z_min: float = 0.05

class ForestNavEnv(gym.Env):
    
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
            low = -1.0, high = 1.0, shape=(obs_dim), dtype=np.float32
        )

        # Action space: a[0] = forward speed command, a[1] = yaw rate command, a[2] = vertical speed command
        # all 3 normalized in their respective v max. E.g. v = a[0] * v_max
        # SAC in SB3 is built for continuous Box actions
        self.action_space = spaces.Box(
            low = -1.0, high = 1.0, shape = (3), dtype = np.float32
        )

        self._t = 0.0
        self._step_count = 0

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

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        
        # gym seeding contract
        super().reset(seed=seed)
        self._t = 0.0
        self._step_count = 0

        # Sample world, start, goal
        self.trees = self._sample_forest()
        self.pos, self.yaw = self._sample_start_pose() # do I really need that?
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

        # reward and termination
        dist = self._dist_to_goal()
        prev_dist = self._prev_dist if self._prev_dist is not None else dist
        d_progress = prev_dist - dist
        self._prev_dist = dist

        min_range = float(np.min(lidar))
        collision = int(min_range < self.p.collision_threshold)
        success = int(dist < self.p.goal_tolerance)

        # I should hyperparameterize all weights probably, this looks hella junky
        reward = 0.0
        reward += self.p.reward_progress_scale * d_progress # encourage progress towards goal
        reward += self.p.reward_speed_scale * (safe_v / self.p.v_max) # encourage faster speeds
        reward -= self.p.reward_step_penalty # small penalty for each step to encourage faster completion

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
        return f"t={self._t:2f} pos = {self.pos} yaw = {float(self.yaw):.2f} goal = {self.goal} v = {float(self.v):.2f}"
    
    def close(self):
        pass

    # ~~~ HELPER FUNCTIONS DEFINED BELOW ~~~

    def _pack_obs(self, lidar: np.ndarray, dist: float, v: float, wz: float) -> np.ndarray:
        # normalize lidar
        lidar_n = np.clip(lidar / self.p.lidar_range_max, 0.0, 1.0). astype(np.float32) # not sure where that'd need clipping but safe

        # goal direction in body frame
        dx = float(self.goal[0] - self.pos[0])
        dy = float(self.goal[1] - self.pos[1])
        theta = np.arctan2(dy, dx) - float(self.yaw)
        c, s = np.cos(theta), np.sin(theta)

        dist_n = np.clip(dist / (2.0 * self.p.world_radius), 0.0, 1.0)
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
        info = {
            "dist_to_goal": float(self._dist_to_goal()),
            "min_range": float(np.min(self._lidar_scan())),
        }
        info.update(kwargs)
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

        # horizontal tree avoidance
        if self.trees is not None and len(self.trees) > 0:
            vel_dir = np.array(
                [np.cos(float(self.yaw)), np.sin(float(self.yaw))],
                dtype=np.float64,
            )
            pos_xy = self.pos[:2].astype(np.float64)

            # FIXME checking all trees every step is not super efficient, but should be fine for now with small numbers. Can optimize later with some spatial data structure if needed
            for tree in self.trees:
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

        # normalised intervention magnitude
        cmd_norm = abs(v) + abs(wz) + abs(vz)
        delta_norm = abs(v - safe_v) + abs(wz - safe_wz) + abs(vz - safe_vz)
        shield_delta = delta_norm / cmd_norm if cmd_norm > 1e-6 else 0.0

        return safe_v, safe_wz, safe_vz, int(shield_active), float(shield_delta)

    def _integrate(self, v: float, wz: float, vz: float):
        # simple kinematics
        self.yaw = np.float32(float(self.yaw) + wz * self.p.dt)
        self.pos[0] = np.float32(float(self.pos[0]) + v * np.cos(float(self.yaw)) * self.p.dt)
        self.pos[1] = np.float32(float(self.pos[1]) + v * np.sin(float(self.yaw)) * self.p.dt)
        self.pos[2] = np.float32(float(self.pos[2]) + vz * self.p.dt)

    def _lidar_scan(self) -> np.ndarray:
        # simulate lidar scan by raycasting against the cylinders
        # return shape (N,) float32 with values in [0, range_max] or range_max if no hit
        return np.full((self.p.lidar_num_beams,), self.p.lidar_range_max, dtype=np.float32)
    
    def _sample_forest(self):
        return np.zeros((0,3), dtype=np.float32)
    
    def _sample_start_pose(self):
        return np.array([0.0, 0.0, 2.0], dtype=np.float32), np.float32(0.0)
    
    def _sample_goal_pose(self):
        return np.array([10.0, 0.0, 2.0], dtype=np.float32)
    
if __name__ == "__main__":
    params = SimParams()
    env = ForestNavEnv(params)
    check_env(env, warn=True)