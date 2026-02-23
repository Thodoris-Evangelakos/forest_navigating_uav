from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import math
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fastsim_forest_nav.envs.forest_nav_env import SimParams


@dataclass
class GazeboParams(SimParams):
    odom_topic: str = "/odom"
    scan_topic: str = "/scan"
    cmd_vel_topic: str = "/cmd_vel"

    use_sim_reset_service: bool = False
    reset_service_name: str = "/reset_simulation"

    spin_timeout_sec: float = 2.0
    settle_time_sec: float = 0.05

    # goal handling (only external absolute state + goal are assumed)
    fixed_goal: list[float] = field(default_factory=lambda: [8.0, 0.0, 2.0])
    randomize_goal_on_reset: bool = True

    # lidar guardrails
    lidar_min_valid_range: float = 0.03

    # lidar-based safety shield
    shield_front_arc_deg: float = 70.0
    shield_ttc_threshold_sec: float = 1.0


class GazeboForestNavEnv(gym.Env):
    def __init__(self, params: GazeboParams, render_mode: Optional[str] = None):
        super().__init__()
        self.p = params
        self.render_mode = render_mode

        obs_dim = self.p.lidar_num_beams + 6
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self._t = 0.0
        self._step_count = 0

        self.pos = np.zeros(3, dtype=np.float32)
        self.yaw = np.float32(0.0)
        self.v = np.float32(0.0)
        self.wz = np.float32(0.0)
        self.vz = np.float32(0.0)

        self.goal = np.asarray(self.p.fixed_goal, dtype=np.float32)
        if self.goal.shape != (3,):
            raise ValueError("GazeboParams.fixed_goal must be [x, y, z]")

        self.z_target = np.float32(self.goal[2])
        self.trees = None
        self._prev_dist: Optional[float] = None
        self._world_half_extent = float(self.p.world_radius)

        self._latest_scan_raw: np.ndarray = np.full((self.p.lidar_num_beams,), self.p.lidar_range_max, dtype=np.float32)
        self._latest_scan_angles: np.ndarray = np.linspace(-np.pi, np.pi, self.p.lidar_num_beams, endpoint=False, dtype=np.float32)
        self._have_scan = False
        self._have_odom = False

        self._ros = self._init_ros_interfaces()

    def _init_ros_interfaces(self) -> dict[str, Any]:
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import qos_profile_sensor_data
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import LaserScan

            reset_client = None
            if self.p.use_sim_reset_service:
                from std_srvs.srv import Empty

            if not rclpy.ok():
                rclpy.init(args=None)

            node_name = f"forest_nav_gazebo_env_{int(time.time() * 1e6) % 1_000_000_000}"
            node = Node(node_name)

            cmd_pub = node.create_publisher(Twist, self.p.cmd_vel_topic, 10)

            def odom_callback(msg: Odometry) -> None:
                self.pos[0] = np.float32(msg.pose.pose.position.x)
                self.pos[1] = np.float32(msg.pose.pose.position.y)
                self.pos[2] = np.float32(msg.pose.pose.position.z)

                q = msg.pose.pose.orientation
                self.yaw = np.float32(math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)))
                self._have_odom = True

            def scan_callback(msg: LaserScan) -> None:
                ranges = np.asarray(msg.ranges, dtype=np.float32)
                if ranges.size == 0:
                    return

                sanitized = np.where(np.isfinite(ranges), ranges, self.p.lidar_range_max)
                sanitized = np.clip(sanitized, self.p.lidar_min_valid_range, self.p.lidar_range_max)

                angle_min = float(msg.angle_min)
                angle_increment = float(msg.angle_increment)
                angles = angle_min + angle_increment * np.arange(sanitized.shape[0], dtype=np.float32)
                angles = np.arctan2(np.sin(angles), np.cos(angles))

                self._latest_scan_raw = sanitized
                self._latest_scan_angles = angles.astype(np.float32)
                self._have_scan = True

            node.create_subscription(Odometry, self.p.odom_topic, odom_callback, qos_profile_sensor_data)
            node.create_subscription(LaserScan, self.p.scan_topic, scan_callback, qos_profile_sensor_data)

            if self.p.use_sim_reset_service:
                from std_srvs.srv import Empty

                reset_client = node.create_client(Empty, self.p.reset_service_name)

            return {
                "enabled": True,
                "rclpy": rclpy,
                "node": node,
                "Twist": Twist,
                "cmd_pub": cmd_pub,
                "reset_client": reset_client,
            }
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "GazeboForestNavEnv requires ROS2 Python interfaces (rclpy, geometry_msgs, nav_msgs, sensor_msgs). "
                "Install ROS2 and source the environment before running Gazebo backend."
            ) from exc

    def _spin_until_ready(self, timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec
        rclpy = self._ros["rclpy"]
        node = self._ros["node"]

        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.01)
            if self._have_odom and self._have_scan:
                return

        raise TimeoutError(
            f"Timed out waiting for odom/scan topics ({self.p.odom_topic}, {self.p.scan_topic}). "
            "Check Gazebo and ROS2 bridges are running."
        )

    def _spin_for(self, duration_sec: float) -> None:
        end_t = time.monotonic() + max(0.0, duration_sec)
        rclpy = self._ros["rclpy"]
        node = self._ros["node"]
        while time.monotonic() < end_t:
            rclpy.spin_once(node, timeout_sec=0.001)

    def _call_reset_service_if_enabled(self) -> None:
        client = self._ros.get("reset_client")
        if client is None:
            return

        from std_srvs.srv import Empty

        if not client.wait_for_service(timeout_sec=self.p.spin_timeout_sec):
            raise TimeoutError(f"Reset service not available: {self.p.reset_service_name}")

        req = Empty.Request()
        future = client.call_async(req)
        rclpy = self._ros["rclpy"]
        node = self._ros["node"]
        deadline = time.monotonic() + self.p.spin_timeout_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.01)
            if future.done():
                _ = future.result()
                return

        raise TimeoutError(f"Reset service call timed out: {self.p.reset_service_name}")

    def _publish_cmd(self, v: float, wz: float, vz: float) -> None:
        Twist = self._ros["Twist"]
        msg = Twist()
        msg.linear.x = float(v)
        msg.linear.y = 0.0
        msg.linear.z = float(vz)
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(wz)
        self._ros["cmd_pub"].publish(msg)

    def _resample_lidar(self) -> tuple[np.ndarray, np.ndarray]:
        src_ranges = self._latest_scan_raw
        src_angles = self._latest_scan_angles

        if src_ranges.shape[0] == self.p.lidar_num_beams:
            return src_ranges.astype(np.float32), src_angles.astype(np.float32)

        target_angles = np.linspace(-np.pi, np.pi, self.p.lidar_num_beams, endpoint=False, dtype=np.float32)

        order = np.argsort(src_angles)
        sorted_angles = src_angles[order]
        sorted_ranges = src_ranges[order]

        wrapped_angles = np.concatenate([sorted_angles - 2.0 * np.pi, sorted_angles, sorted_angles + 2.0 * np.pi])
        wrapped_ranges = np.concatenate([sorted_ranges, sorted_ranges, sorted_ranges])

        interp = np.interp(target_angles, wrapped_angles, wrapped_ranges)
        interp = np.clip(interp, self.p.lidar_min_valid_range, self.p.lidar_range_max)
        return interp.astype(np.float32), target_angles

    def _dist_to_goal(self) -> float:
        return float(np.linalg.norm(self.goal[:2] - self.pos[:2]))

    def _pack_obs(self, lidar: np.ndarray, dist: float, v: np.float32, wz: np.float32) -> np.ndarray:
        lidar_n = np.clip(lidar / self.p.lidar_range_max, 0.0, 1.0).astype(np.float32)

        dx = float(self.goal[0] - self.pos[0])
        dy = float(self.goal[1] - self.pos[1])
        theta = np.arctan2(dy, dx) - float(self.yaw)
        c, s = np.cos(theta), np.sin(theta)

        dist_n = np.clip(dist / (2.0 * self._world_half_extent), 0.0, 1.0)
        v_n = np.clip(v / self.p.v_max, -1.0, 1.0)
        wz_n = np.clip(wz / self.p.wz_max, -1.0, 1.0)
        z_err = np.clip((float(self.z_target) - float(self.pos[2])) / self.p.z_error_scale, -1.0, 1.0)

        tail = np.array([c, s, dist_n, v_n, wz_n, z_err], dtype=np.float32)
        return np.concatenate([lidar_n, tail], axis=0).astype(np.float32)

    def _apply_lidar_shield(
        self,
        v: float,
        wz: float,
        vz: float,
        lidar_ranges: np.ndarray,
        lidar_angles: np.ndarray,
    ) -> tuple[np.float32, np.float32, np.float32, int, float]:
        safe_v = float(v)
        safe_wz = float(wz)
        safe_vz = float(vz)
        shield_active = 0

        front_arc = np.deg2rad(max(1.0, float(self.p.shield_front_arc_deg)))
        front_mask = np.abs(lidar_angles) <= (front_arc / 2.0)

        if np.any(front_mask) and safe_v > 0.0:
            front_min = float(np.min(lidar_ranges[front_mask]))

            # limit speed so next step cannot penetrate safety radius
            v_clearance_cap = max(0.0, (front_min - self.p.r_safe) / max(self.p.dt, 1e-4))

            # time-to-collision cap for better dampening at speed
            v_ttc_cap = max(0.0, front_min / max(self.p.shield_ttc_threshold_sec, 1e-3))

            v_cap = min(v_clearance_cap, v_ttc_cap)
            if safe_v > v_cap:
                safe_v = v_cap
                shield_active = 1

        if float(self.pos[2]) <= self.p.shield_floor_z_min and safe_vz < 0.0:
            safe_vz = 0.0
            shield_active = 1

        cmd_norm = abs(v) + abs(wz) + abs(vz)
        delta_norm = abs(v - safe_v) + abs(wz - safe_wz) + abs(vz - safe_vz)
        shield_delta = delta_norm / cmd_norm if cmd_norm > 1e-6 else 0.0

        return np.float32(safe_v), np.float32(safe_wz), np.float32(safe_vz), int(shield_active), float(shield_delta)

    def _get_info(self, **kwargs) -> dict[str, Any]:
        lidar, _ = self._resample_lidar()
        info = {
            "dist_to_goal": float(self._dist_to_goal()),
            "min_range": float(np.min(lidar)),
            "tree_count": 0,
            "worldgen_seed": None,
        }
        info.update(kwargs)
        info["is_success"] = bool(info.get("success", False))
        return info

    def _sample_goal_pose(self) -> np.ndarray:
        if not self.p.randomize_goal_on_reset:
            return np.asarray(self.p.fixed_goal, dtype=np.float32)

        for _ in range(self.p.spawn_max_attempts):
            gx = float(self.np_random.uniform(-self._world_half_extent, self._world_half_extent))
            gy = float(self.np_random.uniform(-self._world_half_extent, self._world_half_extent))
            candidate = np.array([gx, gy, float(self.p.default_z_target)], dtype=np.float32)
            if np.linalg.norm(candidate[:2] - self.pos[:2]) >= self.p.min_start_goal_distance:
                return candidate

        fallback = np.array([self._world_half_extent * 0.5, 0.0, float(self.p.default_z_target)], dtype=np.float32)
        return fallback.astype(np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        super().reset(seed=seed)
        self._t = 0.0
        self._step_count = 0

        self._call_reset_service_if_enabled()
        self._spin_until_ready(self.p.spin_timeout_sec)
        self._spin_for(self.p.settle_time_sec)

        self.goal = self._sample_goal_pose()
        self.z_target = np.float32(self.goal[2])

        self.v = np.float32(0.0)
        self.wz = np.float32(0.0)
        self.vz = np.float32(0.0)
        self._publish_cmd(0.0, 0.0, 0.0)

        self._prev_dist = self._dist_to_goal()
        lidar, _ = self._resample_lidar()
        obs = self._pack_obs(lidar, self._prev_dist, self.v, self.wz)
        info = self._get_info(shield_active=0, collision=0, success=0, shield_delta=0.0)
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)

        cmd_v = float(action[0]) * self.p.v_max
        cmd_wz = float(action[1]) * self.p.wz_max
        cmd_vz = float(action[2]) * self.p.vz_max

        lidar_ranges, lidar_angles = self._resample_lidar()
        safe_v, safe_wz, safe_vz, shield_active, shield_delta = self._apply_lidar_shield(
            cmd_v,
            cmd_wz,
            cmd_vz,
            lidar_ranges,
            lidar_angles,
        )

        self._publish_cmd(float(safe_v), float(safe_wz), float(safe_vz))
        self.v = np.float32(safe_v)
        self.wz = np.float32(safe_wz)
        self.vz = np.float32(safe_vz)

        self._spin_for(self.p.dt)

        lidar_ranges, _ = self._resample_lidar()

        dist = self._dist_to_goal()
        prev_dist = self._prev_dist if self._prev_dist is not None else dist
        d_progress = prev_dist - dist
        self._prev_dist = dist

        min_range = float(np.min(lidar_ranges))
        collision = int(min_range < self.p.collision_threshold)
        success = int(dist < self.p.goal_tolerance)

        reward = 0.0
        reward += self.p.reward_progress_scale * d_progress
        reward += self.p.reward_speed_scale * (safe_v / self.p.v_max)
        reward -= self.p.reward_step_penalty

        if min_range < self.p.r_safe:
            reward -= self.p.reward_proximity_scale * (self.p.r_safe - min_range) / self.p.r_safe
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

        obs = self._pack_obs(lidar_ranges, dist, self.v, self.wz)
        info = self._get_info(
            shield_active=shield_active,
            collision=collision,
            success=success,
            shield_delta=shield_delta,
        )
        return obs, float(reward), terminated, truncated, info

    def render(self):
        return f"t={self._t:.2f} pos={self.pos} yaw={float(self.yaw):.2f} goal={self.goal} v={float(self.v):.2f}"

    def close(self):
        try:
            self._publish_cmd(0.0, 0.0, 0.0)
        except Exception:
            pass

        if self._ros.get("enabled", False):
            node = self._ros.get("node")
            if node is not None:
                node.destroy_node()
