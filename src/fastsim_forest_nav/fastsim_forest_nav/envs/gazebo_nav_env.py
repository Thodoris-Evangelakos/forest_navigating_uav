from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import math
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fastsim_forest_nav.dynamics.controls import (
    accel_limit_velocity as _accel_limit_velocity,
    approach_speed_cap as _approach_speed_cap,
    map_normalized_accel as _map_normalized_accel,
)
from fastsim_forest_nav.envs.params import SimParams
from fastsim_forest_nav.safety.geometry import (
    effective_drone_radius as _effective_drone_radius,
    protected_radius as _protected_radius,
    soft_clearance_margin as _soft_clearance_margin,
)


@dataclass
class GazeboParams(SimParams):
    """Gazebo-specific parameters - all values must come from config.

    All fields carry sensible defaults so that Python's dataclass inheritance
    rules are satisfied (SimParams gains optional fields with defaults at the
    end; child-class fields without defaults are not permitted after them).
    The defaults here match the values in sac_gazebo.yaml and serve as
    documentation; they should always be explicitly set via that config.
    """
    odom_topic: str = "/odom"
    scan_topic: str = "/scan"
    cmd_vel_topic: str = "/model/uav1/cmd_vel"
    use_sim_reset_service: bool = False
    reset_service_name: str = "/reset_simulation"
    spin_timeout_sec: float = 2.0
    settle_time_sec: float = 0.05
    fixed_goal: list = field(default_factory=lambda: [8.0, 0.0, 2.0])
    randomize_goal_on_reset: bool = False
    lidar_min_valid_range: float = 0.03
    shield_front_arc_deg: float = 55.0
    shield_ttc_threshold_sec: float = 0.75


class GazeboForestNavEnv(gym.Env):
    def __init__(self, params: GazeboParams, render_mode: Optional[str] = None):
        super().__init__()
        self.p = params
        self.render_mode = render_mode

        if self.p.action_mode != "hybrid":
            raise ValueError(
                f"Unsupported action_mode='{self.p.action_mode}'. "
                "Only action_mode='hybrid' is supported."
            )

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

        self._z_hold = np.float32(0.3)
        self.z_target = self._z_hold
        self.trees = None
        self._prev_dist: Optional[float] = None
        self._world_half_extent = float(self.p.world_radius)

        self._latest_scan_raw: np.ndarray = np.full((self.p.lidar_num_beams,), self.p.lidar_range_max, dtype=np.float32)
        self._latest_scan_angles: np.ndarray = np.linspace(-np.pi, np.pi, self.p.lidar_num_beams, endpoint=False, dtype=np.float32)
        self._have_scan = False
        self._have_odom = False

        # odometry twist feedback (used by hybrid dynamics to measure applied velocity)
        self._odom_v = np.float32(0.0)
        self._odom_wz = np.float32(0.0)
        self._odom_vz = np.float32(0.0)
        self._have_odom_twist = False

        self._ros = self._init_ros_interfaces()

    def _effective_world_half_extent(self) -> float:
        return max(1e-3, float(self._world_half_extent) - float(self.p.boundary_margin))

    def _distance_to_world_boundary(self) -> float:
        half = self._effective_world_half_extent()
        dx = half - abs(float(self.pos[0]))
        dy = half - abs(float(self.pos[1]))
        return float(min(dx, dy))

    def _distance_to_world_boundary_along_motion(self, signed_speed: float) -> float:
        if abs(float(signed_speed)) <= 1e-9:
            return float("inf")

        half = self._effective_world_half_extent()
        x = float(self.pos[0])
        y = float(self.pos[1])

        motion_sign = 1.0 if float(signed_speed) >= 0.0 else -1.0
        dx = motion_sign * float(np.cos(float(self.yaw)))
        dy = motion_sign * float(np.sin(float(self.yaw)))

        candidates: list[float] = []
        eps = 1e-9

        if abs(dx) > eps:
            wall_x = half if dx > 0.0 else -half
            tx = (wall_x - x) / dx
            if tx >= 0.0:
                y_hit = y + tx * dy
                if -half <= y_hit <= half:
                    candidates.append(float(tx))

        if abs(dy) > eps:
            wall_y = half if dy > 0.0 else -half
            ty = (wall_y - y) / dy
            if ty >= 0.0:
                x_hit = x + ty * dx
                if -half <= x_hit <= half:
                    candidates.append(float(ty))

        if not candidates:
            return float("inf")
        return float(min(candidates))

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

                # capture twist for hybrid dynamics feedback
                self._odom_v = np.float32(msg.twist.twist.linear.x)
                self._odom_wz = np.float32(msg.twist.twist.angular.z)
                self._odom_vz = np.float32(msg.twist.twist.linear.z)
                self._have_odom_twist = True

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

            # also treat configured world boundary as virtual front obstacle
            boundary_front = self._distance_to_world_boundary_along_motion(safe_v)
            front_min = min(front_min, boundary_front)

            v_decel_cap = float(self.p.decel_v_max) if float(self.p.decel_v_max) > 0.0 else float(self.p.accel_v_max)

            # dynamics-aware speed cap so one-step travel + braking distance stays safe
            protected_radius = _protected_radius(self.p)
            gap = max(0.0, front_min - protected_radius)
            v_clearance_cap = _approach_speed_cap(gap, v_decel_cap, self.p.dt)

            # time-to-collision cap for better dampening at speed
            v_ttc_cap = max(0.0, front_min / max(self.p.shield_ttc_threshold_sec, 1e-3))

            v_cap = min(v_clearance_cap, v_ttc_cap)
            if safe_v > v_cap:
                safe_v = v_cap
                shield_active = 1

        if float(self.pos[2]) <= self.p.shield_floor_z_min and safe_vz < 0.0:
            safe_vz = 0.0
            shield_active = 1

        if float(self.p.shield_ceiling_z_max) > 0.0 and float(self.pos[2]) >= float(self.p.shield_ceiling_z_max) and safe_vz > 0.0:
            safe_vz = 0.0
            shield_active = 1

        cmd_norm = abs(v) + abs(wz) + abs(vz)
        delta_norm = abs(v - safe_v) + abs(wz - safe_wz) + abs(vz - safe_vz)
        shield_delta = delta_norm / cmd_norm if cmd_norm > 1e-6 else 0.0

        return np.float32(safe_v), np.float32(safe_wz), np.float32(safe_vz), int(shield_active), float(shield_delta)

    def _get_info(self, **kwargs) -> dict[str, Any]:
        lidar, _ = self._resample_lidar()
        boundary_range = self._distance_to_world_boundary()
        info = {
            "dist_to_goal": float(self._dist_to_goal()),
            "min_range": float(min(float(np.min(lidar)), boundary_range)),
            "tree_count": 0,
            "worldgen_seed": None,
        }
        info.update(kwargs)
        info["is_success"] = bool(info.get("success", False))
        return info

    def _sample_xy_in_bounds(self, x_bounds: tuple[float, float], y_bounds: tuple[float, float]) -> np.ndarray:
        half = self._effective_world_half_extent()
        x_lo = float(max(-half, min(x_bounds[0], x_bounds[1])))
        x_hi = float(min(half, max(x_bounds[0], x_bounds[1])))
        y_lo = float(max(-half, min(y_bounds[0], y_bounds[1])))
        y_hi = float(min(half, max(y_bounds[0], y_bounds[1])))

        if x_lo >= x_hi or y_lo >= y_hi:
            return np.array([0.0, 0.0], dtype=np.float32)

        gx = float(self.np_random.uniform(x_lo, x_hi))
        gy = float(self.np_random.uniform(y_lo, y_hi))
        return np.array([gx, gy], dtype=np.float32)

    def _sample_goal_pose(self) -> np.ndarray:
        if not self.p.randomize_goal_on_reset:
            return np.asarray(self.p.fixed_goal, dtype=np.float32)

        half = float(self._effective_world_half_extent())
        band_inner = 0.55 * half

        start_xy = self.pos[:2].astype(np.float32)
        start_side_x = 1.0 if float(start_xy[0]) >= 0.0 else -1.0
        start_side_y = 1.0 if float(start_xy[1]) >= 0.0 else -1.0
        goal_side_x = -start_side_x
        goal_side_y = -start_side_y

        for _ in range(self.p.spawn_max_attempts):
            x_bounds = (goal_side_x * band_inner, goal_side_x * half)
            y_bounds = (goal_side_y * band_inner, goal_side_y * half)

            goal_xy = self._sample_xy_in_bounds(x_bounds=x_bounds, y_bounds=y_bounds)
            if np.allclose(goal_xy, 0.0):
                continue

            candidate = np.array([goal_xy[0], goal_xy[1], float(self.p.default_z_target)], dtype=np.float32)
            if np.linalg.norm(candidate[:2] - self.pos[:2]) >= self.p.min_start_goal_distance:
                return candidate

        fallback = np.array([
            goal_side_x * band_inner,
            goal_side_y * band_inner,
            float(self.p.default_z_target),
        ], dtype=np.float32)
        return fallback.astype(np.float32)

    def _action_to_velocity_command(
        self,
        action: np.ndarray,
    ) -> tuple[float, float, float, int]:
        """Convert normalized action to velocity command locally via acceleration integration.

        Gazebo listens to velocity commands on ``/cmd_vel``. To keep control dynamics
        acceleration-based, we integrate acceleration against the current measured
        (or last commanded) velocity state inside the environment, then publish the
        resulting velocity command. Vertical action is intentionally ignored in
        Gazebo, and altitude is held at z=0.3m using a local feedback command.

        Returns ``(cmd_v, cmd_wz, cmd_vz, accel_clipped)``.
        """
        accel_v = _map_normalized_accel(float(action[0]), self.p.accel_v_max, self.p.decel_v_max)
        accel_wz = _map_normalized_accel(float(action[1]), self.p.accel_wz_max, self.p.decel_wz_max)

        pre_clip_v = float(self.v) + accel_v * float(self.p.dt)
        pre_clip_wz = float(self.wz) + accel_wz * float(self.p.dt)

        cmd_v = float(np.clip(pre_clip_v, -self.p.v_max, self.p.v_max))
        cmd_wz = float(np.clip(pre_clip_wz, -self.p.wz_max, self.p.wz_max))

        z_err = float(self._z_hold - float(self.pos[2]))
        cmd_vz = float(np.clip(2.0 * z_err, -self.p.vz_max, self.p.vz_max))

        accel_clipped = int(
            (cmd_v != pre_clip_v) or (cmd_wz != pre_clip_wz)
        )

        return cmd_v, cmd_wz, cmd_vz, accel_clipped

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        super().reset(seed=seed)
        self._t = 0.0
        self._step_count = 0

        self._call_reset_service_if_enabled()
        self._spin_until_ready(self.p.spin_timeout_sec)
        self._spin_for(self.p.settle_time_sec)

        self.goal = self._sample_goal_pose()
        self.z_target = self._z_hold

        self.v = np.float32(0.0)
        self.wz = np.float32(0.0)
        self.vz = np.float32(0.0)
        self._have_odom_twist = False  # discard stale twist until first post-reset odom arrives
        self._publish_cmd(0.0, 0.0, 0.0)

        self._prev_dist = self._dist_to_goal()
        lidar, _ = self._resample_lidar()
        obs = self._pack_obs(lidar, self._prev_dist, self.v, self.wz)
        info = self._get_info(shield_active=0, collision=0, success=0, shield_delta=0.0)
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)

        # start each control step from measured state when available
        if self._have_odom_twist:
            self.v = self._odom_v
            self.wz = self._odom_wz
            self.vz = self._odom_vz

        cmd_v, cmd_wz, cmd_vz, accel_clipped = self._action_to_velocity_command(action)

        lidar_ranges, lidar_angles = self._resample_lidar()
        safe_v, safe_wz, safe_vz, shield_active, shield_delta = self._apply_lidar_shield(
            cmd_v,
            cmd_wz,
            cmd_vz,
            lidar_ranges,
            lidar_angles,
        )
        applied_v, slew_clipped_v = _accel_limit_velocity(
            desired=float(safe_v),
            current=float(self.v),
            accel_max=float(self.p.accel_v_max),
            dt=float(self.p.dt),
            decel_max=float(self.p.decel_v_max),
        )
        applied_wz, slew_clipped_wz = _accel_limit_velocity(
            desired=float(safe_wz),
            current=float(self.wz),
            accel_max=float(self.p.accel_wz_max),
            dt=float(self.p.dt),
            decel_max=float(self.p.decel_wz_max),
        )
        applied_vz, slew_clipped_vz = _accel_limit_velocity(
            desired=float(safe_vz),
            current=float(self.vz),
            accel_max=float(self.p.accel_vz_max),
            dt=float(self.p.dt),
            decel_max=float(self.p.decel_vz_max),
        )
        accel_clipped = int(
            bool(accel_clipped)
            or slew_clipped_v
            or slew_clipped_wz
            or slew_clipped_vz
        )

        self._publish_cmd(applied_v, applied_wz, applied_vz)
        # optimistic model-based state; will be overridden by odom feedback below
        self.v = np.float32(applied_v)
        self.wz = np.float32(applied_wz)
        self.vz = np.float32(applied_vz)

        self._spin_for(self.p.dt)

        # update velocity state from odometry feedback (closes sim-to-real loop)
        if self._have_odom_twist:
            self.v = self._odom_v
            self.wz = self._odom_wz
            self.vz = self._odom_vz

        lidar_ranges, _ = self._resample_lidar()

        dist = self._dist_to_goal()
        prev_dist = self._prev_dist if self._prev_dist is not None else dist
        d_progress = prev_dist - dist
        self._prev_dist = dist

        tree_min_range = float(np.min(lidar_ranges))
        boundary_range = self._distance_to_world_boundary()
        min_range = float(min(tree_min_range, boundary_range))
        drone_radius = _effective_drone_radius(self.p)
        clearance = min_range - drone_radius
        collision = int(clearance < 0.0)
        success = int(dist < self.p.goal_tolerance)

        reward = 0.0
        reward += self.p.reward_progress_scale * d_progress
        reward += self.p.reward_speed_scale * (float(self.v) / self.p.v_max)  # use measured/applied velocity
        reward -= self.p.reward_step_penalty

        if accel_clipped:
            reward -= self.p.reward_accel_clip_penalty

        soft_margin = _soft_clearance_margin(self.p)
        if clearance < soft_margin and soft_margin > 1e-6:
            reward -= self.p.reward_proximity_scale * (soft_margin - clearance) / soft_margin
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
            accel_clipped=accel_clipped,
            clearance=clearance,
            drone_radius=drone_radius,
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
