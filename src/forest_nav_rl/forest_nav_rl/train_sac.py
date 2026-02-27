"""Train SAC agents for forest navigation and manage run artifacts."""

from __future__ import annotations

import argparse
import signal
import re
import time
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections import defaultdict
from collections import deque
import torch
import numpy as np

# SB3 imports
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

from forest_nav_rl.gym_async_vec_env import GymAsyncVecEnv
from forest_nav_rl.utils import build_env_ctor_and_kwargs, get_env_backend
from forest_nav_rl.visualize_training import generate_single_run_report


MONITOR_INFO_KEYS = (
    "success",
    "collision",
    "shield_active",
    "shield_delta",
    "accel_clipped",
    "min_range",
    "dist_to_goal",
)


class CommandCenterTensorBoardCallback(BaseCallback):
    """Aggregate and log rich training safety/performance metrics to TensorBoard."""

    def __init__(self, log_every_steps: int = 1_000, episode_window: int = 200):
        """Initialize periodic logging buffers and aggregation windows."""
        super().__init__()
        self.log_every_steps = max(1, int(log_every_steps))
        self.episode_window = max(20, int(episode_window))
        self._step_accumulators: dict[str, list[float]] = defaultdict(list)
        self._episode_records: deque[dict[str, float | str | None]] = deque(
            maxlen=self.episode_window
        )
        self._episode_returns: np.ndarray | None = None
        self._episode_lengths: np.ndarray | None = None
        self._episode_shield_steps: np.ndarray | None = None
        self._episode_min_clearance: np.ndarray | None = None
        self._last_dump_walltime: float = 0.0
        self._last_dump_timesteps: int = 0

    @staticmethod
    def _tagify(value: Any) -> str:
        text = str(value) if value is not None else "unknown"
        text = text.strip().lower()
        text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
        return text.strip("_") or "unknown"

    @staticmethod
    def _safe_mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    @staticmethod
    def _safe_quantile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        return float(np.quantile(np.asarray(values, dtype=np.float64), q))

    def _on_training_start(self) -> None:
        n_envs = int(getattr(self.training_env, "num_envs", 1))
        self._episode_returns = np.zeros(n_envs, dtype=np.float64)
        self._episode_lengths = np.zeros(n_envs, dtype=np.int64)
        self._episode_shield_steps = np.zeros(n_envs, dtype=np.int64)
        self._episode_min_clearance = np.full(n_envs, np.inf, dtype=np.float64)
        self._last_dump_walltime = time.monotonic()
        self._last_dump_timesteps = int(self.num_timesteps)

    def _on_step(self) -> bool:
        assert self._episode_returns is not None
        assert self._episode_lengths is not None
        assert self._episode_shield_steps is not None
        assert self._episode_min_clearance is not None

        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        for idx, info in enumerate(infos):
            self._episode_lengths[idx] += 1
            if idx < len(rewards):
                self._episode_returns[idx] += float(rewards[idx])

            for key in (
                "success",
                "collision",
                "shield_active",
                "shield_delta",
                "accel_clipped",
                "min_range",
                "dist_to_goal",
                "clearance",
            ):
                value = info.get(key)
                if value is None:
                    continue
                self._step_accumulators[key].append(float(value))

            shield_active = bool(info.get("shield_active", 0))
            if shield_active:
                self._episode_shield_steps[idx] += 1

            clearance_val = info.get("clearance")
            if clearance_val is not None:
                self._episode_min_clearance[idx] = min(
                    self._episode_min_clearance[idx],
                    float(clearance_val),
                )

            done = bool(dones[idx]) if idx < len(dones) else False
            if not done:
                continue

            success = float(info.get("success", 0.0))
            collision = float(info.get("collision", 0.0))
            truncated = 1.0 if success < 0.5 and collision < 0.5 else 0.0

            self._episode_records.append(
                {
                    "return": float(self._episode_returns[idx]),
                    "length": float(self._episode_lengths[idx]),
                    "shield_step_ratio": float(self._episode_shield_steps[idx])
                    / max(1.0, float(self._episode_lengths[idx])),
                    "min_clearance": float(self._episode_min_clearance[idx])
                    if np.isfinite(self._episode_min_clearance[idx])
                    else 0.0,
                    "success": success,
                    "collision": collision,
                    "truncated": truncated,
                    "worldgen_seed": info.get("worldgen_seed"),
                    "worldgen_layout": str(info.get("worldgen_layout") or "none"),
                }
            )

            self._episode_returns[idx] = 0.0
            self._episode_lengths[idx] = 0
            self._episode_shield_steps[idx] = 0
            self._episode_min_clearance[idx] = np.inf

        if self.num_timesteps % self.log_every_steps == 0:
            self._flush_to_logger()

        return True

    def _on_training_end(self) -> None:
        self._flush_to_logger()

    def _flush_to_logger(self) -> None:
        safety_keys = (
            "success",
            "collision",
            "shield_active",
            "shield_delta",
            "accel_clipped",
            "min_range",
            "dist_to_goal",
            "clearance",
        )
        for key in safety_keys:
            values = self._step_accumulators.get(key, [])
            if not values:
                continue
            self.logger.record(f"command_center/step/{key}_mean", self._safe_mean(values))

        min_range_values = self._step_accumulators.get("min_range", [])
        if min_range_values:
            self.logger.record(
                "command_center/safety/min_range_p10", self._safe_quantile(min_range_values, 0.10)
            )
            self.logger.record(
                "command_center/safety/min_range_p50", self._safe_quantile(min_range_values, 0.50)
            )
            self.logger.record(
                "command_center/safety/min_range_p90", self._safe_quantile(min_range_values, 0.90)
            )

        clearance_values = self._step_accumulators.get("clearance", [])
        if clearance_values:
            self.logger.record("command_center/safety/clearance_min", float(min(clearance_values)))
            self.logger.record(
                "command_center/safety/clearance_p10", self._safe_quantile(clearance_values, 0.10)
            )
            self.logger.record(
                "command_center/safety/clearance_p50", self._safe_quantile(clearance_values, 0.50)
            )
            self.logger.record(
                "command_center/safety/clearance_p90", self._safe_quantile(clearance_values, 0.90)
            )

        if self._episode_records:
            returns = [float(rec["return"]) for rec in self._episode_records]
            lengths = [float(rec["length"]) for rec in self._episode_records]
            shield_ratios = [float(rec["shield_step_ratio"]) for rec in self._episode_records]
            min_clearances = [float(rec["min_clearance"]) for rec in self._episode_records]
            success_vals = [float(rec["success"]) for rec in self._episode_records]
            collision_vals = [float(rec["collision"]) for rec in self._episode_records]
            trunc_vals = [float(rec["truncated"]) for rec in self._episode_records]

            self.logger.record("command_center/episodes/return_mean", self._safe_mean(returns))
            self.logger.record(
                "command_center/episodes/return_p10", self._safe_quantile(returns, 0.10)
            )
            self.logger.record(
                "command_center/episodes/return_p90", self._safe_quantile(returns, 0.90)
            )
            self.logger.record("command_center/episodes/length_mean", self._safe_mean(lengths))
            self.logger.record(
                "command_center/episodes/shield_ratio_mean", self._safe_mean(shield_ratios)
            )
            self.logger.record(
                "command_center/episodes/min_clearance_mean", self._safe_mean(min_clearances)
            )
            self.logger.record(
                "command_center/episodes/success_rate", self._safe_mean(success_vals)
            )
            self.logger.record(
                "command_center/episodes/collision_rate", self._safe_mean(collision_vals)
            )
            self.logger.record(
                "command_center/episodes/truncation_rate", self._safe_mean(trunc_vals)
            )

            unique_seeds = {
                rec.get("worldgen_seed")
                for rec in self._episode_records
                if rec.get("worldgen_seed") is not None
            }
            self.logger.record(
                "command_center/worldgen/unique_seed_count", float(len(unique_seeds))
            )

            layout_counts: dict[str, int] = defaultdict(int)
            for rec in self._episode_records:
                layout_name = self._tagify(rec.get("worldgen_layout", "none"))
                layout_counts[layout_name] += 1
            total_layout = max(1, sum(layout_counts.values()))
            for layout_name, count in layout_counts.items():
                self.logger.record(
                    f"command_center/worldgen/layout_share/{layout_name}",
                    float(count) / float(total_layout),
                )

        model = self.model
        replay_buffer = getattr(model, "replay_buffer", None)
        if replay_buffer is not None:
            current_size = float(getattr(replay_buffer, "size", lambda: 0)())
            capacity = float(getattr(replay_buffer, "buffer_size", 0.0))
            if capacity > 0:
                self.logger.record("command_center/replay/fill_ratio", current_size / capacity)
            self.logger.record("command_center/replay/size", current_size)

        actor_opt = getattr(getattr(model, "actor", None), "optimizer", None)
        if actor_opt is not None and actor_opt.param_groups:
            self.logger.record(
                "command_center/optimizer/actor_lr", float(actor_opt.param_groups[0].get("lr", 0.0))
            )

        critic = getattr(model, "critic", None)
        critic_opt = getattr(critic, "optimizer", None)
        if critic_opt is not None and critic_opt.param_groups:
            self.logger.record(
                "command_center/optimizer/critic_lr",
                float(critic_opt.param_groups[0].get("lr", 0.0)),
            )

        ent_coef_tensor = getattr(model, "log_ent_coef", None)
        if ent_coef_tensor is not None:
            with torch.no_grad():
                self.logger.record(
                    "command_center/optimizer/ent_coef",
                    float(torch.exp(ent_coef_tensor.detach()).item()),
                )

        self.logger.record(
            "command_center/training/n_updates", float(getattr(model, "_n_updates", 0))
        )

        now = time.monotonic()
        elapsed = max(1e-6, now - self._last_dump_walltime)
        steps = max(0, int(self.num_timesteps) - int(self._last_dump_timesteps))
        self.logger.record("command_center/system/steps_per_sec", float(steps) / elapsed)
        self._last_dump_walltime = now
        self._last_dump_timesteps = int(self.num_timesteps)

        self.logger.dump(self.num_timesteps)
        self._step_accumulators.clear()


class FinalModelCallback(BaseCallback):
    """Periodically saves current model to final/ directory."""

    def __init__(
        self,
        save_path: Path,
        save_freq: int,
        save_vecnormalize: bool = False,
        save_replay_buffer: bool = True,
    ):
        """Configure periodic snapshot persistence for final artifacts."""
        super().__init__()
        self.save_path = Path(save_path)
        self.save_freq = save_freq
        self.save_vecnormalize = save_vecnormalize
        self.save_replay_buffer = save_replay_buffer

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self._save_model()
        return True

    def _on_training_end(self) -> None:
        self._save_model()

    def _save_model(self) -> None:
        model_path = self.save_path / "sac_final_model.zip"
        self.model.save(str(model_path))

        if self.save_replay_buffer:
            rb_path = self.save_path / "replay_buffer.pkl"
            save_rb = getattr(self.model, "save_replay_buffer", None)
            if callable(save_rb):
                save_rb(str(rb_path))

        if self.save_vecnormalize and isinstance(self.training_env, VecNormalize):
            norm_path = self.save_path / "vecnormalize.pkl"
            self.training_env.save(str(norm_path))


class GracefulInterruptCallback(BaseCallback):
    """Catches Ctrl+C (SIGINT) and stops training gracefully."""

    def __init__(self):
        """Initialize interrupt-tracking state and original signal handler."""
        super().__init__()
        self._interrupted = False
        self._original_handler = None

    def _init_callback(self) -> None:
        self._original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print("\n[GracefulInterrupt] Ctrl+C received. Stopping training after current step...")
        self._interrupted = True

    def _on_step(self) -> bool:
        # Return False to stop training
        return not self._interrupted

    def _on_training_end(self) -> None:
        # Restore original signal handler
        if self._original_handler is not None:
            signal.signal(signal.SIGINT, self._original_handler)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load and return YAML content as a dictionary."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data


def make_run_dir(base_dir: str | Path, exp_name: str) -> Path:
    """Create and return the next numbered run directory for an experiment."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    i = 1
    while (base / f"{exp_name}_{i:03d}").exists():
        i += 1
    run_dir = base / f"{exp_name}_{i:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_vec_env(env_cfg: dict[str, Any], n_envs: int, seed: int, monitor_dir: str):
    """Build a vectorized environment with monitor logging enabled."""
    env_ctor, env_kwargs = build_env_ctor_and_kwargs(env_cfg)
    return GymAsyncVecEnv(
        env_ctor=env_ctor,
        env_kwargs=env_kwargs,
        n_envs=n_envs,
        seed=seed,
        monitor_dir=monitor_dir,
        monitor_kwargs={"info_keywords": MONITOR_INFO_KEYS},
    )


def maybe_wrap_vecnorm(venv, norm_cfg: dict[str, Any] | bool | None):
    """Optionally wrap an environment with ``VecNormalize`` based on config."""
    if isinstance(norm_cfg, bool):
        norm_cfg = {"enabled": norm_cfg}
    elif norm_cfg is None:
        norm_cfg = {}

    if not norm_cfg.get("enabled", False):
        return venv, None

    venv = VecNormalize(
        venv,
        norm_obs=norm_cfg.get("norm_obs", True),
        norm_reward=norm_cfg.get("norm_reward", True),
        clip_obs=norm_cfg.get("clip_obs", 10.0),
        clip_reward=norm_cfg.get("clip_reward", 10.0),
    )
    return venv, venv


def _count_episodes_in_monitor_dir(monitor_dir: Path) -> int:
    if not monitor_dir.exists():
        return 0

    total = 0
    for monitor_file in monitor_dir.glob("*.monitor.csv"):
        with monitor_file.open("r") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or stripped.startswith("r,"):
                    continue
                total += 1
    return total


def configure_torch_for_device(device_arg: str) -> None:
    """Enable CUDA/TF32 performance flags when training on GPU."""
    requested_cuda = device_arg == "auto" or str(device_arg).startswith("cuda")
    if not requested_cuda or not torch.cuda.is_available():
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def main():
    """Run SAC training, checkpointing, and report generation workflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to run directory to resume from (e.g., outputs/runs/sac_fastsim_005)",
    )
    args = parser.parse_args()
    configure_torch_for_device(args.device)

    cfg = load_yaml(args.config)
    backend = get_env_backend(cfg["env"])
    if backend == "gazebo":
        raise ValueError(
            "Gazebo backend is for demonstration only; training is not supported. "
            "Use fastsim for training and gazebo for rollout/visualization."
        )

    # Enforce hybrid control mode for all training runs.
    _raw_params = cfg.get("env", {}).get("env_kwargs", {}).get("params", {})
    _action_mode = _raw_params.get("action_mode", "hybrid")
    if _action_mode != "hybrid":
        raise ValueError(
            f"Training requires action_mode='hybrid' in env.env_kwargs.params "
            f"(got '{_action_mode}'). "
            "Set `action_mode: hybrid` in your training config."
        )

    exp_name = cfg["experiment"]["name"]
    base_runs_dir = cfg["experiment"]["runs_dir"]
    seed = int(cfg["experiment"].get("seed", 0))

    # Handle resume from previous run
    if args.resume_from:
        run_dir = Path(args.resume_from)
        if not run_dir.exists():
            raise ValueError(f"Resume directory does not exist: {run_dir}")
        print(f"Resuming training from: {run_dir}")
        is_resume = True
    else:
        run_dir = make_run_dir(base_runs_dir, exp_name)
        is_resume = False

    (run_dir / "eval").mkdir(exist_ok=True)
    (run_dir / "best").mkdir(exist_ok=True)
    (run_dir / "tb").mkdir(exist_ok=True)
    (run_dir / "monitors" / "train").mkdir(parents=True, exist_ok=True)
    (run_dir / "monitors" / "eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "final").mkdir(exist_ok=True)

    # save final merged config used for this run
    with open(run_dir / "config_used.yaml", "w") as f:
        yaml.dump(cfg, f)

    train_monitor_dir = run_dir / "monitors" / "train"
    episodes_before = _count_episodes_in_monitor_dir(train_monitor_dir)
    started_at = datetime.now(timezone.utc)
    print(f"Training started at: {started_at.isoformat()}")

    env_ctor, env_kwargs = build_env_ctor_and_kwargs(cfg["env"])

    if bool(cfg["env"].get("check_env", backend != "gazebo")):
        single_env = env_ctor(**env_kwargs)
        check_env(single_env, warn=True)
        single_env.close()

    # build vectorized training env
    n_envs = cfg["training"].get("n_envs", 8)
    if backend == "gazebo" and int(n_envs) != 1:
        raise ValueError("Gazebo backend currently supports n_envs=1 only")

    train_env = build_vec_env(
        env_cfg=cfg["env"], n_envs=n_envs, seed=seed, monitor_dir=str(train_monitor_dir)
    )
    norm_cfg = cfg["training"].get("norm", cfg["training"].get("normalize", {}))
    train_env, _ = maybe_wrap_vecnorm(train_env, norm_cfg)

    # build vectorized eval env
    eval_env = build_vec_env(
        env_cfg=cfg["env"],
        n_envs=1,
        seed=seed + 1000,
        monitor_dir=str(run_dir / "monitors" / "eval"),
    )
    eval_env, _ = maybe_wrap_vecnorm(eval_env, norm_cfg)
    if isinstance(eval_env, VecNormalize):
        eval_env.training = False
        eval_env.norm_reward = False

    # callbacks
    # Explicitly disable historical checkpoint snapshots.
    # Delivery policy keeps only best/ and final/ artifacts.
    if bool(cfg["training"].get("save_checkpoints", False)):
        print("Note: training.save_checkpoints is ignored; checkpoints/ output is disabled.")
    save_freq = cfg["training"].get("checkpoint_freq_step", 10000)
    save_freq = max(save_freq // n_envs, 1)

    final_snapshot_freq_step = int(
        cfg["training"].get("final_snapshot_freq_step", save_freq * n_envs)
    )
    final_snapshot_freq = (
        max(final_snapshot_freq_step // n_envs, 1) if final_snapshot_freq_step > 0 else 0
    )

    save_vecnorm_cfg = cfg.get("save_vecnorm", False)
    if isinstance(save_vecnorm_cfg, dict):
        save_vecnormalize = bool(save_vecnorm_cfg.get("enabled", False))
    else:
        save_vecnormalize = bool(save_vecnorm_cfg)

    # Allow disabling replay buffer saves in checkpoints (heavy I/O)
    save_replay_buffer = cfg["training"].get("save_replay_buffer", True)

    checkpoint_cb = None

    eval_freq = int(cfg["logging"]["eval_freq_step"])
    eval_freq = max(eval_freq // n_envs, 1)  # adjust for number of envs

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best"),
        n_eval_episodes=cfg["logging"]["n_eval_episodes"],
        eval_freq=eval_freq,
        log_path=str(run_dir / "eval"),
        deterministic=True,
        render=False,
    )

    command_center_cb = CommandCenterTensorBoardCallback(
        log_every_steps=int(cfg["logging"].get("metrics_log_freq_step", 1_000)),
        episode_window=int(cfg["logging"].get("command_center_episode_window", 200)),
    )

    final_model_cb = None
    if final_snapshot_freq > 0:
        final_model_cb = FinalModelCallback(
            save_path=run_dir / "final",
            save_freq=final_snapshot_freq,
            save_vecnormalize=save_vecnormalize,
            save_replay_buffer=save_replay_buffer,
        )

    # SAC model - either load from checkpoint or create new
    sac_cfg = cfg["sac"]
    policy_kwargs = sac_cfg.get("policy_kwargs", {})

    if is_resume:
        # Load model from final or best checkpoint
        model_path = run_dir / "final" / "sac_final_model.zip"
        if not model_path.exists():
            raise ValueError(f"Model file not found for resume: {model_path}")

        print(f"Loading model from: {model_path}")
        model = SAC.load(str(model_path), env=train_env, device=args.device)

        # Load replay buffer if it exists
        rb_path = run_dir / "final" / "replay_buffer.pkl"
        if rb_path.exists():
            print(f"Loading replay buffer from: {rb_path}")
            model.load_replay_buffer(str(rb_path))
        else:
            print(f"Warning: Replay buffer not found at {rb_path}. Starting with empty buffer.")

        # Load VecNormalize stats if they exist
        if isinstance(train_env, VecNormalize):
            norm_path = run_dir / "final" / "vecnormalize.pkl"
            if norm_path.exists():
                print(f"Loading VecNormalize stats from: {norm_path}")
                train_env = VecNormalize.load(str(norm_path), train_env)
    else:
        # Create new model
        model = SAC(
            policy=sac_cfg.get("policy", "MlpPolicy"),
            env=train_env,
            learning_rate=sac_cfg.get("learning_rate", 3e-4),
            buffer_size=sac_cfg.get("buffer_size", 1_000_000),
            learning_starts=int(sac_cfg.get("learning_starts", 10_000)),
            batch_size=sac_cfg.get("batch_size", 256),
            tau=sac_cfg.get("tau", 0.005),
            gamma=sac_cfg.get("gamma", 0.99),
            train_freq=sac_cfg.get("train_freq", (1, "step")),
            gradient_steps=sac_cfg.get("gradient_steps", -1),  # -1 means match env steps in rollout
            ent_coef=sac_cfg.get("ent_coef", "auto"),
            target_update_interval=sac_cfg.get("target_update_interval", 1),
            tensorboard_log=str(run_dir / "tb"),
            verbose=sac_cfg.get("verbose", 1),
            seed=seed,
            policy_kwargs=policy_kwargs,
            device=args.device,
        )

    total_timesteps = int(cfg["experiment"]["total_timesteps"])

    # Graceful interrupt handler for Ctrl+C
    interrupt_cb = GracefulInterruptCallback()

    interrupted = False
    try:
        callbacks = [eval_cb, command_center_cb, interrupt_cb]
        if checkpoint_cb is not None:
            callbacks.append(checkpoint_cb)
        if final_model_cb is not None:
            callbacks.append(final_model_cb)

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            tb_log_name=run_dir.name,
            reset_num_timesteps=not is_resume,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\n[GracefulInterrupt] KeyboardInterrupt caught. Saving model...")

    # save final model and optionally VecNormalize stats
    model_path = run_dir / "final" / "sac_final_model.zip"
    model.save(str(model_path))

    # explicitly save replay buffer
    if save_replay_buffer:
        rb_path = run_dir / "final" / "replay_buffer.pkl"
        model.save_replay_buffer(str(rb_path))

    # save VecNormalize stats if applicable
    if isinstance(train_env, VecNormalize):
        norm_path = run_dir / "final" / "vecnormalize.pkl"
        train_env.save(str(norm_path))

    ended_at = datetime.now(timezone.utc)
    episodes_after = _count_episodes_in_monitor_dir(train_monitor_dir)
    episodes_trained = max(0, episodes_after - episodes_before)
    completed = not (interrupted or interrupt_cb._interrupted)

    summary = {
        "training_started_at": started_at.isoformat(),
        "training_ended_at": ended_at.isoformat(),
        "duration_seconds": round((ended_at - started_at).total_seconds(), 3),
        "completed": completed,
        "interrupted_by_ctrl_c": bool(interrupted or interrupt_cb._interrupted),
        "episodes_trained": int(episodes_trained),
        "episodes_before": int(episodes_before),
        "episodes_after": int(episodes_after),
        "run_dir": str(run_dir),
        "final_model": str(model_path),
    }

    summary_path = run_dir / "final" / "training_summary.yaml"
    with summary_path.open("w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    # Auto-generate training report at run end (including graceful Ctrl+C stop).
    report_dir = run_dir / "report"
    rolling_window = int(cfg.get("logging", {}).get("report_rolling_window", 50))
    try:
        generate_single_run_report(run_dir, report_dir, rolling_window=rolling_window)
        print(f"Run report generated at: {report_dir}")
    except Exception as exc:
        print(f"Warning: failed to generate run report automatically: {exc}")

    print("Training summary:")
    print(f"  started_at: {summary['training_started_at']}")
    print(f"  ended_at: {summary['training_ended_at']}")
    print(f"  duration_seconds: {summary['duration_seconds']}")
    print(f"  completed: {summary['completed']}")
    print(f"  interrupted_by_ctrl_c: {summary['interrupted_by_ctrl_c']}")
    print(f"  episodes_trained: {summary['episodes_trained']}")
    print(f"  summary_file: {summary_path}")

    if completed:
        print(f"Training completed. Final model saved to: {model_path}. Run directory: {run_dir}")
    else:
        print(f"Training interrupted. Model saved to: {model_path}. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
