from __future__ import annotations

import argparse
import signal
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections import defaultdict
import torch

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


class SafetyMetricsCallback(BaseCallback):
    def __init__(self, log_every_steps: int = 1_000):
        super().__init__()
        self.log_every_steps = max(1, int(log_every_steps))
        self._accumulators: dict[str, list[float]] = defaultdict(list)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            for key in ("success", "collision", "shield_active", "shield_delta", "accel_clipped", "min_range", "dist_to_goal"):
                value = info.get(key)
                if value is None:
                    continue
                self._accumulators[key].append(float(value))

        if self.num_timesteps % self.log_every_steps == 0:
            self._flush_to_logger()

        return True

    def _on_training_end(self) -> None:
        self._flush_to_logger()

    def _flush_to_logger(self) -> None:
        if not self._accumulators:
            return

        for key, values in self._accumulators.items():
            if not values:
                continue
            self.logger.record(f"safety/{key}_mean", float(sum(values) / len(values)))

        self.logger.dump(self.num_timesteps)
        self._accumulators.clear()


class FinalModelCallback(BaseCallback):
    """Periodically saves current model to final/ directory."""
    def __init__(
        self,
        save_path: Path,
        save_freq: int,
        save_vecnormalize: bool = False,
        save_replay_buffer: bool = True,
    ):
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
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data

def make_run_dir(base_dir: str | Path, exp_name: str) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    i = 1
    while (base / f"{exp_name}_{i:03d}").exists():
        i += 1
    run_dir = base / f"{exp_name}_{i:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def build_vec_env(env_cfg: dict[str, Any], n_envs: int, seed: int, monitor_dir: str):
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
        clip_reward=norm_cfg.get("clip_reward", 10.0)
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
    requested_cuda = device_arg == "auto" or str(device_arg).startswith("cuda")
    if not requested_cuda or not torch.cuda.is_available():
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda")
    parser.add_argument("--resume-from", type=str, default=None, 
                        help="Path to run directory to resume from (e.g., outputs/runs/sac_fastsim_005)")
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
        env_cfg=cfg["env"],
        n_envs=n_envs,
        seed=seed,
        monitor_dir=str(train_monitor_dir)
    )
    norm_cfg = cfg["training"].get("norm", cfg["training"].get("normalize", {}))
    train_env, _ = maybe_wrap_vecnorm(train_env, norm_cfg)

    # build vectorized eval env
    eval_env = build_vec_env(
        env_cfg=cfg["env"],
        n_envs=1,
        seed=seed+1000,
        monitor_dir=str(run_dir / "monitors" / "eval")
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

    final_snapshot_freq_step = int(cfg["training"].get("final_snapshot_freq_step", save_freq * n_envs))
    final_snapshot_freq = max(final_snapshot_freq_step // n_envs, 1) if final_snapshot_freq_step > 0 else 0

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
        render=False
    )

    safety_cb = SafetyMetricsCallback(log_every_steps=int(cfg["logging"].get("metrics_log_freq_step", 1_000)))

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
            gradient_steps=sac_cfg.get("gradient_steps", -1), # -1 means match env steps in rollout
            ent_coef=sac_cfg.get("ent_coef", "auto"),
            target_update_interval=sac_cfg.get("target_update_interval", 1),
            tensorboard_log=str(run_dir / "tb"),
            verbose=sac_cfg.get("verbose", 1),
            seed=seed,
            policy_kwargs=policy_kwargs,
            device=args.device
        )

    total_timesteps = int(cfg["experiment"]["total_timesteps"])

    # Graceful interrupt handler for Ctrl+C
    interrupt_cb = GracefulInterruptCallback()

    interrupted = False
    try:
        callbacks = [eval_cb, safety_cb, interrupt_cb]
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