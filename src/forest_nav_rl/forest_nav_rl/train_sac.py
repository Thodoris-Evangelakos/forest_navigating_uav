from __future__ import annotations

import argparse
import signal
import yaml
from pathlib import Path
from typing import Any
from collections import defaultdict

# SB3 imports
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv

from forest_nav_rl.utils import build_env_ctor_and_kwargs, get_env_backend


MONITOR_INFO_KEYS = (
    "success",
    "collision",
    "shield_active",
    "shield_delta",
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
            for key in ("success", "collision", "shield_active", "shield_delta", "min_range", "dist_to_goal"):
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
    def __init__(self, save_path: Path, save_freq: int, save_vecnormalize: bool = False):
        super().__init__()
        self.save_path = Path(save_path)
        self.save_freq = save_freq
        self.save_vecnormalize = save_vecnormalize

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self._save_model()
        return True
    
    def _on_training_end(self) -> None:
        self._save_model()

    def _save_model(self) -> None:
        model_path = self.save_path / "sac_final_model.zip"
        self.model.save(str(model_path))
        
        rb_path = self.save_path / "replay_buffer.pkl"
        self.model.save_replay_buffer(str(rb_path))
        
        if self.save_vecnormalize and hasattr(self.training_env, 'save'):
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

    # Use SubprocVecEnv for parallel stepping (better CPU utilization)
    vec_env_cls = SubprocVecEnv if n_envs > 1 else None

    venv = make_vec_env(
        env_ctor,
        n_envs = n_envs,
        seed = seed,
        monitor_dir=monitor_dir,
        monitor_kwargs={"info_keywords": MONITOR_INFO_KEYS},
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls
    )

    return venv

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    backend = get_env_backend(cfg["env"])
    if backend == "gazebo":
        raise ValueError(
            "Gazebo backend is for demonstration only; training is not supported. "
            "Use fastsim for training and gazebo for rollout/visualization."
        )

    exp_name = cfg["experiment"]["name"]
    base_runs_dir = cfg["experiment"]["runs_dir"]
    seed = int(cfg["experiment"].get("seed", 0))

    run_dir = make_run_dir(base_runs_dir, exp_name)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "eval").mkdir(exist_ok=True)
    (run_dir / "best").mkdir(exist_ok=True)
    (run_dir / "tb").mkdir(exist_ok=True)
    (run_dir / "monitors" / "train").mkdir(parents=True, exist_ok=True)
    (run_dir / "monitors" / "eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "final").mkdir(exist_ok=True)

    # save final merged config used for this run
    with open(run_dir / "config_used.yaml", "w") as f:
        yaml.dump(cfg, f)

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
        monitor_dir=str(run_dir / "monitors" / "train")
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

    # callbacks (checkpoints + eval)
    # CheckpointCallback can save replay buffer and VecNormalize stats if needed
    save_freq = cfg["training"].get("checkpoint_freq_step", 10000)
    save_freq = max(save_freq // n_envs, 1)  # adjust for number of envs

    save_vecnorm_cfg = cfg.get("save_vecnorm", False)
    if isinstance(save_vecnorm_cfg, dict):
        save_vecnormalize = bool(save_vecnorm_cfg.get("enabled", False))
    else:
        save_vecnormalize = bool(save_vecnorm_cfg)

    # Allow disabling replay buffer saves in checkpoints (heavy I/O)
    save_replay_buffer = cfg["training"].get("save_replay_buffer", True)

    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="sac_checkpoint",
        save_replay_buffer=save_replay_buffer,
        save_vecnormalize=save_vecnormalize
    )

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

    final_model_cb = FinalModelCallback(
        save_path=run_dir / "final",
        save_freq=save_freq,
        save_vecnormalize=save_vecnormalize
    )

    # SAC model
    sac_cfg = cfg["sac"]
    policy_kwargs = sac_cfg.get("policy_kwargs", {})
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
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_cb, eval_cb, safety_cb, final_model_cb, interrupt_cb],
            progress_bar=True
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\n[GracefulInterrupt] KeyboardInterrupt caught. Saving model...")
    
    # save final model and optionally VecNormalize stats
    model_path = run_dir / "final" / "sac_final_model.zip"
    model.save(str(model_path))

    # explicitly save replay buffer
    rb_path = run_dir / "final" / "replay_buffer.pkl"
    model.save_replay_buffer(str(rb_path))

    # save VecNormalize stats if applicable
    if isinstance(train_env, VecNormalize):
        norm_path = run_dir / "final" / "vecnormalize.pkl"
        train_env.save(str(norm_path))

    if interrupted or interrupt_cb._interrupted:
        print(f"Training interrupted. Model saved to: {model_path}. Run directory: {run_dir}")
    else:
        print(f"Training completed. Final model saved to: {model_path}. Run directory: {run_dir}")

if __name__ == "__main__":
    main()