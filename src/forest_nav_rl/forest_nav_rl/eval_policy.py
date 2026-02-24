from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from forest_nav_rl.utils import build_env_ctor_and_kwargs


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate a trained SAC policy")
	parser.add_argument("--model", type=Path, required=True, help="Path to model .zip")
	parser.add_argument("--config", type=Path, default=None, help="Path to config_used.yaml or training config")
	parser.add_argument("--num-episodes", type=int, default=20, help="Number of episodes to evaluate")
	parser.add_argument("--seed", type=int, default=None, help="Optional evaluation seed")
	parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda")
	parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions (default)")
	parser.add_argument("--stochastic", action="store_false", dest="deterministic", help="Use stochastic actions")
	parser.add_argument(
		"--output-json",
		type=Path,
		default=None,
		help="Optional output path for metrics JSON (default: <model_dir>/eval/eval_summary.json)",
	)
	parser.set_defaults(deterministic=True)
	return parser.parse_args()


def resolve_config_path(model_path: Path, config_path: Path | None) -> Path | None:
	if config_path is not None:
		return config_path

	candidate_run_config = model_path.parent.parent / "config_used.yaml"
	if candidate_run_config.exists():
		return candidate_run_config

	return None


def resolve_vecnormalize_path(model_path: Path) -> Path | None:
	model_parent = model_path.parent
	candidates = [
		model_parent / "vecnormalize.pkl",
		model_parent.parent / "vecnormalize.pkl",
		model_parent.parent / "final" / "vecnormalize.pkl",
	]
	for candidate in candidates:
		if candidate.exists():
			return candidate
	return None


def load_env_ctor_and_kwargs(config_path: Path | None):
	if config_path is None or not config_path.exists():
		cfg: dict[str, dict] = {"env": {"backend": "fastsim", "env_kwargs": {"params": {}}}}
	else:
		with config_path.open("r", encoding="utf-8") as handle:
			cfg = yaml.safe_load(handle)

	env_cfg = cfg.get("env", {"backend": "fastsim", "env_kwargs": {"params": {}}})
	return build_env_ctor_and_kwargs(env_cfg)


def load_obs_normalizer(vecnormalize_path: Path | None, env_ctor, env_kwargs: dict) -> VecNormalize | None:
	if vecnormalize_path is None:
		return None

	dummy_env = DummyVecEnv([lambda: env_ctor(**env_kwargs)])
	vecnorm = VecNormalize.load(str(vecnormalize_path), dummy_env)
	vecnorm.training = False
	vecnorm.norm_reward = False
	return vecnorm


def main() -> None:
	args = parse_args()

	model = SAC.load(args.model, device=args.device)

	resolved_config = resolve_config_path(args.model, args.config)
	if resolved_config is not None:
		print(f"Using config: {resolved_config}")

	env_ctor, env_kwargs = load_env_ctor_and_kwargs(resolved_config)
	env = env_ctor(**env_kwargs)

	vecnormalize_path = resolve_vecnormalize_path(args.model)
	obs_normalizer = load_obs_normalizer(vecnormalize_path, env_ctor, env_kwargs)
	if vecnormalize_path is not None:
		print(f"Using VecNormalize stats: {vecnormalize_path}")
	else:
		print("No VecNormalize stats found; running with raw observations.")

	successes: list[float] = []
	collisions: list[float] = []
	rewards: list[float] = []
	lengths: list[float] = []
	min_ranges: list[float] = []

	for episode_idx in range(args.num_episodes):
		episode_seed = None if args.seed is None else args.seed + episode_idx
		obs, _ = env.reset(seed=episode_seed)
		done = False
		total_reward = 0.0
		steps = 0
		final_info: dict = {}

		while not done:
			policy_obs = obs
			if obs_normalizer is not None:
				normalized_obs = obs_normalizer.normalize_obs(np.asarray([obs], dtype=np.float32))
				policy_obs = np.asarray(normalized_obs, dtype=np.float32)[0]

			action, _states = model.predict(policy_obs, deterministic=args.deterministic)
			obs, reward, terminated, truncated, info = env.step(action)
			total_reward += float(reward)
			steps += 1
			final_info = info
			done = bool(terminated or truncated)

		rewards.append(total_reward)
		lengths.append(float(steps))
		successes.append(float(bool(final_info.get("success", False))))
		collisions.append(float(bool(final_info.get("collision", False))))
		min_ranges.append(float(final_info.get("min_range", np.nan)))

	metrics = {
		"model": str(args.model),
		"episodes": int(args.num_episodes),
		"deterministic": bool(args.deterministic),
		"reward_mean": float(np.mean(rewards)) if rewards else float("nan"),
		"reward_std": float(np.std(rewards)) if rewards else float("nan"),
		"length_mean": float(np.mean(lengths)) if lengths else float("nan"),
		"success_rate": float(np.mean(successes)) if successes else float("nan"),
		"collision_rate": float(np.mean(collisions)) if collisions else float("nan"),
		"min_range_mean": float(np.nanmean(min_ranges)) if min_ranges else float("nan"),
	}

	print("\nEvaluation summary")
	print(f"  Episodes:      {metrics['episodes']}")
	print(f"  Reward mean:   {metrics['reward_mean']:.3f} ± {metrics['reward_std']:.3f}")
	print(f"  Length mean:   {metrics['length_mean']:.2f}")
	print(f"  Success rate:  {100.0 * metrics['success_rate']:.1f}%")
	print(f"  Collision rate:{100.0 * metrics['collision_rate']:.1f}%")
	print(f"  Min range mean:{metrics['min_range_mean']:.3f}")

	output_json = args.output_json
	if output_json is None:
		output_json = args.model.parent / "eval" / "eval_summary.json"
	output_json.parent.mkdir(parents=True, exist_ok=True)

	with output_json.open("w", encoding="utf-8") as handle:
		json.dump(metrics, handle, indent=2)

	print(f"Saved evaluation metrics: {output_json}")

	env.close()
	if obs_normalizer is not None:
		obs_normalizer.close()


if __name__ == "__main__":
	main()
