"""Measure MILD gait quality from actual forces and sole positions, without a viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from instinct_rl.runners import OnPolicyRunner
from mjlab.utils.torch import configure_torch_backends

import instinct_mj.tasks  # noqa: F401
from instinct_mj.tasks.locomotion.mild.walking_mdp import desired_support
from instinct_mj.tasks.registry import (
    load_env_cfg, load_env_cls, load_instinct_rl_cfg, load_runner_cls, load_vecenv_cls,
)


def summarize(data, *, dt, force_threshold=5.0):
    """Columns: Fz(2), sole height(2), body velocity(3), command(3),
    terminated, timeout, episode age, desired support(2).
    """
    valid = (data[:, :, 10:12].sum(-1) == 0) & (data[:, :, 12] > 1.0)
    valid &= np.linalg.norm(data[:, :, 7:9], axis=-1) > 0.1
    support = data[:, :, :2] > force_threshold
    x = data[valid]
    if not len(x):
        return {"passed": False, "valid_samples": 0, "reason": "No walking samples after settling"}
    count = support[valid].sum(-1)
    # Detect exclusive touchdown events; do not count reset boundaries.
    touchdown = support[1:] & ~support[:-1]
    adjacent_valid = valid[1:] & valid[:-1]
    per_env = []
    for index in range(data.shape[1]):
        events = touchdown[:, index] & adjacent_valid[:, index, None]
        exclusive = events.sum(-1) == 1
        sides = events[exclusive].argmax(-1)
        per_env.append({
            "left_touchdowns": int(events[:, 0].sum()),
            "right_touchdowns": int(events[:, 1].sum()),
            "simultaneous_touchdowns": int((events.sum(-1) == 2).sum()),
            "alternation_fraction": float(np.mean(sides[1:] != sides[:-1])) if len(sides) > 1 else 0.0,
        })
    summary = {
        "valid_samples": int(len(x)),
        "simulated_seconds_per_env": data.shape[0] * dt,
        "failures": int(data[:, :, 10].sum()),
        "timeouts": int(data[:, :, 11].sum()),
        "flight_fraction": float(np.mean(count == 0)),
        "single_support_fraction": float(np.mean(count == 1)),
        "double_support_fraction": float(np.mean(count == 2)),
        "both_soles_above_surface_fraction": float(np.mean((x[:, 2:4] > 0).all(-1))),
        "mean_vx_body": float(x[:, 4].mean()),
        "vx_rmse": float(np.sqrt(np.mean((x[:, 4] - x[:, 7]) ** 2))),
        "per_env_touchdowns": per_env,
    }
    if np.isfinite(x[:, 13:15]).all():
        summary["phase_agreement"] = float(np.mean(support[valid] == x[:, 13:15].astype(bool)))
        swing = ~x[:, 13:15].astype(bool)
        summary["swing_clearance_fraction_per_foot"] = [
            float(np.mean(x[swing[:, foot], 2 + foot] > 0.02)) if swing[:, foot].any() else 0.0
            for foot in range(2)
        ]
    checks = {
        "no_falls": summary["failures"] == 0,
        "flight_below_5_percent": summary["flight_fraction"] < 0.05,
        "single_support_above_50_percent": summary["single_support_fraction"] > 0.5,
        "vx_rmse_below_0_2_mps": summary["vx_rmse"] < 0.2,
        "repeated_alternation_each_env": all(
            min(e["left_touchdowns"], e["right_touchdowns"]) >= 6
            and e["alternation_fraction"] > 0.8 for e in per_env
        ),
    }
    if "phase_agreement" in summary:
        checks["phase_agreement_above_80_percent"] = summary["phase_agreement"] > 0.8
        checks["each_swing_foot_clears_2cm_over_half_swing"] = all(
            fraction > 0.5 for fraction in summary["swing_clearance_fraction_per_foot"]
        )
    summary.update(checks=checks, passed=all(checks.values()))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Instinct-Locomotion-Mild-Walk-G1-Play-v0")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--randomize-material", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.num_envs < 1:
        parser.error("steps and num-envs must be positive")
    configure_torch_backends()
    cfg = load_env_cfg(args.task, play=True)
    cfg.seed, cfg.scene.num_envs = args.seed, args.num_envs
    cfg.deformable_contact.randomize_material = args.randomize_material
    cfg.deformable_contact.debug_vis = False
    cfg.commands["base_velocity"].debug_vis = False
    env = load_env_cls(args.task)(cfg, device="cuda:0")
    try:
        agent = load_instinct_rl_cfg(args.task)
        vec = load_vecenv_cls(args.task)(env, policy_group=agent.policy_observation_group,
                                        critic_group=agent.critic_observation_group)
        runner = (load_runner_cls(args.task) or OnPolicyRunner)(vec, agent.to_dict(), log_dir=None, device="cuda:0")
        runner.load(str(args.checkpoint.resolve()))
        policy = runner.get_inference_policy(device="cuda:0")
        observations, _ = vec.get_observations()
        rows = []
        with torch.inference_mode():
            for _ in range(args.steps):
                observations, _, _, _ = vec.step(policy(observations))
                c = env.mild_contact
                height = c.positions[:, :, 2].reshape(args.num_envs, 2, -1).amin(-1)
                height = height - env.scene.env_origins[:, 2:3] - c.cfg.surface_height
                target = desired_support(env).float() if hasattr(cfg, "gait_period") else torch.full_like(height, float("nan"))
                rows.append(torch.cat((
                    c.foot_force[:, :, 2], height,
                    env.scene["robot"].data.root_link_lin_vel_b,
                    env.command_manager.get_command("base_velocity"),
                    env.reset_terminated[:, None].float(), env.reset_time_outs[:, None].float(),
                    env.episode_length_buf[:, None] * env.step_dt, target,
                ), dim=-1).cpu().numpy())
        data = np.stack(rows)
        result = summarize(data, dt=env.step_dt)
        result.update(task=args.task, checkpoint=str(args.checkpoint.resolve()), seed=args.seed,
                      num_envs=args.num_envs, randomize_material=args.randomize_material,
                      sampling_hz=1.0 / env.step_dt, support_force_threshold_N=5.0,
                      scope="Forward 0.6 m/s; policy-boundary samples; first second after each reset excluded.")
        args.output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output / "trajectory.npz", data=data)
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
