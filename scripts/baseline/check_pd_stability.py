"""Isolate G1 PD stability with a small hip-yaw pulse while both legs are airborne."""

import argparse
import json
from pathlib import Path

import torch

from instinct_mj.tasks.registry import load_env_cfg, load_env_cls


def measure(pd_decimation, steps=30):
    task = "Instinct-Locomotion-vae-G1-Play-v0"
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = 1
    cfg.auto_reset = False
    cfg.pd_decimation = pd_decimation
    cfg.latency_range_ms = (10.0, 10.0)
    cfg.scene.entities["robot"].init_state.pos = (0.0, 0.0, 3.0)
    cfg.deformable_contact.debug_vis = False
    env = load_env_cls(task)(cfg, device="cuda:0")
    samples = []
    try:
        env.reset()
        with torch.inference_mode():
            for i in range(steps):
                action = torch.zeros(1, 12, device=env.device)
                if i == 0:
                    action[0, 2] = 0.02  # Left hip yaw, only the first 10 ms command.
                _, _, terminated, truncated, _ = env.step(action)
                data = env.scene["robot"].data
                samples.append(
                    {
                        "time_s": (i + 1) * env.step_dt,
                        "max_joint_speed_rad_s": float(data.joint_vel.abs().max()),
                        "left_hip_yaw_rad": float(data.joint_pos[0, env.leg_ids[2]]),
                        "maximum_foot_force_N": float(env.mild_contact.foot_force.norm(dim=-1).max()),
                    }
                )
                if terminated.any() or truncated.any():
                    break
        return {
            "pd_hz": 1000 / pd_decimation,
            "samples": samples,
            "peak_joint_speed_rad_s": max(s["max_joint_speed_rad_s"] for s in samples),
            "airborne_without_foot_force": all(s["maximum_foot_force_N"] < 1e-6 for s in samples),
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/change1/pd_stability.json"))
    args = parser.parse_args()
    result = {
        "command_hz": 100,
        "motor_target_delay_ms": 10,
        "hip_yaw_pulse_rad": 0.02,
        "controllers": [measure(10), measure(1)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
