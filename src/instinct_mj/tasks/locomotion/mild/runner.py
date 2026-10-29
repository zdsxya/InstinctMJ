"""Local model construction, auditable counters and budgeted training."""

import copy
import json
import os
import statistics
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import instinct_rl
import torch
import torch.distributed as dist
from instinct_rl.runners import OnPolicyRunner

from .actor_critic import MildActorCritic
from .ppo import MildVaePPO


class _WallTimeComplete(Exception):
    pass


class _RewardConverged(Exception):
    pass


class MildVaeRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        if dist.is_initialized():
            raise NotImplementedError("The budgeted MILD runner currently supports a single GPU")
        self.started_at = time.monotonic()
        self.cfg = copy.deepcopy(train_cfg)
        self.policy_cfg, self.alg_cfg = self.cfg["policy"], self.cfg["algorithm"]
        self.env, self.device, self.log_dir = env, device, log_dir
        if self.cfg.get("normalizers"):
            raise ValueError("MILD uses fixed physical scaling; running normalization would rescale regression targets")
        if self.policy_cfg["history_length"] != env.cfg.history_length:
            raise ValueError("Policy and environment history lengths must match")
        policy_options, algorithm_options = self.policy_cfg.copy(), self.alg_cfg.copy()
        policy_options.pop("class_name")
        algorithm_options.pop("class_name")
        obs_format = env.get_obs_format()
        actor_critic = MildActorCritic(obs_format, env.num_actions, num_rewards=env.num_rewards, **policy_options).to(
            device
        )
        self.alg = MildVaePPO(actor_critic, device=device, **algorithm_options)
        self.num_steps_per_env, self.save_interval = self.cfg["num_steps_per_env"], self.cfg["save_interval"]
        self.log_interval = 1  # Also the safe budget-check boundary after an update.
        self.normalizers = {}
        self.alg.init_storage(env.num_envs, self.num_steps_per_env, obs_format, env.num_actions, env.num_rewards)
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.current_learning_iteration = 0
        self.completed_updates = 0
        self.completed_episodes = 0
        self.training_seconds = 0.0
        self.stop_reason = "not_started"
        self.reward_history = deque(maxlen=2 * self.cfg.get("convergence_window", 500))
        self.convergence = {}
        self.rollout_reward = torch.zeros((), device=device)
        self.rollout_count = 0
        self.git_status_repos = [instinct_rl.__file__, __file__]
        self.schema = {
            "version": 4,
            "obs_format": {
                group: {name: list(shape) for name, shape in terms.items()} for group, terms in obs_format.items()
            },
            "joint_names": list(env.unwrapped.scene["robot"].joint_names),
            "actions": env.num_actions,
            "history_length": env.cfg.history_length,
            "physics_dt": env.unwrapped.physics_dt,
            "policy_dt": env.unwrapped.step_dt,
            "target_scales": [1.0] * 11,
            "leg_pd_hz": 1.0 / (env.unwrapped.physics_dt * env.cfg.pd_decimation),
            "upper_body_hold": "native_pd_1khz",
            "motor_latency": "target_arrival_fresh_feedback",
            "amp": False,
        }
        env.reset()

    def rollout_step(self, obs, critic_obs):
        result = super().rollout_step(obs, critic_obs)
        self.completed_episodes += int(result[3].count_nonzero().item())
        self.rollout_reward += result[2].mean().detach()
        self.rollout_count += 1
        return result

    def log(self, locs, **kwargs):
        self.completed_updates = self.current_learning_iteration + 1
        duration = locs["collection_time"] + locs["learn_time"]
        self.training_seconds += duration
        super().log(locs, **kwargs)
        reward = float(self.rollout_reward / max(1, self.rollout_count))
        self.rollout_reward.zero_()
        self.rollout_count = 0
        self.reward_history.append(reward)
        quality = (
            {
                key: float(torch.stack([item[key].reshape(()) for item in locs["step_infos"]]).mean())
                for key in ("Walk/phase_agreement", "Walk/single_support_fraction", "Walk/vx_error_abs")
            }
            if locs["step_infos"]
            else {}
        )
        episode_length = statistics.mean(locs["lenbuffer"]) if locs["lenbuffer"] else 0.0
        window = self.cfg.get("convergence_window", 500)
        if len(self.reward_history) == 2 * window:
            history = list(self.reward_history)
            before, after = statistics.mean(history[:window]), statistics.mean(history[window:])
            relative_change = abs(after - before) / max(abs(before), 1e-8)
            relative_noise = statistics.stdev(history[window:]) / max(abs(after), 1e-8)
            self.convergence = {
                "window": window,
                "previous_mean": before,
                "recent_mean": after,
                "relative_change": relative_change,
                "relative_noise": relative_noise,
                "episode_length": episode_length,
                **quality,
            }
        if self.log_dir is not None:
            summary = {
                "completed_updates": self.completed_updates,
                "completed_episodes": self.completed_episodes,
                "policy_transitions": self.tot_timesteps,
                "physics_transitions": self.tot_timesteps * self.env.cfg.decimation,
                "iteration_seconds": duration,
                "reward_per_step": reward,
                "mean_episode_reward": statistics.mean(locs["rewbuffer"][0]) if locs["rewbuffer"][0] else 0.0,
                "mean_episode_length": episode_length,
                "quality": quality,
                "convergence": self.convergence,
                "wall_seconds_this_run": time.monotonic() - self.started_at,
                "losses": {key: float(value) for key, value in locs["losses"].items()},
            }
            with (Path(self.log_dir) / "progress.jsonl").open("a") as handle:
                handle.write(json.dumps(summary, allow_nan=False) + "\n")
            for key in ("completed_episodes", "policy_transitions", "physics_transitions"):
                self.writer.add_scalar(f"Counters/{key}", summary[key], self.completed_updates)
        if (
            self.completed_updates >= self.cfg.get("convergence_min_updates", 3000)
            and self.convergence
            and self.convergence["relative_change"] < self.cfg.get("convergence_relative_tolerance", 0.02)
            and self.convergence["relative_noise"] < 0.05
            and episode_length > 0.9 * self.env.max_episode_length
            and quality.get("Walk/phase_agreement", 0.0) > 0.8
            and quality.get("Walk/single_support_fraction", 0.0) > 0.5
        ):
            self.current_learning_iteration = self.completed_updates
            raise _RewardConverged
        limit = self.cfg.get("wall_time_limit_s", 0.0)
        if limit and time.monotonic() - self.started_at + max(60.0, 2 * duration) >= limit:
            self.current_learning_iteration = self.completed_updates
            raise _WallTimeComplete

    def learn(self, *args, **kwargs):
        self.stop_reason = "running"
        # A reset-defined gait target must not receive an unobserved random age.
        kwargs["init_at_random_ep_len"] = False
        try:
            super().learn(*args, **kwargs)
            self.stop_reason = "iterations_complete"
        except _RewardConverged:
            self.stop_reason = "reward_converged"
            print(f"[MILD] Reward plateau and gait gates met at {self.completed_updates} updates.", flush=True)
        except _WallTimeComplete:
            self.stop_reason = "wall_time_budget_complete"
            print(f"[MILD] Wall-time budget reached after {self.completed_updates} complete updates.", flush=True)
        except KeyboardInterrupt:
            self.stop_reason = "interrupted"
            raise
        except Exception:
            self.stop_reason = "failed"
            raise
        finally:
            if self.log_dir:
                # Even on SIGTERM, retain completed work. The regular periodic
                # checkpoints remain available if an update itself fails.
                if self.stop_reason != "failed":
                    self.save(str(Path(self.log_dir) / f"model_{self.completed_updates}.pt"))
                status = {
                    "status": self.stop_reason,
                    "convergence": self.convergence,
                    "completed_updates": self.completed_updates,
                    "completed_episodes": self.completed_episodes,
                    "policy_transitions": self.tot_timesteps,
                    "wall_seconds_this_run": time.monotonic() - self.started_at,
                }
                (Path(self.log_dir) / "training_status.json").write_text(json.dumps(status, indent=2) + "\n")
            if self.writer:
                self.writer.flush()
                self.writer.close()

    def save(self, path, infos=None):
        state = self.alg.state_dict()
        state.update(
            {
                "iter": self.completed_updates,
                "infos": infos,
                "mild_schema": self.schema,
                "train_cfg": self.cfg,
                "contact_cfg": asdict(self.env.cfg.deformable_contact),
                "completed_episodes": self.completed_episodes,
                "policy_transitions": self.tot_timesteps,
                "training_seconds": self.training_seconds,
                "reward_history": list(self.reward_history),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(self.device) if str(self.device).startswith("cuda") else None,
            }
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        temporary = str(path) + ".tmp"
        torch.save(state, temporary)
        os.replace(temporary, path)

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=True)
        if state.get("mild_schema") != self.schema:
            raise ValueError("Checkpoint MILD schema differs (joint order/history/physics/observations); cannot resume")
        self.alg.load_state_dict(state)
        self.current_learning_iteration = self.completed_updates = state["iter"]
        self.completed_episodes = state.get("completed_episodes", 0)
        self.tot_timesteps = state.get("policy_transitions", 0)
        self.training_seconds = state.get("training_seconds", 0.0)
        self.reward_history.extend(state.get("reward_history", []))
        torch.set_rng_state(state["torch_rng"].cpu())
        if state.get("cuda_rng") is not None and str(self.device).startswith("cuda"):
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.device)
        return state.get("infos")
