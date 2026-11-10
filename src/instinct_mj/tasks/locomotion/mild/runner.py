"""Existing PPO rollout/update with task-local construction and diagnostics."""

import copy
import json
import os
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import instinct_rl
import torch
import torch.distributed as dist
from instinct_rl.runners import OnPolicyRunner

from .actor_critic import MildActorCritic
from .deployment import MILD_SCHEMA_VERSION, gait_schema
from .ppo import MildVaePPO


class _WallTimeComplete(Exception):
    pass


def _scalar(value):
    return float(value.detach().float().mean().item()) if isinstance(value, torch.Tensor) else float(value)


def _per_joint(value, width):
    """Serialize resolved control values in action order, independent of batch size."""
    if isinstance(value, torch.Tensor):
        return (value[0] if value.ndim > 1 else value).detach().cpu().tolist()
    return [float(value)] * width


class MildBaselineRunner(OnPolicyRunner):
    """Ordinary ActorCritic/PPO with shared ablation logs and checkpoint checks."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self._validate_execution(train_cfg)
        if env.cfg.use_vae:
            raise ValueError("Baseline runner requires the baseline environment (use_vae=False)")
        super().__init__(env, copy.deepcopy(train_cfg), log_dir, device)
        self._setup_diagnostics(train_cfg)

    @staticmethod
    def _validate_execution(train_cfg):
        if dist.is_initialized():
            raise NotImplementedError("The MILD estimator ablations currently support a single GPU")
        if train_cfg.get("normalizers"):
            raise ValueError("MILD ablations use fixed observation scales, without running normalization")

    def _setup_diagnostics(self, train_cfg):
        # The parent constructor removes class names from its config; retain
        # the original settings for checkpoints and deployment metadata.
        self.cfg = copy.deepcopy(train_cfg)
        self.started_at = time.monotonic()
        self.log_interval = 1
        self.checkpoint_interval = self.cfg["save_interval"]
        if self.checkpoint_interval < 1:
            raise ValueError("save_interval must be positive")
        # The parent saves at zero-based iteration indices. Save from log()
        # after complete updates so model_100.pt actually represents 100 updates.
        self.save_interval = sys.maxsize
        self.completed_updates = 0
        self.completed_episodes = 0
        self.fallen_episodes = 0
        self.training_seconds = 0.0
        self.stop_reason = "not_started"
        self.git_status_repos = [instinct_rl.__file__, __file__]
        env = self.env.unwrapped
        action = env.action_manager.get_term("joint_pos")
        self.schema = {
            "version": MILD_SCHEMA_VERSION,
            **gait_schema(env.cfg),
            "obs_format": {
                group: {name: list(shape) for name, shape in terms.items()}
                for group, terms in self.env.get_obs_format().items()
            },
            "joint_names": list(env.scene["robot"].joint_names),
            "action_names": list(action.target_names),
            "actions": self.env.num_actions,
            "action_scale": copy.deepcopy(env.cfg.actions["joint_pos"].scale),
            "resolved_action_scale": _per_joint(action.scale, self.env.num_actions),
            "resolved_action_offset": _per_joint(action.offset, self.env.num_actions),
            "action_clip": copy.deepcopy(action.cfg.clip),
            "default_joint_pos": env.scene["robot"].data.default_joint_pos[0].detach().cpu().tolist(),
            "actuators": [
                {
                    "class": type(actuator).__name__,
                    **{
                        name: getattr(actuator, name, None)
                        for name in ("target_names_expr", "stiffness", "damping", "effort_limit", "armature")
                    },
                }
                for actuator in env.cfg.scene.entities["robot"].articulation.actuators
            ],
            "history_length": env.cfg.history_length if env.cfg.use_vae else 1,
            "history_layout": "term_major_oldest_to_newest_reset_backfill",
            "observation_scales": {
                group: {name: term.scale for name, term in cfg.terms.items() if term is not None}
                for group, cfg in env.cfg.observations.items()
            },
            "physics_dt": env.physics_dt,
            "policy_dt": env.step_dt,
            "policy": copy.deepcopy(train_cfg["policy"]),
            "actuation": "inherited_g1_native_pd_position_targets",
            "state_targets": "SI_root_frame_velocities_undeformed_surface_sole_heights",
            "amp": False,
        }

    def rollout_step(self, obs, critic_obs):
        result = super().rollout_step(obs, critic_obs)
        self.completed_episodes += int(result[3].count_nonzero().item())
        metrics = result[4].get("step", {})
        self.fallen_episodes += int(_scalar(metrics.get("Evaluation/Fallen_Episodes", 0)))
        return result

    def log(self, locs, **kwargs):
        """Use the existing writer; these metrics work on CPU and GPU alike."""
        self.completed_updates = self.current_learning_iteration + 1
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        duration = locs["collection_time"] + locs["learn_time"]
        self.training_seconds += duration
        self.tot_time = self.training_seconds
        scalars = {f"Loss/{key}": _scalar(value) for key, value in locs["losses"].items()}
        for key, value in locs["stats"].items():
            tag = key if key.startswith(("Train/", "VAE/")) else f"Train/{key}"
            scalars[tag] = _scalar(value)
        scalars.update({
            "Train/PPO_Actor_Loss": _scalar(locs["losses"]["surrogate_loss"]),
            "Train/PPO_Critic_Loss": _scalar(locs["losses"]["value_loss"]),
            "Train/Policy_Entropy": -_scalar(locs["losses"].get("entropy", 0.0)),
            "Loss/learning_rate": self.alg.learning_rate,
            "Policy/mean_noise_std": _scalar(self.alg.actor_critic.action_std),
            "Perf/total_fps": self.num_steps_per_env * self.env.num_envs / max(duration, 1e-9),
            "Perf/collection_time": locs["collection_time"],
            "Perf/learning_time": locs["learn_time"],
            "Evaluation/Fall_Rate": self.fallen_episodes / max(1, self.completed_episodes),
            "Counters/completed_episodes": self.completed_episodes,
            "Counters/fallen_episodes": self.fallen_episodes,
            "Counters/policy_transitions": self.tot_timesteps,
        })
        for infos, prefix in ((locs["ep_infos"], "Episode/"), (locs["step_infos"], "Step/")):
            keys = {key for item in infos for key, value in item.items() if not isinstance(value, str)}
            for key in keys:
                values = [_scalar(item[key]) for item in infos if key in item]
                tag = key if key.startswith(("Episode/", "Evaluation/")) else prefix + key
                scalars[tag] = statistics.mean(values)
        if locs["rewbuffer"][0]:
            scalars["Evaluation/Episode_Reward"] = statistics.mean(locs["rewbuffer"][0])
        if locs["lenbuffer"]:
            scalars["Evaluation/Episode_Length"] = statistics.mean(locs["lenbuffer"])
        for index, buffer in enumerate(locs["rframebuffer"]):
            if buffer:
                scalars[f"Train/mean_reward_each_timestep_{index}"] = statistics.mean(buffer)
        for key, value in scalars.items():
            self.writer.add_scalar(key, value, self.completed_updates)
        summary = {
            "completed_updates": self.completed_updates,
            "completed_episodes": self.completed_episodes,
            "policy_transitions": self.tot_timesteps,
            "iteration_seconds": duration,
            "metrics": scalars,
        }
        with (Path(self.log_dir) / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(summary, allow_nan=False) + "\n")
        print(
            f"[MILD] update={self.completed_updates} transitions={self.tot_timesteps} "
            f"loss={scalars['Loss/total_loss']:.5g} fps={scalars['Perf/total_fps']:.0f}",
            flush=True,
        )
        if self.completed_updates % self.checkpoint_interval == 0:
            self.save(str(Path(self.log_dir) / f"model_{self.completed_updates}.pt"))
        limit = self.cfg.get("wall_time_limit_s", 0.0)
        if limit and time.monotonic() - self.started_at >= limit:
            self.current_learning_iteration = self.completed_updates
            raise _WallTimeComplete

    def learn(self, *args, **kwargs):
        if self.log_dir is None:
            raise ValueError("A log directory is required for training; inference can omit it")
        self.stop_reason = "running"
        try:
            super().learn(*args, **kwargs)
            self.stop_reason = "iterations_complete"
        except _WallTimeComplete:
            self.stop_reason = "wall_time_budget_complete"
        except KeyboardInterrupt:
            self.stop_reason = "interrupted"
            raise
        except Exception:
            self.stop_reason = "failed"
            raise
        finally:
            if self.stop_reason in {"interrupted", "wall_time_budget_complete"}:
                self.save(str(Path(self.log_dir) / f"model_{self.completed_updates}.pt"))
            status = {
                "status": self.stop_reason,
                "completed_updates": self.completed_updates,
                "completed_episodes": self.completed_episodes,
                "policy_transitions": self.tot_timesteps,
                "wall_seconds_this_run": time.monotonic() - self.started_at,
            }
            (Path(self.log_dir) / "training_status.json").write_text(json.dumps(status, indent=2) + "\n")
            if self.writer:
                self.writer.flush()
                self.writer.close()
                self.writer = None

    def save(self, path, infos=None):
        state = self.alg.state_dict()
        state.update({
            "iter": self.completed_updates,
            "infos": infos,
            "mild_schema": self.schema,
            "train_cfg": self.cfg,
            "contact_cfg": asdict(self.env.cfg.deformable_contact),
            "completed_episodes": self.completed_episodes,
            "fallen_episodes": self.fallen_episodes,
            "policy_transitions": self.tot_timesteps,
            "training_seconds": self.training_seconds,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(self.device) if str(self.device).startswith("cuda") else None,
        })
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        temporary = str(path) + ".tmp"
        torch.save(state, temporary)
        os.replace(temporary, path)

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=True)
        if not isinstance(state.get("mild_schema"), dict) or state["mild_schema"].get("version") != MILD_SCHEMA_VERSION:
            raise ValueError(
                "Checkpoint requires MILD schema version-6 with gait_clock; older no-clock checkpoints cannot load"
            )
        if state.get("mild_schema") != self.schema:
            raise ValueError("Checkpoint MILD schema differs (joint order/history/physics/observations/policy/gait); cannot load")
        self.alg.load_state_dict(state)
        # The parent PPO restores optimizer state, but not this schedule scalar.
        self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]
        self.current_learning_iteration = self.completed_updates = state["iter"]
        self.completed_episodes = state.get("completed_episodes", 0)
        self.fallen_episodes = state.get("fallen_episodes", 0)
        self.tot_timesteps = state.get("policy_transitions", 0)
        self.training_seconds = state.get("training_seconds", 0.0)
        torch.set_rng_state(state["torch_rng"].cpu())
        if state.get("cuda_rng") is not None and str(self.device).startswith("cuda"):
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.device)
        return state.get("infos")

    def export_as_onnx(self, obs, export_model_dir):
        super().export_as_onnx(obs, export_model_dir)
        Path(export_model_dir, "observation_contract.json").write_text(json.dumps(self.schema, indent=2) + "\n")

    @staticmethod
    def configure_inference(env_cfg, agent_cfg, checkpoint):
        from .deployment import configure_inference

        return configure_inference(env_cfg, agent_cfg, checkpoint)


class MildVaeRunner(MildBaselineRunner):
    """Construct local VAE modules, reuse the original PPO training loop."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self._validate_execution(train_cfg)
        self.cfg = copy.deepcopy(train_cfg)
        self.policy_cfg, self.alg_cfg = self.cfg["policy"], self.cfg["algorithm"]
        self.env, self.device, self.log_dir = env, device, log_dir
        if not env.cfg.use_vae or self.policy_cfg["history_length"] != env.cfg.history_length:
            raise ValueError("VAE policy and environment must both enable the same history length")
        policy_options, algorithm_options = self.policy_cfg.copy(), self.alg_cfg.copy()
        policy_options.pop("class_name")
        algorithm_options.pop("class_name")
        obs_format = env.get_obs_format()
        actor_critic = MildActorCritic(obs_format, env.num_actions, num_rewards=env.num_rewards, **policy_options).to(device)
        self.alg = MildVaePPO(actor_critic, device=device, **algorithm_options)
        self.num_steps_per_env, self.save_interval = self.cfg["num_steps_per_env"], self.cfg["save_interval"]
        self.normalizers = {}
        self.alg.init_storage(env.num_envs, self.num_steps_per_env, obs_format, env.num_actions, env.num_rewards)
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.current_learning_iteration = 0
        self._setup_diagnostics(train_cfg)
        env.reset()
