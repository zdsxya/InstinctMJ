from __future__ import annotations

from collections.abc import Sequence

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sim import Simulation
from mjlab.utils.logging import print_info
from mjlab.viewer.debug_visualizer import DebugVisualizer
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from prettytable import PrettyTable

from instinct_mj.managers import MultiRewardCfg


class InstinctRlEnv(ManagerBasedRlEnv):
    """This class adds additional logging mechanism on sensors to get more
    comprehensive running statistics.
    """

    def __init__(
        self,
        cfg,
        device: str,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        del kwargs  # Unused.
        self.cfg = cfg
        if self.cfg.seed is not None:
            self.cfg.seed = self.seed(self.cfg.seed)
        self._sim_step_counter = 0
        self._instinct_body_lin_acc_cache: dict[str, dict[str, object]] = {}
        self.extras = {}
        self.obs_buf = {}
        # Initialize the manual-reset state here because InstinctRlEnv
        # customizes scene construction instead of calling ManagerBasedRlEnv.__init__.
        self._manual_reset_pending = torch.zeros(self.cfg.scene.num_envs, dtype=torch.bool, device=device)
        # Scratch buffer for per-env command dt; see ManagerBasedRlEnv.step().
        self._command_dt = torch.zeros(self.cfg.scene.num_envs, device=device)

        # Use InstinctScene so terrain cfg.class_type is honored (e.g. hacked_generator importer).
        self.scene = self.cfg.scene_class_type(self.cfg.scene, device=device)
        self.sim = Simulation(
            num_envs=self.scene.num_envs,
            cfg=self.cfg.sim,
            device=device,
            spec=self.scene.spec,
            variant_info=self.scene.collect_variant_info(),
        )

        self.scene.initialize(
            mj_model=self.sim.mj_model,
            model=self.sim.model,
            data=self.sim.data,
        )
        if self.scene.sensor_context is not None:
            self.sim.set_sensor_context(self.scene.sensor_context)

        print_info("")
        table = PrettyTable()
        table.title = "Base Environment"
        table.field_names = ["Property", "Value"]
        table.align["Property"] = "l"
        table.align["Value"] = "l"
        table.add_row(["Number of environments", self.num_envs])
        table.add_row(["Environment device", self.device])
        table.add_row(["Environment seed", self.cfg.seed])
        table.add_row(["Physics step-size", self.physics_dt])
        table.add_row(["Environment step-size", self.step_dt])
        print_info(table.get_string())
        print_info("")

        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(cfg.scene.num_envs, device=device, dtype=torch.long)
        self.render_mode = render_mode
        self._offline_renderer: OffscreenRenderer | None = None
        if self.render_mode == "rgb_array":
            renderer = OffscreenRenderer(
                model=self.sim.mj_model,
                cfg=self.cfg.viewer,
                scene=self.scene,
                sim_model=self.sim.model,
                expanded_fields=self.sim.expanded_fields,
            )
            renderer.initialize()
            self._offline_renderer = renderer
        self.metadata["render_fps"] = 1.0 / self.step_dt

        self.load_managers()
        self.setup_manager_visualizers()

    def load_managers(self) -> None:
        """Extend mjlab manager loading with InstinctLab multi-reward routing."""
        if isinstance(self.cfg.rewards, MultiRewardCfg):
            reward_group_cfg = self.cfg.rewards
            self.cfg.rewards = {}

        super().load_managers()

        if "reward_group_cfg" in locals():
            self.cfg.rewards = reward_group_cfg
            self.reward_manager = self.cfg.multi_reward_manager_class_type(
                self.cfg.rewards,
                self,
                scale_by_dt=self.cfg.scale_rewards_by_dt,
            )
            print_info(f"[INFO] {self.reward_manager}")

        self.monitor_manager = self.cfg.monitor_manager_class_type(self.cfg.monitors, self)
        print_info(f"[INFO] Monitor Manager: {self.monitor_manager}")

    def setup_manager_visualizers(self) -> None:
        super().setup_manager_visualizers()
        self.manager_visualizers["monitor_manager"] = self.monitor_manager

    def step(self, action: torch.Tensor):
        obs, reward, terminated, truncated, extras = super().step(action)
        monitor_infos = self.monitor_manager.update(dt=self.step_dt)
        extras.setdefault("step", {})
        extras["step"].update(monitor_infos)
        return obs, reward, terminated, truncated, extras

    def update_visualizers(self, visualizer: DebugVisualizer) -> None:
        super().update_visualizers(visualizer)
        terrain = self.scene.terrain
        # Custom terrain importers provide this hook; mjlab's plane
        # TerrainEntity does not.
        debug_vis = getattr(terrain, "debug_vis", None)
        if callable(debug_vis):
            debug_vis(visualizer)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        if isinstance(env_ids, Sequence):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.int64)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.int64)

        monitor_infos = self.monitor_manager.reset(env_ids, is_episode=True)

        super()._reset_idx(env_ids)

        self.extras["log"] = self.extras.get("log", {})
        self.extras["log"].update(monitor_infos)

    """
  Properties.
  """

    @property
    def num_rewards(self) -> int:
        return getattr(self.reward_manager, "num_rewards", 1)

    @property
    def body_lin_acc_cache(self) -> dict[str, dict[str, object]]:
        """Get the per-entity body linear acceleration cache."""
        return self._instinct_body_lin_acc_cache
