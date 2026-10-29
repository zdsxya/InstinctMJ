"""Soft arm limits estimated from the same files used by walking AMP."""

import hashlib
import math
from pathlib import Path

import numpy as np
import yaml


def load_shoulder_roll_envelope(dataset, selection, quantile=0.01, position_margin=math.radians(2), velocity_margin=0.1):
    """Pool mirrored shoulders; differentiate within clips, never across joins.

    This walking task uses selected retargeted NPZ files at 50 Hz. The range is
    a soft guardrail, not a target pose or a phase-aligned imitation reward.
    """
    if not 0 <= quantile < .5 or position_margin < 0 or velocity_margin < 0:
        raise ValueError("Invalid reference-envelope quantile or margin")
    selected = yaml.safe_load(Path(selection).read_text())["selected_files"]
    if not selected:
        raise ValueError("The AMP selection must contain motion files")
    positions, velocities, sources = [], [], []
    for filename in selected:
        path = Path(dataset) / filename
        with np.load(path, allow_pickle=False) as motion:
            names = motion["joint_names"].tolist()
            ids = [names.index(f"{side}_shoulder_roll_joint") for side in ("left", "right")]
            q = np.asarray(motion["joint_pos"][:, ids], dtype=np.float64) * [1, -1]
            fps = float(motion["framerate"])
        if len(q) < 2 or not np.isfinite(q).all() or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid reference motion: {path}")
        positions.append(q.ravel())
        velocities.append((np.abs(np.diff(q, axis=0)) * fps).ravel())
        sources.append({"file": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    lower, upper = np.quantile(np.concatenate(positions), [quantile, 1 - quantile])
    return {
        "lower_rad": float(lower - position_margin),
        "upper_rad": float(upper + position_margin),
        "speed_rad_s": float(np.quantile(np.concatenate(velocities), 1 - quantile) + velocity_margin),
        "quantile": quantile, "position_margin_rad": position_margin, "velocity_margin_rad_s": velocity_margin,
        "sources": sources,
    }


def shoulder_roll_excess(position, velocity, lower, upper, speed):
    """Zero cost inside the reference band; penalize excess angle and speed."""
    outward = position * position.new_tensor([1., -1.])
    angle_error = (lower - outward).clamp(min=0) + (outward - upper).clamp(min=0)
    speed_error = (velocity.abs() - speed).clamp(min=0)
    return ((angle_error / math.radians(5)).square() + .1 * (speed_error / .5).square()).mean(dim=-1)


def load_shoulder_pitch_envelope(dataset, selection, quantile=0.01, position_margin=math.radians(2)):
    """Estimate the forward-raise limit from selected AMP shoulder positions.

    Both G1 shoulder pitch axes have the same sign: more negative means more
    forward raise. Pool both shoulders for a mirror-symmetric lower bound,
    including the same walking, turning and standing clips as the AMP buffer.
    """
    if not 0 <= quantile < .5 or not math.isfinite(position_margin) or position_margin < 0:
        raise ValueError("Invalid reference-envelope quantile or margin")
    selected = yaml.safe_load(Path(selection).read_text())["selected_files"]
    if not selected:
        raise ValueError("The AMP selection must contain motion files")
    positions, sources = [], []
    for filename in selected:
        path = Path(dataset) / filename
        with np.load(path, allow_pickle=False) as motion:
            names = motion["joint_names"].tolist()
            ids = [names.index(f"{side}_shoulder_pitch_joint") for side in ("left", "right")]
            q = np.asarray(motion["joint_pos"][:, ids], dtype=np.float64)
        if len(q) == 0 or not np.isfinite(q).all():
            raise ValueError(f"Invalid reference motion: {path}")
        positions.append(q.ravel())
        sources.append({"file": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return {
        "lower_rad": float(np.quantile(np.concatenate(positions), quantile) - position_margin),
        "quantile": quantile, "position_margin_rad": position_margin, "sources": sources,
    }


def shoulder_pitch_excess(position, lower):
    """Penalize forward raise beyond the reference limit, leaving backward swing free."""
    angle_error = (lower - position).clamp(min=0)
    return (angle_error / math.radians(5)).square().mean(dim=-1)


class ReferenceShoulderRollPenalty:
    """Cache a reference-derived range once when the reward manager is built."""

    def __init__(self, cfg, env):
        reference = env.scene[cfg.params["reference_name"]]
        buffer_cfg = reference.cfg.motion_buffers[cfg.params["buffer_name"]]
        if buffer_cfg.filtered_motion_selection_filepath is None or buffer_cfg.subset_selection is not None:
            raise ValueError("Walking arm envelope requires an explicit AMP selected-files manifest")
        self.reference_stats = load_shoulder_roll_envelope(
            buffer_cfg.path, buffer_cfg.filtered_motion_selection_filepath,
        )
        names = list(env.scene["robot"].joint_names)
        self.joint_ids = [names.index(f"{side}_shoulder_roll_joint") for side in ("left", "right")]
        print("[AMP arm envelope]", {k: v for k, v in self.reference_stats.items() if k != "sources"}, flush=True)

    def __call__(self, env, reference_name, buffer_name):
        del reference_name, buffer_name  # Resolved once at construction.
        robot = env.scene["robot"]
        return shoulder_roll_excess(
            robot.data.joint_pos[:, self.joint_ids], robot.data.joint_vel[:, self.joint_ids],
            self.reference_stats["lower_rad"], self.reference_stats["upper_rad"],
            self.reference_stats["speed_rad_s"],
        )


class ReferenceShoulderPitchPenalty:
    """Cache a soft forward-raise limit; do not pull the arms to a fixed pose."""

    def __init__(self, cfg, env):
        reference = env.scene[cfg.params["reference_name"]]
        buffer_cfg = reference.cfg.motion_buffers[cfg.params["buffer_name"]]
        if buffer_cfg.filtered_motion_selection_filepath is None or buffer_cfg.subset_selection is not None:
            raise ValueError("Walking arm envelope requires an explicit AMP selected-files manifest")
        self.reference_stats = load_shoulder_pitch_envelope(
            buffer_cfg.path, buffer_cfg.filtered_motion_selection_filepath,
            quantile=cfg.params.get("quantile", 0.01),
            position_margin=cfg.params.get("position_margin", math.radians(2)),
        )
        names = list(env.scene["robot"].joint_names)
        self.joint_ids = [names.index(f"{side}_shoulder_pitch_joint") for side in ("left", "right")]
        print("[AMP arm pitch envelope]", {k: v for k, v in self.reference_stats.items() if k != "sources"}, flush=True)

    def __call__(self, env, reference_name, buffer_name, quantile=0.01, position_margin=math.radians(2)):
        del reference_name, buffer_name, quantile, position_margin  # Resolved at construction.
        return shoulder_pitch_excess(
            env.scene["robot"].data.joint_pos[:, self.joint_ids], self.reference_stats["lower_rad"],
        )
