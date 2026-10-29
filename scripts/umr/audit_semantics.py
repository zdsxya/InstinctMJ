"""Compare selected UMR motions to source SMPL-X and exact training MJCF.

Run in the UMR conda environment; only reads existing motions. Joint angle
values alone cannot diagnose elbow posture because the MJCF zero is bent.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path.home() / "UMR/scripts"))
from smpl_surface_retarget_common import load_smplx_npz_motion, smplx_motion_vertices_joints

sys.path.insert(0, str(Path(__file__).parent))
from convert_and_audit import collision_bottom, resample_qpos


def elbow_bend(shoulder, elbow, wrist):
    a, b = elbow - shoulder, wrist - elbow
    cosine = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/change1/umr_semantics.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    directory = ROOT / "outputs/umr"
    manifest = json.loads((directory / "manifest.json").read_text())
    xml = Path(manifest["robot_xml"])
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    joints = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    result = {
        "robot_sha256_matches": hashlib.sha256(xml.read_bytes()).hexdigest() == manifest["robot_sha256"],
        "method": (
            "Elbow bend from shoulder/elbow/wrist world points (0 deg straight), not raw joint angles. Source and robot"
            " have different morphology. Source ankle excursion is not sole clearance."
        ),
        "clips": [],
    }
    selected = set(yaml.safe_load((directory / "motions/motions.yaml").read_text())["selected_files"])
    for motion_path in sorted((directory / "motions").glob("*.npz")):
        name = motion_path.stem.removesuffix("_retargeted")
        raw = np.load(directory / "retargeted" / (name + ".npz"), allow_pickle=True)
        converted = np.load(motion_path)
        qpos = raw["qpos"]
        expected = resample_qpos(qpos, float(raw["fps"].item()), float(converted["framerate"].item()))
        conversion_error = float(np.abs(expected[:, 7:] - converted["joint_pos"]).max())
        source = load_smplx_npz_motion(directory / "sources" / (name + ".npz"))
        frame_ids = raw["frame_ids"].astype(int)
        _, source_joints, _ = smplx_motion_vertices_joints(
            source,
            frame_ids,
            Path.home() / "GMR/assets/body_models/smplx",
            device="cpu",
            smplx_batch_size=32,
            smplx_batch_size_max=32,
        )
        geometry = []
        for frame in qpos:
            data.qpos[:] = frame
            mujoco.mj_forward(model, data)
            row = []
            for side in ("left", "right"):
                shoulder = data.xanchor[model.joint(side + "_shoulder_roll_joint").id]
                elbow = data.xanchor[model.joint(side + "_elbow_joint").id]
                wrist = data.xanchor[model.joint(side + "_wrist_yaw_joint").id]
                bend = elbow_bend(shoulder, elbow, wrist)
                foot = model.body(side + "_ankle_roll_link").id
                gids = [
                    g
                    for g in range(model.ngeom)
                    if model.geom_bodyid[g] == foot and (model.geom_contype[g] or model.geom_conaffinity[g])
                ]
                bottom = min(collision_bottom(model, data, g) for g in gids)
                hip = data.xanchor[model.joint(side + "_hip_pitch_joint").id]
                knee = data.xanchor[model.joint(side + "_knee_joint").id]
                ankle = data.xanchor[model.joint(side + "_ankle_roll_joint").id]
                row.extend(
                    (
                        float(bend),
                        bottom,
                        float(elbow_bend(hip, knee, ankle)),
                        ankle[2],
                        np.linalg.norm(knee - hip) + np.linalg.norm(ankle - knee),
                    )
                )
            geometry.append(row)
        geometry = np.array(geometry)
        source_bend = np.stack(
            [
                elbow_bend(source_joints[:, 16], source_joints[:, 18], source_joints[:, 20]),
                elbow_bend(source_joints[:, 17], source_joints[:, 19], source_joints[:, 21]),
            ],
            axis=-1,
        )
        robot_bend = geometry[:, [0, 5]]
        robot_soles = geometry[:, [1, 6]]
        speed = np.linalg.norm(np.diff(qpos[:, :2], axis=0), axis=-1) * float(raw["fps"].item())
        source_knee = np.stack(
            [
                elbow_bend(source_joints[:, 1], source_joints[:, 4], source_joints[:, 7]),
                elbow_bend(source_joints[:, 2], source_joints[:, 5], source_joints[:, 8]),
            ],
            axis=-1,
        )
        robot_knee = geometry[:, [2, 7]]
        source_leg_lengths = np.stack(
            [
                np.linalg.norm(source_joints[:, k] - source_joints[:, h], axis=-1)
                + np.linalg.norm(source_joints[:, a] - source_joints[:, k], axis=-1)
                for h, k, a in ((1, 4, 7), (2, 5, 8))
            ],
            axis=-1,
        ).mean(0)
        item = {
            "id": name,
            "selected_for_amp": motion_path.name in selected,
            "source_knee_bend_p95_deg": np.percentile(source_knee, 95, axis=0).tolist(),
            "robot_knee_bend_p95_deg": np.percentile(robot_knee, 95, axis=0).tolist(),
            "knee_bend_mae_deg": np.abs(source_knee - robot_knee).mean(0).tolist(),
            "source_ankle_excursion_per_leg_length": (
                np.ptp(source_joints[:, [7, 8], 2], axis=0) / source_leg_lengths
            ).tolist(),
            "robot_ankle_excursion_per_leg_length": (
                np.ptp(geometry[:, [3, 8]], axis=0) / geometry[:, [4, 9]].mean(0)
            ).tolist(),
            "joint_order_matches": list(raw["robot_joint_names"]) == joints == list(converted["joint_names"]),
            "resampling_joint_max_error_rad": conversion_error,
            "resampling_root_position_max_error_m": float(np.abs(expected[:, :3] - converted["base_pos_w"]).max()),
            "resampling_quaternion_abs_dot_min": float(
                np.abs((expected[:, 3:7] * converted["base_quat_w"]).sum(-1)).min()
            ),
            "source_elbow_bend_median_deg": np.median(source_bend, axis=0).tolist(),
            "robot_elbow_bend_median_deg": np.median(robot_bend, axis=0).tolist(),
            "elbow_bend_mae_deg": np.abs(source_bend - robot_bend).mean(0).tolist(),
            "robot_sole_height_max_m": robot_soles.max(0).tolist(),
            "source_ankle_vertical_excursion_m": np.ptp(source_joints[:, [7, 8], 2], axis=0).tolist(),
            "root_speed_mean_m_s": float(speed.mean()),
            "knee_at_lower_bound_fraction": [
                (qpos[:, 7 + joints.index(side + "_knee_joint")] < 0.101).mean().item() for side in ("left", "right")
            ],
        }
        result["clips"].append(item)
        print(json.dumps(item), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
