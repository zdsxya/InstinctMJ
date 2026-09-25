"""Convert UMR qpos to Instinct motion files and gate the training selection.

Run with InstinctMJ's Python so validation uses the training MuJoCo version.
No per-frame pose correction or joint clipping hides invalid retargeting.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[2]
LIMITS = {
    "joint_limit_excess_rad": 1e-4,
    "joint_step_rad": 0.4,
    "joint_speed_rad_s": 20.0,
    "root_step_rad": 0.25,
    "root_speed_m_s": 2.0,
    "root_angular_speed_rad_s": 5.0,
    "ground_penetration_m": 0.02,
    "self_penetration_m": 0.02,
    "stance_slip_p95_m_s": 0.35,
    "unsupported_fraction": 0.1,
}


def resample_qpos(qpos, source_fps, target_fps=50.0):
    source_times = np.arange(len(qpos)) / source_fps
    times = np.arange(int(np.floor(source_times[-1] * target_fps)) + 1) / target_fps
    result = np.column_stack([np.interp(times, source_times, qpos[:, i]) for i in range(qpos.shape[1])])
    rotations = Rotation.from_quat(qpos[:, [4, 5, 6, 3]])
    result[:, 3:7] = Slerp(source_times, rotations)(times).as_quat()[:, [3, 0, 1, 2]]
    for i in range(1, len(result)):
        if np.dot(result[i - 1, 3:7], result[i, 3:7]) < 0:
            result[i, 3:7] *= -1
    return result


def collision_bottom(model, data, gid):
    kind = model.geom_type[gid]
    size = model.geom_size[gid]
    rotation = data.geom_xmat[gid].reshape(3, 3)
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        extent = size[0] + size[1] * abs(rotation[2, 2])
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        extent = size[0]
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        extent = np.abs(rotation[2]) @ size
    else:
        raise ValueError(f"Unsupported collision geometry {kind}; extend exact ground audit")
    return float(data.geom_xpos[gid, 2] - extent)


def sole_contact_point(model, data, gid):
    """World-space support point of an actual sole collision capsule."""
    if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_CAPSULE:
        raise ValueError("G1 sole audit expects capsules")
    axis = data.geom_xmat[gid].reshape(3, 3)[:, 2]
    point = data.geom_xpos[gid].copy()
    point -= np.sign(axis[2]) * model.geom_size[gid, 1] * axis
    point[2] -= model.geom_size[gid, 0]
    return point


def audit(model, qpos, fps):
    if qpos.shape[1] != model.nq or len(qpos) < 20 or not np.isfinite(qpos).all():
        raise ValueError("Invalid qpos shape, length, or nonfinite values")
    hinge_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    pos = qpos[:, model.jnt_qposadr[hinge_ids]]
    joint_delta = np.diff(pos, axis=0)
    rotations = Rotation.from_quat(qpos[:, [4, 5, 6, 3]])
    root_delta = (rotations[:-1].inv() * rotations[1:]).magnitude()
    foot_ids = [model.body(name).id for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
    colliding = np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    foot_geoms = [[g for g in colliding if model.geom_bodyid[g] == body] for body in foot_ids]
    data = mujoco.MjData(model)
    heights, contact_velocities, penetration, self_depth = [], [], [], []
    jacobian = np.empty((3, model.nv))
    for index, frame in enumerate(qpos):
        data.qpos[:] = frame
        before, after = max(index - 1, 0), min(index + 1, len(qpos) - 1)
        mujoco.mj_differentiatePos(model, data.qvel, (after - before) / fps, qpos[before], qpos[after])
        mujoco.mj_forward(model, data)
        bottom = {g: collision_bottom(model, data, g) for g in colliding}
        heights.append([min(bottom[g] for g in gids) for gids in foot_geoms])
        foot_velocities = []
        for body, gids in zip(foot_ids, foot_geoms):
            gid = min(gids, key=bottom.get)
            point = sole_contact_point(model, data, gid)
            mujoco.mj_jac(model, data, jacobian, None, point, body)
            foot_velocities.append(jacobian @ data.qvel)
        contact_velocities.append(foot_velocities)
        penetration.append(max(0.0, -min(bottom.values())))
        self_depth.append(max([0.0] + [-float(c.dist) for c in data.contact[:data.ncon]]))
    heights = np.asarray(heights)
    foot_velocity = np.asarray(contact_velocities)
    # Use material velocity at the sole's ground support point. Ankle velocity
    # alone mistakes natural heel/toe rolling for sliding.
    # A 2.5 cm window incorrectly labels low-clearance SWING as contact.
    # Use a 5 mm geometric contact tolerance and report clearance separately.
    stance = (heights < 0.005) & (np.abs(foot_velocity[:, :, 2]) < 0.15)
    slip = np.linalg.norm(foot_velocity[:, :, :2], axis=-1)[stance]
    ranges = model.jnt_range[hinge_ids]
    excess = np.maximum(ranges[:, 0] - pos, pos - ranges[:, 1])
    metrics = {
        "joint_limit_excess_rad": float(max(0, excess.max())),
        "joint_step_rad": float(np.abs(joint_delta).max()),
        "joint_speed_rad_s": float(np.abs(joint_delta).max() * fps),
        "root_step_rad": float(root_delta.max()),
        "root_speed_m_s": float(np.linalg.norm(np.diff(qpos[:, :3], axis=0) * fps, axis=-1).max()),
        "root_angular_speed_rad_s": float(root_delta.max() * fps),
        "ground_penetration_m": float(max(penetration)),
        "self_penetration_m": float(max(self_depth)),
        "stance_slip_p95_m_s": float(np.percentile(slip, 95)) if len(slip) else None,
        "unsupported_fraction": float(np.mean(np.min(heights, axis=1) > 0.04)),
    }
    checks = {key: value is not None and value <= LIMITS[key] for key, value in metrics.items()}
    yaw = np.unwrap(rotations.as_euler("xyz")[:, 2])
    return dict(metrics=metrics, checks=checks, passed=all(checks.values()), frames=len(qpos), fps=fps,
                duration_s=(len(qpos) - 1) / fps, stance_samples=int(stance.sum()),
                foot_clearance_max_m=heights.max(axis=0).tolist(),
                median_speed_m_s=float(np.median(np.linalg.norm(np.diff(qpos[:, :2], axis=0) * fps, axis=-1))),
                heading_change_rad=float(yaw[-1] - yaw[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/umr")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    manifest = json.loads((args.output / "manifest.json").read_text())
    model_path = Path(manifest["robot_xml"])
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != manifest["robot_sha256"]:
        raise ValueError("Training MJCF changed after retarget configuration was generated")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    joint_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    joint_names = [model.joint(int(i)).name for i in joint_ids]
    if len(joint_names) != 29 or model.nq != 36:
        raise ValueError("Expected floating-base 29-DOF G1")
    destination = args.output / "motions"
    destination.mkdir(exist_ok=True)
    records, selected = [], []
    for clip in manifest["clips"]:
        if args.only and clip["id"] not in args.only:
            continue
        # Official UMR writes its joint-name array with object dtype. These
        # files are locally produced by the pinned UMR run, not untrusted input.
        with np.load(args.output / "retargeted" / (clip["id"] + ".npz"), allow_pickle=True) as src:
            if Path(str(src["robot_xml"].item())).resolve() != model_path.resolve():
                raise ValueError("UMR output uses another robot model; explicit frame conversion is required")
            if "robot_joint_names" in src and list(src["robot_joint_names"]) != joint_names:
                raise ValueError("UMR joint order differs from the training MJCF")
            qpos = resample_qpos(src["qpos"], float(src["fps"].item()))
        report = dict(id=clip["id"], kind=clip["kind"], **audit(model, qpos, 50.0))
        records.append(report)
        filename = clip["id"] + "_retargeted.npz"
        np.savez_compressed(destination / filename, framerate=50.0, joint_names=np.asarray(joint_names),
                            joint_pos=qpos[:, model.jnt_qposadr[joint_ids]].astype(np.float32),
                            base_pos_w=qpos[:, :3].astype(np.float32), base_quat_w=qpos[:, 3:7].astype(np.float32))
        if report["passed"]:
            selected.append(filename)
        print(clip["id"], "PASS" if report["passed"] else "FAIL",
              {k: report["metrics"][k] for k, ok in report["checks"].items() if not ok})
    audit_path = args.output / ("audit_partial.json" if args.only else "audit.json")
    audit_path.write_text(json.dumps(dict(thresholds=LIMITS, clips=records), indent=2) + "\n")
    if not args.only:
        # Failed clips stay on disk for diagnosis, but are never selected for training.
        (destination / "motions.yaml").write_text(yaml.safe_dump({"selected_files": selected}))
        kinds = {r["kind"] for r in records if r["passed"]}
        if len(selected) <= 10 or kinds != {"walk", "turn", "stand"}:
            raise SystemExit(f"Dataset not ready: {len(selected)} passing clips, categories={kinds}")


if __name__ == "__main__":
    main()
