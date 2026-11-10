"""Export a saved InstinctMJ Parkour run for instinct_onboard without a simulator."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from instinct_rl.modules import EncoderMoEActorCritic

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_MJCF = REPO_ROOT / "src/instinct_mj/tasks/parkour/mjcf/g1_29dof_torsoBase_popsicle_with_shoe.xml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_values(values, names: list[str], default: float = 0.0) -> list[float]:
    if not isinstance(values, dict):
        return [float(values)] * len(names)
    result = []
    for name in names:
        matches = [float(value) for pattern, value in values.items() if re.fullmatch(pattern, name)]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous joint configuration for {name}: {values}")
        result.append(matches[0] if matches else default)
    return result


def build_configuration(env: dict, robot_mjcf: Path) -> tuple[dict, dict, dict]:
    """Translate only the saved fields consumed by ParkourAgent and its ROS node."""
    joints = [joint for joint in ET.parse(robot_mjcf).findall(".//worldbody//joint") if joint.get("type") != "free"]
    joint_names = [joint.attrib["name"] for joint in joints]
    robot = env["scene"]["entities"]["robot"]
    if robot["sort_actuators"]:
        raise ValueError("This exporter requires the Parkour MJCF actuator order (sort_actuators=False).")
    camera = copy.deepcopy(next(sensor for sensor in env["scene"]["sensors"] if sensor["name"] == "camera"))
    camera["pattern_cfg"] = camera.pop("pattern")
    control_dt = float(env["sim"]["mujoco"]["timestep"]) * int(env["decimation"])
    # A zero update_period means every sense() call; Parkour calls sense once per control step.
    sensor_period = float(camera["update_period"]) or control_dt
    camera["update_period"] = sensor_period
    height, width = camera["pattern_cfg"]["height"], camera["pattern_cfg"]["width"]
    crop = camera["noise_pipeline"].get("crop_and_resize")
    if crop:
        top, bottom, left, right = crop["crop_region"]
        height, width = height - top - bottom, width - left - right
        if crop["resize_shape"] is not None:
            height, width = crop["resize_shape"]

    actions = copy.deepcopy(env["actions"])
    if list(actions) != ["joint_pos"] or actions["joint_pos"]["actuator_names"] != [".*"]:
        raise ValueError("This exporter supports the full 29-joint Parkour position policy only.")
    action = actions["joint_pos"]
    action["class_type"] = "JointPositionAction"
    action["asset_name"] = action.pop("entity_name")
    action["joint_names"] = action.pop("actuator_names")
    actuators = {}
    p_gains, d_gains, effort_limits = [0.0] * len(joints), [0.0] * len(joints), [0.0] * len(joints)
    for index, source in enumerate(robot["articulation"]["actuators"]):
        actuators[f"pd_{index}"] = {
            "class_type": "BuiltinPdActuator",
            "joint_names_expr": source["target_names_expr"],
            "stiffness": source["stiffness"],
            "damping": source["damping"],
            "effort_limit": source["effort_limit"],
        }
        for joint_index, name in enumerate(joint_names):
            if any(re.fullmatch(pattern, name) for pattern in source["target_names_expr"]):
                if p_gains[joint_index] != 0:
                    raise ValueError(f"Multiple actuators configured for {name}")
                p_gains[joint_index] = float(source["stiffness"])
                d_gains[joint_index] = float(source["damping"])
                effort_limits[joint_index] = float(source["effort_limit"])
    if len(joints) != 29 or any(value <= 0 for value in p_gains + d_gains + effort_limits):
        raise ValueError("Incomplete G1 actuator configuration")

    formats = {}
    dimensions = {
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "velocity_commands": 3,
        "joint_pos": len(joints),
        "joint_vel": len(joints),
        "actions": len(joints),
    }
    for group_name in ("policy", "critic"):
        group = env["observations"][group_name]
        if group["history_length"] is not None:
            raise ValueError("Expected per-term observation histories")
        formats[group_name] = {}
        for name, term in group["terms"].items():
            if name == "depth_image":
                formats[group_name][name] = (term["params"]["num_output_frames"], height, width)
            else:
                formats[group_name][name] = (dimensions[name] * max(term["history_length"], 1),)
    policy = copy.deepcopy(env["observations"]["policy"]["terms"])
    depth = policy["depth_image"]["params"]
    frame_offsets = list(
        range((depth["num_output_frames"] - 1) * depth["history_skip_frames"], -1, -depth["history_skip_frames"])
    )
    metadata = {
        "format": "instinct_mj_parkour_v1",
        "robot_class_name": "G1_29Dof_TorsoBase_MJ",
        "joint_names": joint_names,
        "action_joint_names": joint_names,
        "control_dt": control_dt,
        "sensor_period_s": sensor_period,
        "depth_frame_offsets": frame_offsets,
        "depth_frame_offsets_s": [offset * sensor_period for offset in frame_offsets],
        "obs_format": formats,
        "num_actions": len(joints),
        "normalization": "none",
        "history_order": "oldest_to_newest_per_term",
        "p_gains": p_gains,
        "d_gains": d_gains,
        "effort_limits": effort_limits,
        "default_joint_pos": resolve_values(robot["init_state"]["joint_pos"], joint_names),
        "action_scale": resolve_values(action["scale"], joint_names, 1.0),
        "robot_mjcf_sha256": sha256(robot_mjcf),
    }
    onboard_env = {
        "deployment": {"source": "InstinctMJ", "robot_class_name": "G1_29Dof_TorsoBase_MJ", "control_dt": control_dt},
        "decimation": env["decimation"],
        "sim": {"dt": float(env["sim"]["mujoco"]["timestep"])},
        "scene": {"robot": {"init_state": robot["init_state"], "actuators": actuators}, "camera": camera},
        "observations": {"policy": policy},
        "actions": actions,
    }
    return onboard_env, metadata, formats


def verify_export(model, exported: Path, formats: dict) -> dict:
    """Compare the complete encoder + actor chain to the checkpoint's PyTorch policy."""
    for filename in ("0-depth_encoder.onnx", "actor.onnx"):
        onnx.checker.check_model(str(exported / filename))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    sessions = {
        name: ort.InferenceSession(str(exported / filename), sess_options=options, providers=["CPUExecutionProvider"])
        for name, filename in (("depth", "0-depth_encoder.onnx"), ("actor", "actor.onnx"))
    }
    proprio_dim = sum(math.prod(shape) for name, shape in formats["policy"].items() if name != "depth_image")
    depth_shape = formats["policy"]["depth_image"]
    rng = np.random.default_rng(20260909)
    max_depth_error, max_action_error = 0.0, 0.0
    fixture_proprio, fixture_depth, fixture_actions = [], [], []
    with torch.inference_mode():
        for index in range(32):
            proprio = rng.normal(0.0, 0.5, (1, proprio_dim)).astype(np.float32)
            depth = rng.uniform(0.0, 1.0, (1, *depth_shape)).astype(np.float32)
            if index < 2:
                proprio.fill(0.0)
                depth.fill(float(index))
            observation = torch.from_numpy(np.concatenate((proprio, depth.reshape(1, -1)), axis=-1))
            expected_latent = model.encoders._parallel_blocks["depth_encoder"](torch.from_numpy(depth)).numpy()
            expected_action = model.act_inference(observation).numpy()
            latent = sessions["depth"].run(None, {"input": depth})[0]
            action = sessions["actor"].run(None, {"input": np.concatenate((proprio, latent), axis=-1)})[0]
            np.testing.assert_allclose(latent, expected_latent, rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(action, expected_action, rtol=1e-4, atol=1e-4)
            max_depth_error = max(max_depth_error, float(np.max(np.abs(latent - expected_latent))))
            max_action_error = max(max_action_error, float(np.max(np.abs(action - expected_action))))
            if index < 8:
                fixture_proprio.append(proprio[0])
                fixture_depth.append(depth[0])
                fixture_actions.append(expected_action[0])
    np.savez_compressed(
        exported / "verification_inputs.npz",
        proprio=np.asarray(fixture_proprio),
        depth=np.asarray(fixture_depth),
        actions=np.asarray(fixture_actions),
    )
    return {
        "samples": 32,
        "provider": "CPUExecutionProvider",
        "max_depth_abs_error": max_depth_error,
        "max_action_abs_error": max_action_error,
        "atol": 1e-4,
        "rtol": 1e-4,
    }


def export(checkpoint: Path, output_dir: Path, robot_mjcf: Path = ROBOT_MJCF) -> dict:
    checkpoint, output_dir = checkpoint.resolve(), output_dir.resolve()
    run_dir = checkpoint.parent
    if output_dir == run_dir:
        raise ValueError("The deployment bundle must not overwrite the original run's params directory.")
    agent = yaml.safe_load((run_dir / "params/agent.yaml").read_text())
    env = yaml.safe_load((run_dir / "params/env.yaml").read_text())
    if agent["normalizers"] or agent["empirical_normalization"]:
        raise ValueError("This Parkour exporter requires a policy without observation normalization.")
    if agent["policy"]["class_name"] != "EncoderMoEActorCritic":
        raise ValueError("Expected an EncoderMoEActorCritic checkpoint")
    onboard_env, metadata, formats = build_configuration(env, robot_mjcf)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy_cfg = copy.deepcopy(agent["policy"])
    policy_cfg.pop("class_name")
    model = EncoderMoEActorCritic(formats, metadata["num_actions"], **policy_cfg).eval()
    model.load_state_dict(state["model_state_dict"], strict=True)
    exported = output_dir / "exported"
    exported.mkdir(parents=True, exist_ok=True)
    encoder = model.encoders._parallel_blocks["depth_encoder"]
    depth_shape = formats["policy"]["depth_image"]
    proprio_dim = sum(math.prod(shape) for name, shape in formats["policy"].items() if name != "depth_image")
    actor_dim = proprio_dim + agent["policy"]["encoder_configs"]["depth_encoder"]["output_size"]
    with torch.inference_mode():
        for module, shape, filename in (
            (encoder, (1, *depth_shape), "0-depth_encoder.onnx"),
            (model.actor, (1, actor_dim), "actor.onnx"),
        ):
            torch.onnx.export(
                module,
                torch.zeros(shape),
                str(exported / filename),
                input_names=["input"],
                output_names=["output"],
                opset_version=17,
                dynamo=False,
            )
    verification = verify_export(model, exported, formats)
    metadata.update(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "checkpoint_iteration": int(state["iter"]),
            "training_env_sha256": sha256(run_dir / "params/env.yaml"),
            "training_agent_sha256": sha256(run_dir / "params/agent.yaml"),
            "policy_class": agent["policy"]["class_name"],
            "actor_input_shape": [1, actor_dim],
            "depth_input_shape": [1, *depth_shape],
            "onnx_opset": 17,
            "verification": verification,
        }
    )
    (exported / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "params").mkdir(exist_ok=True)
    (output_dir / "params/env.yaml").write_text(yaml.safe_dump(onboard_env, sort_keys=False))
    (output_dir / "params/agent.yaml").write_text(yaml.safe_dump(agent, sort_keys=False))
    source = output_dir / "source"
    source.mkdir(exist_ok=True)
    for filename in ("agent.yaml", "env.yaml"):
        shutil.copy2(run_dir / "params" / filename, source / filename)
    shutil.copy2(robot_mjcf, source / robot_mjcf.name)
    for diff in (run_dir / "git").glob("*.diff"):
        shutil.copy2(diff, source / diff.name)
    run_exported = run_dir / "exported"
    if run_exported.resolve() != exported:
        run_exported.mkdir(exist_ok=True)
        for artifact in exported.iterdir():
            shutil.copy2(artifact, run_exported / artifact.name)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "depth_input_shape": metadata["depth_input_shape"],
                "actor_input_shape": metadata["actor_input_shape"],
                "verification": verification,
            },
            indent=2,
        )
    )
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robot-mjcf", type=Path, default=ROBOT_MJCF)
    args = parser.parse_args()
    export(args.checkpoint, args.output_dir, args.robot_mjcf)
