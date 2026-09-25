"""Crop licensed local CMU/AMASS SMPL-X motions and configure official UMR.

Run in the separate UMR environment. Raw source files and body models are never
modified. Each output keeps its original shape, gender, units and timing.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmu-root", type=Path, default=Path.home() / "CMU")
    parser.add_argument("--umr-root", type=Path, default=Path.home() / "UMR")
    parser.add_argument("--body-model-dir", type=Path, default=Path.home() / "GMR/assets/body_models/smplx")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/umr")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    sys.path.insert(0, str(args.umr_root / "scripts"))
    from smpl_surface_retarget_common import load_smplx_npz_motion, smplx_motion_vertices_joints
    output = args.output.resolve()
    for name in ("sources", "configs", "retargeted", "logs"):
        (output / name).mkdir(parents=True, exist_ok=True)
    clips = json.loads(Path(__file__).with_name("clips.json").read_text())
    model = ROOT / "src/instinct_mj/assets/resources/unitree_g1/xml/g1_29dof_torsobase_popsicle.xml"
    defaults = json.loads((args.umr_root / "humanoid_retarget_defaults.json").read_text())
    robot = json.loads((args.umr_root / "robot_configs/humanoid_retarget_unitree_g1_example.json").read_text())["robot"]
    # Retarget directly to the training MJCF: no pelvis/torso rerooting or
    # hard-coded waist sign conversion is necessary downstream.
    robot.update(name="instinct_g1_mild", xml=str(model), point_cloud_center="body:waist_yaw_link")
    robot["xml_policy"] = {"add_freejoint_root": False}
    manifest = {"umr_commit": subprocess.check_output(
        ["git", "-C", str(args.umr_root), "rev-parse", "HEAD"], text=True).strip(),
        "robot_xml": str(model), "robot_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "clips": []}
    if args.only and (output / "manifest.json").exists():
        previous = json.loads((output / "manifest.json").read_text())
        if previous["robot_sha256"] != manifest["robot_sha256"] or previous["umr_commit"] != manifest["umr_commit"]:
            raise ValueError("Model/UMR changed; rebuild the entire dataset")
        manifest["clips"] = [c for c in previous["clips"] if c["id"] not in args.only]
    for clip in clips:
        if args.only and clip["id"] not in args.only:
            continue
        src = (args.cmu_root / clip["source"]).resolve()
        with np.load(src, allow_pickle=False) as data:
            fps = float(data["mocap_frame_rate"])
            stride = round(fps / 30)
            if not np.isclose(fps / stride, 30):
                raise ValueError(f"Expected a source rate divisible by 30: {src}: {fps}")
            start, end = round(clip["start"] * fps), round(clip["end"] * fps)
            if end > len(data["poses"]):
                raise ValueError(f"Crop exceeds source length: {src}")
            selected = slice(start, end, stride)
            source_path = output / "sources" / (clip["id"] + ".npz")
            np.savez_compressed(source_path,
                poses=data["poses"][selected], trans=data["trans"][selected],
                betas=data["betas"], gender=data["gender"],
                mocap_frame_rate=30.0, output_up="z")
        # CMU SMPL-X fits can place the body surface below the capture floor.
        # Calibrate ONE world-z offset per clip using UMR's exact body model
        # (including its ten-beta convention); never flatten motion framewise.
        sequence = load_smplx_npz_motion(source_path)
        vertices, _, _ = smplx_motion_vertices_joints(
            sequence, np.arange(len(sequence["pose_aa"])), args.body_model_dir,
            device="cuda", smplx_batch_size=64, smplx_batch_size_max=64,
        )
        ground_offset = -float(vertices[:, :, 2].min())
        with np.load(source_path) as prepared:
            arrays = dict(prepared)
        arrays["trans"][:, 2] += ground_offset
        np.savez_compressed(source_path, **arrays)
        config = json.loads(json.dumps(defaults))
        config["robot"].update(robot)
        config["smplx_model_dir"] = str(args.body_model_dir.resolve())
        config["correspondence"]["dataset"]["smpl_models"][0]["dir"] = str(args.body_model_dir.resolve())
        config["correspondence"]["train"]["device"] = "cuda"
        config["retarget"].update(smplx_device="cuda", smplx_batch_size=64, smplx_batch_size_max=64,
            out=str(output / "retargeted" / (clip["id"] + ".npz")))
        config["solver"]["batch_size"] = 8
        config["solver"]["iters"] = 4
        config["solver"]["trajectory_warm_start_mode"] = "bidirectional"
        config["motion"].update(data=str(output / "sources"), seq_key=clip["id"])
        config["view"]["enabled"] = False
        cfg_path = output / "configs" / (clip["id"] + ".json")
        cfg_path.write_text(json.dumps(config, indent=2) + "\n")
        subject = int(src.parent.name)
        manifest["clips"].append(dict(clip, source_path=str(src),
            source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
            source_url=f"http://mocap.cs.cmu.edu/search.php?subjectnumber={subject}",
            fitted_source="AMASS CMU SMPL-X (existing local download)",
            config=str(cfg_path), frames=round((clip["end"] - clip["start"]) * 30),
            source_ground_offset_m=ground_offset,
            prepared_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest()))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest contains {len(manifest['clips'])} clips under {output}")


if __name__ == "__main__":
    main()
