"""Render all converted references with the actual training G1 collision model."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "outputs/umr")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    manifest = json.loads((args.output / "manifest.json").read_text())
    spec = mujoco.MjSpec.from_file(manifest["robot_xml"])
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, .1], rgba=[.7, .73, .76, 1], group=0)
    spec.worldbody.add_light(pos=[0, -3, 5], dir=[0, .4, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    model = spec.compile()
    model.vis.global_.offwidth, model.vis.global_.offheight = 640, 480
    data = mujoco.MjData(model)
    render_dir = args.output / "renders"
    render_dir.mkdir(exist_ok=True)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.8, 130, -15
    options = mujoco.MjvOption()
    options.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    clips = [c for c in manifest["clips"] if not args.only or c["id"] in args.only]
    sheet = Image.new("RGB", (4 * 320, len(clips) * 260), "white")
    draw = ImageDraw.Draw(sheet)
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for row, clip in enumerate(clips):
            src = np.load(args.output / "motions" / (clip["id"] + "_retargeted.npz"))
            qpos = np.concatenate([src["base_pos_w"], src["base_quat_w"], src["joint_pos"]], axis=-1)
            indices = np.linspace(0, len(qpos) - 1, 4).astype(int)
            images = {}
            with imageio.get_writer(render_dir / (clip["id"] + ".mp4"), fps=25, codec="libx264") as writer:
                for index in sorted(set(range(0, len(qpos), 2)) | set(indices)):
                    data.qpos[:] = qpos[index]
                    mujoco.mj_forward(model, data)
                    camera.lookat[:] = qpos[index, :3]
                    camera.lookat[2] = .7
                    renderer.update_scene(data, camera=camera, scene_option=options)
                    frame = renderer.render().copy()
                    if index % 2 == 0:
                        writer.append_data(frame)
                    if index in indices:
                        images[index] = Image.fromarray(frame).resize((320, 240))
            draw.text((8, row * 260 + 3), f"{clip['id']} | {clip['kind']} | {clip['source']} {clip['start']}:{clip['end']}s", fill="black")
            for column, index in enumerate(indices):
                sheet.paste(images[index], (column * 320, row * 260 + 20))
            print(f"Rendered {clip['id']}", flush=True)
    sheet.save(render_dir / ("contact_sheet_partial.jpg" if args.only else "contact_sheet.jpg"), quality=94)


if __name__ == "__main__":
    main()
