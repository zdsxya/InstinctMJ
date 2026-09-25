"""Run official UMR sequentially, retaining reusable correspondence caches."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--umr-root", type=Path, default=Path.home() / "UMR")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "outputs/umr")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    manifest = json.loads((args.output / "manifest.json").read_text())
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    for clip in manifest["clips"]:
        if args.only and clip["id"] not in args.only:
            continue
        command = [sys.executable, str(args.umr_root / "scripts/humanoid_retarget_pipeline.py"),
                   "--config", clip["config"], "--skip-view", "--force-retarget"]
        print(f"Retargeting {clip['id']}", flush=True)
        with (args.output / "logs" / (clip["id"] + ".log")).open("w") as log:
            subprocess.run(command, cwd=args.umr_root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("Retargeting completed", flush=True)


if __name__ == "__main__":
    main()
