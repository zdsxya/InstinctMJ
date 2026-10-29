"""Launch the MILD VAE baseline with a wall-clock watchdog (including startup)."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--max-hours", type=float, default=9.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    args = parser.parse_args()
    if args.num_envs < 1 or args.iterations < 1 or not 0 < args.max_hours <= 9:
        parser.error("Require positive envs/iterations and 0 < max-hours <= 9")
    root = Path(__file__).resolve().parents[2]
    destination = (root / args.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = destination / f"train_{stamp}.log"
    status_path = destination / f"launch_{stamp}.json"
    total_seconds = args.max_hours * 3600
    grace = min(120.0, total_seconds / 10)
    command = [
        sys.executable,
        "-u",
        "-m",
        "instinct_mj.scripts.instinct_rl.train",
        "Instinct-Locomotion-vae-G1-v0",
        "--num-envs",
        str(args.num_envs),
        "--agent.max-iterations",
        str(args.iterations),
        "--agent.save-interval",
        "100",
        "--agent.wall-time-limit-s",
        str(max(1.0, total_seconds - max(1800.0, 2 * grace))),
    ]
    start = time.monotonic()
    started = datetime.now().astimezone()
    status = {
        "status": "running",
        "started_at": started.isoformat(),
        "deadline": (started + timedelta(seconds=total_seconds)).isoformat(),
        "num_envs": args.num_envs,
        "max_iterations": args.iterations,
        "supervisor_pid": os.getpid(),
        "stdout": str(log_path),
        "command": command,
    }

    def write_status():
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(status_path)

    with log_path.open("w") as output:
        child = subprocess.Popen(command, cwd=root, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        status["training_pid"] = child.pid
        write_status()
        print(json.dumps({"launch": str(status_path), **status}, indent=2), flush=True)

        def forward_stop(signum, _frame):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)

        signal.signal(signal.SIGTERM, forward_stop)
        signal.signal(signal.SIGINT, forward_stop)
        try:
            child.wait(timeout=max(1.0, total_seconds - grace - (time.monotonic() - start)))
        except subprocess.TimeoutExpired:
            status["status"] = "watchdog_stopping"
            write_status()
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=max(0.1, total_seconds - (time.monotonic() - start)))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                status["status"] = "watchdog_killed"
        if status["status"] == "running":
            status["status"] = "exited" if child.returncode == 0 else "failed"
        status["returncode"] = child.returncode
        status["elapsed_seconds"] = time.monotonic() - start
        # Link the trainer's richer status/checkpoints when initialization succeeded.
        for line in log_path.read_text().splitlines():
            if line.startswith("[INFO] Logging to: "):
                status["run_dir"] = line.removeprefix("[INFO] Logging to: ")
        if status.get("run_dir"):
            training_status = Path(status["run_dir"]) / "training_status.json"
            if training_status.exists():
                status["training_status"] = json.loads(training_status.read_text())
        write_status()
        print(json.dumps(status, indent=2), flush=True)
    return child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
