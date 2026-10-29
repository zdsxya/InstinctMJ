"""Monitor a change1 run, evaluate its final checkpoint, and record acceptance.

Does not label a time/iteration limit as convergence. Safe to restart: completed
final evaluations are reused only when they refer to the same checkpoint.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
START = "<!-- CHANGE1_TRAINING_STATUS_BEGIN -->"
END = "<!-- CHANGE1_TRAINING_STATUS_END -->"


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def progress_rows(run):
    path = run / "progress.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A concurrently appended last line can be incomplete.
    return rows


def write_doc(run, latest, status=None, evaluations=None):
    text = [f"- 当前唯一 VAE 训练目录：`{run.relative_to(ROOT)}`。", "- 训练 stdout：`outputs/change1/train.log`。"]
    if latest:
        text += [
            f'- 已完成 {latest["completed_updates"]} 次更新，'
            f'{latest["policy_transitions"]:,} 次策略交互；'
            f'当前平均每步 reward={latest["reward_per_step"]:.6f}，'
            f'平均 episode 长度={latest["mean_episode_length"] / 100:.2f} s。'
        ]
    if status is None:
        text.append("- **训练仍在进行；尚未确认 reward 收敛或通过独立步态验收。**")
    else:
        text.append(
            f'- 训练结束状态：`{status["status"]}`；最终 checkpoint：`model_{status["completed_updates"]}.pt`。'
        )
        converged = status["status"] == "reward_converged"
        passed = converged and evaluations and all(e.get("passed", False) for e in evaluations.values())
        text.append(
            "- **验收通过：reward 窗口与独立步态评估均达标。**"
            if passed
            else "- **尚未通过完整验收。时间/迭代上限退出或步态评估失败均不视为完成。**"
        )
        if evaluations:
            for name, value in evaluations.items():
                text.append(
                    f'- `{name}`：passed={value.get("passed", False)}；详见 `outputs/change1/{name}/summary.json`。'
                )
        if status.get("convergence"):
            c = status["convergence"]
            text.append(
                f'- reward 相邻窗口均值 {c["previous_mean"]:.6f} → {c["recent_mean"]:.6f}，'
                f'相对变化 {c["relative_change"]:.2%}，最近窗口相对标准差 {c["relative_noise"]:.2%}。'
            )
    path = ROOT / "docs/change1.md"
    doc = path.read_text()
    before, after = doc.split(START, 1)
    _, after = after.split(END, 1)
    path.write_text(before + START + "\n" + "\n".join(text) + "\n" + END + after)


def plot_training(rows, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    steps = [r["completed_updates"] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, layout="constrained")
    values = np.array([r["reward_per_step"] for r in rows])
    axes[0].plot(steps, values, alpha=0.35, lw=0.7, label="Per update")
    window = min(100, len(values))
    if window:
        axes[0].plot(
            steps[window - 1 :],
            np.convolve(values, np.ones(window) / window, mode="valid"),
            label=f"{window}-update mean",
        )
    axes[0].set_ylabel("Mean reward / step")
    axes[0].legend()
    axes[1].plot(steps, [r["mean_episode_length"] / 100 for r in rows])
    axes[1].set_ylabel("Episode length (s)")
    for key, label in (("Walk/phase_agreement", "Phase agreement"), ("Walk/single_support_fraction", "Single support")):
        axes[2].plot(steps, [r["quality"].get(key, 0) for r in rows], label=label)
    axes[2].set(xlabel="PPO update", ylabel="Fraction", ylim=(0, 1))
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def evaluate(checkpoint, name, *, randomize=False):
    directory = ROOT / "outputs/change1" / name
    summary = directory / "summary.json"
    if summary.exists():
        previous = json.loads(summary.read_text())
        if previous.get("checkpoint") == str(checkpoint):
            return previous
    command = [
        sys.executable,
        "-m",
        "instinct_mj.scripts.instinct_rl.evaluate_mild_walk",
        "--task",
        "Instinct-Locomotion-vae-G1-Play-v0",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(directory),
        "--steps",
        "1500",
        "--num-envs",
        "8",
    ]
    if randomize:
        command.append("--randomize-material")
    with (ROOT / "outputs/change1" / (name + ".log")).open("w") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode or not summary.exists():
        return {
            "passed": False,
            "error": f"evaluation process exited {result.returncode}",
            "checkpoint": str(checkpoint),
        }
    return json.loads(summary.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Refresh current progress without waiting or evaluating")
    args = parser.parse_args()
    output = ROOT / "outputs/change1"
    metadata_path = output / "run.json"
    metadata = json.loads(metadata_path.read_text())
    run = Path(metadata["run_dir"])
    previous_update = -1
    while True:
        rows = progress_rows(run)
        latest = rows[-1] if rows else None
        status_file = run / "training_status.json"
        status = json.loads(status_file.read_text()) if status_file.exists() else None
        update = latest["completed_updates"] if latest else 0
        if update != previous_update or status:
            metadata.update(status=status["status"] if status else "running", latest=latest)
            atomic_json(metadata_path, metadata)
            write_doc(run, latest, status)
            print(
                json.dumps(
                    {
                        "status": metadata["status"],
                        "updates": update,
                        "mean_episode_reward": latest.get("mean_episode_reward") if latest else None,
                    }
                ),
                flush=True,
            )
            previous_update = update
        if args.once:
            return
        if status:
            break
        time.sleep(30)
    checkpoint = run / f'model_{status["completed_updates"]}.pt'
    evaluations = {}
    if checkpoint.exists():
        evaluations["final_eval"] = evaluate(checkpoint, "final_eval")
        evaluations["final_eval_random_material"] = evaluate(checkpoint, "final_eval_random_material", randomize=True)
    if rows:
        plot_training(rows, output / "training_curve.png")
    passed = (
        status["status"] == "reward_converged"
        and bool(evaluations)
        and all(x.get("passed", False) for x in evaluations.values())
    )
    result = {"accepted": passed, "training": status, "checkpoint": str(checkpoint), "evaluations": evaluations}
    atomic_json(output / "result.json", result)
    write_doc(run, latest, status, evaluations)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
