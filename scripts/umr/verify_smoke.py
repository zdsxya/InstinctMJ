"""Verify actual PPO/discriminator updates and finite short-training outputs."""

import argparse
import json
import math
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def finite_tensors(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tensors(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tensors(v) for v in value)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    checkpoint = args.run / f"model_{args.iterations}.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    steps = {}
    for name in ("optimizer_state_dict", "discriminator_optimizer"):
        values = [float(s["step"]) for s in state[name]["state"].values()]
        if not values or min(values) <= 0:
            raise ValueError(f"No optimizer updates: {name}")
        steps[name] = dict(min=min(values), max=max(values))
    if not state.get("discriminator") or not finite_tensors(state):
        raise ValueError("Missing discriminator or nonfinite checkpoint tensors")
    accumulator = EventAccumulator(str(args.run))
    accumulator.Reload()
    metrics = {}
    for tag in ("Loss/discriminator_loss", "Loss/discriminator_gradient_penalty", "Loss/amp_total_loss",
                "Step/discriminator_reward", "Perf/gpu_global_free_mem"):
        values = accumulator.Scalars(tag)
        if len(values) < args.iterations or not all(math.isfinite(v.value) for v in values):
            raise ValueError(f"Missing/nonfinite training metric: {tag}")
        metrics[tag] = [v.value for v in values]
    first = torch.load(args.run / "model_1.pt", map_location="cpu", weights_only=False)
    changed = any(not torch.equal(first["discriminator"][k], v) for k, v in state["discriminator"].items())
    if not changed:
        raise ValueError("Discriminator did not change after the first saved checkpoint")
    report = dict(passed=True, checkpoint=str(checkpoint.resolve()), iterations=int(state["iter"]),
                  finite_checkpoint=True, discriminator_changed=True, optimizer_steps=steps, metrics=metrics)
    (args.run / "smoke_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
