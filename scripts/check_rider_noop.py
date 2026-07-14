"""No-op check for the passive rider: prove it never touches the training run.

Runs count_based.py twice with identical seeds — once plain, once with
--passive-rnd + --step-log + --artifact-interval — and asserts that the final
checkpoint's agent weights, optimizer state, and global_step are BIT-IDENTICAL
between the two runs, and that the step logs' shared columns (extrinsic
reward, active bonus, room, episode id, done) match exactly. Any influence of
the rider on the reward, the RNG streams, or the update math would break
exact equality within three iterations.

Also verifies the rider actually did something: the rider run's step log must
contain finite, non-degenerate passive bonuses, and its artefact files must
exist (their sizes are printed for the checkpointing size budget).

Run from the project root with the venv active (takes a few minutes, CPU):
    python scripts/check_rider_noop.py
"""

import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

COMMON = [
    "--total-timesteps", "1536",   # 3 iterations at num_envs=4 * num_steps=128
    "--num-envs", "4",
    "--seed", "7",
    "--sync-envs",
    "--no-cuda",
    "--step-log",
]


def run_variant(workdir: Path, extra_args: list[str]) -> Path:
    cmd = [sys.executable, "src/agents/count_based.py", *COMMON,
           "--runs-dir", str(workdir / "runs"),
           "--videos-dir", str(workdir / "videos"),
           "--checkpoint-dir", str(workdir / "checkpoints"),
           *extra_args]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)
    return workdir


def load_ckpt(workdir: Path) -> dict:
    paths = glob.glob(str(workdir / "checkpoints" / "**" / "ckpt_*.pt"), recursive=True)
    assert len(paths) == 1, f"expected exactly one checkpoint, found {paths}"
    return torch.load(paths[0], weights_only=False)


def load_step_log(workdir: Path) -> dict:
    paths = glob.glob(str(workdir / "runs" / "**" / "step_log" / "*.npz"), recursive=True)
    assert len(paths) == 1, f"expected exactly one step-log shard, found {paths}"
    return dict(np.load(paths[0]))


def assert_tensors_equal(a, b, path=""):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), f"{path}: key mismatch"
        for k in a:
            assert_tensors_equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, torch.Tensor):
        assert torch.equal(a, b), f"{path}: tensor mismatch"
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: length mismatch"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_tensors_equal(x, y, f"{path}[{i}]")
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        print("=== run A: plain count_based (no rider) ===")
        a = run_variant(tmp / "a", [])
        print("\n=== run B: count_based + passive RND rider ===")
        b = run_variant(tmp / "b", ["--passive-rnd",
                                    "--rider-obs-norm-init-steps", "2",
                                    "--artifact-interval", "3"])

        ckpt_a, ckpt_b = load_ckpt(a), load_ckpt(b)

        assert ckpt_a["global_step"] == ckpt_b["global_step"], "global_step differs"
        assert_tensors_equal(ckpt_a["agent_state_dict"], ckpt_b["agent_state_dict"],
                             "agent_state_dict")
        assert_tensors_equal(ckpt_a["optimizer_state_dict"]["state"],
                             ckpt_b["optimizer_state_dict"]["state"],
                             "optimizer_state")
        assert "rider_state" in ckpt_b and "rider_state" not in ckpt_a
        print("\nOK: agent weights, optimizer state, and global_step are "
              "bit-identical with and without the rider.")

        log_a, log_b = load_step_log(a), load_step_log(b)
        for key in ("reward_ext", "bonus_active", "room", "episode_id", "done",
                    "global_step", "iteration"):
            assert np.array_equal(log_a[key], log_b[key]), f"step log {key} differs"
        print("OK: step-log trajectories (reward, active bonus, room, episode id, "
              "done) are identical across both runs.")

        passive = log_b["bonus_passive"]
        assert np.isfinite(passive).all(), "rider bonuses contain non-finite values"
        assert passive.std() > 0, "rider bonuses are constant -- rider not computing?"
        assert np.isnan(log_a["bonus_passive"]).all(), \
            "run A has passive bonuses despite no rider"
        assert np.isfinite(log_b["norm_passive"]).all() and (log_b["norm_passive"] > 0).all()
        print(f"OK: rider produced a live signal "
              f"(raw bonus mean={passive.mean():.4f}, std={passive.std():.4f}).")

        rooms = log_b["room"]
        assert (rooms >= 0).all(), "step log contains unresolved room ids (-1)"
        print(f"OK: room ids resolved for every step (values: {sorted(set(rooms.flatten().tolist()))}).")

        art = sorted(glob.glob(str(b / "checkpoints" / "**" / "artifacts" / "*"),
                               recursive=True))
        assert art, "no artefact files written"
        print("\nArtefact sizes:")
        for f in art:
            print(f"  {os.path.basename(f):32s} {os.path.getsize(f) / 1e6:8.2f} MB")

        print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
