"""SimHash bucket-occupancy probe: is the hash degenerate on Montezuma?

Rolls a random policy through the real training wrapper stack and feeds the
identical observation stream to several SimHashCounter configurations at
once, then reports bucket-occupancy statistics for each. The sanity criterion
for a *usable* count signal (see the dual-signal rider work): occupancy is
neither all-unique (singleton_frac ~ 1.0 -> bonus is a flat beta everywhere,
no signal) nor collapsed (top_bucket_share ~ 1.0 -> every state in one
bucket, also no signal).

Run from the project root with the venv active:
    python scripts/simhash_occupancy_probe.py --steps 50000 --num-envs 8
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gymnasium as gym
import numpy as np

from agents.base import make_env
from agents.count_based import SimHashCounter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=50_000,
                   help="Total observations to hash (summed over envs)")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()

    configs = [
        ("index  k=64  (count_based.py default)", dict(hash_dim=64,  mode="index")),
        ("pool16 k=64", dict(hash_dim=64,  mode="pool", pool_size=16)),
        ("pool16 k=128", dict(hash_dim=128, mode="pool", pool_size=16)),
        ("pool16 k=32", dict(hash_dim=32,  mode="pool", pool_size=16)),
        ("pool8  k=64", dict(hash_dim=64,  mode="pool", pool_size=8)),
    ]
    counters = [SimHashCounter(seed=args.seed, **kw) for _, kw in configs]

    envs = gym.vector.SyncVectorEnv(
        [make_env("ALE/MontezumaRevenge-v5", i, capture_video=False, run_name="occupancy_probe")
         for i in range(args.num_envs)]
    )
    obs, _ = envs.reset(seed=args.seed)

    t0 = time.time()
    steps_done = 0
    while steps_done < args.steps:
        obs, _, _, _, _ = envs.step(envs.action_space.sample())
        for i in range(args.num_envs):
            for c in counters:
                c.increment(obs[i])
        steps_done += args.num_envs
        if steps_done % 10_000 < args.num_envs:
            print(f"  {steps_done}/{args.steps} steps "
                  f"({steps_done / (time.time() - t0):.0f} obs/s)")
    envs.close()

    print(f"\nRandom-policy stream, {steps_done} observations, seed={args.seed}\n")
    header = (f"{'config':38s} {'unique':>9s} {'singleton':>10s} "
              f"{'mean_n':>8s} {'max_n':>7s} {'top_share':>10s}")
    print(header)
    print("-" * len(header))
    for (name, _), c in zip(configs, counters):
        s = c.occupancy_stats()
        print(f"{name:38s} {s['unique']:>9d} {s['singleton_frac']:>10.3f} "
              f"{s['mean_count']:>8.1f} {s['max_count']:>7d} {s['top_bucket_share']:>10.3f}")
    print("\nsingleton = fraction of buckets holding exactly one visit "
          "(~1.0 -> degenerate all-unique);\ntop_share = share of all visits "
          "in the single largest bucket (~1.0 -> collapsed).")


if __name__ == "__main__":
    main()
