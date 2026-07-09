"""Throughput profiler for the Montezuma's Revenge training pipeline.

Measures where training time actually goes -- emulation/wrapper overhead,
AsyncVectorEnv subprocess IPC, NN inference, NN backward -- so the standing
"CPU/emulation-bound, not GPU-bound" hypothesis (based on a flat 144-183 SPS
observed across a prior 10M-step RND run) can be checked against real numbers
instead of reasoning alone.

Run from project root with the venv active:
    source .venv/bin/activate
    python scripts/profile_throughput.py                              # local sanity check
    python scripts/profile_throughput.py --num-envs 32 --measure-steps 300   # cluster, real numbers
    python scripts/profile_throughput.py --cprofile                   # function-level breakdown
"""

import os
import sys

import argparse
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from agents.base import NatureCNN, RoomTracker, layer_init, make_env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="ALE/MontezumaRevenge-v5")
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--num-steps", type=int, default=128,
                    help="Env-steps per backward pass, matching the agents' default rollout length")
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--measure-steps", type=int, default=50)
    p.add_argument("--sync-envs", action="store_true",
                    help="Use SyncVectorEnv instead of AsyncVectorEnv (also forced on by --cprofile)")
    p.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-room-tracker-ablation", action="store_true",
                    help="Skip the getRAM() with/without comparison (roughly halves phase-1 runtime)")
    p.add_argument("--cprofile", action="store_true",
                    help="Wrap env.step() in cProfile and print a function-level breakdown. Forces "
                         "--sync-envs: under AsyncVectorEnv, cProfile in the main process only sees "
                         "IPC-wait time, not what happens inside env.step() in the worker subprocess.")
    return p.parse_args()


class TinyAgent(nn.Module):
    """Mirrors ppo.py's Agent class shape, for realistic forward/backward FLOPs."""

    def __init__(self, num_actions):
        super().__init__()
        self.cnn = NatureCNN()
        self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def forward(self, x):
        features = self.cnn(x)
        return self.actor(features), self.critic(features)


def make_vector_env(args, context=None):
    VecCls = gym.vector.SyncVectorEnv if args.sync_envs else gym.vector.AsyncVectorEnv
    fns = [make_env(args.env_id, i, capture_video=False, run_name="profile") for i in range(args.num_envs)]
    if VecCls is gym.vector.AsyncVectorEnv and context is not None:
        return VecCls(fns, context=context)
    return VecCls(fns)


def random_step(envs, num_envs):
    action_space = envs.single_action_space
    actions = np.array([action_space.sample() for _ in range(num_envs)])
    envs.step(actions)


def time_env_only(args):
    envs = make_vector_env(args)
    try:
        envs.reset()
        for _ in range(args.warmup_steps):
            random_step(envs, args.num_envs)
        start = time.time()
        for _ in range(args.measure_steps):
            random_step(envs, args.num_envs)
        elapsed = time.time() - start
    finally:
        envs.close()
    return args.num_envs * args.measure_steps / elapsed


def time_room_tracker_ablation(args):
    """Compares env-only SPS with RoomTracker's getRAM() call stubbed out.

    The stub must be applied to the RoomTracker class *before* AsyncVectorEnv
    spawns its workers, and the workers must inherit that patched state via
    fork (not macOS's spawn default, which re-imports agents.base fresh in
    each worker and would silently make this ablation a no-op). This runs
    before any CUDA context exists in this process, so forking here is safe.
    """
    original_room = RoomTracker._room
    RoomTracker._room = lambda self: 1
    try:
        context = None if args.sync_envs else "fork"
        envs = make_vector_env(args, context=context)
        try:
            envs.reset()
            for _ in range(args.warmup_steps):
                random_step(envs, args.num_envs)
            start = time.time()
            for _ in range(args.measure_steps):
                random_step(envs, args.num_envs)
            elapsed = time.time() - start
        finally:
            envs.close()
    finally:
        RoomTracker._room = original_room
    return args.num_envs * args.measure_steps / elapsed


def inference_step(envs, agent, device, next_obs):
    with torch.no_grad():
        logits, _ = agent(next_obs)
        action = Categorical(logits=logits).sample()
    obs_np, _, _, _, _ = envs.step(action.cpu().numpy())
    return torch.tensor(obs_np, dtype=torch.float32, device=device)


def time_inference_only(args, envs, agent, device):
    next_obs, _ = envs.reset()
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    for _ in range(args.warmup_steps):
        next_obs = inference_step(envs, agent, device, next_obs)
    start = time.time()
    for _ in range(args.measure_steps):
        next_obs = inference_step(envs, agent, device, next_obs)
    elapsed = time.time() - start
    return args.num_envs * args.measure_steps / elapsed


def time_full_iteration(args, envs, agent, optimizer, device):
    """Rollout collection + one backward pass every num_steps env-steps,
    mirroring the real PPO loop's cadence (see ppo.py's train())."""
    obs_shape = envs.single_observation_space.shape

    def run_chunk(next_obs, chunk_len):
        obs_buf = torch.zeros((chunk_len, args.num_envs) + obs_shape, device=device)
        for t in range(chunk_len):
            obs_buf[t] = next_obs
            next_obs = inference_step(envs, agent, device, next_obs)
        b_obs = obs_buf.reshape((-1,) + obs_shape)
        logits, values = agent(b_obs)
        dummy_loss = Categorical(logits=logits).entropy().mean() + values.mean()
        optimizer.zero_grad()
        dummy_loss.backward()
        optimizer.step()
        return next_obs

    next_obs, _ = envs.reset()
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_obs = run_chunk(next_obs, min(args.warmup_steps, args.num_steps) or 1)

    start = time.time()
    steps_done = 0
    while steps_done < args.measure_steps:
        chunk_len = min(args.num_steps, args.measure_steps - steps_done)
        next_obs = run_chunk(next_obs, chunk_len)
        steps_done += chunk_len
    elapsed = time.time() - start
    return args.num_envs * steps_done / elapsed


def run_cprofile(args):
    import cProfile
    import pstats

    envs = make_vector_env(args)
    try:
        envs.reset()
        for _ in range(args.warmup_steps):
            random_step(envs, args.num_envs)
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(args.measure_steps):
            random_step(envs, args.num_envs)
        profiler.disable()
    finally:
        envs.close()
    pstats.Stats(profiler).sort_stats("cumulative").print_stats(20)


def main():
    args = parse_args()
    if args.cprofile and not args.sync_envs:
        print("(--cprofile forces --sync-envs: AsyncVectorEnv's main process only sees IPC-wait "
              "time, not what happens inside env.step() in the worker subprocess)")
        args.sync_envs = True

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"=== Throughput Profile: {args.env_id} ===")
    print(f"num_envs={args.num_envs}  sync_envs={args.sync_envs}  device={device}  "
          f"warmup={args.warmup_steps}  measured={args.measure_steps}")
    print()

    if args.cprofile:
        print(f"[cProfile] top 20 cumulative-time functions inside env.step() ({args.measure_steps} steps):")
        run_cprofile(args)
        return

    sps_env_only = time_env_only(args)
    print(f"[1]  Env-only, random actions (RoomTracker active)    : {sps_env_only:8.1f} SPS")

    if not args.skip_room_tracker_ablation:
        sps_stubbed = time_room_tracker_ablation(args)
        tax_pct = 100.0 * (sps_stubbed - sps_env_only) / sps_stubbed if sps_stubbed else 0.0
        print(f"[1b] Env-only, random actions (RoomTracker stubbed)   : {sps_stubbed:8.1f} SPS   "
              f"(getRAM() tax: ~{tax_pct:.1f}%)")

    envs = make_vector_env(args)
    try:
        num_actions = envs.single_action_space.n
        agent = TinyAgent(num_actions).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)

        sps_inference = time_inference_only(args, envs, agent, device)
        print(f"[2]  Full step, forward only (inference)               : {sps_inference:8.1f} SPS")

        sps_full = time_full_iteration(args, envs, agent, optimizer, device)
        print(f"[3]  Full step + backward (every {args.num_steps} steps)           : {sps_full:8.1f} SPS")
    finally:
        envs.close()

    if sps_env_only:
        env_share = 100.0 * sps_full / sps_env_only
        print(f"     -> full-step throughput is ~{env_share:.0f}% of the env-only baseline; "
              f"a number close to 100% confirms the bottleneck is env/emulation, not NN compute")


if __name__ == "__main__":
    main()
