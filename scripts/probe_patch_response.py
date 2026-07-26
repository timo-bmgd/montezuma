"""Post-hoc RND patch-response probe — measures, from a saved checkpoint and with
NO new training, how the trained predictor responds to the noisy-TV patch. This
is the H1-mechanism read-out that the noisy-TV plan needs (doc/regression-findings.md
TASK F item F1) and that the room-1-confined runs can already supply.

For each checkpoint it loads the RND predictor+target and obs_rms, rolls the
trained agent for a few thousand room-1 steps, and reports the normalised RND
prediction error under three versions of every collected frame:

  displayed  — the frame exactly as shown during the rollout (patch present in
               remote/static; game HUD in off/sham)
  fresh(K)   — the same frame with the patch region re-stamped by K freshly
               sampled uniform patches (mean +/- std over the K draws)
  blank      — the same frame with the patch region set to its per-frame mean
               (the "no stimulus" baseline: normalises to ~0 in that region)

Derived quantities:
  patch_contribution = err(displayed) - err(blank)
      how much prediction error (=> intrinsic reward) the patch region adds at
      end of training. This is the per-checkpoint version of charts/tv_intrinsic_share.
  content_sensitivity = std_K(err(fresh)) / mean_K(err(fresh))
      does the predictor's error depend on the patch CONTENT, or has it collapsed
      to the content-invariant conditional-mean solution (H1's G=0 regime)? Near
      zero => content-invariant => no memorisation channel => no behavioural-capture
      lever, regardless of stimulus strength.
  G_proxy = mean_K(err(fresh)) - err(displayed)
      the memorisation gap G AT THIS RUN'S refresh setting. IMPORTANT: all the
      seed-42 runs used --tv-refresh-every 1 (patch resampled every step), so no
      patch ever persists long enough to be memorised and G_proxy is EXPECTED
      near zero (this is the T=1 end of P3's non-monotonic G(T) curve, not a
      falsification of H1). Recovering the full non-monotonic G(T) needs runs at
      T in {1, 64, inf} (inf = a real 'frozen' patch) -- those runs do not exist
      and the frozen/T=inf path is not yet in the code (F3/F4). This probe gives
      the T=1 datapoint and the content-sensitivity read for free.

Usage (run on the cluster where the checkpoints live, or anywhere with the venv):
    ./.venv/bin/python scripts/probe_patch_response.py \
        --checkpoints $SCRATCH/montezuma/checkpoints/*/ckpt_002441.pt \
        --num-frames 4096 --num-fresh 8 --out patch_response.csv

Point --checkpoints at one ckpt per category (off/remote/sham-remote/static). The
off/sham numbers are the floor: their region is game HUD, so 'fresh' stamps a
synthetic patch there -- read them as the calibration baseline, not memorisation.
"""
import argparse
import glob
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from gymnasium.wrappers.utils import RunningMeanStd
from agents.base import make_env, tv_region_slices
from agents.rnd import Agent, RNDModel, _normalize_obs, _rnd_error


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Checkpoint .pt paths (globs ok), one per TV category")
    p.add_argument("--num-frames", type=int, default=4096,
                   help="Room-1 frames to collect per checkpoint")
    p.add_argument("--num-fresh", type=int, default=8,
                   help="Freshly-sampled patches per frame (content-sensitivity avg)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="Optional CSV path")
    return p.parse_args()


def load_ckpt(path, device):
    ckpt = torch.load(path, weights_only=False, map_location=device)
    a = ckpt["args"]
    env = make_env(a["env_id"], idx=1, capture_video=False, run_name="probe",
                   clip_reward=a.get("clip_reward", True), tv_mode=a.get("tv_mode", "off"),
                   tv_size=tuple(a.get("tv_size", (12, 84))),
                   tv_position=tuple(a.get("tv_position", (0, 0))),
                   tv_refresh_every=a.get("tv_refresh_every", 1))()
    shim = types.SimpleNamespace(single_action_space=env.action_space,
                                 single_observation_space=env.observation_space)
    agent = Agent(shim).to(device)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()
    rnd = RNDModel().to(device)
    rnd.load_state_dict(ckpt["rnd_model_state_dict"])
    rnd.eval()
    obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
    obs_rms.mean = ckpt["obs_rms_mean"]
    obs_rms.var = ckpt["obs_rms_var"]
    obs_rms.count = ckpt["obs_rms_count"]
    return ckpt, a, env, agent, rnd, obs_rms


@torch.no_grad()
def collect_frames(env, agent, device, n, seed):
    obs, _ = env.reset(seed=seed)
    frames = []
    for _ in range(n):
        x = torch.tensor(np.asarray(obs), dtype=torch.float32, device=device).unsqueeze(0)
        action, *_ = agent.get_action_and_value(x)
        obs, _, term, trunc, _ = env.step(int(action.item()))
        frames.append(np.asarray(obs)[3].astype(np.float32).copy())  # latest 84x84 frame
        if term or trunc:
            obs, _ = env.reset()
    return np.stack(frames)[:, None, :, :]  # (N,1,84,84)


@torch.no_grad()
def err_mean(rnd, frames_np, obs_rms, device):
    return float(_rnd_error(rnd, _normalize_obs(frames_np, obs_rms, device)).mean().item())


def main():
    args = parse_args()
    paths = [p for pat in args.checkpoints for p in sorted(glob.glob(pat))]
    if not paths:
        raise SystemExit(f"no checkpoints matched: {args.checkpoints}")
    device = torch.device(args.device)
    rows = []
    for path in paths:
        ckpt, a, env, agent, rnd, obs_rms = load_ckpt(path, device)
        mode = a.get("tv_mode", "off")
        rs, cs = tv_region_slices(tuple(a.get("tv_position", (0, 0))),
                                  tuple(a.get("tv_size", (12, 84))))
        rng = np.random.default_rng([a.get("seed", 1), 0xC0FFEE])
        frames = collect_frames(env, agent, device, args.num_frames, a.get("seed", 1))
        env.close()

        err_disp = err_mean(rnd, frames, obs_rms, device)
        # blank: region -> per-frame mean of that region
        blank = frames.copy()
        reg_mean = blank[:, :, rs, cs].mean(axis=(2, 3), keepdims=True)
        blank[:, :, rs, cs] = reg_mean
        err_blank = err_mean(rnd, blank, obs_rms, device)
        # fresh: K resampled patches
        fresh_errs = []
        ph = rs.stop - rs.start
        pw = cs.stop - cs.start
        for _ in range(args.num_fresh):
            f = frames.copy()
            f[:, :, rs, cs] = rng.integers(0, 256, size=(len(f), 1, ph, pw)).astype(np.float32)
            fresh_errs.append(err_mean(rnd, f, obs_rms, device))
        fresh_errs = np.array(fresh_errs)

        row = {
            "checkpoint": os.path.basename(os.path.dirname(path)),
            "tv_mode": mode,
            "err_displayed": err_disp,
            "err_blank": err_blank,
            "err_fresh_mean": float(fresh_errs.mean()),
            "err_fresh_std": float(fresh_errs.std()),
            "patch_contribution": err_disp - err_blank,
            "content_sensitivity": float(fresh_errs.std() / max(fresh_errs.mean(), 1e-8)),
            "G_proxy": float(fresh_errs.mean()) - err_disp,
        }
        rows.append(row)
        print(f"[{mode:12s}] err_disp={err_disp:.4f} err_blank={err_blank:.4f} "
              f"err_fresh={fresh_errs.mean():.4f}+/-{fresh_errs.std():.4f}  "
              f"patch_contrib={row['patch_contribution']:+.4f} "
              f"content_sens={row['content_sensitivity']:.3f} G_proxy={row['G_proxy']:+.4f}")

    if args.out:
        import csv
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
