"""Generate the real example images + numbers used by the RND concept figures
(doc/figures/concept-diagrams/concept-figures.html).

Recipe matches scripts/visualize_preprocessing.py: deterministic env (seed 1,
noop_max=0), 2,000-frame random-action warm-up for the whitening statistics,
whitening (x - mu) / sigma clipped to +/-5, and the untrained RNDModel at torch
seed 1 -- i.e. exactly the state at step 0 of training. Per-pixel attribution
is the input gradient |d err / d pixel| of the scalar prediction error.

Run from the project root with the venv active:
    python scripts/rnd_concept_examples.py
Outputs land in doc/figures/concept-diagrams/examples/.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "doc" / "figures" / "concept-diagrams" / "examples"
OUT.mkdir(parents=True, exist_ok=True)

import gymnasium as gym
import ale_py
import torch
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers.utils import RunningMeanStd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents.rnd import RNDModel

gym.register_envs(ale_py)


def make():
    env = gym.make("ALE/MontezumaRevenge-v5", frameskip=1)
    # noop_max=0: deterministic reset, so "the example frame" is well-defined
    return AtariPreprocessing(env, noop_max=0, frame_skip=4, screen_size=84,
                              grayscale_obs=True, terminal_on_life_loss=False)


def save_gray(arr, name, scale=6):
    up = np.kron(arr, np.ones((scale, scale)))
    plt.imsave(OUT / name, up, cmap="gray", vmin=0, vmax=255)


def save_cmap(arr, name, cmap, vmin, vmax, scale=6):
    up = np.kron(arr, np.ones((scale, scale)))
    plt.imsave(OUT / name, up, cmap=cmap, vmin=vmin, vmax=vmax)


# 1) obs-stats warm-up: 2,000 random-action frames
env = make()
env.action_space.seed(1)
obs, _ = env.reset(seed=1)
rms = RunningMeanStd(shape=(84, 84))
buf = []
for _ in range(2000):
    obs, _, term, trunc, _ = env.step(env.action_space.sample())
    buf.append(obs.astype(np.float32))
    if term or trunc:
        obs, _ = env.reset()
    if len(buf) == 128:
        rms.update(np.stack(buf))
        buf = []
if buf:
    rms.update(np.stack(buf))

# 2) deterministic example frame: fresh reset, 8x RIGHT (action 3)
obs, _ = env.reset(seed=1)
for _ in range(8):
    obs, _, _, _, _ = env.step(3)
frame = obs.astype(np.float32)
env.close()

white = np.clip((frame - rms.mean) / np.sqrt(rms.var + 1e-8), -5, 5)

# 3) untrained RNDModel: outputs, error, input-gradient attribution
torch.manual_seed(1)
rnd = RNDModel()
x = torch.tensor(white, dtype=torch.float32).reshape(1, 1, 84, 84)
x.requires_grad_(True)
pred, tgt = rnd(x)
err = ((tgt - pred) ** 2).sum() / 2  # same formula as rnd.py's _rnd_error
err.backward()
sal = x.grad.abs().squeeze().numpy()

# 4) TV patch on the same frame (NoisyTVWrapper's own RNG recipe, seed 1)
patch = np.random.default_rng([1, 0x7F00D]).integers(0, 256, (12, 84)).astype(np.float32)
stamped = frame.copy()
stamped[0:12, :] = patch
white_stamped = np.clip((stamped - rms.mean) / np.sqrt(rms.var + 1e-8), -5, 5)

save_gray(frame, "frame_plain.png")
save_gray(stamped, "frame_stamped.png")
save_cmap(white, "whitened.png", "RdBu_r", -5, 5)
save_cmap(white_stamped, "whitened_stamped.png", "RdBu_r", -5, 5)
save_cmap(sal, "saliency.png", "Blues", 0.0, float(sal.max()))

nums = {
    "err": round(float(err.item()), 2),
    "tgt6": [round(float(v), 2) for v in tgt[0, :6]],
    "pred6": [round(float(v), 2) for v in pred[0, :6]],
    "sal_max": float(sal.max()),
    "coverage_pct": round(100 * 12 * 84 / (84 * 84), 1),
    "obs_std_mean": round(float(np.sqrt(rms.var).mean()), 2),
}
with open(OUT / "numbers.json", "w") as f:
    json.dump(nums, f, indent=1)
print(json.dumps(nums, indent=1))
print("saved to", OUT)
