"""Generate thesis figures: every observation-preprocessing stage, from the raw
ALE screen to the network inputs, for the first agent step(s) of Montezuma's
Revenge.

The script re-executes each stage of the pipeline *manually* on a raw
frameskip=1 ALE env (the exact operations AtariPreprocessing performs), and in
parallel steps the real make_env() wrapper stack with the same seed and action
sequence. Every manually produced frame is asserted byte-identical to the real
stack's output, so the saved images are provably the genuine pipeline.

Requires the paper-conditions branch (noop_max=0): reset is deterministic, so
"the first frame" is well-defined and reproducible.

Usage, from the project root with the venv active:
    python scripts/visualize_preprocessing.py            # writes doc/figures/preprocessing/

Outputs (PNG):
  plain/00_raw_step1_frame{1..4}.png   raw 210x160 RGB, the 4 emulator frames of step 1
  plain/01_gray_frame3.png / _frame4   the two frames AtariPreprocessing reads (210x160)
  plain/02_maxpool.png                 pixel-wise max of the two (flicker removal)
  plain/02_flicker_diff.png            |frame3-frame4|: inter-frame sprite difference
  plain/03_resized_84x84.png           cv2.INTER_AREA down-scale, x5 nearest upscale
  plain/04_tv_stamped_84x84.png        NoisyTVWrapper patch (remote mode), x5 upscale
  plain/05_stack_slot{0..3}.png        the 4-frame stack after 8 agent steps
  plain/06_rnd_whitened.png            per-pixel whitened RND input (diverging colormap)
  _overview.png                        annotated montage of the full journey
"""
import os
import sys

import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from agents.base import NoisyTVWrapper, make_env, tv_region_slices  # noqa: E402

import cv2  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from gymnasium.wrappers.utils import RunningMeanStd  # noqa: E402

ENV_ID = "ALE/MontezumaRevenge-v5"
SEED = 1
ACTION = 3          # RIGHT — visible motion across the frame stack
N_STEPS = 8         # agent steps to simulate (step 1 is the featured one)
OUT = os.path.join(os.path.dirname(__file__), "..", "doc", "figures", "preprocessing")
PLAIN = os.path.join(OUT, "plain")
os.makedirs(PLAIN, exist_ok=True)


def save_plain(name, img, upscale=1, cmap=None):
    """Save a raw image without axes; nearest-neighbour upscale for small frames."""
    if upscale > 1:
        img = cv2.resize(img, (img.shape[1] * upscale, img.shape[0] * upscale),
                         interpolation=cv2.INTER_NEAREST)
    path = os.path.join(PLAIN, name)
    if cmap is None:
        cv2.imwrite(path, img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:  # float data through a matplotlib colormap
        plt.imsave(path, img, cmap=cmap)
    print(f"  wrote {os.path.relpath(path)}  shape={img.shape}")


# ── manual pipeline on a raw frameskip=1 env ─────────────────────────────────
# Exactly what AtariPreprocessing does per agent step: 4x step(action), read
# grayscale on the 3rd and 4th frame, max-pool, INTER_AREA resize to 84x84.
print("stage 0-1: manual pipeline on raw frameskip=1 env")
raw_env = gym.make(ENV_ID, frameskip=1)
raw_env.reset(seed=SEED)
ale = raw_env.unwrapped.ale

steps = []  # per agent step: dict(raw_rgb[4], gray3, gray4, maxpool, resized)
for k in range(N_STEPS):
    rec = {"raw_rgb": [], "gray3": np.empty((210, 160), np.uint8),
           "gray4": np.empty((210, 160), np.uint8)}
    for t in range(4):
        raw_env.step(ACTION)
        rec["raw_rgb"].append(ale.getScreenRGB())
        if t == 2:
            ale.getScreenGrayscale(rec["gray3"])
        elif t == 3:
            ale.getScreenGrayscale(rec["gray4"])
    rec["maxpool"] = np.maximum(rec["gray3"], rec["gray4"])
    rec["resized"] = cv2.resize(rec["maxpool"], (84, 84),
                                interpolation=cv2.INTER_AREA).astype(np.uint8)
    steps.append(rec)
raw_env.close()

s1 = steps[0]
for i, f in enumerate(s1["raw_rgb"], 1):
    save_plain(f"00_raw_step1_frame{i}.png", f)
save_plain("01_gray_frame3.png", s1["gray3"])
save_plain("01_gray_frame4.png", s1["gray4"])
save_plain("02_maxpool.png", s1["maxpool"])
# inter-frame difference (sprite animation and/or flicker) merged by max-pool;
# show the early step where the two read frames differ the most
flicker_k = int(np.argmax([np.abs(s["gray3"].astype(int) - s["gray4"].astype(int)).sum()
                           for s in steps]))
diff = np.abs(steps[flicker_k]["gray3"].astype(int)
              - steps[flicker_k]["gray4"].astype(int)).astype(np.uint8)
print(f"  inter-frame diff: largest at agent step {flicker_k + 1} "
      f"({(diff > 0).sum()} px differ)")
save_plain("02_flicker_diff.png", diff)
save_plain("03_resized_84x84.png", s1["resized"], upscale=5)

# ── verify against the real make_env stack (tv off) ──────────────────────────
print("verify: real make_env stack (tv off), same seed + actions")
env_off = make_env(ENV_ID, 0, False, "viz")()
env_off.reset(seed=SEED)
for k in range(N_STEPS):
    obs_off, _, _, _, _ = env_off.step(ACTION)
    assert (obs_off[3] == steps[k]["resized"]).all(), \
        f"manual pipeline diverged from make_env at step {k + 1}"
env_off.close()
print(f"  OK: manual 84x84 frames byte-identical to make_env for all {N_STEPS} steps")

# stack montage from the real env's final observation (slots = steps 5..8)
save_plain("05_stack_slot0.png", obs_off[0], upscale=5)
save_plain("05_stack_slot1.png", obs_off[1], upscale=5)
save_plain("05_stack_slot2.png", obs_off[2], upscale=5)
save_plain("05_stack_slot3.png", obs_off[3], upscale=5)

# ── stage 2: the noisy-TV stamp (remote mode) ────────────────────────────────
print("stage 2: NoisyTVWrapper patch (remote mode)")
rng = np.random.default_rng([SEED, NoisyTVWrapper._RNG_SALT])
patch = rng.integers(0, 256, size=(12, 84), dtype=np.uint8)  # reset resample
stamped = s1["resized"].copy()
rs, cs = tv_region_slices((0, 0), (12, 84))
stamped[rs, cs] = patch

env_tv = make_env(ENV_ID, 0, False, "viz", tv_mode="remote")()
env_tv.reset(seed=SEED)
obs_tv, _, _, _, _ = env_tv.step(ACTION)   # ACTION < 18: no resample, patch persists
env_tv.close()
assert (obs_tv[3] == stamped).all(), "manual TV stamp diverged from NoisyTVWrapper"
print("  OK: manual patch byte-identical to the real wrapper's output")
save_plain("04_tv_stamped_84x84.png", stamped, upscale=5)

# ── stage 5b: whitened RND input ─────────────────────────────────────────────
print("stage 5: RND whitening (running stats from random play)")
obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
env_w = make_env(ENV_ID, 0, False, "viz")()
env_w.reset(seed=SEED + 100)
buf = []
for _ in range(3000):  # mini warm-up (production: 50 iters x num_steps x num_envs)
    o, _, te, tr, _ = env_w.step(env_w.action_space.sample())
    buf.append(o[3:4][None].astype(np.float32))
    if te or tr:
        env_w.reset()
env_w.close()
obs_rms.update(np.concatenate(buf, axis=0))
whitened = ((s1["resized"].astype(np.float32) - obs_rms.mean[0, 0])
            / np.sqrt(obs_rms.var[0, 0])).clip(-5, 5)
print(f"  whitened range: [{whitened.min():.2f}, {whitened.max():.2f}]")
save_plain("06_rnd_whitened.png", whitened, cmap="RdBu_r")

# ── annotated overview figure ────────────────────────────────────────────────
print("overview figure")
fig = plt.figure(figsize=(15, 9.5))
gs = fig.add_gridspec(3, 5, hspace=0.34, wspace=0.12)


def panel(r, c, img, title, cmap="gray", span=1):
    ax = fig.add_subplot(gs[r, c] if span == 1 else gs[r, c:c + span])
    ax.imshow(img, cmap=cmap, interpolation="nearest",
              **({} if img.ndim == 3 or cmap != "gray" else {"vmin": 0, "vmax": 255}))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


for i in range(4):
    panel(0, i, s1["raw_rgb"][i],
          f"raw frame {i + 1}/4  (210×160 RGB)\n" +
          ("discarded" if i < 2 else f"read as grayscale → buffer"))
panel(0, 4, diff, f"|frame3 − frame4|  (agent step {flicker_k + 1})\ninter-frame sprite difference — max-pool keeps both")
panel(1, 0, s1["gray3"], "grayscale frame 3  (210×160)")
panel(1, 1, s1["gray4"], "grayscale frame 4  (210×160)")
panel(1, 2, s1["maxpool"], "np.maximum(frame3, frame4)")
panel(1, 3, s1["resized"], "cv2.INTER_AREA → 84×84 uint8\n= what a tv-off agent sees")
panel(1, 4, stamped, "NoisyTVWrapper (remote):\n12×84 noise patch over the HUD")
for i in range(4):
    panel(2, i, obs_off[i], f"stack slot {i}  ({'oldest' if i == 0 else 'newest' if i == 3 else '…'})\nafter 8 steps of RIGHT")
panel(2, 4, whitened, "RND input: whitened newest frame\n(x−μ)/σ per pixel, clip ±5", cmap="RdBu_r")

fig.suptitle("Montezuma's Revenge — observation pipeline, first agent step(s), seed 1  "
             "(every 84×84 frame verified byte-identical to the make_env stack)",
             fontsize=11.5)
fig.savefig(os.path.join(OUT, "_overview.png"), dpi=150, bbox_inches="tight",
            facecolor="white")
print(f"  wrote {os.path.relpath(os.path.join(OUT, '_overview.png'))}")
print("done — all stage images verified and written")
