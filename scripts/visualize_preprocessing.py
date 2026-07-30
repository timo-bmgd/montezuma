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
  plain/06_rnd_whitened.png            whitened RND input, diverging colormap centered
                                       at 0 (vmin/vmax = -5/+5, the clip bounds)
  plain/06_rnd_whitened_gray.png       same data, neutral grayscale (0 = mid-gray)
  _overview.png                        annotated top-to-bottom flowchart of the journey
  _pipeline_horizontal.png / .pdf      compact left-to-right pipeline strip (paper figure)
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
from matplotlib.patches import (ConnectionPatch, FancyArrowPatch,  # noqa: E402
                                FancyBboxPatch, Rectangle)
from gymnasium.wrappers.utils import RunningMeanStd  # noqa: E402

ENV_ID = "ALE/MontezumaRevenge-v5"
SEED = 1
ACTION = 3          # RIGHT — visible motion across the frame stack
N_STEPS = 8         # agent steps to simulate (step 1 is the featured one)
OUT = os.path.join(os.path.dirname(__file__), "..", "doc", "figures", "preprocessing")
PLAIN = os.path.join(OUT, "plain")
os.makedirs(PLAIN, exist_ok=True)


def save_plain(name, img, upscale=1, cmap=None, vmin=None, vmax=None):
    """Save a raw image without axes; nearest-neighbour upscale for small frames."""
    if upscale > 1:
        img = cv2.resize(img, (img.shape[1] * upscale, img.shape[0] * upscale),
                         interpolation=cv2.INTER_NEAREST)
    path = os.path.join(PLAIN, name)
    if cmap is None:
        cv2.imwrite(path, img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:  # float data through a matplotlib colormap
        plt.imsave(path, img, cmap=cmap, vmin=vmin, vmax=vmax)
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
# symmetric limits so 0 (= "pixel matches its running average") is the neutral colour
save_plain("06_rnd_whitened.png", whitened, upscale=5, cmap="RdBu_r", vmin=-5, vmax=5)
save_plain("06_rnd_whitened_gray.png", whitened, upscale=5, cmap="gray", vmin=-5, vmax=5)

# ── annotated overview: strict top-to-bottom flowchart ───────────────────────
print("overview figure")
GRAY = {"cmap": "gray", "vmin": 0, "vmax": 255}
EDGE = "#3a3f4a"
LBL = "#8a4a10"
MUTED = "#5b6270"

fig = plt.figure(figsize=(13, 18))
gs = fig.add_gridspec(6, 12, height_ratios=[1.25, 1.25, 1.25, 1.0, 1.0, 1.15],
                      left=0.03, right=0.97, top=0.925, bottom=0.03,
                      hspace=0.55, wspace=0.18)


def panel(r, c0, c1, img, caption, **imshow_kw):
    ax = fig.add_subplot(gs[r, c0:c1])
    ax.imshow(img, interpolation="nearest", **imshow_kw)
    ax.axis("off")
    ax.text(0.5, -0.045, caption, transform=ax.transAxes, ha="center", va="top",
            fontsize=9.2, color="#22262e")
    return ax


def group_box(axs, label, dashed=False):
    bxs = [a.get_position() for a in axs]
    x0 = min(b.x0 for b in bxs) - 0.011
    x1 = max(b.x1 for b in bxs) + 0.011
    y0 = min(b.y0 for b in bxs) - 0.033
    y1 = max(b.y1 for b in bxs) + 0.007
    fig.add_artist(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure,
        boxstyle="round,pad=0.004,rounding_size=0.008", fill=False,
        edgecolor="#8a93a3", linewidth=1.3,
        linestyle="--" if dashed else "-", zorder=1.5))
    fig.text(x0 + 0.002, y1 + 0.010, label, fontsize=11.5, fontweight="bold",
             va="bottom", ha="left", color="#16181d", bbox=dict(facecolor="white", edgecolor="none", pad=1.2), zorder=6)


def arrow(axA, axB, xyA=(0.5, -0.17), xyB=(0.5, 1.06), label=None, dx=0.010):
    fig.add_artist(ConnectionPatch(
        xyA=xyA, coordsA=axA.transAxes, xyB=xyB, coordsB=axB.transAxes,
        arrowstyle="-|>", mutation_scale=16, lw=1.7, color=EDGE, zorder=5))
    if label:
        pa, pb = axA.get_position(), axB.get_position()
        xm = pa.x0 + (pa.x1 - pa.x0) * xyA[0]
        ym = pb.y1 + 0.62 * (pa.y0 - pb.y1)   # nearer the source: clears headers
        fig.text(xm + dx, ym, label, fontsize=8.8,
                 family="monospace", color=LBL, va="center", ha="left", bbox=dict(facecolor="white", edgecolor="none", pad=1.2), zorder=6)


# STAGE 1: the 4 raw frames of one agent step
raw_axs = []
for i in range(4):
    cap = f"frame {i + 1}/4 — " + ("discarded ✗" if i < 2 else "READ ✓")
    raw_axs.append(panel(0, 3 * i, 3 * i + 3, s1["raw_rgb"][i], cap))
group_box(raw_axs, "①  ONE AGENT STEP — the same action is fed to 4 consecutive "
                   "raw emulator frames (each 210×160 RGB)")

# STAGE 2: grayscale reads of frames 3+4, aligned under their raw frames
g3 = panel(1, 6, 9, s1["gray3"], "grayscale frame 3  (210×160)", **GRAY)
g4 = panel(1, 9, 12, s1["gray4"], "grayscale frame 4  (210×160)", **GRAY)
group_box([g3, g4], "②  GRAYSCALE — frames 3 + 4, two separate 2-D images")
arrow(raw_axs[2], g3, label="getScreenGrayscale")
arrow(raw_axs[3], g4)
fig.text(0.17, (raw_axs[0].get_position().y0 + g3.get_position().y1) / 2,
         "frames 1 & 2 are never observed —\nthey exist only inside the emulator",
         fontsize=9.6, color=MUTED, ha="center", va="center", style="italic")

# STAGE 3: max-pool merges the two frames into ONE 2-D image
dif = panel(2, 2, 5, diff, f"|frame3 − frame4|  (largest at step {flicker_k + 1}):\n"
                           "what differs between the two — here the moving sprite", **GRAY)
mp = panel(2, 6, 9, s1["maxpool"], "np.maximum(frame3, frame4)  →  ONE image,\n"
                                   "still 2-D (210×160) — nothing visible is lost", **GRAY)
group_box([dif, mp], "③  MAX-POOL — 2 images → 1 image (pixel-wise maximum)")
arrow(g3, mp, xyA=(0.5, -0.17), xyB=(0.30, 1.06))
arrow(g4, mp, xyA=(0.5, -0.17), xyB=(0.75, 1.06))

# STAGE 4 + 5: resize; TV stamp as a side branch
rz = panel(3, 6, 9, s1["resized"], "84×84 uint8 — what a tv-off agent sees", **GRAY)
group_box([rz], "④  RESIZE → 84×84")
tv = panel(3, 9, 12, stamped, "in TV runs, THIS frame is stacked instead", **GRAY)
group_box([tv], "⑤  NOISY-TV (TV runs only)", dashed=True)
arrow(mp, rz)
fig.add_artist(ConnectionPatch(
    xyA=(1.03, 0.5), coordsA=rz.transAxes, xyB=(-0.03, 0.5), coordsB=tv.transAxes,
    arrowstyle="-|>", mutation_scale=16, lw=1.7, color=EDGE,
    linestyle=(0, (4, 3)), zorder=5))

# STAGE 6: the frame stack — one processed frame per agent step
slot_axs = []
for i in range(4):
    cap = f"slot {i}  =  step {i + 5}" + ("  (oldest)" if i == 0 else "  (newest)" if i == 3 else "")
    slot_axs.append(panel(4, 3 * i, 3 * i + 3, obs_off[i], cap, **GRAY))
group_box(slot_axs, "⑥  FRAME STACK — one frame from each of the LAST 4 AGENT STEPS → (4, 84, 84)\n"
                    "(these four ≠ the 4 emulator frames of ① — they span four separate steps)")
arrow(rz, slot_axs[2], xyA=(0.5, -0.30), xyB=(0.5, 1.10))
_p = rz.get_position()
fig.text(_p.x0 + _p.width / 2 + 0.012, _p.y0 - 0.031,
         "one frame per agent step → deque",
         fontsize=8.6, family="monospace", color=LBL, va="center", ha="left", bbox=dict(facecolor="white", edgecolor="none", pad=1.2), zorder=6)

# STAGE 7: network inputs
txt = fig.add_subplot(gs[5, 0:7])
txt.axis("off")
txt.text(0, 0.97, "⑦  NETWORK INPUTS", fontsize=11.5, fontweight="bold", va="top")
txt.text(0, 0.84, (
    "Policy path:  the whole (4,84,84) stack, ÷255 → NatureCNN → 512 features.\n\n"
    "RND path:  ONLY the newest frame (slot 3), per-pixel whitened —\n"
    "(x − μ)/σ, clip ±5;  μ, σ = running statistics of each pixel over play.\n\n"
    "The RND image is NOT colour — it is one float channel; the colormap\n"
    "encodes the signed value:  red = brighter than this pixel's usual value,\n"
    "blue = darker, white ≈ 0 (pixel matches its average).\n\n"
    "The background vanishes because it never changes (x ≈ μ ⇒ 0): whitening\n"
    "erases static pixels, so RND sees only deviations — sprites, HUD changes.\n"
    "The same mechanism tames any stationary noise to ~unit variance, which\n"
    "is why the TV's stimulus strength is patch AREA, not amplitude."),
    fontsize=9.5, va="top", color="#22262e", linespacing=1.35)
wh = fig.add_subplot(gs[5, 8:11])
im = wh.imshow(whitened, cmap="RdBu_r", vmin=-5, vmax=5, interpolation="nearest")
wh.set_xticks([]); wh.set_yticks([])
for sp in wh.spines.values():
    sp.set_color("#b8bec8")
wh.text(0.5, -0.045, "RND input: whitened newest frame", transform=wh.transAxes,
        ha="center", va="top", fontsize=9.2, color="#22262e")
cax = wh.inset_axes([1.06, 0.02, 0.06, 0.96])
cb = fig.colorbar(im, cax=cax)
cb.ax.tick_params(labelsize=8)
cb.set_label("(x − μ) / σ", fontsize=8.5)
arrow(slot_axs[3], wh, xyA=(0.5, -0.30), xyB=(0.5, 1.06),
      label="newest frame only:\n(x−μ)/σ, clip ±5", dx=-0.155)

fig.suptitle("Montezuma's Revenge — the observation pipeline, stage by stage "
             "(read top → bottom)\nfirst agent step(s), seed 1 · every 84×84 frame "
             "verified byte-identical to the real make_env stack", fontsize=13, y=0.985)
fig.savefig(os.path.join(OUT, "_overview.png"), dpi=150, facecolor="white")
print(f"  wrote {os.path.relpath(os.path.join(OUT, '_overview.png'))}")

# ── compact left-to-right pipeline strip (paper figure) ──────────────────────
print("horizontal pipeline figure")
fig2 = plt.figure(figsize=(16, 3.7))
gs2 = fig2.add_gridspec(1, 6, width_ratios=[0.80, 0.90, 0.80, 1.0, 1.0, 1.24],
                        left=0.012, right=0.988, top=0.90, bottom=0.17, wspace=0.42)
PORTRAIT = 160 / 210   # width/height of a raw 210x160 frame drawn at unit height


h_captions = []


def hpanel(c, caption):
    ax = fig2.add_subplot(gs2[0, c])
    ax.axis("off")
    ax.set_aspect("equal")
    h_captions.append((ax, caption))
    return ax


def place(ax, img, x0, y0, w, h, dashed=False):
    kw = {"cmap": "gray", "vmin": 0, "vmax": 255} if img.ndim == 2 else {}
    ax.imshow(img, extent=(x0, x0 + w, y0, y0 + h), interpolation="nearest", **kw)
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor="#8a93a3",
                           linewidth=0.9, linestyle="--" if dashed else "-"))


axs2 = []
a = hpanel(0, "raw frame\n(210, 160, 3)  RGB")
place(a, s1["raw_rgb"][3], 0, 0, PORTRAIT, 1)
a.set_xlim(-0.01, PORTRAIT + 0.01); a.set_ylim(-0.01, 1.01); axs2.append(a)

a = hpanel(1, "grayscale frames 3 & 4\n2 × (210, 160)")
d = 0.05
place(a, s1["gray3"], d, d, PORTRAIT, 1)          # frame 3 behind
place(a, s1["gray4"], 0, 0, PORTRAIT, 1)          # frame 4 in front
a.set_xlim(-0.01, PORTRAIT + d + 0.01); a.set_ylim(-0.01, 1 + d + 0.01); axs2.append(a)

a = hpanel(2, "max-pooled\n(210, 160)")
place(a, s1["maxpool"], 0, 0, PORTRAIT, 1)
a.set_xlim(-0.01, PORTRAIT + 0.01); a.set_ylim(-0.01, 1.01); axs2.append(a)

a = hpanel(3, "resized\n(84, 84)  uint8")
place(a, s1["resized"], 0, 0, 1, 1)
a.set_xlim(-0.01, 1.01); a.set_ylim(-0.01, 1.01); axs2.append(a)

a = hpanel(4, "noisy-TV stamped\n(84, 84) — TV runs only")
place(a, stamped, 0, 0, 1, 1, dashed=True)
a.set_xlim(-0.01, 1.01); a.set_ylim(-0.01, 1.01); axs2.append(a)

a = hpanel(5, "frame stack, last 4 agent steps\n(4, 84, 84)  uint8")
d = 0.055
for j, sl in enumerate((0, 1, 2, 3)):             # oldest deepest in the deck
    off = d * (3 - j)
    place(a, obs_off[sl], off, off, 1, 1)
a.set_xlim(-0.01, 1 + 3 * d + 0.01); a.set_ylim(-0.01, 1 + 3 * d + 0.01); axs2.append(a)

for _ax, _cap in h_captions:
    _p = _ax.get_position()
    fig2.text((_p.x0 + _p.x1) / 2, 0.115, _cap, ha="center", va="top",
              fontsize=9.3, color="#22262e")

OPS = ["grayscale\n(read frames 3 & 4)", "np.maximum\n(2 → 1)",
       "resize\n(INTER_AREA)", "+ noise patch\n(TV runs only)",
       "push into deque\n(1 frame / step)"]
for i, op in enumerate(OPS):
    pa, pb = axs2[i].get_position(), axs2[i + 1].get_position()
    ym = (pa.y0 + pa.y1) / 2
    fig2.add_artist(FancyArrowPatch(
        (pa.x1 + 0.002, ym), (pb.x0 - 0.002, ym), transform=fig2.transFigure,
        arrowstyle="-|>", mutation_scale=14, lw=1.4, color=EDGE,
        linestyle=(0, (4, 3)) if i == 3 else "-"))
    fig2.text((pa.x1 + pb.x0) / 2, 0.915, op, ha="center", va="bottom",
              fontsize=8.8, color=LBL, family="monospace")

for ext in ("png", "pdf"):
    fig2.savefig(os.path.join(OUT, f"_pipeline_horizontal.{ext}"),
                 dpi=200 if ext == "png" else None, facecolor="white")
print(f"  wrote {os.path.relpath(os.path.join(OUT, '_pipeline_horizontal.png'))} (+ .pdf)")

print("done — all stage images verified and written")
