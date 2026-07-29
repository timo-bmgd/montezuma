"""Draw the observation/reward preprocessing pipeline diagram for the thesis."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else "doc/figures/preprocessing-pipeline"

FILL = "#f2f4f7"
FILL_TV = "#fdf3e3"
FILL_NET = "#e8eef8"
EDGE = "#3a3f4a"
ARROW = "#3a3f4a"
SHAPE_C = "#8a4a10"
TITLE_C = "#16181d"
TEXT_C = "#2a2e36"
MUTED = "#555b66"

fig, ax = plt.subplots(figsize=(8.6, 12.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 172)
ax.axis("off")

CX, W = 34, 62          # main flow column
LINE = 3.1              # line spacing inside boxes
GAP = 7.0               # vertical gap between boxes (arrow + label)


def boxheight(n_lines, line=LINE):
    return 8.8 + line * (n_lines - 1)


def draw_box(top_y, title, lines, cx=CX, w=W, fill=FILL, dashed=False,
             title_fs=10.5, fs=8.6, line=LINE):
    h = boxheight(len(lines), line)
    bb = FancyBboxPatch((cx - w / 2, top_y - h), w, h,
                        boxstyle="round,pad=0.6,rounding_size=1.4",
                        linewidth=1.1, edgecolor=EDGE, facecolor=fill,
                        linestyle="--" if dashed else "-")
    ax.add_patch(bb)
    ty = top_y - 2.6
    ax.text(cx, ty, title, ha="center", va="center", fontsize=title_fs,
            fontweight="bold", color=TITLE_C)
    for i, ln in enumerate(lines):
        ax.text(cx, ty - 3.4 - i * line, ln, ha="center", va="center",
                fontsize=fs, color=TEXT_C)
    return top_y - h  # bottom y


def draw_arrow(y1, y2, label=None, x=CX):
    ax.add_patch(FancyArrowPatch((x, y1 - 0.6), (x, y2 + 0.6),
                                 arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.3, color=ARROW))
    if label:
        ax.text(x + 2.5, (y1 + y2) / 2, label, ha="left", va="center",
                fontsize=8.3, family="monospace", color=SHAPE_C)


ax.text(50, 169.0, "Observation & reward pipeline", ha="center", fontsize=14.5,
        fontweight="bold", color=TITLE_C)
ax.text(50, 165.6, "make_env(), src/agents/base.py  —  one agent step, RND-paper conditions",
        ha="center", fontsize=9.5, color=MUTED)

y = 162.0
stages = [
    dict(title="ALE emulator  (ale_py AtariEnv)", lines=[
        "Atari 2600 Montezuma's Revenge, 60 Hz",
        "action space: minimal set = Discrete(18)",
        "sticky actions p = 0.25  (replace no-op starts)",
        "frameskip=1  ·  episode cap 18,000 frames (5 min)",
    ], label="(210,160,3) uint8  raw screen"),
    dict(title="RoomTracker", lines=[
        "reads RAM byte 3 → rooms-visited metric",
        "observation passes through unchanged",
    ], label=None),
    dict(title="AtariPreprocessing  (gymnasium)", lines=[
        "repeat action ×4 raw frames,  rewards summed",
        "grayscale last 2 frames (210×160, off the ALE object)",
        "pixel-wise max of the 2 frames  (sprite-flicker removal)",
        "cv2 area-resize → 84×84,  uint8",
    ], label="(84,84) uint8  one frame / step"),
    dict(title="NoisyTVWrapper  —  TV runs only (--tv-mode ≠ off)", lines=[
        "stamps 12×84 uniform-noise patch (HUD band, rows 0–11)",
        "remote / sham-remote: +1 NOOP-mapped action → Discrete(19)",
    ], label="(84,84) uint8  (patch stamped)", dashed=True, fill=FILL_TV, title_fs=9.8),
    dict(title="FrameStackObservation(4)", lines=[
        "rolling deque of the last 4 processed frames",
        "slot 0 = oldest  …  slot 3 = newest",
    ], label="(4,84,84) uint8"),
    dict(title="RecordEpisodeStatistics  →  ClipReward", lines=[
        "episodic return logged first  =  true game score",
        "then training reward clipped to [−1, 1]",
    ], label=None),
]

bottom = None
for st in stages:
    if bottom is not None:
        draw_arrow(bottom, y, prev_label)
    bottom = draw_box(y, st["title"], st["lines"],
                      dashed=st.get("dashed", False),
                      fill=st.get("fill", FILL),
                      title_fs=st.get("title_fs", 10.5))
    prev_label = st["label"]
    y = bottom - GAP

# fork from the last box to the two network consumers
LX, RX, BW = 25.5, 74.5, 45
fork_y = bottom - 3.6
ax.plot([CX, CX], [bottom - 0.6, fork_y], color=ARROW, lw=1.3)
ax.plot([LX, RX], [fork_y, fork_y], color=ARROW, lw=1.3)

net_top = fork_y - 9.0
for x_ in (LX, RX):
    ax.add_patch(FancyArrowPatch((x_, fork_y), (x_, net_top + 0.6),
                                 arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.3, color=ARROW))
mid_y = (fork_y + net_top) / 2
ax.text(LX + 2.0, mid_y, "(4,84,84) / 255.0", ha="left", va="center",
        fontsize=8.1, family="monospace", color=SHAPE_C)
ax.text(RX + 2.0, mid_y, "(1,84,84), (x−μ)/σ, clip ±5", ha="left", va="center",
        fontsize=8.1, family="monospace", color=SHAPE_C)

NLINE = 2.9
nb = draw_box(net_top, "Policy & value nets (NatureCNN)", [
    "conv 8×8 s4 → (32,20,20)",
    "conv 4×4 s2 → (64,9,9)",
    "conv 3×3 s1 → (64,7,7)",
    "flatten 3136 → FC → 512 features",
    "→ π(a|s),  V_ext,  V_int",
], cx=LX, w=BW, fill=FILL_NET, title_fs=9.6, fs=8.1, line=NLINE)
draw_box(net_top, "RND target + predictor", [
    "input: newest frame only, whitened (obs_rms)",
    "frozen random target  → 512-dim",
    "trained predictor  → 512-dim",
    "‖target − predictor‖² / 2",
    "=  intrinsic reward (novelty bonus)",
], cx=RX, w=BW, fill=FILL_NET, title_fs=9.6, fs=8.1, line=NLINE)

fy = nb - 5.0
for i, ln in enumerate([
    "Conditions match Burda et al. 2018 (RND): sticky actions instead of random no-op starts,",
    "episode cap 4,500 agent steps (18,000 raw frames), full-episode training (no life-loss termination),",
    "reward clipped after true-score logging. RND networks see only the newest, whitened frame.",
]):
    ax.text(50, fy - i * 2.8, ln, ha="center", fontsize=8.3, color=MUTED)

fig.tight_layout(pad=0.4)
for ext in ("png", "svg", "pdf"):
    fig.savefig(f"{OUT}.{ext}", dpi=170 if ext == "png" else None,
                facecolor="white", bbox_inches="tight")
print("written:", OUT, "+ .png/.svg/.pdf")
