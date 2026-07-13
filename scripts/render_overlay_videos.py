"""Offline reconstruction of --overlay-video's gameplay+dashboard videos.

--overlay-video no longer renders/draws/encodes during training (see
agents/video_overlay.py's EpisodeOverlayLogger) -- it only logs a cheap (ALE state
snapshot, action sequence, per-step metrics) triple per recorded episode to
<videos-dir>/<run_name>/overlay_logs/ep*.pkl. This script replays each logged episode
through a fresh env -- restore_state() + step() the logged actions -- to reconstruct
the exact frames (ALE is fully deterministic given its cloned state, including the
RNG that drives repeat_action_probability sticky actions), then draws the bar meter
and dashboard and encodes both videos, exactly as the old inline recorder used to,
but entirely off the training critical path.

Run any time after (or during, for already-flushed episodes) training:
    python scripts/render_overlay_videos.py videos/ALE/MontezumaRevenge-v5__rnd__1__1234
    python scripts/render_overlay_videos.py videos/ALE/MontezumaRevenge-v5__rnd__1__1234 --episode 300
"""
import argparse
import os
import pathlib
import pickle
import sys

import numpy as np
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing
from PIL import Image, ImageDraw, ImageFont

gym.register_envs(ale_py)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _draw_bar_meter(img, value, vmin, vmax, label="r"):
    """Draws a thin vertical fill bar against the right edge of img (mutates and returns it).

    Fill height is (value - vmin) / (vmax - vmin), clamped to [0, 1] so out-of-range
    spikes pin the bar at empty/full rather than distorting the scale. vmin/vmax are
    fixed, metric-specific bounds chosen by the caller (see rnd.py/count_based.py),
    not derived from the data being drawn.
    """
    W, H = img.size
    bar_w = max(4, round(W * 0.05))
    label_h = 9
    pad = 2
    x1, x0 = W - pad, W - pad - bar_w
    y0, y1 = pad, H - pad - label_h

    frac = 0.0 if vmax <= vmin else (value - vmin) / (vmax - vmin)
    frac = min(1.0, max(0.0, frac))
    fill_top = y1 - frac * (y1 - y0)

    draw = ImageDraw.Draw(img)
    draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255), width=1)
    if frac > 0:
        draw.rectangle([x0 + 1, fill_top, x1 - 1, y1 - 1], fill=(255, 200, 0))
    font = ImageFont.load_default(size=label_h)
    draw.text((x0, y1 + 1), label, fill=(255, 255, 255), font=font,
              stroke_width=1, stroke_fill=(0, 0, 0))
    return img


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="<videos-dir>/<run_name> directory (contains overlay_logs/)")
    p.add_argument("--episode", type=int, default=None,
                   help="Render only this episode number (default: all logged episodes)")
    p.add_argument("--out-dir", default=None,
                   help="Output directory for gameplay/dashboard mp4s (default: <run_dir>/overlay)")
    return p.parse_args()


def _build_replay_env(env_id):
    """Same stepping semantics as base.make_env (frame_skip=4 maps one logged action
    to 4 raw ALE frames), minus the training-only wrappers (FrameStackObservation,
    RecordEpisodeStatistics, ClipReward, RoomTracker) which don't affect the action ->
    frame mapping and aren't needed here."""
    env = gym.make(env_id, frameskip=1, render_mode="rgb_array")
    env = AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84,
                              grayscale_obs=True, terminal_on_life_loss=False)
    return env


def _reconstruct_frames(log):
    env = _build_replay_env(log["env_id"])
    try:
        env.reset()
        env.unwrapped.restore_state(log["state"])
        frames = []
        for action in log["actions"]:
            env.step(action)
            frames.append(env.render())
        return frames
    finally:
        env.close()


def _write_gameplay(frames, log, out_dir, imageio):
    vmin, vmax = log["main_metric_range"]
    main_vals = log["metrics"][log["main_metric"]]
    out = [
        np.asarray(_draw_bar_meter(Image.fromarray(frame), val, vmin, vmax))
        for frame, val in zip(frames, main_vals)
    ]
    path = out_dir / f"gameplay_ep{log['episode']:05d}.mp4"
    imageio.mimwrite(str(path), out, fps=log["fps"])


def _write_dashboard(log, out_dir, imageio):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    metric_names = log["metric_names"]
    metrics = log["metrics"]
    n = len(log["actions"])
    fig = Figure(figsize=(4.8, 3.6), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    lines = {name: ax.plot([], [], label=name)[0] for name in metric_names}
    ax.set_xlim(0, max(n - 1, 1))
    all_vals = [v for name in metric_names for v in metrics[name]]
    lo, hi = min(all_vals), max(all_vals)
    pad = (hi - lo) * 0.05 or 1.0
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("step")
    ax.legend(loc="upper right", fontsize=6)
    fig.tight_layout()

    out = []
    xs = list(range(n))
    for t in range(n):
        for name in metric_names:
            lines[name].set_data(xs[: t + 1], metrics[name][: t + 1])
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())[:, :, :3]
        out.append(buf.copy())
    path = out_dir / f"dashboard_ep{log['episode']:05d}.mp4"
    imageio.mimwrite(str(path), out, fps=log["fps"])


def render_one(log_path, out_dir):
    import imageio

    with open(log_path, "rb") as f:
        log = pickle.load(f)
    frames = _reconstruct_frames(log)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_gameplay(frames, log, out_dir, imageio)
    _write_dashboard(log, out_dir, imageio)
    print(f"  episode {log['episode']}: {len(frames)} frames -> {out_dir}")


def main():
    args = parse_args()
    run_dir = pathlib.Path(args.run_dir)
    log_dir = run_dir / "overlay_logs"
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else run_dir / "overlay"

    if args.episode is not None:
        log_paths = [log_dir / f"ep{args.episode:05d}.pkl"]
    else:
        log_paths = sorted(log_dir.glob("ep*.pkl"))

    if not log_paths:
        raise SystemExit(f"No overlay logs found in {log_dir}")

    print(f"Reconstructing {len(log_paths)} episode(s) from {log_dir} -> {out_dir}")
    for log_path in log_paths:
        if not log_path.exists():
            print(f"  {log_path.name}: not found, skipping")
            continue
        render_one(log_path, out_dir)


if __name__ == "__main__":
    main()
