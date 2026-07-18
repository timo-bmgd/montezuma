"""Synced gameplay + metrics-dashboard video pairs for a single training run.

For a triggered episode, produces two video files of identical length/timing:
  - gameplay_ep<N>.mp4  — rendered gameplay with a thin vertical bar meter for
    the "main" metric on the right edge of the frame
  - dashboard_ep<N>.mp4 — a scrolling line plot of several metrics, frame-synced
    to the gameplay video

Env 0 must be constructed via base.make_env(..., overlay_video=True), which
attaches OverlayFrameProbe to every sub-env so envs.call("overlay_render")
is safe to invoke regardless of AsyncVectorEnv vs SyncVectorEnv.

The gameplay bar meter uses a *fixed* min/max per agent (passed in as
main_metric_range), not a per-episode autoscale like the dashboard's y-axis.
This is deliberate: autoscaling would make the bar always look "half full"
regardless of the metric's actual magnitude, hiding exactly the kind of
collapse (e.g. RND's intrinsic reward shrinking 20x over training) that the
bar is meant to make visible at a glance across many recorded episodes.
"""

import functools
import pathlib

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@functools.lru_cache(maxsize=None)
def _label_font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # Pillow < 10.1 has no size parameter; use matplotlib's bundled DejaVu Sans
        # so the label still renders at the requested size on older installs.
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)


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
    font = _label_font(label_h)
    draw.text((x0, y1 + 1), label, fill=(255, 255, 255), font=font,
              stroke_width=1, stroke_fill=(0, 0, 0))
    return img


class EpisodeOverlayRecorder:
    """Buffers env-0 frames and per-step metrics for triggered episodes.

    capture_step must be called once per env.step(), unconditionally, for
    every training step (it internally gates itself via episode_trigger).
    """

    def __init__(self, videos_dir, run_name, metric_names, main_metric, episode_trigger,
                 main_metric_range, fps=30):
        self._folder = pathlib.Path(videos_dir) / run_name / "overlay"
        self._folder.mkdir(parents=True, exist_ok=True)
        self._metric_names = list(metric_names)
        self._main_metric = main_metric
        self._main_metric_range = main_metric_range
        self._episode_trigger = episode_trigger
        self._fps = fps

        self._ep = 0
        self._frames: list = []
        self._metrics = {name: [] for name in self._metric_names}
        self.is_recording = bool(self._episode_trigger(self._ep))

    def capture_step(self, envs, metrics: dict, terminated0: bool, truncated0: bool):
        if self.is_recording:
            frame = envs.call("overlay_render")[0]
            if frame is not None:
                self._frames.append(np.asarray(frame))
                for name in self._metric_names:
                    self._metrics[name].append(float(metrics[name]))

        if terminated0 or truncated0:
            if self.is_recording:
                self._flush()
            self._frames = []
            self._metrics = {name: [] for name in self._metric_names}
            self._ep += 1
            self.is_recording = bool(self._episode_trigger(self._ep))

    def _flush(self):
        if not self._frames:
            return
        import imageio
        self._write_gameplay(imageio)
        self._write_dashboard(imageio)

    def _write_gameplay(self, imageio):
        main_vals = self._metrics[self._main_metric]
        vmin, vmax = self._main_metric_range
        out = [
            np.asarray(_draw_bar_meter(Image.fromarray(frame), val, vmin, vmax))
            for frame, val in zip(self._frames, main_vals)
        ]
        path = self._folder / f"gameplay_ep{self._ep:05d}.mp4"
        imageio.mimwrite(str(path), out, fps=self._fps)

    def _write_dashboard(self, imageio):
        n = len(self._frames)
        fig = Figure(figsize=(4.8, 3.6), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        lines = {name: ax.plot([], [], label=name)[0] for name in self._metric_names}
        ax.set_xlim(0, max(n - 1, 1))
        all_vals = [v for name in self._metric_names for v in self._metrics[name]]
        lo, hi = min(all_vals), max(all_vals)
        pad = (hi - lo) * 0.05 or 1.0
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("step")
        ax.legend(loc="upper right", fontsize=6)
        fig.tight_layout()

        out = []
        xs = list(range(n))
        for t in range(n):
            for name in self._metric_names:
                lines[name].set_data(xs[: t + 1], self._metrics[name][: t + 1])
            canvas.draw()
            buf = np.asarray(canvas.buffer_rgba())[:, :, :3]
            out.append(buf.copy())
        path = self._folder / f"dashboard_ep{self._ep:05d}.mp4"
        imageio.mimwrite(str(path), out, fps=self._fps)
