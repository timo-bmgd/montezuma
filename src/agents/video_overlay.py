"""Lightweight per-episode logging for --overlay-video, decoupled from rendering.

EpisodeOverlayLogger records only what's cheap to capture during training: a single
ALE state snapshot (taken right before the first logged action of a triggered
episode, via OverlayStateProbe.overlay_clone_state()) plus that episode's action
sequence and per-step metric values. It writes one small pickle file per recorded
episode -- no rendering, no image drawing, no video encoding happen during training.

The actual gameplay+dashboard videos are reconstructed afterward, by
scripts/render_overlay_videos.py: restore_state() + replay the logged actions through
a fresh env recovers the exact frames (ALE is fully deterministic given its cloned
state, including the RNG that drives repeat_action_probability sticky actions -- see
OverlayStateProbe's docstring in base.py), then the bar meter / dashboard are drawn
and encoded there, entirely off the training critical path.
"""

import pathlib
import pickle


class EpisodeOverlayLogger:
    """Logs a (state, action-sequence, metrics) triple per triggered episode.

    before_step/after_step must be called once per env.step(), unconditionally, for
    every training step, bracketing the envs.step() call:
      - before_step, right before envs.step(), so the state snapshot reflects env 0
        exactly as of the first logged action (not one step late).
      - after_step, right after envs.step(), with the action that was just applied
        to env 0 and this step's metric values (pass {} when not recording -- callers
        can check .is_recording to skip computing metrics that aren't needed).
    Both internally gate themselves via episode_trigger.
    """

    def __init__(self, videos_dir, run_name, env_id, metric_names, main_metric,
                 episode_trigger, main_metric_range, fps=30):
        self._folder = pathlib.Path(videos_dir) / run_name / "overlay_logs"
        self._folder.mkdir(parents=True, exist_ok=True)
        self._env_id = env_id
        self._metric_names = list(metric_names)
        self._main_metric = main_metric
        self._main_metric_range = main_metric_range
        self._episode_trigger = episode_trigger
        self._fps = fps

        self._ep = 0
        self._state = None
        self._actions: list = []
        self._metrics = {name: [] for name in self._metric_names}
        self.is_recording = bool(self._episode_trigger(self._ep))

    def before_step(self, envs):
        if self.is_recording and self._state is None:
            self._state = envs.call("overlay_clone_state")[0]

    def after_step(self, action0: int, metrics: dict, terminated0: bool, truncated0: bool):
        if self.is_recording:
            self._actions.append(int(action0))
            for name in self._metric_names:
                self._metrics[name].append(float(metrics[name]))

        if terminated0 or truncated0:
            if self.is_recording:
                self._flush()
            self._state = None
            self._actions = []
            self._metrics = {name: [] for name in self._metric_names}
            self._ep += 1
            self.is_recording = bool(self._episode_trigger(self._ep))

    def _flush(self):
        if not self._actions:
            return
        path = self._folder / f"ep{self._ep:05d}.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "env_id": self._env_id,
                "episode": self._ep,
                "state": self._state,
                "actions": self._actions,
                "metrics": self._metrics,
                "metric_names": self._metric_names,
                "main_metric": self._main_metric,
                "main_metric_range": self._main_metric_range,
                "fps": self._fps,
            }, f)
