"""Lightweight per-episode logging for --overlay-video, decoupled from rendering.

EpisodeOverlayLogger records only what's cheap to capture during training: a single
ALE state snapshot (taken right before the first logged action of a triggered
episode, via OverlayStateProbe.overlay_clone_state() -- deferred one step when
needed to land after gymnasium's vector-env autoreset actually happens, see the
class docstring below) plus that episode's action sequence and per-step metric
values. It writes one small pickle file per recorded episode -- no rendering, no
image drawing, no video encoding happen during training.

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
      - before_step(envs), right before envs.step().
      - after_step(envs, action0, metrics, terminated0, truncated0), right after
        envs.step(), with the action that was PASSED for env 0 this step and this
        step's metric values (pass {} when not recording -- callers can check
        .is_recording to skip computing metrics that aren't needed). after_step also
        takes envs, since it may need to clone state itself (see below).
    Both internally gate themselves via episode_trigger.

    Autoreset-aware: gymnasium's default vector-env autoreset mode (AutoresetMode.
    NEXT_STEP, used by both SyncVectorEnv and AsyncVectorEnv here) does NOT reset a
    just-terminated sub-env immediately -- it defers the reset to the *next*
    envs.step() call, which also silently discards whatever action was passed for
    that env that step. So the state snapshot for a newly-triggered episode can't be
    taken on the step right after the previous episode ends (env 0 hasn't actually
    been reset yet at that point -- it's still sitting at the old episode's terminal
    state) -- it has to be deferred to the following step, after that reset has
    actually happened inside envs.step(), and that step's discarded phantom action
    must not be logged as part of the new episode's action sequence. _awaiting_reset
    tracks this one-step lag. The very first episode is unaffected: the caller does a
    real, synchronous envs.reset() right before the main training loop starts, so env
    0 is already genuinely fresh by the first before_step call.
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
        self._awaiting_reset = False

    def before_step(self, envs):
        if self.is_recording and self._state is None and not self._awaiting_reset:
            self._state = envs.call("overlay_clone_state")[0]

    def after_step(self, envs, action0: int, metrics: dict, terminated0: bool, truncated0: bool):
        if self._awaiting_reset:
            # This step's envs.step() call was the deferred autoreset for env 0:
            # action0 was silently discarded by the vector env and never applied to
            # any real state, so it must not be logged. The real reset just
            # happened, so this is the first point where the state is actually fresh.
            self._awaiting_reset = False
            if self.is_recording and self._state is None:
                self._state = envs.call("overlay_clone_state")[0]
        elif self.is_recording:
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
            self._awaiting_reset = True

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
