import cv2
import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers import (
    AtariPreprocessing,
    ClipReward,
    FrameStackObservation,
    RecordEpisodeStatistics,
    RecordVideo,
)

gym.register_envs(ale_py)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NatureCNN(nn.Module):
    """Nature DQN CNN backbone. Input: (N, 4, 84, 84) uint8. Output: (N, 512) float."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x / 255.0)


class RoomTracker(gym.Wrapper):
    """Tracks unique rooms visited in Montezuma's Revenge per episode.

    Room number is read from Atari RAM byte 3 after every step.
    Appends rooms_visited to info when the episode ends.
    """

    _ROOM_RAM_ADDR = 3

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._rooms: set[int] = {self._room()}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._rooms.add(self._room())
        info["rooms_visited"] = len(self._rooms)
        return obs, reward, terminated, truncated, info

    def _room(self) -> int:
        return int(self.unwrapped.ale.getRAM()[self._ROOM_RAM_ADDR])


class NewRoomRecorder(gym.Wrapper):
    """Records a video for each episode that sets a new room high-water mark.

    Buffers rgb_array frames in memory during each episode. On episode end,
    writes an mp4 if rooms_visited exceeds the previous best, otherwise
    discards the buffer. Requires render_mode="rgb_array" and RoomTracker
    in the wrapper stack.
    """

    def __init__(self, env: gym.Env, video_folder: str, fps: int = 30):
        super().__init__(env)
        import pathlib
        self._folder = pathlib.Path(video_folder)
        self._folder.mkdir(parents=True, exist_ok=True)
        self._fps = fps
        self._best_rooms = 0
        self._frames: list = []
        self._ep = 0

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._frames = [self.render()]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._frames.append(self.render())
        if (terminated or truncated) and "rooms_visited" in info:
            if info["rooms_visited"] > self._best_rooms:
                self._best_rooms = info["rooms_visited"]
                self._write_video()
            self._frames = []
            self._ep += 1
        return obs, reward, terminated, truncated, info

    def _write_video(self):
        if not self._frames:
            return
        import imageio
        path = self._folder / f"new_room_ep{self._ep:05d}_r{self._best_rooms:02d}.mp4"
        imageio.mimwrite(str(path), self._frames, fps=self._fps)


class OverlayFrameProbe(gym.Wrapper):
    """Exposes a uniform overlay_render() method across all vector sub-envs.

    Only the active env (idx 0) actually renders; all others return None
    without touching self.render(). This lets envs.call("overlay_render") be
    invoked safely on the whole vector env even though only idx 0 was
    constructed with render_mode="rgb_array" — calling .render() on an env
    built with render_mode=None raises, and AsyncVectorEnv/SyncVectorEnv's
    .call() always dispatches to every sub-env with no per-index targeting.
    """

    def __init__(self, env: gym.Env, active: bool):
        super().__init__(env)
        self._active = active

    def overlay_render(self):
        return self.render() if self._active else None


class NoisyTVWrapper(gym.Wrapper):
    """Injects a synthetic "noisy TV" — a rectangular patch of per-pixel uniform
    noise — into the processed 84x84 observation, to test whether intrinsic-motivation
    agents get captured by agent-irrelevant stochasticity (the noisy-TV thought
    experiment of Burda et al. 2018).

    Must sit between AtariPreprocessing and FrameStackObservation:
    AtariPreprocessing reads the screen directly from the ALE object
    (getScreenGrayscale into internal buffers), so a wrapper below it never
    influences the processed frames, and stamping here gives exactly one noise
    sample per agent step (no frameskip max-pool / resize attenuation) with
    per-step noise history preserved across the frame stack.

    Modes:
      static      — always-on patch, resampled every `refresh_every` agent steps.
                    No behavioral trap (the patch is screen-fixed and unavoidable);
                    tests intrinsic-signal degradation only.
      remote      — action space grows by one: the added action maps to NOOP(0)
                    in the game and resamples the patch ("pressing the TV
                    remote"). Agent-controllable stochasticity — the paper's
                    actual thought experiment, and the mode where capture is
                    observable as behavior.
      sham-remote — the added NOOP-mapped action but NO patch: the
                    action-space-matched control for remote.

    The noise RNG is derived from the per-sub-env reset seed (salted so it is
    decorrelated from the env's own RNG) and persists across autoresets, so
    noise streams are reproducible per seed. A fresh patch is drawn on every
    episode start. render() composites the current patch (nearest-neighbour
    upscaled to raw-frame coordinates) onto a copy of the rendered frame so the
    TV is visible in RecordVideo / NewRoomRecorder / overlay output.
    """

    _RNG_SALT = 0x7F00D

    def __init__(self, env: gym.Env, mode: str, size: tuple[int, int] = (12, 84),
                 position: tuple[int, int] = (0, 0), refresh_every: int = 1):
        super().__init__(env)
        assert mode in ("static", "remote", "sham-remote"), f"unknown tv mode {mode!r}"
        row, col = position
        ph, pw = size
        h, w = env.observation_space.shape  # (84, 84) after AtariPreprocessing
        assert 0 <= row and row + ph <= h and 0 <= col and col + pw <= w, \
            f"TV patch {ph}x{pw} at {position} exceeds the {h}x{w} frame"
        self.mode = mode
        self.size = (ph, pw)
        self.position = (row, col)
        self.refresh_every = refresh_every
        self._rng: np.random.Generator | None = None
        self._patch: np.ndarray | None = None
        self._steps = 0
        if mode in ("remote", "sham-remote"):
            self.tv_action = env.action_space.n
            self.action_space = gym.spaces.Discrete(env.action_space.n + 1)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng([seed, self._RNG_SALT])
        elif self._rng is None:
            self._rng = np.random.default_rng()
        self._steps = 0
        if self.mode != "sham-remote":
            self._patch = self._sample_patch()
            self._stamp(obs)
        return obs, info

    def step(self, action):
        if self.mode in ("remote", "sham-remote"):
            # Resample on the *chosen* action, before ALE's sticky-action lottery:
            # the remote is the one perfectly reliable action in the game.
            pressed = action == self.tv_action
            env_action = 0 if pressed else action  # 0 = NOOP in ALE's action set
        else:
            self._steps += 1
            pressed = self._steps % self.refresh_every == 0
            env_action = action
        obs, reward, terminated, truncated, info = super().step(env_action)
        if self.mode != "sham-remote":
            if pressed:
                self._patch = self._sample_patch()
            self._stamp(obs)
        return obs, reward, terminated, truncated, info

    def _sample_patch(self) -> np.ndarray:
        return self._rng.integers(0, 256, size=self.size, dtype=np.uint8)

    def _stamp(self, obs: np.ndarray) -> None:
        # In-place is safe: AtariPreprocessing returns a freshly allocated
        # cv2.resize output per call, never a shared buffer.
        row, col = self.position
        obs[row : row + self.size[0], col : col + self.size[1]] = self._patch

    def render(self):
        frame = self.env.render()
        if frame is None or self._patch is None:
            return frame
        row, col = self.position
        h, w = self.observation_space.shape
        fh, fw = frame.shape[:2]
        r0, r1 = round(row * fh / h), round((row + self.size[0]) * fh / h)
        c0, c1 = round(col * fw / w), round((col + self.size[1]) * fw / w)
        up = cv2.resize(self._patch, (c1 - c0, r1 - r0), interpolation=cv2.INTER_NEAREST)
        # Copy before compositing: NewRoomRecorder buffers rendered frames for a
        # whole episode, so mutating the returned frame could rewrite history.
        frame = frame.copy()
        frame[r0:r1, c0:c1] = up[..., None]  # gray → broadcast over RGB
        return frame


_TV_ARG_DEFAULTS = {"tv_mode": "off", "tv_size": (12, 84), "tv_position": (0, 0),
                    "tv_refresh_every": 1}


def check_tv_args_match(ckpt_args: dict, args) -> None:
    """Abort a --resume whose --tv-* flags differ from the checkpoint's.

    remote and sham-remote load into identical network shapes (same extended
    action space), so a wrong-mode resume would load cleanly and silently
    change the experiment mid-run. Checkpoints predating the TV feature carry
    no tv_* keys and are treated as tv_mode="off".
    """
    for name, default in _TV_ARG_DEFAULTS.items():
        old, new = ckpt_args.get(name, default), getattr(args, name)
        if isinstance(old, (list, tuple)) or isinstance(new, (list, tuple)):
            old = tuple(old) if isinstance(old, (list, tuple)) else (old,)
            new = tuple(new) if isinstance(new, (list, tuple)) else (new,)
        if old != new:
            raise SystemExit(
                f"--resume checkpoint was trained with --{name.replace('_', '-')}={old} "
                f"but this invocation sets {new}; a mismatched resume silently changes "
                f"the experiment. Pass the checkpoint's own --tv-* flags."
            )


def make_env(
    env_id: str,
    idx: int,
    capture_video: bool,
    run_name: str,
    videos_dir: str = "videos",
    video_episode_interval: int = 1,
    record_room_discovery: bool = False,
    clip_reward: bool = True,
    overlay_video: bool = False,
    tv_mode: str = "off",
    tv_size: tuple[int, int] = (12, 84),
    tv_position: tuple[int, int] = (0, 0),
    tv_refresh_every: int = 1,
):
    """Returns a thunk for gym.vector.AsyncVectorEnv (or SyncVectorEnv).

    Wrapper stack (inner → outer):
      ALE env → RoomTracker → AtariPreprocessing → [NoisyTVWrapper]
        → FrameStackObservation → RecordEpisodeStatistics → [ClipReward]
        → [OverlayFrameProbe for idx 0 when overlay_video]
        → [RecordVideo for idx 0 when capture_video] → [NewRoomRecorder on top when record_room_discovery]

    overlay_video and capture_video are mutually exclusive (enforced by each
    agent's train() at the CLI level, not here) -- overlay_video takes priority
    if both are somehow set.

    NoisyTVWrapper (tv_mode != "off") is added to *every* sub-env, not just
    idx 0: vector envs require homogeneous spaces (remote/sham-remote extend
    the action space) and the TV must exist in every env's observations. With
    tv_mode="off" the wrapper is not constructed at all, keeping the default
    stack byte-identical to the pre-TV code path.

    terminal_on_life_loss=False so the agent experiences full episodes —
    important for exploration research where we want to reward reaching new rooms,
    not just surviving individual lives.

    ClipReward is applied *after* RecordEpisodeStatistics so episodic_return in infos
    still reports true game score; only the reward the agent trains on is clipped to
    [-1, 1], matching the original RND paper's preprocessing (Burda et al., 2018,
    Appendix A.3) and CleanRL's ppo_atari.py / ppo_rnd_envpool.py.
    """

    def thunk():
        render_mode = "rgb_array" if (idx == 0 and (capture_video or overlay_video)) else None
        # frameskip=1 disables ALE's built-in repeat; AtariPreprocessing does the skipping instead
        env = gym.make(env_id, frameskip=1, render_mode=render_mode)
        env = RoomTracker(env)
        env = AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            grayscale_obs=True,
            terminal_on_life_loss=False,
        )
        if tv_mode != "off":
            env = NoisyTVWrapper(env, mode=tv_mode, size=tuple(tv_size),
                                 position=tuple(tv_position), refresh_every=tv_refresh_every)
        env = FrameStackObservation(env, 4)
        env = RecordEpisodeStatistics(env)
        if clip_reward:
            env = ClipReward(env, -1, 1)
        if overlay_video:
            env = OverlayFrameProbe(env, active=(idx == 0))
        elif capture_video and idx == 0:
            # Stacked, not either/or: room-discovery mode adds sparse "new room" videos
            # on top of (not instead of) periodic sampling, so a production run gets both.
            if record_room_discovery:
                env = NewRoomRecorder(env, f"{videos_dir}/{run_name}/room_discovery")
            trigger = lambda ep, n=video_episode_interval: ep % n == 0
            env = RecordVideo(env, f"{videos_dir}/{run_name}", episode_trigger=trigger, disable_logger=True)
        return env

    return thunk
