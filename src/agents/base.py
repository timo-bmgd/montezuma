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


def make_env(
    env_id: str,
    idx: int,
    capture_video: bool,
    run_name: str,
    videos_dir: str = "videos",
    video_episode_interval: int = 1,
    record_room_discovery: bool = False,
    clip_reward: bool = True,
):
    """Returns a thunk for gym.vector.AsyncVectorEnv (or SyncVectorEnv).

    Wrapper stack (inner → outer):
      ALE env → RoomTracker → AtariPreprocessing → FrameStackObservation
        → RecordEpisodeStatistics → [ClipReward] [→ RecordVideo for idx 0 when capture_video]

    terminal_on_life_loss=False so the agent experiences full episodes —
    important for exploration research where we want to reward reaching new rooms,
    not just surviving individual lives.

    ClipReward is applied *after* RecordEpisodeStatistics so episodic_return in infos
    still reports true game score; only the reward the agent trains on is clipped to
    [-1, 1], matching the original RND paper's preprocessing (Burda et al., 2018,
    Appendix A.3) and CleanRL's ppo_atari.py / ppo_rnd_envpool.py.
    """

    def thunk():
        render_mode = "rgb_array" if (capture_video and idx == 0) else None
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
        env = FrameStackObservation(env, 4)
        env = RecordEpisodeStatistics(env)
        if clip_reward:
            env = ClipReward(env, -1, 1)
        if capture_video and idx == 0:
            if record_room_discovery:
                env = NewRoomRecorder(env, f"{videos_dir}/{run_name}/room_discovery")
            else:
                trigger = lambda ep, n=video_episode_interval: ep % n == 0
                env = RecordVideo(env, f"{videos_dir}/{run_name}", episode_trigger=trigger, disable_logger=True)
        return env

    return thunk
