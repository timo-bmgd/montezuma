import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers import (
    AtariPreprocessing,
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
        if terminated or truncated:
            info["rooms_visited"] = len(self._rooms)
        return obs, reward, terminated, truncated, info

    def _room(self) -> int:
        return int(self.unwrapped.ale.getRAM()[self._ROOM_RAM_ADDR])


def make_env(env_id: str, idx: int, capture_video: bool, run_name: str, videos_dir: str = "videos"):
    """Returns a thunk for gym.vector.AsyncVectorEnv (or SyncVectorEnv).

    Wrapper stack (inner → outer):
      ALE env → RoomTracker → AtariPreprocessing → FrameStackObservation
        → RecordEpisodeStatistics [→ RecordVideo for idx 0 when capture_video]

    terminal_on_life_loss=False so the agent experiences full episodes —
    important for exploration research where we want to reward reaching new rooms,
    not just surviving individual lives.
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
        if capture_video and idx == 0:
            env = RecordVideo(env, f"{videos_dir}/{run_name}", disable_logger=True)
        return env

    return thunk
