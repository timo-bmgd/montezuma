"""Passive-rider infrastructure for the dual-signal novelty comparison.

Scientific motivation (thesis): RND (prediction error) and SimHash counting
(visit counts) define novelty differently. Comparing the two *definitions*
requires both signals evaluated on identical observations — two separate runs
cannot provide this, because each agent visits different states. A passive
rider computes the second method's bonus on the exact same observation stream
the driving agent trains on, logs it per step, and NEVER adds it to the
reward.

Both riders isolate their randomness from the driving run (own numpy
Generator; network init under a forked torch CPU RNG), so a run with a rider
enabled is bit-identical to the same run with the rider disabled — verified
by scripts/check_rider_noop.py.

Components:
  RNDRider     — passive RND: own RNDModel, predictor optimizer, and
                 obs/reward running stats, trained online on the driving
                 agent's rollout batches exactly the way rnd.py trains its
                 active RND (same frame convention, same normalisation, same
                 update_proportion masking).
  SimHashRider — passive SimHash visit counting (thin wrapper over
                 count_based.SimHashCounter).
  StepLogger   — per-(step, env) record of both bonus signals plus room id,
                 episode id, extrinsic reward, done flag, and global_step;
                 written as compressed .npz shards for offline correlation.
  save_rnd_artifact / save_simhash_artifact — small-artefact checkpointing:
                 everything needed to re-evaluate each bonus offline, and
                 nothing else (no optimizer states, no policy weights).
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.wrappers.utils import RunningMeanStd


class RNDRider:
    """Passive RND observer: computes (and learns) the RND bonus on the
    driving agent's observation stream without ever touching its reward.

    Mirrors rnd.py's active RND mechanics one-to-one so the logged signal is
    what an RND-driven run *would* compute on these observations:
      - bonus on the latest frame of the 4-stack (next_obs[:, 3:4]),
        normalised by running obs stats, clipped to ±5,
      - predictor trained each iteration on the rollout batch's obs
        (obs_rms updated with the batch first), update_proportion masking,
      - reward normalisation via RewardForwardFilter + RunningMeanStd.

    RNG isolation (what makes the no-op guarantee hold): the RNDModel is
    initialised under a forked torch CPU RNG seeded with the rider's own
    seed, and all stochastic choices during training (minibatch permutation,
    update-proportion mask) come from the rider's private numpy Generator —
    the driving run's global torch/numpy RNG streams are never consumed.
    """

    def __init__(self, device, seed, int_gamma=0.99, lr=1e-4,
                 update_proportion=0.25, update_epochs=4, num_minibatches=4,
                 max_grad_norm=0.5):
        # Lazy import: rnd.py imports this module at top level, so importing
        # agents.rnd here at module scope would be circular.
        from agents.rnd import RNDModel, RewardForwardFilter, _normalize_obs
        self._normalize_obs = _normalize_obs
        self.device = device
        self.seed = seed
        self.update_proportion = update_proportion
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.max_grad_norm = max_grad_norm

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = RNDModel()
        self.model = self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.predictor.parameters(), lr=lr, eps=1e-5)
        self.obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
        self.reward_rms = RunningMeanStd()
        self.reward_filter = RewardForwardFilter(int_gamma)
        self.rng = np.random.default_rng(seed)
        self.last_fwd_loss = float("nan")

    def init_obs_rms(self, envs, num_init_iterations, num_steps, seed):
        """Random rollouts to initialise obs running stats — mirrors rnd.py's
        init phase. Steps the envs; the caller MUST envs.reset(seed=...)
        again afterwards. Because that reset reseeds the env RNG, the
        subsequent training stream is identical to a run without this init
        phase (rnd.py relies on the same property)."""
        envs.reset(seed=seed)
        frames_buf = []
        for _ in range(num_init_iterations * num_steps):
            acs = envs.action_space.sample()
            next_obs_np, _, _, _, _ = envs.step(acs)
            frames_buf.append(next_obs_np[:, 3:4, :, :].astype(np.float32))
            if len(frames_buf) == num_steps:
                self.obs_rms.update(np.concatenate(frames_buf, axis=0))
                frames_buf = []

    @torch.no_grad()
    def compute_bonus(self, next_obs_np) -> np.ndarray:
        """Raw (unnormalised) RND bonus per env for the latest stack frame."""
        last_frames = next_obs_np[:, 3:4, :, :].astype(np.float32)
        rnd_obs = self._normalize_obs(last_frames, self.obs_rms, self.device)
        pred, tgt = self.model(rnd_obs)
        return ((tgt - pred).pow(2).sum(1) / 2).cpu().numpy()

    def end_iteration_norm(self, raw_bonus_tn: np.ndarray) -> float:
        """Update reward-normalisation stats from this iteration's (T, N) raw
        bonuses and return the divisor sqrt(reward_rms.var) — the normalised
        bonus rnd.py would train on is raw / divisor."""
        num_envs = raw_bonus_tn.shape[1]
        discounted_per_env = np.array(
            [self.reward_filter.update(raw_bonus_tn[:, i]) for i in range(num_envs)]
        )
        self.reward_rms.update_from_moments(
            discounted_per_env.mean(), discounted_per_env.std() ** 2, discounted_per_env.size
        )
        return float(np.sqrt(self.reward_rms.var))

    def update(self, b_obs: torch.Tensor) -> float:
        """Train the predictor on the rollout batch (b_obs: (B, 4, 84, 84)).

        Same order of operations as rnd.py: obs_rms is updated with the batch
        first, then the batch is normalised with the updated stats. All
        randomness comes from the rider's own Generator."""
        last_frames = b_obs[:, 3:4, :, :].cpu().numpy().astype(np.float32)
        self.obs_rms.update(last_frames)
        rnd_obs_batch = self._normalize_obs(last_frames, self.obs_rms, self.device)
        batch_size = rnd_obs_batch.shape[0]
        minibatch_size = batch_size // self.num_minibatches
        losses = []
        for _ in range(self.update_epochs):
            mb_inds = self.rng.permutation(batch_size)
            for start in range(0, batch_size, minibatch_size):
                mb = mb_inds[start:start + minibatch_size]
                pred, tgt = self.model(rnd_obs_batch[mb])
                fwd_loss_per = F.mse_loss(pred, tgt.detach(), reduction="none").mean(-1)
                mask = torch.from_numpy(
                    (self.rng.random(len(fwd_loss_per)) < self.update_proportion).astype(np.float32)
                ).to(self.device)
                fwd_loss = (fwd_loss_per * mask).sum() / mask.sum().clamp(min=1)
                self.optimizer.zero_grad()
                fwd_loss.backward()
                nn.utils.clip_grad_norm_(self.model.predictor.parameters(), self.max_grad_norm)
                self.optimizer.step()
                losses.append(fwd_loss.item())
        self.last_fwd_loss = float(np.mean(losses))
        return self.last_fwd_loss

    def checkpoint_state(self) -> dict:
        """Full rider state for embedding in the driving agent's checkpoint,
        so --resume continues the rider instead of resetting it."""
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "obs_rms_mean": self.obs_rms.mean,
            "obs_rms_var": self.obs_rms.var,
            "obs_rms_count": self.obs_rms.count,
            "reward_rms_mean": self.reward_rms.mean,
            "reward_rms_var": self.reward_rms.var,
            "reward_rms_count": self.reward_rms.count,
            "reward_filter_rewems": self.reward_filter.rewems,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_checkpoint_state(self, state: dict):
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.obs_rms.mean = state["obs_rms_mean"]
        self.obs_rms.var = state["obs_rms_var"]
        self.obs_rms.count = state["obs_rms_count"]
        self.reward_rms.mean = state["reward_rms_mean"]
        self.reward_rms.var = state["reward_rms_var"]
        self.reward_rms.count = state["reward_rms_count"]
        self.reward_filter.rewems = state["reward_filter_rewems"]
        self.rng.bit_generator.state = state["rng_state"]


class SimHashRider:
    """Passive SimHash visit counting on the driving agent's observation
    stream. Purely observational: counting mutates only the rider's own
    table, and computing the hash consumes no RNG at all (the projection
    matrix is fixed at construction from the rider's own seed)."""

    def __init__(self, hash_dim, seed, beta, mode="pool", pool_size=16):
        # Lazy import: count_based.py imports this module at top level.
        from agents.count_based import SimHashCounter
        self.counter = SimHashCounter(hash_dim=hash_dim, seed=seed, mode=mode, pool_size=pool_size)
        self.beta = beta
        self.seed = seed

    def compute_bonus(self, next_obs_np) -> np.ndarray:
        """Increment counts for each env's observation, return β/√n bonuses —
        the same increment-then-bonus order count_based.py's active loop uses."""
        bonuses = np.zeros(len(next_obs_np), dtype=np.float32)
        for i in range(len(next_obs_np)):
            self.counter.increment(next_obs_np[i])
            bonuses[i] = self.counter.bonus(next_obs_np[i], self.beta)
        return bonuses


class StepLogger:
    """Per-step dual-bonus log for offline signal correlation.

    One row per (rollout step, env): extrinsic reward (as trained on, i.e.
    post-ClipReward when clipping is enabled), the driving method's bonus,
    the passive rider's bonus (NaN when no rider), current room id (from
    RoomTracker via infos["room"]; -1 if unavailable for a step), a per-env
    episode counter, done flag, and global_step. Bonuses are stored RAW; the
    per-iteration normalisation divisors (`norm_active` / `norm_passive`, the
    sqrt(reward_rms.var) values RND divides by — NaN for SimHash, which has
    no normalisation) are stored once per iteration, so normalised signals
    are exactly recoverable as raw / divisor.

    Episode ids count episodes per env from the start of *this process*; a
    --resume run restarts them at 0 (disambiguate with global_step).

    Written as compressed .npz shards covering `flush_every` iterations each,
    named steps_<first_iter>_<last_iter>.npz. Roughly 10–25 kB/iteration
    uncompressed at num_envs=8–32, i.e. tens of MB over a 10M-step run.
    """

    ARRAY_KEYS = ("reward_ext", "bonus_active", "bonus_passive", "room", "episode_id", "done")

    def __init__(self, out_dir, num_steps, num_envs, flush_every=50):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.flush_every = flush_every
        self._iters: list[dict] = []
        self._episode_ids = np.zeros(num_envs, dtype=np.int64)
        self._cur = None
        self._cur_iteration = None

    def begin_iteration(self, iteration: int):
        T, N = self.num_steps, self.num_envs
        self._cur_iteration = iteration
        self._cur = {
            "global_step":   np.zeros(T, dtype=np.int64),
            "reward_ext":    np.zeros((T, N), dtype=np.float32),
            "bonus_active":  np.full((T, N), np.nan, dtype=np.float32),
            "bonus_passive": np.full((T, N), np.nan, dtype=np.float32),
            "room":          np.full((T, N), -1, dtype=np.int16),
            "episode_id":    np.zeros((T, N), dtype=np.int64),
            "done":          np.zeros((T, N), dtype=bool),
        }

    def log_step(self, step, global_step, reward, done, infos,
                 bonus_active=None, bonus_passive=None):
        c = self._cur
        c["global_step"][step] = global_step
        c["reward_ext"][step] = reward
        if bonus_active is not None:
            c["bonus_active"][step] = bonus_active
        if bonus_passive is not None:
            c["bonus_passive"][step] = bonus_passive
        room = infos.get("room")
        if room is not None:
            room = np.asarray(room, dtype=np.int16)
            mask = infos.get("_room")
            if mask is not None:
                room = np.where(np.asarray(mask), room, np.int16(-1))
            c["room"][step] = room
        c["episode_id"][step] = self._episode_ids
        done = np.asarray(done, dtype=bool)
        c["done"][step] = done
        self._episode_ids[done] += 1

    def end_iteration(self, norm_active=float("nan"), norm_passive=float("nan")):
        self._cur["iteration"] = self._cur_iteration
        self._cur["norm_active"] = norm_active
        self._cur["norm_passive"] = norm_passive
        self._iters.append(self._cur)
        self._cur = None
        if len(self._iters) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self._iters:
            return
        first, last = self._iters[0]["iteration"], self._iters[-1]["iteration"]
        out = {key: np.concatenate([it[key] for it in self._iters], axis=0)
               for key in self.ARRAY_KEYS}
        out["global_step"] = np.concatenate([it["global_step"] for it in self._iters])
        out["iteration"] = np.array([it["iteration"] for it in self._iters], dtype=np.int64)
        out["norm_active"] = np.array([it["norm_active"] for it in self._iters], dtype=np.float64)
        out["norm_passive"] = np.array([it["norm_passive"] for it in self._iters], dtype=np.float64)
        out["num_steps_per_iteration"] = np.int64(self.num_steps)
        np.savez_compressed(self.dir / f"steps_{first:06d}_{last:06d}.npz", **out)
        self._iters = []

    def close(self):
        self.flush()


def save_rnd_artifact(path, iteration, global_step, rnd_model, obs_rms, reward_rms, extra=None):
    """Small RND artefact: predictor + target state_dicts + the running stats
    needed to reproduce the bonus offline. Deliberately NO optimizer state
    and NO policy weights (~16 MB vs ~48 MB for a full rnd.py checkpoint)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "iteration": iteration,
        "global_step": global_step,
        "predictor_state_dict": rnd_model.predictor.state_dict(),
        "target_state_dict": rnd_model.target.state_dict(),
        "obs_rms_mean": obs_rms.mean,
        "obs_rms_var": obs_rms.var,
        "obs_rms_count": obs_rms.count,
        "reward_rms_mean": reward_rms.mean,
        "reward_rms_var": reward_rms.var,
        "reward_rms_count": reward_rms.count,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def save_simhash_artifact(path, iteration, global_step, counter, beta):
    """Small SimHash artefact: bit-packed count table + projection matrix +
    hash config as compressed .npz — everything needed to re-evaluate the
    count bonus offline."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = counter.artifact_arrays()
    np.savez_compressed(path, iteration=np.int64(iteration),
                        global_step=np.int64(global_step),
                        beta=np.float64(beta), **arrays)
