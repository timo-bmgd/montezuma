"""Count-based exploration (SimHash) + PPO for Montezuma's Revenge.

Two hash pipelines (--hash-mode): the original "index" mode is degenerate on
Montezuma, but not in the direction this docstring used to claim (all-unique
buckets) — measured on a 50k-observation random-policy stream
(scripts/simhash_occupancy_probe.py, 2026-07-14), a single bucket absorbs
~50% of all visits while half the remaining buckets are singletons, so the
bonus is dominated by one mega-bucket. "pool" mode (Tang et al. 2017-style:
last stack frame area-pooled to --hash-pool-size squared) measured healthy
occupancy (top bucket ~3%, mean count ~16 at k=64) and is what production
count runs should use; "index" stays the default only for backward
compatibility. Watch charts/hash_top_bucket_share and
charts/hash_singleton_frac in TensorBoard for either degeneracy.

Supports a passive RND rider (--passive-rnd) for the dual-signal novelty
comparison — see doc/dual-signal-rider.md and agents/riders.py.

Run from project root with the venv active:
    source .venv/bin/activate
    python src/agents/count_based.py
    python src/agents/count_based.py --hash-mode pool --hash-pool-size 16
    python src/agents/count_based.py --exploration-coef 0.1 --hash-dim 128
"""

import os
import sys

import argparse
import random
import time
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.base import NatureCNN, layer_init, make_env
from agents.riders import RNDRider, StepLogger, save_rnd_artifact, save_simhash_artifact


def parse_args():
    p = argparse.ArgumentParser()
    # experiment
    p.add_argument("--exp-name", default=os.path.basename(__file__)[:-3])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--track", action="store_true")
    p.add_argument("--wandb-project", default="montezuma-thesis")
    p.add_argument("--capture-video", action="store_true")
    p.add_argument("--record-room-discovery", action="store_true",
                   help="In addition to periodic recording, also record video whenever "
                        "agent sets a new room high-water mark")
    p.add_argument("--video-episode-interval", type=int, default=100,
                   help="Record a video every N episodes (env 0 only, when --capture-video is set)")
    p.add_argument("--overlay-video", action="store_true",
                   help="Record synced gameplay+dashboard overlay videos (env 0 only)")
    p.add_argument("--overlay-episode-interval", type=int, default=100,
                   help="Record overlay video pair every N episodes when --overlay-video is set")
    p.add_argument("--clip-reward", action=argparse.BooleanOptionalAction, default=True,
                   help="Clip extrinsic reward to [-1, 1] (standard Atari preprocessing)")
    # env
    p.add_argument("--env-id", default="ALE/MontezumaRevenge-v5")
    p.add_argument("--total-timesteps", type=int, default=10_000_000)
    # ppo (same defaults as ppo.py)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--clip-coef", type=float, default=0.1)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    # count-based specific
    p.add_argument("--exploration-coef", type=float, default=0.01,
                   help="Intrinsic reward scale: beta / sqrt(n(s))")
    p.add_argument("--hash-dim", type=int, default=64,
                   help="Number of bits in SimHash code")
    p.add_argument("--hash-mode", choices=["index", "pool"], default="index",
                   help="'index' (default, original behaviour): mean over the 4-stack, 128 "
                        "linspace-sampled pixels. 'pool' (Tang et al. 2017-style): last stack "
                        "frame, area-downsampled to --hash-pool-size squared before projection "
                        "-- coarser, gives meaningful (non-singleton) visit counts")
    p.add_argument("--hash-pool-size", type=int, default=16,
                   help="Side length of the area-pooled input image when --hash-mode pool")
    # passive RND rider (dual-signal novelty comparison)
    p.add_argument("--passive-rnd", action="store_true",
                   help="Run a passive RND observer on the same observation stream: its bonus "
                        "is computed, trained, and logged per step, but NEVER added to the "
                        "reward. Combine with --step-log for the dual-signal analysis")
    p.add_argument("--rider-seed", type=int, default=None,
                   help="Seed for the rider's own RNG streams (default: --seed + 1000)")
    p.add_argument("--rider-lr", type=float, default=1e-4,
                   help="Adam lr for the rider RND predictor (rnd.py's default lr)")
    p.add_argument("--rider-update-proportion", type=float, default=0.25,
                   help="Fraction of minibatch samples used to train the rider predictor")
    p.add_argument("--rider-int-gamma", type=float, default=0.99,
                   help="Discount for the rider's RewardForwardFilter normalisation")
    p.add_argument("--rider-obs-norm-init-steps", type=int, default=50,
                   help="Iterations of random rollouts to initialise the rider's obs running "
                        "stats (mirrors rnd.py's --obs-norm-init-steps; envs are re-reset with "
                        "the run seed afterwards, so the training stream is unaffected)")
    # per-step dual-bonus logging
    p.add_argument("--step-log", action="store_true",
                   help="Write per-(step, env) records of extrinsic reward, active bonus, "
                        "passive bonus, room id, episode id, done to compressed .npz shards "
                        "under {runs-dir}/{run_name}/step_log/")
    p.add_argument("--step-log-flush-iters", type=int, default=50,
                   help="Iterations per step-log shard")
    # selective small-artefact checkpointing
    p.add_argument("--artifact-interval", type=int, default=0,
                   help="Every N iterations (and at the final iteration), save small analysis "
                        "artefacts: SimHash count table + projection (.npz) and, with "
                        "--passive-rnd, the rider's predictor/target + running stats (.pt). "
                        "No optimizer states. 0 (default) disables")
    # infrastructure
    p.add_argument("--sync-envs", action="store_true",
                   help="Use SyncVectorEnv instead of AsyncVectorEnv (easier debugging)")
    p.add_argument("--runs-dir", default="runs", help="Directory for TensorBoard logs")
    p.add_argument("--videos-dir", default="videos", help="Directory for recorded videos")
    p.add_argument("--checkpoint-dir", default="checkpoints", help="Directory to save checkpoints")
    p.add_argument("--checkpoint-interval", type=int, default=100,
                   help="Save a checkpoint every N iterations")
    p.add_argument("--resume", default=None, help="Path to checkpoint .pt file to resume from")
    return p.parse_args()


class SimHashCounter:
    """State visit counter using random projection (SimHash).

    Two hash pipelines, selected by `mode`:
      - "index" (original behaviour): mean over the 4-frame stack, 128
        linspace-sampled pixels, standardised, projected to hash_dim bits.
      - "pool" (Tang et al. 2017-style): the *last* stack frame (matching
        RND's frame convention), area-downsampled to pool_size x pool_size,
        standardised, projected to hash_dim bits. Coarser input -> more hash
        collisions -> meaningful (non-singleton) visit counts.

    Visits are counted per binary hash code.
    Intrinsic reward = exploration_coef / sqrt(n(hash(s)))
    """

    _DOWNSAMPLE = 128  # number of pixels kept after downsampling ("index" mode)

    def __init__(self, hash_dim: int = 64, seed: int = 42,
                 mode: str = "index", pool_size: int = 16):
        if mode not in ("index", "pool"):
            raise ValueError(f"unknown SimHash mode: {mode!r}")
        self.hash_dim = hash_dim
        self.seed = seed
        self.mode = mode
        self.pool_size = pool_size
        input_dim = self._DOWNSAMPLE if mode == "index" else pool_size * pool_size
        rng = np.random.default_rng(seed)
        self._A = rng.standard_normal((hash_dim, input_dim)).astype(np.float32)
        self._counts: dict[bytes, int] = {}

    def _hash(self, obs: np.ndarray) -> bytes:
        # obs: (4, 84, 84) uint8
        if self.mode == "index":
            frame = obs.astype(np.float32).mean(axis=0)      # (84, 84)
            idx = np.linspace(0, frame.size - 1, self._DOWNSAMPLE, dtype=int)
            flat = frame.flatten()[idx]
        else:  # "pool"
            frame = obs[-1].astype(np.float32)               # last stack frame
            flat = cv2.resize(frame, (self.pool_size, self.pool_size),
                              interpolation=cv2.INTER_AREA).flatten()
        flat = (flat - flat.mean()) / (flat.std() + 1e-8)    # normalise
        return (self._A @ flat > 0).tobytes()

    def increment(self, obs: np.ndarray) -> int:
        key = self._hash(obs)
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key]

    def bonus(self, obs: np.ndarray, beta: float) -> float:
        key = self._hash(obs)
        return beta / np.sqrt(max(self._counts.get(key, 1), 1))

    @property
    def num_unique(self) -> int:
        return len(self._counts)

    def occupancy_stats(self) -> dict:
        """Bucket-occupancy summary — the degeneracy check for the hash: a
        singleton_frac near 1.0 means nearly every state got its own bucket
        (bonus ~ flat beta, no signal); a top_bucket_share near 1.0 means the
        hash collapsed everything together."""
        if not self._counts:
            return {"unique": 0, "total": 0, "singleton_frac": 0.0,
                    "max_count": 0, "mean_count": 0.0, "top_bucket_share": 0.0}
        counts = np.fromiter(self._counts.values(), dtype=np.int64)
        total = int(counts.sum())
        return {
            "unique": int(counts.size),
            "total": total,
            "singleton_frac": float((counts == 1).mean()),
            "max_count": int(counts.max()),
            "mean_count": float(counts.mean()),
            "top_bucket_share": float(counts.max() / total),
        }

    def state_arrays(self):
        """Count table as (bit-packed keys (U, hash_dim/8) uint8, counts (U,)
        int64) — for checkpoints and artefacts."""
        n = len(self._counts)
        packed_width = (self.hash_dim + 7) // 8
        if n == 0:
            return (np.zeros((0, packed_width), dtype=np.uint8),
                    np.zeros(0, dtype=np.int64))
        # keys are bool-array .tobytes() -> one 0x00/0x01 byte per bit
        raw = np.frombuffer(b"".join(self._counts.keys()), dtype=np.uint8)
        raw = raw.reshape(n, self.hash_dim)
        counts = np.fromiter(self._counts.values(), dtype=np.int64)
        return np.packbits(raw, axis=1), counts

    def load_state_arrays(self, packed_keys: np.ndarray, counts: np.ndarray):
        """Inverse of state_arrays — restores the count table on --resume."""
        self._counts = {}
        if len(packed_keys) == 0:
            return
        unpacked = np.unpackbits(packed_keys, axis=1)[:, :self.hash_dim].astype(bool)
        for row, c in zip(unpacked, counts):
            self._counts[row.tobytes()] = int(c)

    def artifact_arrays(self) -> dict:
        """Everything needed to re-evaluate the count bonus offline."""
        packed_keys, counts = self.state_arrays()
        return {
            "packed_keys": packed_keys,
            "counts": counts,
            "projection": self._A,
            "hash_dim": np.int64(self.hash_dim),
            "seed": np.int64(self.seed),
            "mode": np.str_(self.mode),
            "pool_size": np.int64(self.pool_size),
        }


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.cnn = NatureCNN()
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.cnn(x))

    def get_action_and_value(self, x, action=None):
        features = self.cnn(x)
        logits = self.actor(features)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(features)


def _save_checkpoint(path, iteration, global_step, agent, optimizer, counter, rider, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    packed_keys, counts = counter.state_arrays()
    payload = {
        "iteration": iteration,
        "global_step": global_step,
        "agent_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "counter_packed_keys": packed_keys,
        "counter_counts": counts,
        "args": vars(args),
    }
    if rider is not None:
        payload["rider_state"] = rider.checkpoint_state()
    torch.save(payload, path)


def _load_checkpoint(path, agent, optimizer, counter, rider):
    """Returns (iteration, global_step, rider_restored)."""
    ckpt = torch.load(path, weights_only=False)
    agent.load_state_dict(ckpt["agent_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "counter_packed_keys" in ckpt:
        counter.load_state_arrays(ckpt["counter_packed_keys"], ckpt["counter_counts"])
    else:
        print("WARNING: checkpoint predates count-table saving -- visit counts "
              "restart from empty, so bonuses will be inflated after resume.")
    rider_restored = False
    if rider is not None:
        if "rider_state" in ckpt:
            rider.load_checkpoint_state(ckpt["rider_state"])
            rider_restored = True
        else:
            print("WARNING: --passive-rnd set but checkpoint has no rider state -- "
                  "the rider starts fresh (obs-norm init will run again).")
    return ckpt["iteration"], ckpt["global_step"], rider_restored


def train():
    args = parse_args()
    if args.overlay_video and args.capture_video:
        raise SystemExit("--overlay-video and --capture-video are mutually exclusive in this first version")

    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_iterations = args.total_timesteps // batch_size
    if args.resume:
        # Recover the original run_name from the checkpoint path (same fix as
        # rnd.py/ppo.py -- see doc/decisions.md 2026-07-14): run_name can
        # contain "/" (env_id), so it must be taken relative to checkpoint_dir.
        run_name = str(Path(args.resume).resolve().parent.relative_to(
            Path(args.checkpoint_dir).resolve()
        ))
    else:
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(project=args.wandb_project, sync_tensorboard=True,
                   config=vars(args), name=run_name, save_code=True)

    writer = SummaryWriter(f"{args.runs_dir}/{run_name}")
    writer.add_text("hyperparameters",
                    "|param|value|\n|-|-|\n" +
                    "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    VecCls = gym.vector.SyncVectorEnv if args.sync_envs else gym.vector.AsyncVectorEnv
    envs = VecCls(
        [make_env(args.env_id, i, args.capture_video, run_name, args.videos_dir, args.video_episode_interval,
                  args.record_room_discovery, clip_reward=args.clip_reward,
                  overlay_video=args.overlay_video)
         for i in range(args.num_envs)]
    )

    overlay_recorder = None
    if args.overlay_video:
        from agents.video_overlay import EpisodeOverlayRecorder
        overlay_recorder = EpisodeOverlayRecorder(
            args.videos_dir, run_name,
            metric_names=["raw_count", "applied_bonus", "unique_states"],
            main_metric="applied_bonus",
            episode_trigger=lambda ep, n=args.overlay_episode_interval: ep % n == 0,
            # bonus = exploration_coef / sqrt(n); n >= 1 always, so exploration_coef
            # at n=1 (brand-new state) is the theoretical ceiling, decaying toward 0
            # as a state gets revisited more.
            main_metric_range=(0.0, args.exploration_coef),
        )

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    counter = SimHashCounter(hash_dim=args.hash_dim, seed=args.seed,
                             mode=args.hash_mode, pool_size=args.hash_pool_size)

    rider = None
    if args.passive_rnd:
        rider_seed = args.rider_seed if args.rider_seed is not None else args.seed + 1000
        rider = RNDRider(device, rider_seed, int_gamma=args.rider_int_gamma,
                         lr=args.rider_lr,
                         update_proportion=args.rider_update_proportion,
                         update_epochs=args.update_epochs,
                         num_minibatches=args.num_minibatches,
                         max_grad_norm=args.max_grad_norm)
        print(f"Passive RND rider enabled (rider_seed={rider_seed})")

    step_logger = None
    if args.step_log:
        step_logger = StepLogger(f"{args.runs_dir}/{run_name}/step_log",
                                 args.num_steps, args.num_envs,
                                 flush_every=args.step_log_flush_iters)

    obs_shape = envs.single_observation_space.shape  # (4, 84, 84)
    obs_buf   = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    act_buf   = torch.zeros((args.num_steps, args.num_envs), device=device)
    logp_buf  = torch.zeros((args.num_steps, args.num_envs), device=device)
    rew_buf   = torch.zeros((args.num_steps, args.num_envs), device=device)
    done_buf  = torch.zeros((args.num_steps, args.num_envs), device=device)
    val_buf   = torch.zeros((args.num_steps, args.num_envs), device=device)

    start_iteration = 1
    global_step = 0
    rider_restored = False

    if args.resume:
        start_iteration, global_step, rider_restored = _load_checkpoint(
            args.resume, agent, optimizer, counter, rider
        )
        start_iteration += 1
        print(f"Resumed from {args.resume} at iteration {start_iteration - 1}, global_step={global_step}")

    if rider is not None and not rider_restored:
        # Random rollouts to initialise the rider's obs running stats, exactly
        # like rnd.py's own init phase. The training envs.reset(seed=...) below
        # reseeds the env RNG, so the training stream is unaffected.
        print(f"Initialising rider obs normalisation ({args.rider_obs_norm_init_steps} iterations)...")
        rider.init_obs_rms(envs, args.rider_obs_norm_init_steps, args.num_steps, seed=args.seed)
        print("Done.")

    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs  = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    passive_buf = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)

    for iteration in range(start_iteration, num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.lr

        # ── rollout collection ──────────────────────────────────────────────
        intrinsic_log = []
        if step_logger is not None:
            step_logger.begin_iteration(iteration)
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step]  = next_obs
            done_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                val_buf[step] = value.flatten()
            act_buf[step]  = action
            logp_buf[step] = logprob

            next_obs_np, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminated, truncated)

            # count-based intrinsic reward: increment counter, then compute bonus
            intrinsic = np.zeros(args.num_envs, dtype=np.float32)
            raw_count0 = None
            for i in range(args.num_envs):
                n = counter.increment(next_obs_np[i])
                intrinsic[i] = counter.bonus(next_obs_np[i], args.exploration_coef)
                if i == 0:
                    raw_count0 = n
            intrinsic_log.append(intrinsic.mean())

            # passive rider: bonus computed and logged on the same observations,
            # but NEVER added to combined_reward below
            passive_bonus = None
            if rider is not None:
                passive_bonus = rider.compute_bonus(next_obs_np)
                passive_buf[step] = passive_bonus

            combined_reward = reward + intrinsic
            rew_buf[step] = torch.tensor(combined_reward, dtype=torch.float32, device=device)
            next_obs  = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done_np, dtype=torch.float32, device=device)

            if step_logger is not None:
                step_logger.log_step(step, global_step, reward, next_done_np, infos,
                                     bonus_active=intrinsic, bonus_passive=passive_bonus)

            if overlay_recorder is not None:
                metrics0 = {
                    "raw_count": float(raw_count0),
                    "applied_bonus": float(intrinsic[0]),
                    "unique_states": float(counter.num_unique),
                }
                overlay_recorder.capture_step(envs, metrics0, bool(terminated[0]), bool(truncated[0]))

            if "_episode" in infos:
                for i, ended in enumerate(infos["_episode"]):
                    if not ended:
                        continue
                    r = float(infos["episode"]["r"][i])
                    l = int(infos["episode"]["l"][i])
                    print(f"  step={global_step}  return={r:.1f}  length={l}")
                    writer.add_scalar("charts/episodic_return", r, global_step)
                    writer.add_scalar("charts/episodic_length", l, global_step)
                    if "rooms_visited" in infos:
                        writer.add_scalar("charts/rooms_visited", int(infos["rooms_visited"][i]), global_step)

        # ── passive rider: reward-normalisation stats + step-log finalise ───
        norm_passive = float("nan")
        if rider is not None:
            norm_passive = rider.end_iteration_norm(passive_buf)
        if step_logger is not None:
            # active count bonus has no normalisation -> norm_active stays NaN
            step_logger.end_iteration(norm_passive=norm_passive)

        # ── GAE ─────────────────────────────────────────────────────────────
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rew_buf, device=device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - done_buf[t + 1]
                    nextvalues = val_buf[t + 1]
                delta = rew_buf[t] + args.gamma * nextvalues * nextnonterminal - val_buf[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + val_buf

        # ── PPO update ──────────────────────────────────────────────────────
        b_obs  = obs_buf.reshape((-1,) + obs_shape)
        b_logp = logp_buf.reshape(-1)
        b_act  = act_buf.reshape(-1)
        b_adv  = advantages.reshape(-1)
        b_ret  = returns.reshape(-1)
        b_val  = val_buf.reshape(-1)

        clipfracs = []
        for _ in range(args.update_epochs):
            mb_inds = np.random.permutation(batch_size)
            for start in range(0, batch_size, minibatch_size):
                mb = mb_inds[start : start + minibatch_size]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb], b_act.long()[mb]
                )
                logratio = newlogprob - b_logp[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv_norm = b_adv[mb]
                mb_adv_norm = (mb_adv_norm - mb_adv_norm.mean()) / (mb_adv_norm.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv_norm * ratio,
                    -mb_adv_norm * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()

                newvalue = newvalue.view(-1)
                v_clipped = b_val[mb] + torch.clamp(newvalue - b_val[mb], -args.clip_coef, args.clip_coef)
                v_loss = 0.5 * torch.max(
                    (newvalue - b_ret[mb]) ** 2,
                    (v_clipped - b_ret[mb]) ** 2,
                ).mean()

                loss = pg_loss - args.ent_coef * entropy.mean() + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # ── passive rider: train the rider predictor on this rollout batch ──
        # (own optimizer + own RNG streams; cannot influence agent/optimizer)
        if rider is not None:
            rider.update(b_obs)

        y_pred, y_true = b_val.cpu().numpy(), b_ret.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        sps = int(global_step / (time.time() - start_time))
        print(f"iteration={iteration}/{num_iterations}  SPS={sps}  unique_states={counter.num_unique}")
        writer.add_scalar("charts/learning_rate",      optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS",                sps,                             global_step)
        writer.add_scalar("charts/unique_states",      counter.num_unique,              global_step)
        writer.add_scalar("charts/mean_intrinsic_rew", np.mean(intrinsic_log),          global_step)
        writer.add_scalar("losses/value_loss",         v_loss.item(),                   global_step)
        writer.add_scalar("losses/policy_loss",        pg_loss.item(),                  global_step)
        writer.add_scalar("losses/entropy",            entropy.mean().item(),           global_step)
        writer.add_scalar("losses/approx_kl",          approx_kl.item(),                global_step)
        writer.add_scalar("losses/clipfrac",           np.mean(clipfracs),              global_step)
        writer.add_scalar("losses/explained_variance", explained_var,                   global_step)

        if iteration % 10 == 0:
            occ = counter.occupancy_stats()
            writer.add_scalar("charts/hash_singleton_frac", occ["singleton_frac"], global_step)
            writer.add_scalar("charts/hash_mean_count",     occ["mean_count"],     global_step)
            writer.add_scalar("charts/hash_max_count",      occ["max_count"],      global_step)
            writer.add_scalar("charts/hash_top_bucket_share", occ["top_bucket_share"], global_step)

        if rider is not None:
            writer.add_scalar("rider/raw_bonus_mean",        float(passive_buf.mean()),  global_step)
            writer.add_scalar("rider/raw_bonus_std",         float(passive_buf.std()),   global_step)
            writer.add_scalar("rider/normalized_bonus_mean", float(passive_buf.mean()) / norm_passive, global_step)
            writer.add_scalar("rider/fwd_loss",              rider.last_fwd_loss,        global_step)
            writer.add_scalar("rider/obs_rms_std",   float(np.sqrt(rider.obs_rms.var.mean())), global_step)
            writer.add_scalar("rider/reward_rms_std", norm_passive,                      global_step)

        if args.artifact_interval and (iteration % args.artifact_interval == 0
                                       or iteration == num_iterations):
            art_dir = os.path.join(args.checkpoint_dir, run_name, "artifacts")
            save_simhash_artifact(os.path.join(art_dir, f"simhash_{iteration:06d}.npz"),
                                  iteration, global_step, counter, beta=args.exploration_coef)
            if rider is not None:
                save_rnd_artifact(os.path.join(art_dir, f"rider_rnd_{iteration:06d}.pt"),
                                  iteration, global_step, rider.model,
                                  rider.obs_rms, rider.reward_rms,
                                  extra={"rider_seed": rider.seed})

        if iteration % args.checkpoint_interval == 0 or iteration == num_iterations:
            ckpt_path = os.path.join(args.checkpoint_dir, run_name, f"ckpt_{iteration:06d}.pt")
            _save_checkpoint(ckpt_path, iteration, global_step, agent, optimizer, counter, rider, args)

    if step_logger is not None:
        step_logger.close()
    envs.close()
    writer.close()


if __name__ == "__main__":
    train()
