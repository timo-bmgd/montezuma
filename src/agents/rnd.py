"""PPO + RND (Random Network Distillation) for Montezuma's Revenge.

Intrinsic reward = prediction error of a trainable network trying to match a
fixed random target network. Novel states have high error → high intrinsic reward.
Separate value heads for extrinsic (episodic, gamma=0.999) and intrinsic
(non-episodic, int_gamma=0.99) streams.

Reference: Burda et al., 2018 — https://arxiv.org/abs/1810.12894

Run from project root with the venv active:
    source .venv/bin/activate
    python src/agents/rnd.py
    python src/agents/rnd.py --total-timesteps 2000000000 --num-envs 128
"""

import os
import sys

import argparse
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.wrappers.utils import RunningMeanStd
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.base import (NoisyTVWrapper, check_tv_args_match, check_tv_geometry,
                         compute_gae, layer_init, make_env, masked_mean,
                         tv_region_slices)


def parse_args():
    p = argparse.ArgumentParser()
    # experiment
    p.add_argument("--exp-name", default=os.path.basename(__file__)[:-3])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--track", action="store_true", help="log to Weights & Biases")
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
    # noisy TV (see doc/noisy-tv-experiment.md)
    p.add_argument("--tv-mode", choices=["off", "static", "remote", "sham-remote"], default="off",
                   help="Inject a synthetic noise patch into observations. static = always-on, "
                        "resampled every --tv-refresh-every steps; remote = extra NOOP-mapped "
                        "action that resamples the patch (agent-controllable stochasticity, the "
                        "paper's noisy-TV thought experiment); sham-remote = the extra action but "
                        "no patch (action-space-matched control for remote)")
    p.add_argument("--tv-size", type=int, nargs=2, default=[12, 84], metavar=("H", "W"),
                   help="TV patch height x width in 84x84-frame pixels. Default spans the full "
                        "HUD band: per-pixel obs normalization tames any stationary noise to "
                        "~unit variance, so patch AREA is the only stimulus-strength lever -- a "
                        "12x12 patch measurably contributes ~1%% of prediction error (probed), "
                        "too weak to ever dominate the intrinsic signal")
    p.add_argument("--tv-position", type=int, nargs=2, default=[0, 0], metavar=("ROW", "COL"),
                   help="TV patch top-left corner in the 84x84 frame. The default 12x84 strip at "
                        "(0,0) covers exactly the HUD band (score/lives display -- HUD, not "
                        "gameplay; rows >=12 are playfield in every room and must stay clear)")
    p.add_argument("--tv-refresh-every", type=int, default=1,
                   help="static mode only: resample the patch every N agent steps")
    p.add_argument("--tv-diag-interval", type=int, default=10,
                   help="Run the TV occlusion diagnostic every N iterations (0 disables): a "
                        "second no-grad RND pass with the patch region mean-imputed, logging "
                        "charts/intrinsic_rew_full_diag, charts/intrinsic_rew_occluded and "
                        "charts/tv_intrinsic_share. Runs in every mode -- off/sham-remote give "
                        "the ~0 calibration floor for that screen region")
    # env
    p.add_argument("--env-id", default="ALE/MontezumaRevenge-v5")
    p.add_argument("--total-timesteps", type=int, default=10_000_000)
    # ppo
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gamma", type=float, default=0.999,
                   help="Discount factor for extrinsic rewards")
    p.add_argument("--int-gamma", type=float, default=0.99,
                   help="Discount factor for intrinsic rewards (non-episodic)")
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--clip-coef", type=float, default=0.1)
    p.add_argument("--ent-coef", type=float, default=0.001,
                   help="Matches the RND paper's published Atari value (Burda et al. 2018); "
                        "see doc/decisions.md -- a 2026-07-13 10x bump was discredited "
                        "same-day and reverted 2026-07-14")
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    # rnd
    p.add_argument("--int-coef", type=float, default=1.0,
                   help="Weight of intrinsic advantages in combined advantage")
    p.add_argument("--ext-coef", type=float, default=2.0,
                   help="Weight of extrinsic advantages in combined advantage")
    p.add_argument("--update-proportion", type=float, default=0.25,
                   help="Fraction of minibatch samples used to train the RND predictor")
    p.add_argument("--obs-norm-init-steps", type=int, default=50,
                   help="Iterations of random rollouts to initialize obs running stats")
    # infrastructure
    p.add_argument("--sync-envs", action="store_true",
                   help="Use SyncVectorEnv instead of AsyncVectorEnv (easier debugging)")
    p.add_argument("--runs-dir", default="runs", help="Directory for TensorBoard logs")
    p.add_argument("--videos-dir", default="videos", help="Directory for recorded videos")
    p.add_argument("--checkpoint-dir", default="checkpoints", help="Directory to save checkpoints")
    p.add_argument("--checkpoint-interval", type=int, default=100,
                   help="Save a checkpoint every N iterations")
    p.add_argument("--resume", default=None, help="Path to checkpoint .pt file to resume from")
    # collapse auto-stop
    p.add_argument("--auto-stop", action=argparse.BooleanOptionalAction, default=True,
                   help="Stop training early if a collapse signature (entropy collapse + "
                        "intrinsic-reward collapse + frozen PPO updates, sustained together) is "
                        "detected; see doc/10M-RND-run-failure-documentation.md for the failure "
                        "mode this guards against. Writes a checkpoint and exits with code 42 "
                        "on trigger, to avoid burning the rest of the walltime budget on a run "
                        "that has already stopped learning.")
    p.add_argument("--auto-stop-patience", type=int, default=100,
                   help="Consecutive iterations the collapse signature must hold before stopping")
    p.add_argument("--auto-stop-entropy-frac", type=float, default=0.15,
                   help="Trigger threshold: entropy / ln(action_space_n) below this value")
    p.add_argument("--auto-stop-intrinsic-drop", type=float, default=10.0,
                   help="Trigger threshold: raw_intrinsic_rew_mean down by at least this factor "
                        "from its running peak so far this run")
    p.add_argument("--auto-stop-kl-eps", type=float, default=1e-3,
                   help="Trigger threshold: approx_kl below this value (near-zero PPO updates)")
    p.add_argument("--auto-stop-clipfrac-eps", type=float, default=0.01,
                   help="Trigger threshold: clipfrac below this value (near-zero PPO updates)")
    return p.parse_args()


class Agent(nn.Module):
    """PPO agent with dual critic heads for extrinsic and intrinsic value streams."""

    def __init__(self, envs):
        super().__init__()
        # CNN backbone: (N, 4, 84, 84) → (N, 448)
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 448)),
            nn.ReLU(),
        )
        self.extra_layer = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.1), nn.ReLU()
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(448, envs.single_action_space.n), std=0.01),
        )
        self.critic_ext = layer_init(nn.Linear(448, 1), std=0.01)
        self.critic_int = layer_init(nn.Linear(448, 1), std=0.01)

    def get_value(self, x):
        hidden = self.network(x / 255.0)
        features = self.extra_layer(hidden)
        return self.critic_ext(features + hidden), self.critic_int(features + hidden)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        features = self.extra_layer(hidden)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic_ext(features + hidden),
            self.critic_int(features + hidden),
        )


class RNDModel(nn.Module):
    """Random Network Distillation — fixed target + trainable predictor.

    Both networks take a single normalized grayscale frame (N, 1, 84, 84).
    Intrinsic reward = squared L2 distance between predictor and target outputs.
    """

    def __init__(self):
        super().__init__()
        feature_dim = 64 * 7 * 7

        self.target = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_dim, 512)),
        )
        for p in self.target.parameters():
            p.requires_grad = False

        self.predictor = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
        )

    def forward(self, x):
        return self.predictor(x), self.target(x)


class RewardForwardFilter:
    """Tracks a discounted running estimate of intrinsic rewards for normalisation."""

    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


def _save_checkpoint(path, iteration, global_step, agent, rnd_model, optimizer,
                     obs_rms, reward_rms, reward_filter, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "iteration": iteration,
        "global_step": global_step,
        "agent_state_dict": agent.state_dict(),
        "rnd_model_state_dict": rnd_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "obs_rms_mean": obs_rms.mean,
        "obs_rms_var": obs_rms.var,
        "obs_rms_count": obs_rms.count,
        "reward_rms_mean": reward_rms.mean,
        "reward_rms_var": reward_rms.var,
        "reward_rms_count": reward_rms.count,
        "reward_filter_rewems": reward_filter.rewems,
        "args": vars(args),
    }, path)


def _load_checkpoint(path, agent, rnd_model, optimizer, obs_rms, reward_rms, reward_filter, args):
    ckpt = torch.load(path, weights_only=False)
    # Guard BEFORE restoring any state: an action-space-changing tv mismatch
    # would otherwise die in load_state_dict with a raw shape error instead of
    # the guard's actionable message.
    check_tv_args_match(ckpt["args"], args)
    agent.load_state_dict(ckpt["agent_state_dict"])
    rnd_model.load_state_dict(ckpt["rnd_model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    obs_rms.mean = ckpt["obs_rms_mean"]
    obs_rms.var = ckpt["obs_rms_var"]
    obs_rms.count = ckpt["obs_rms_count"]
    reward_rms.mean = ckpt["reward_rms_mean"]
    reward_rms.var = ckpt["reward_rms_var"]
    reward_rms.count = ckpt["reward_rms_count"]
    reward_filter.rewems = ckpt["reward_filter_rewems"]
    return ckpt["iteration"], ckpt["global_step"]


def _normalize_obs(obs_np, obs_rms, device):
    """Normalise a (N, 1, 84, 84) float32 array with running stats, clip to ±5."""
    mean = torch.from_numpy(obs_rms.mean).to(device)
    std = torch.sqrt(torch.from_numpy(obs_rms.var).to(device))
    return ((torch.from_numpy(obs_np).to(device) - mean) / std).clip(-5, 5).float()


def _rnd_error(rnd_model, x):
    """Per-sample RND prediction error — the intrinsic reward before normalisation."""
    pred, tgt = rnd_model(x)
    return (tgt - pred).pow(2).sum(1) / 2


def train():
    args = parse_args()
    if args.overlay_video and args.capture_video:
        raise SystemExit("--overlay-video and --capture-video are mutually exclusive in this first version")
    # Fail fast on invalid TV geometry in every mode — with --tv-mode off the
    # wrapper never runs its own check, but the occlusion diagnostic still
    # slices with these values (numpy would silently clamp an oversized region).
    check_tv_geometry(tuple(args.tv_size), tuple(args.tv_position), args.tv_refresh_every)

    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_iterations = args.total_timesteps // batch_size
    if args.resume:
        # Recover the original run_name from the checkpoint path
        # ({checkpoint_dir}/{run_name}/ckpt_XXXXXX.pt) so a resumed run continues
        # writing to the same TensorBoard/W&B run instead of fragmenting into a
        # fresh timestamped directory every time a walltime-limited job requeues.
        # run_name itself can contain "/" (env_id is e.g. "ALE/MontezumaRevenge-v5"),
        # so this must be relative to checkpoint_dir, not just the immediate parent.
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
                  overlay_video=args.overlay_video, tv_mode=args.tv_mode,
                  tv_size=tuple(args.tv_size), tv_position=tuple(args.tv_position),
                  tv_refresh_every=args.tv_refresh_every) for i in range(args.num_envs)]
    )

    overlay_recorder = None
    if args.overlay_video:
        from agents.video_overlay import EpisodeOverlayRecorder
        overlay_recorder = EpisodeOverlayRecorder(
            args.videos_dir, run_name,
            metric_names=["raw_intrinsic", "normalized_intrinsic", "obs_rms_std", "reward_rms_std"],
            main_metric="normalized_intrinsic",
            episode_trigger=lambda ep, n=args.overlay_episode_interval: ep % n == 0,
            # normalized_intrinsic can't go below 0 (it's a normalized squared
            # prediction error). Upper bound of 0.3 gives headroom above the highest
            # healthy value observed in the 10M-step production runs (~0.217 early in
            # training, collapsing to ~0.009 as the RND predictor saturated). A bar
            # that consistently sits near-empty across a run's recorded episodes is
            # the same collapse signature, made visible at a glance instead of
            # requiring a TensorBoard pull.
            main_metric_range=(0.0, 0.3),
        )

    agent = Agent(envs).to(device)
    rnd_model = RNDModel().to(device)
    combined_params = list(agent.parameters()) + list(rnd_model.predictor.parameters())
    optimizer = optim.Adam(combined_params, lr=args.lr, eps=1e-5)

    obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
    reward_rms = RunningMeanStd()
    reward_filter = RewardForwardFilter(args.int_gamma)

    obs_shape = envs.single_observation_space.shape  # (4, 84, 84)
    obs_buf       = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    act_buf       = torch.zeros((args.num_steps, args.num_envs), device=device)
    logp_buf      = torch.zeros((args.num_steps, args.num_envs), device=device)
    rew_buf       = torch.zeros((args.num_steps, args.num_envs), device=device)
    intr_buf      = torch.zeros((args.num_steps, args.num_envs), device=device)
    done_buf      = torch.zeros((args.num_steps, args.num_envs), device=device)
    ext_val_buf   = torch.zeros((args.num_steps, args.num_envs), device=device)
    int_val_buf   = torch.zeros((args.num_steps, args.num_envs), device=device)

    start_iteration = 1
    global_step = 0

    if args.resume:
        start_iteration, global_step = _load_checkpoint(
            args.resume, agent, rnd_model, optimizer, obs_rms, reward_rms, reward_filter, args
        )
        start_iteration += 1
        print(f"Resumed from {args.resume} at iteration {start_iteration - 1}, global_step={global_step}")

    if not args.resume:
        # ── obs normalisation init: random rollouts ─────────────────────────
        print(f"Initialising obs normalisation ({args.obs_norm_init_steps} iterations)...")
        next_obs_np, _ = envs.reset(seed=args.seed)
        frames_buf = []
        for _ in range(args.obs_norm_init_steps * args.num_steps):
            acs = envs.action_space.sample()
            next_obs_np, _, _, _, _ = envs.step(acs)
            frames_buf.append(next_obs_np[:, 3:4, :, :].astype(np.float32))  # (N, 1, 84, 84)
            if len(frames_buf) == args.num_steps:
                obs_rms.update(np.concatenate(frames_buf, axis=0))
                frames_buf = []
        print("Done.")

    # full reset for training
    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs  = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)
    start_time = time.time()

    collapse_streak = 0
    raw_intrinsic_peak = 0.0
    action_space_n = envs.single_action_space.n

    for iteration in range(start_iteration, num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.lr

        # ── rollout collection ──────────────────────────────────────────────
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step]  = next_obs
            done_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, ext_val, int_val = agent.get_action_and_value(next_obs)
                ext_val_buf[step] = ext_val.flatten()
                int_val_buf[step] = int_val.flatten()
            act_buf[step]  = action
            logp_buf[step] = logprob

            next_obs_np, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminated, truncated)
            rew_buf[step] = torch.tensor(reward, dtype=torch.float32, device=device)
            next_obs  = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done_np, dtype=torch.float32, device=device)

            # intrinsic reward from RND prediction error on the latest frame
            last_frames = next_obs_np[:, 3:4, :, :].astype(np.float32)  # (N, 1, 84, 84)
            rnd_obs = _normalize_obs(last_frames, obs_rms, device)
            with torch.no_grad():
                intr_buf[step] = _rnd_error(rnd_model, rnd_obs).flatten()

            if overlay_recorder is not None:
                raw0 = float(intr_buf[step, 0].item())
                # live snapshot of reward_rms.var (≤num_steps stale) — the true
                # normalized value isn't known until this iteration's post-hoc
                # normalization below runs, so this is an approximation for video only
                normalized0 = raw0 / float(np.sqrt(reward_rms.var))
                metrics0 = {
                    "raw_intrinsic": raw0,
                    "normalized_intrinsic": normalized0,
                    "obs_rms_std": float(np.sqrt(obs_rms.var.mean())),
                    "reward_rms_std": float(np.sqrt(reward_rms.var)),
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

        # ── normalise intrinsic rewards ─────────────────────────────────────
        curiosity_np = intr_buf.cpu().numpy()  # (T, N)
        # update discounted running estimate per env, then normalise with std
        discounted_per_env = np.array(
            [reward_filter.update(curiosity_np[:, i]) for i in range(args.num_envs)]
        )
        mean = discounted_per_env.mean()
        std = discounted_per_env.std()
        count = discounted_per_env.size
        reward_rms.update_from_moments(mean, std**2, count)
        intr_buf /= np.sqrt(reward_rms.var)

        # ── dual GAE ────────────────────────────────────────────────────────
        # NEXT_STEP-autoreset-correct GAE (see agents.base.compute_gae): the
        # discarded-action reset step (done_buf[t]==1) is walled off in both streams.
        # Extrinsic is episodic; intrinsic is non-episodic (RND) so it never cuts the
        # bootstrap at episode boundaries -- but the wall still stops the fake terminal
        # frame from bridging two episodes' intrinsic returns.
        with torch.no_grad():
            next_ext_val, next_int_val = agent.get_value(next_obs)
            next_ext_val = next_ext_val.reshape(1, -1)
            next_int_val = next_int_val.reshape(1, -1)

            ext_adv = compute_gae(rew_buf, ext_val_buf, done_buf, next_ext_val, next_done,
                                  args.gamma, args.gae_lambda, episodic=True)
            int_adv = compute_gae(intr_buf, int_val_buf, done_buf, next_int_val, next_done,
                                  args.int_gamma, args.gae_lambda, episodic=False)
            ext_returns = ext_adv + ext_val_buf
            int_returns = int_adv + int_val_buf

        # ── PPO + RND update ────────────────────────────────────────────────
        b_obs      = obs_buf.reshape((-1,) + obs_shape)
        b_logp     = logp_buf.reshape(-1)
        b_act      = act_buf.reshape(-1)
        b_ext_adv  = ext_adv.reshape(-1)
        b_int_adv  = int_adv.reshape(-1)
        b_ext_ret  = ext_returns.reshape(-1)
        b_int_ret  = int_returns.reshape(-1)
        b_ext_val  = ext_val_buf.reshape(-1)
        b_adv      = b_int_adv * args.int_coef + b_ext_adv * args.ext_coef
        # 1 for real steps, 0 for NEXT_STEP fake (discarded-action) steps; excludes the
        # discarded terminal-frame action and its cross-episode value targets from the
        # policy/value update (the RND predictor still trains on these real frames).
        b_keep     = (1.0 - done_buf).reshape(-1)

        # update obs_rms with this iteration's latest frames for next training step
        newest_np = b_obs[:, 3:4, :, :].cpu().numpy().astype(np.float32)
        obs_rms.update(newest_np)
        rnd_obs_batch = _normalize_obs(newest_np, obs_rms, device)

        # ── TV occlusion diagnostic ─────────────────────────────────────────
        # Second no-grad pass with the patch region occluded, against a "full"
        # pass from the same batch, predictor state and post-update obs_rms —
        # the step-wise raw_intrinsic_rew_mean uses the *previous* iteration's
        # obs_rms, so it can't serve as the full-input reference here. Zeroing
        # the normalized region IS mean-imputation ((mean − mean)/std == 0)
        # without copying and re-normalizing the batch on the CPU. Occlusion is
        # off-distribution for the predictor, so read tv_intrinsic_share as a
        # trend, not an exact decomposition.
        if args.tv_diag_interval > 0 and iteration % args.tv_diag_interval == 0:
            rs, cs = tv_region_slices(tuple(args.tv_position), tuple(args.tv_size))
            rnd_obs_occl = rnd_obs_batch.clone()
            rnd_obs_occl[:, :, rs, cs] = 0.0
            with torch.no_grad():
                full_diag = _rnd_error(rnd_model, rnd_obs_batch).mean().item()
                occluded = _rnd_error(rnd_model, rnd_obs_occl).mean().item()
            writer.add_scalar("charts/intrinsic_rew_full_diag", full_diag, global_step)
            writer.add_scalar("charts/intrinsic_rew_occluded", occluded, global_step)
            writer.add_scalar("charts/tv_intrinsic_share",
                              (full_diag - occluded) / max(full_diag, 1e-8), global_step)

        clipfracs = []
        for _ in range(args.update_epochs):
            mb_inds = np.random.permutation(batch_size)
            for start in range(0, batch_size, minibatch_size):
                mb = mb_inds[start : start + minibatch_size]

                mb_keep = b_keep[mb]  # exclude NEXT_STEP fake steps from policy/value losses

                pred_feat, tgt_feat = rnd_model(rnd_obs_batch[mb])
                fwd_loss_per = F.mse_loss(pred_feat, tgt_feat.detach(), reduction="none").mean(-1)
                # train predictor on a random subset to avoid overfitting (fake-step frames
                # are real observations, so they are NOT excluded from the predictor loss)
                mask = (torch.rand(len(fwd_loss_per), device=device) < args.update_proportion).float()
                fwd_loss = (fwd_loss_per * mask).sum() / mask.sum().clamp(min=1)

                _, newlogprob, entropy, new_ext_val, new_int_val = agent.get_action_and_value(
                    b_obs[mb], b_act.long()[mb]
                )
                logratio = newlogprob - b_logp[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = masked_mean((ratio - 1) - logratio, mb_keep)
                    clipfracs.append(masked_mean(((ratio - 1.0).abs() > args.clip_coef).float(), mb_keep).item())

                # normalise combined advantages over real (kept) samples only
                mb_adv = b_adv[mb]
                adv_mean = masked_mean(mb_adv, mb_keep)
                adv_std = torch.sqrt(masked_mean((mb_adv - adv_mean) ** 2, mb_keep) + 1e-8)
                mb_adv = (mb_adv - adv_mean) / (adv_std + 1e-8)

                pg_loss = masked_mean(torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                ), mb_keep)

                new_ext_val = new_ext_val.view(-1)
                new_int_val = new_int_val.view(-1)
                ext_v_clip = b_ext_val[mb] + torch.clamp(new_ext_val - b_ext_val[mb],
                                                          -args.clip_coef, args.clip_coef)
                ext_v_loss = 0.5 * masked_mean(torch.max(
                    (new_ext_val - b_ext_ret[mb]) ** 2,
                    (ext_v_clip - b_ext_ret[mb]) ** 2,
                ), mb_keep)
                int_v_loss = 0.5 * masked_mean((new_int_val - b_int_ret[mb]) ** 2, mb_keep)
                v_loss = ext_v_loss + int_v_loss

                loss = pg_loss - args.ent_coef * masked_mean(entropy, mb_keep) + v_loss * args.vf_coef + fwd_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(combined_params, args.max_grad_norm)
                optimizer.step()

        y_pred = b_ext_val.cpu().numpy()
        y_true = b_ext_ret.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        sps = int(global_step / (time.time() - start_time))
        print(f"iteration={iteration}/{num_iterations}  SPS={sps}")
        writer.add_scalar("charts/learning_rate",           optimizer.param_groups[0]["lr"],         global_step)
        writer.add_scalar("charts/SPS",                     sps,                                     global_step)
        writer.add_scalar("charts/mean_intrinsic_rew",      intr_buf.mean().item(),                  global_step)
        writer.add_scalar("charts/raw_intrinsic_rew_mean",  float(curiosity_np.mean()),              global_step)
        writer.add_scalar("charts/raw_intrinsic_rew_std",   float(curiosity_np.std()),               global_step)
        writer.add_scalar("charts/obs_rms_std",             float(np.sqrt(obs_rms.var.mean())),      global_step)
        writer.add_scalar("charts/reward_rms_std",          float(np.sqrt(reward_rms.var)),          global_step)
        writer.add_scalar("charts/ext_value_mean",          ext_val_buf.mean().item(),               global_step)
        writer.add_scalar("charts/int_value_mean",          int_val_buf.mean().item(),               global_step)
        if args.tv_mode in ("remote", "sham-remote"):
            # Behavioral-capture metric: fraction of chosen actions that press
            # the TV remote. Uniform-policy chance = 1/n.
            tv_action = NoisyTVWrapper.remote_action_index(action_space_n)
            tv_frac = (act_buf == float(tv_action)).float().mean().item()
            writer.add_scalar("charts/tv_action_frac",      tv_frac,                                 global_step)
        writer.add_scalar("losses/value_loss",              v_loss.item(),                           global_step)
        writer.add_scalar("losses/ext_v_loss",              ext_v_loss.item(),                       global_step)
        writer.add_scalar("losses/int_v_loss",              int_v_loss.item(),                       global_step)
        writer.add_scalar("losses/policy_loss",             pg_loss.item(),                          global_step)
        writer.add_scalar("losses/entropy",                 entropy.mean().item(),                   global_step)
        writer.add_scalar("losses/approx_kl",               approx_kl.item(),                        global_step)
        writer.add_scalar("losses/clipfrac",                np.mean(clipfracs),                      global_step)
        writer.add_scalar("losses/fwd_loss",                fwd_loss.item(),                         global_step)
        writer.add_scalar("losses/explained_variance",      explained_var,                           global_step)

        if args.auto_stop:
            raw_intrinsic_mean = float(curiosity_np.mean())
            raw_intrinsic_peak = max(raw_intrinsic_peak, raw_intrinsic_mean)
            entropy_frac = entropy.mean().item() / np.log(action_space_n)
            intrinsic_drop = raw_intrinsic_peak / max(raw_intrinsic_mean, 1e-8)
            collapsed_now = (
                entropy_frac < args.auto_stop_entropy_frac
                and intrinsic_drop >= args.auto_stop_intrinsic_drop
                and approx_kl.item() < args.auto_stop_kl_eps
                and np.mean(clipfracs) < args.auto_stop_clipfrac_eps
            )
            collapse_streak = collapse_streak + 1 if collapsed_now else 0
            writer.add_scalar("charts/collapse_streak", collapse_streak, global_step)

            if collapse_streak >= args.auto_stop_patience:
                print(f"AUTO-STOP: collapse signature sustained for {args.auto_stop_patience} "
                      f"iterations (entropy_frac={entropy_frac:.3f}, "
                      f"intrinsic_drop={intrinsic_drop:.1f}x, approx_kl={approx_kl.item():.5f}, "
                      f"clipfrac={np.mean(clipfracs):.4f}). Saving checkpoint and exiting.")
                writer.add_scalar("charts/auto_stop_triggered", 1, global_step)
                ckpt_path = os.path.join(args.checkpoint_dir, run_name,
                                         f"ckpt_{iteration:06d}_autostop.pt")
                _save_checkpoint(ckpt_path, iteration, global_step,
                                 agent, rnd_model, optimizer,
                                 obs_rms, reward_rms, reward_filter, args)
                envs.close()
                writer.close()
                sys.exit(42)

        if iteration % args.checkpoint_interval == 0 or iteration == num_iterations:
            ckpt_path = os.path.join(args.checkpoint_dir, run_name, f"ckpt_{iteration:06d}.pt")
            _save_checkpoint(ckpt_path, iteration, global_step,
                             agent, rnd_model, optimizer,
                             obs_rms, reward_rms, reward_filter, args)

    envs.close()
    writer.close()


if __name__ == "__main__":
    train()
