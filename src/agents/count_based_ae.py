"""AE-SimHash count-based exploration + PPO for Montezuma's Revenge.

Sibling to count_based.py's fixed-random-projection SimHash, which is documented there
to fail on Montezuma's Revenge: a random Gaussian projection of raw downsampled pixels
gives nearly every state its own hash bucket (n≈1 everywhere), so the count-based bonus
degenerates to a flat constant with no exploration signal.

This variant replaces the fixed random projection with a trained autoencoder (Tang et al.,
2017 — "#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning",
https://arxiv.org/abs/1703.01310, the AE-SimHash / "SmartHash" variant). The encoder's
sigmoid bottleneck, thresholded at 0.5, is the hash code; a saturation loss plus noise
injected before decoding pushes the code toward confident, perturbation-robust bits, so
perceptually similar frames are more likely to collide into the same bucket than under
raw-pixel hashing.

The autoencoder is trained online in the same PPO minibatch loop, on the same on-policy
rollout batch — no replay buffer, no obs/reward running-normalization (the RND obs-norm
init buffering bug is exactly the class of complexity this variant avoids). Only the
"train an auxiliary network inside the PPO loop with one combined optimizer" mechanic is
borrowed from rnd.py; the dual-value-head architecture and reward/obs normalization are not.

`--ae-sat-coef` and `--ae-noise-std` are reasonable starting points, not values transcribed
from the paper's per-game grid search — treat them as tunable.

Smoke-test finding, first version (84x84 reconstruction target, 20k steps, 4 envs, default
hyperparameters): the encoder collapsed to a near-constant code within 2-4 iterations
(charts/ae_code_bit_variance -> 0, unique_states plateaus immediately), and this persisted
even with --ae-sat-coef 0 --ae-noise-std 0, ruling those two knobs out as the cause. Root
cause hypothesis: plain per-pixel MSE reconstruction is dominated by the large static
background, giving the optimizer no gradient incentive to keep any input-dependent
information in the 64-bit bottleneck. This reproduced the same "no exploration signal"
failure as the raw-pixel baseline, via a different mechanism (bit-collapse instead of
hash-explosion).

Follow-up, second version (42x42 downsampled reconstruction target, same smoke test): the
collapse persists, at essentially the same magnitude and timing (unique_states plateaus at
98 vs. 110 before, bit variance -> 0 by iteration 3-4 either way). This rules out "sprite is
too small a fraction of pixels" as the mechanism -- the pixel-count compression ratio into
the 64-bit bottleneck dropped ~4x (7056 -> 1764 input pixels) with no effect on collapse
speed or severity. Refined root cause: neither the reconstruction MSE nor the saturation
loss penalizes producing the *same* code for *different* inputs -- a large, constant,
input-independent set of logits satisfies the saturation loss trivially, and the decoder can
still keep lowering reconstruction loss over time by learning one memorized "average frame"
from that fixed code (consistent with ae_recon_loss improving smoothly throughout while
bit_variance stays at exactly 0.0). Needs an explicit incentive for the code to vary across
the batch (e.g. a batch-level code-diversity/variance penalty, or a loss that weights
frame-to-frame differences more heavily than static background) -- not yet implemented.

Run from project root with the venv active:
    source .venv/bin/activate
    python src/agents/count_based_ae.py
    python src/agents/count_based_ae.py --exploration-coef 0.1 --hash-dim 128
"""

import os
import sys

import argparse
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.base import NatureCNN, layer_init, make_env


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
                   help="Record video only when agent sets a new room high-water mark")
    p.add_argument("--video-episode-interval", type=int, default=100,
                   help="Record a video every N episodes (env 0 only, when --capture-video is set)")
    p.add_argument("--clip-reward", action=argparse.BooleanOptionalAction, default=True,
                   help="Clip extrinsic reward to [-1, 1] (standard Atari preprocessing)")
    # env
    p.add_argument("--env-id", default="ALE/MontezumaRevenge-v5")
    p.add_argument("--total-timesteps", type=int, default=10_000_000)
    # ppo (same defaults as count_based.py)
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
    # count-based / AE-SimHash specific
    p.add_argument("--exploration-coef", type=float, default=0.01,
                   help="Intrinsic reward scale: beta / sqrt(n(hash(s)))")
    p.add_argument("--hash-dim", type=int, default=64,
                   help="Number of bits in the AE-learned SimHash code")
    p.add_argument("--ae-recon-coef", type=float, default=1.0,
                   help="Weight on the AE reconstruction (MSE) loss")
    p.add_argument("--ae-sat-coef", type=float, default=0.1,
                   help="Weight on the saturation loss pushing latent bits toward 0/1 "
                        "(tunable -- not a value transcribed from the paper, start here and tune)")
    p.add_argument("--ae-noise-std", type=float, default=0.3,
                   help="Std of Gaussian noise added to pre-sigmoid latent logits during "
                        "training (tunable -- not a value transcribed from the paper)")
    p.add_argument("--image-log-interval", type=int, default=100,
                   help="Log original-vs-reconstruction image pairs to TensorBoard every N iterations")
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


class AEHashModel(nn.Module):
    """Autoencoder producing a saturating, thresholdable binary code for SimHash counting
    (Tang et al. 2017, AE-SimHash / "SmartHash").

    Encoder: same conv-trunk shape as NatureCNN/RNDModel (32-64-64 channels), operating on
    a SINGLE normalised grayscale frame (N, 1, 84, 84) in [0, 1] -- the last frame of the
    4-stack, matching RND's convention (rnd.py uses next_obs_np[:, 3:4, :, :]).

    Bottleneck: Linear(3136, hash_dim) -> [+Gaussian noise, train-time only] -> sigmoid.
    The noise forces the saturation loss to push bits toward confident 0/1 values that
    survive the perturbation, which is what makes nearby/perceptually-similar states
    collide onto the same hash code -- the mechanism that fixes raw-pixel SimHash's failure.

    Decoder mirrors the encoder exactly via ConvTranspose2d with matching kernel/stride
    (output_padding=1 on the middle layer compensates for the floor-rounding the forward
    conv introduces going 20->9): 7x7 -> 9x9 -> 20x20 -> 42x42 (encoder: 42x42 -> 20x20 ->
    9x9 -> 7x7 reversed).

    Follow-up to the initial raw-84x84 version: that version's encoder collapsed to a
    near-constant code within a few iterations (bit variance -> 0) regardless of the
    saturation-loss/noise settings (confirmed via ablation with both disabled). This
    version downsamples the frame to 42x42 (exactly half of 84x84, via area interpolation)
    before hashing/reconstructing, matching Tang et al.'s original preprocessing more
    closely than reconstructing the full 84x84 frame. The exact resize resolution/method
    used in the paper's appendix isn't confidently known here -- 42x42/area-interpolation
    is a reasonable, clearly-labelled choice, not a transcribed paper value. Whether this
    resolves the collapse (as opposed to just reproducing it at a smaller resolution, since
    isotropic downsampling alone does not change the sprite's *fraction* of total pixels)
    is an open empirical question -- re-run the smoke test in the module docstring's
    verification section after any change here.
    """

    _DOWNSAMPLE_SIZE = 42  # exactly half of 84x84 -- a clean integer factor, not a paper-verified value

    def __init__(self, hash_dim: int = 64, noise_std: float = 0.3):
        super().__init__()
        self.hash_dim = hash_dim
        self.noise_std = noise_std

        self.encoder_conv = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.encoder_fc = layer_init(nn.Linear(64 * 7 * 7, hash_dim))

        self.decoder_fc = layer_init(nn.Linear(hash_dim, 64 * 7 * 7))
        self.decoder_conv = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            layer_init(nn.ConvTranspose2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=1)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(32, 1, 4, stride=2)),
            nn.Sigmoid(),
        )

    def downsample(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 1, 84, 84) float in [0, 1] -> (N, 1, 42, 42) via area interpolation."""
        return F.interpolate(x, size=(self._DOWNSAMPLE_SIZE, self._DOWNSAMPLE_SIZE), mode="area")

    def _encode_downsampled(self, x_ds: torch.Tensor, add_noise: bool):
        feat = self.encoder_conv(x_ds)
        logits = self.encoder_fc(feat)
        if add_noise and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        b = torch.sigmoid(logits)
        return logits, b

    def encode(self, x: torch.Tensor, add_noise: bool):
        """x: (N, 1, 84, 84) float in [0, 1] (raw, not yet downsampled). Returns (logits, b),
        both (N, hash_dim)."""
        return self._encode_downsampled(self.downsample(x), add_noise)

    def decode(self, b: torch.Tensor) -> torch.Tensor:
        """Returns (N, 1, 42, 42) -- the downsampled resolution, not the raw 84x84 frame."""
        return self.decoder_conv(self.decoder_fc(b))

    def forward(self, x: torch.Tensor):
        """Training forward pass: downsample -> noisy code -> reconstruction.
        Returns (recon, b_noisy, target), where target is the downsampled input the
        reconstruction should be compared against (not the raw 84x84 frame)."""
        x_ds = self.downsample(x)
        _, b = self._encode_downsampled(x_ds, add_noise=True)
        recon = self.decode(b)
        return recon, b, x_ds

    @torch.no_grad()
    def binary_code(self, x: torch.Tensor) -> torch.Tensor:
        """Clean (noise-free) binary hash code for counting: (N, hash_dim), values in {0., 1.}."""
        _, b = self.encode(x, add_noise=False)
        return (b > 0.5).float()


class AEHashCounter:
    """State visit counter keyed on a learned binary code (AE-SimHash).

    Reuses the same dict[bytes, int] counting + beta/sqrt(n) bonus formula as
    SimHashCounter in count_based.py -- only the hash function differs (a trained
    encoder here, vs. a fixed random projection there); this class does not compute
    the hash itself, the caller supplies an already-encoded binary code.

    The hash is non-stationary (encoder weights change every training iteration), so the
    same physical state can map to a different code -- and thus a different counter -- over
    the course of training. This is the same one-iteration staleness RND already accepts
    (its intrinsic reward is computed with predictor/target weights from the end of the
    previous iteration's update); it's an accepted tradeoff here too, not a bug to fix.
    """

    def __init__(self):
        self._counts: dict[bytes, int] = {}

    @staticmethod
    def _key(code: np.ndarray) -> bytes:
        return np.packbits(code.astype(np.uint8)).tobytes()

    def increment(self, code: np.ndarray) -> int:
        key = self._key(code)
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key]

    def bonus(self, code: np.ndarray, beta: float) -> float:
        key = self._key(code)
        return beta / np.sqrt(max(self._counts.get(key, 1), 1))

    @property
    def num_unique(self) -> int:
        return len(self._counts)


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


def _save_checkpoint(path, iteration, global_step, agent, ae_model, optimizer, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "iteration": iteration,
        "global_step": global_step,
        "agent_state_dict": agent.state_dict(),
        "ae_model_state_dict": ae_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }, path)


def _load_checkpoint(path, agent, ae_model, optimizer):
    ckpt = torch.load(path, weights_only=False)
    agent.load_state_dict(ckpt["agent_state_dict"])
    ae_model.load_state_dict(ckpt["ae_model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["iteration"], ckpt["global_step"]


def train():
    args = parse_args()

    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_iterations = args.total_timesteps // batch_size
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
                  args.record_room_discovery, clip_reward=args.clip_reward) for i in range(args.num_envs)]
    )

    agent = Agent(envs).to(device)
    ae_model = AEHashModel(hash_dim=args.hash_dim, noise_std=args.ae_noise_std).to(device)
    combined_params = list(agent.parameters()) + list(ae_model.parameters())
    optimizer = optim.Adam(combined_params, lr=args.lr, eps=1e-5)
    counter = AEHashCounter()

    obs_shape = envs.single_observation_space.shape  # (4, 84, 84)
    obs_buf   = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    act_buf   = torch.zeros((args.num_steps, args.num_envs), device=device)
    logp_buf  = torch.zeros((args.num_steps, args.num_envs), device=device)
    rew_buf   = torch.zeros((args.num_steps, args.num_envs), device=device)
    done_buf  = torch.zeros((args.num_steps, args.num_envs), device=device)
    val_buf   = torch.zeros((args.num_steps, args.num_envs), device=device)

    start_iteration = 1
    global_step = 0

    if args.resume:
        start_iteration, global_step = _load_checkpoint(args.resume, agent, ae_model, optimizer)
        start_iteration += 1
        print(f"Resumed from {args.resume} at iteration {start_iteration - 1}, global_step={global_step}")

    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs  = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    for iteration in range(start_iteration, num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.lr

        # ── rollout collection ──────────────────────────────────────────────
        intrinsic_log = []
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

            # AE-SimHash intrinsic reward: encode last frame (clean, no_grad) -> binary code
            # -> visit count -> beta/sqrt(n) bonus. Encoder weights here are from the end of
            # the *previous* iteration's update (see AEHashCounter docstring).
            last_frame_np = next_obs_np[:, 3:4, :, :].astype(np.float32) / 255.0  # (N, 1, 84, 84)
            codes = ae_model.binary_code(torch.from_numpy(last_frame_np).to(device))
            codes_np = codes.cpu().numpy()

            intrinsic = np.zeros(args.num_envs, dtype=np.float32)
            for i in range(args.num_envs):
                counter.increment(codes_np[i])
                intrinsic[i] = counter.bonus(codes_np[i], args.exploration_coef)
            intrinsic_log.append(intrinsic.mean())

            combined_reward = reward + intrinsic
            rew_buf[step] = torch.tensor(combined_reward, dtype=torch.float32, device=device)
            next_obs  = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done_np, dtype=torch.float32, device=device)

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

        # ── PPO + AE update ─────────────────────────────────────────────────
        b_obs  = obs_buf.reshape((-1,) + obs_shape)
        b_logp = logp_buf.reshape(-1)
        b_act  = act_buf.reshape(-1)
        b_adv  = advantages.reshape(-1)
        b_ret  = returns.reshape(-1)
        b_val  = val_buf.reshape(-1)

        ae_input_batch = b_obs[:, 3:4, :, :] / 255.0  # (batch_size, 1, 84, 84)

        # health-check metric for AE code collapse, on the clean (no-noise) code
        with torch.no_grad():
            _, b_clean_full = ae_model.encode(ae_input_batch, add_noise=False)
            bit_variance = b_clean_full.var(dim=0).mean().item()

        if iteration % args.image_log_interval == 0:
            with torch.no_grad():
                sample = ae_input_batch[:8]
                target_ds = ae_model.downsample(sample)
                _, b_clean_sample = ae_model.encode(sample, add_noise=False)
                recon_clean = ae_model.decode(b_clean_sample)
            # log the downsampled target, not the raw 84x84 frame -- that's what the
            # network actually sees and is being trained to reconstruct
            writer.add_images("debug/ae_original", target_ds, global_step, dataformats="NCHW")
            writer.add_images("debug/ae_reconstruction", recon_clean, global_step, dataformats="NCHW")

        clipfracs = []
        for _ in range(args.update_epochs):
            mb_inds = np.random.permutation(batch_size)
            for start in range(0, batch_size, minibatch_size):
                mb = mb_inds[start : start + minibatch_size]

                recon, b_noisy, target = ae_model(ae_input_batch[mb])
                ae_recon_loss = F.mse_loss(recon, target)
                ae_sat_loss = torch.minimum((1 - b_noisy) ** 2, b_noisy ** 2).sum(dim=1).mean()
                ae_loss = args.ae_recon_coef * ae_recon_loss + args.ae_sat_coef * ae_sat_loss

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

                loss = pg_loss - args.ent_coef * entropy.mean() + v_loss * args.vf_coef + ae_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(combined_params, args.max_grad_norm)
                optimizer.step()

        y_pred, y_true = b_val.cpu().numpy(), b_ret.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        sps = int(global_step / (time.time() - start_time))
        print(f"iteration={iteration}/{num_iterations}  SPS={sps}  unique_states={counter.num_unique}")
        writer.add_scalar("charts/learning_rate",       optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS",                 sps,                             global_step)
        writer.add_scalar("charts/unique_states",       counter.num_unique,              global_step)
        writer.add_scalar("charts/mean_intrinsic_rew",  np.mean(intrinsic_log),          global_step)
        writer.add_scalar("charts/ae_code_bit_variance", bit_variance,                   global_step)
        writer.add_scalar("losses/value_loss",          v_loss.item(),                   global_step)
        writer.add_scalar("losses/policy_loss",         pg_loss.item(),                  global_step)
        writer.add_scalar("losses/entropy",             entropy.mean().item(),           global_step)
        writer.add_scalar("losses/approx_kl",           approx_kl.item(),                global_step)
        writer.add_scalar("losses/clipfrac",            np.mean(clipfracs),              global_step)
        writer.add_scalar("losses/explained_variance",  explained_var,                   global_step)
        writer.add_scalar("losses/ae_recon_loss",       ae_recon_loss.item(),            global_step)
        writer.add_scalar("losses/ae_sat_loss",         ae_sat_loss.item(),              global_step)

        if iteration % args.checkpoint_interval == 0 or iteration == num_iterations:
            ckpt_path = os.path.join(args.checkpoint_dir, run_name, f"ckpt_{iteration:06d}.pt")
            _save_checkpoint(ckpt_path, iteration, global_step, agent, ae_model, optimizer, args)

    envs.close()
    writer.close()


if __name__ == "__main__":
    train()
