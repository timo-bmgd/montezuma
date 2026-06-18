"""Count-based exploration (SimHash) + PPO for Montezuma's Revenge.

Demonstrates that naive visit-counting fails on hard-exploration games:
the state space is so large that nearly every hash bucket has count=1,
making the intrinsic bonus uniformly ~beta everywhere and providing no
meaningful exploration signal.

Run from project root with the venv active:
    source .venv/bin/activate
    python src/agents/count_based.py
    python src/agents/count_based.py --exploration-coef 0.1 --hash-dim 128
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

    Each observation is averaged across the frame stack, downsampled to a
    fixed size, L2-normalised, then projected to hash_dim bits via a fixed
    random matrix. Visits are counted per binary hash code.

    Intrinsic reward = exploration_coef / sqrt(n(hash(s)))

    On Montezuma's Revenge the pixel space is so large that virtually every
    state gets its own bucket (n≈1), so the bonus is ~exploration_coef
    everywhere — demonstrating the limitation of count-based methods on
    hard-exploration games.
    """

    _DOWNSAMPLE = 128  # number of pixels kept after downsampling

    def __init__(self, hash_dim: int = 64, seed: int = 42):
        rng = np.random.default_rng(seed)
        self._A = rng.standard_normal((hash_dim, self._DOWNSAMPLE)).astype(np.float32)
        self._counts: dict[bytes, int] = {}

    def _hash(self, obs: np.ndarray) -> bytes:
        # obs: (4, 84, 84) uint8 — average over frame stack, then downsample
        frame = obs.astype(np.float32).mean(axis=0)          # (84, 84)
        idx = np.linspace(0, frame.size - 1, self._DOWNSAMPLE, dtype=int)
        flat = frame.flatten()[idx]
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


def _save_checkpoint(path, iteration, global_step, agent, optimizer, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "iteration": iteration,
        "global_step": global_step,
        "agent_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }, path)


def _load_checkpoint(path, agent, optimizer):
    ckpt = torch.load(path, weights_only=False)
    agent.load_state_dict(ckpt["agent_state_dict"])
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
        [make_env(args.env_id, i, args.capture_video, run_name, args.videos_dir) for i in range(args.num_envs)]
    )

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    counter = SimHashCounter(hash_dim=args.hash_dim, seed=args.seed)

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
        start_iteration, global_step = _load_checkpoint(args.resume, agent, optimizer)
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

            # count-based intrinsic reward: increment counter, then compute bonus
            intrinsic = np.zeros(args.num_envs, dtype=np.float32)
            for i in range(args.num_envs):
                counter.increment(next_obs_np[i])
                intrinsic[i] = counter.bonus(next_obs_np[i], args.exploration_coef)
            intrinsic_log.append(intrinsic.mean())

            combined_reward = reward + intrinsic
            rew_buf[step] = torch.tensor(combined_reward, dtype=torch.float32, device=device)
            next_obs  = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done_np, dtype=torch.float32, device=device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is None or "episode" not in info:
                        continue
                    ep = info["episode"]
                    print(f"  step={global_step}  return={ep['r']:.1f}  length={ep['l']}")
                    writer.add_scalar("charts/episodic_return", ep["r"], global_step)
                    writer.add_scalar("charts/episodic_length", ep["l"], global_step)
                    if "rooms_visited" in info:
                        writer.add_scalar("charts/rooms_visited", info["rooms_visited"], global_step)

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

        if iteration % args.checkpoint_interval == 0 or iteration == num_iterations:
            ckpt_path = os.path.join(args.checkpoint_dir, run_name, f"ckpt_{iteration:06d}.pt")
            _save_checkpoint(ckpt_path, iteration, global_step, agent, optimizer, args)

    envs.close()
    writer.close()


if __name__ == "__main__":
    train()
