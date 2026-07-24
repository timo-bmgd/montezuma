"""PPO for Montezuma's Revenge — adapted from CleanRL's ppo_atari.py.

Run from project root with the venv active:
    source .venv/bin/activate
    python src/agents/ppo.py
    python src/agents/ppo.py --total-timesteps 1000000 --num-envs 4
    python src/agents/ppo.py --capture-video --track
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
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.base import (NatureCNN, NoisyTVWrapper, check_tv_args_match,
                         check_tv_geometry, compute_gae, layer_init,
                         make_env, masked_mean)


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
    p.add_argument("--video-episode-interval", type=int, default=50,
                   help="Record a video every N episodes (env 0 only, when --capture-video is set)")
    p.add_argument("--clip-reward", action=argparse.BooleanOptionalAction, default=True,
                   help="Clip extrinsic reward to [-1, 1] (standard Atari preprocessing)")
    # noisy TV (control runs for the RND experiment -- see doc/noisy-tv-experiment.md)
    p.add_argument("--tv-mode", choices=["off", "static", "remote", "sham-remote"], default="off",
                   help="Inject a synthetic noise patch into observations. static = always-on, "
                        "resampled every --tv-refresh-every steps; remote = extra NOOP-mapped "
                        "action that resamples the patch; sham-remote = the extra action but no "
                        "patch. PPO has no intrinsic reward, so PPO+TV is the no-curiosity "
                        "control: capture should not occur")
    p.add_argument("--tv-size", type=int, nargs=2, default=[12, 84], metavar=("H", "W"),
                   help="TV patch height x width in 84x84-frame pixels (default: the full HUD "
                        "band -- see rnd.py's help for why area is the stimulus-strength lever)")
    p.add_argument("--tv-position", type=int, nargs=2, default=[0, 0], metavar=("ROW", "COL"),
                   help="TV patch top-left corner in the 84x84 frame (default HUD band; rows "
                        ">=12 are playfield in every room and must stay clear)")
    p.add_argument("--tv-refresh-every", type=int, default=1,
                   help="static mode only: resample the patch every N agent steps")
    # env
    p.add_argument("--env-id", default="ALE/MontezumaRevenge-v5")
    p.add_argument("--total-timesteps", type=int, default=10_000_000)
    # ppo
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
                   help="Stop training early if an entropy + frozen-PPO-update collapse is "
                        "sustained for --auto-stop-patience iterations. Simpler than rnd.py's "
                        "variant (no intrinsic-reward term -- PPO has none); default thresholds "
                        "are provisional since PPO has no prior collapse incident to calibrate "
                        "against (RND's are backed by doc/10M-RND-run-failure-documentation.md, "
                        "this one isn't yet -- re-check after the first production run). Writes "
                        "a checkpoint and exits with code 42 on trigger.")
    p.add_argument("--auto-stop-patience", type=int, default=150,
                   help="Consecutive iterations the collapse signature must hold before stopping")
    p.add_argument("--auto-stop-entropy-frac", type=float, default=0.10,
                   help="Trigger threshold: entropy / ln(action_space_n) below this value")
    p.add_argument("--auto-stop-kl-eps", type=float, default=1e-3,
                   help="Trigger threshold: approx_kl below this value (near-zero PPO updates)")
    p.add_argument("--auto-stop-clipfrac-eps", type=float, default=0.01,
                   help="Trigger threshold: clipfrac below this value (near-zero PPO updates)")
    return p.parse_args()


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


def _load_checkpoint(path, agent, optimizer, args):
    ckpt = torch.load(path, weights_only=False)
    # Guard BEFORE restoring any state: an action-space-changing tv mismatch
    # would otherwise die in load_state_dict with a raw shape error instead of
    # the guard's actionable message.
    check_tv_args_match(ckpt["args"], args)
    agent.load_state_dict(ckpt["agent_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["iteration"], ckpt["global_step"]


def train():
    args = parse_args()
    # Fail fast on invalid TV geometry in every mode (with --tv-mode off the
    # wrapper never runs its own check).
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
                  args.record_room_discovery, clip_reward=args.clip_reward, tv_mode=args.tv_mode,
                  tv_size=tuple(args.tv_size), tv_position=tuple(args.tv_position),
                  tv_refresh_every=args.tv_refresh_every) for i in range(args.num_envs)]
    )

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

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
        start_iteration, global_step = _load_checkpoint(args.resume, agent, optimizer, args)
        start_iteration += 1
        print(f"Resumed from {args.resume} at iteration {start_iteration - 1}, global_step={global_step}")

    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs  = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    collapse_streak = 0
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
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                val_buf[step] = value.flatten()
            act_buf[step]  = action
            logp_buf[step] = logprob

            next_obs_np, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminated, truncated)
            rew_buf[step] = torch.tensor(reward, dtype=torch.float32, device=device)
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

        # ── GAE advantage estimation ────────────────────────────────────────
        # NEXT_STEP-autoreset-correct GAE (see agents.base.compute_gae): the
        # discarded-action reset step (done_buf[t]==1) is walled off.
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = compute_gae(rew_buf, val_buf, done_buf, next_value, next_done,
                                     args.gamma, args.gae_lambda, episodic=True)
            returns = advantages + val_buf

        # ── PPO update ──────────────────────────────────────────────────────
        b_obs  = obs_buf.reshape((-1,) + obs_shape)
        b_logp = logp_buf.reshape(-1)
        b_act  = act_buf.reshape(-1)
        b_adv  = advantages.reshape(-1)
        b_ret  = returns.reshape(-1)
        b_val  = val_buf.reshape(-1)
        # 1 for real steps, 0 for NEXT_STEP fake (discarded-action) steps; excludes the
        # discarded terminal-frame action and its cross-episode value target from the update.
        b_keep = (1.0 - done_buf).reshape(-1)

        clipfracs = []
        for _ in range(args.update_epochs):
            mb_inds = np.random.permutation(batch_size)
            for start in range(0, batch_size, minibatch_size):
                mb = mb_inds[start : start + minibatch_size]

                mb_keep = b_keep[mb]  # exclude NEXT_STEP fake steps from all losses/diagnostics

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb], b_act.long()[mb]
                )
                logratio = newlogprob - b_logp[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = masked_mean((ratio - 1) - logratio, mb_keep)
                    clipfracs.append(masked_mean(((ratio - 1.0).abs() > args.clip_coef).float(), mb_keep).item())

                # normalise advantages over real (kept) samples only
                mb_adv_norm = b_adv[mb]
                adv_mean = masked_mean(mb_adv_norm, mb_keep)
                adv_std = torch.sqrt(masked_mean((mb_adv_norm - adv_mean) ** 2, mb_keep) + 1e-8)
                mb_adv_norm = (mb_adv_norm - adv_mean) / (adv_std + 1e-8)

                pg_loss = masked_mean(torch.max(
                    -mb_adv_norm * ratio,
                    -mb_adv_norm * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                ), mb_keep)

                newvalue = newvalue.view(-1)
                v_clipped = b_val[mb] + torch.clamp(newvalue - b_val[mb], -args.clip_coef, args.clip_coef)
                v_loss = 0.5 * masked_mean(torch.max(
                    (newvalue - b_ret[mb]) ** 2,
                    (v_clipped - b_ret[mb]) ** 2,
                ), mb_keep)

                loss = pg_loss - args.ent_coef * masked_mean(entropy, mb_keep) + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        y_pred, y_true = b_val.cpu().numpy(), b_ret.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        sps = int(global_step / (time.time() - start_time))
        print(f"iteration={iteration}/{num_iterations}  SPS={sps}")
        writer.add_scalar("charts/learning_rate",     optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS",               sps,                             global_step)
        if args.tv_mode in ("remote", "sham-remote"):
            # Behavioral-capture metric: fraction of chosen actions that press
            # the TV remote. Uniform-policy chance = 1/n.
            tv_action = NoisyTVWrapper.remote_action_index(action_space_n)
            tv_frac = (act_buf == float(tv_action)).float().mean().item()
            writer.add_scalar("charts/tv_action_frac", tv_frac,                        global_step)
        writer.add_scalar("losses/value_loss",        v_loss.item(),                   global_step)
        writer.add_scalar("losses/policy_loss",       pg_loss.item(),                  global_step)
        writer.add_scalar("losses/entropy",           entropy.mean().item(),           global_step)
        writer.add_scalar("losses/approx_kl",         approx_kl.item(),                global_step)
        writer.add_scalar("losses/clipfrac",          np.mean(clipfracs),              global_step)
        writer.add_scalar("losses/explained_variance", explained_var,                  global_step)

        if args.auto_stop:
            entropy_frac = entropy.mean().item() / np.log(action_space_n)
            collapsed_now = (
                entropy_frac < args.auto_stop_entropy_frac
                and approx_kl.item() < args.auto_stop_kl_eps
                and np.mean(clipfracs) < args.auto_stop_clipfrac_eps
            )
            collapse_streak = collapse_streak + 1 if collapsed_now else 0
            writer.add_scalar("charts/collapse_streak", collapse_streak, global_step)

            if collapse_streak >= args.auto_stop_patience:
                print(f"AUTO-STOP: collapse signature sustained for {args.auto_stop_patience} "
                      f"iterations (entropy_frac={entropy_frac:.3f}, "
                      f"approx_kl={approx_kl.item():.5f}, clipfrac={np.mean(clipfracs):.4f}). "
                      f"Saving checkpoint and exiting.")
                writer.add_scalar("charts/auto_stop_triggered", 1, global_step)
                ckpt_path = os.path.join(args.checkpoint_dir, run_name,
                                         f"ckpt_{iteration:06d}_autostop.pt")
                _save_checkpoint(ckpt_path, iteration, global_step, agent, optimizer, args)
                envs.close()
                writer.close()
                sys.exit(42)

        if iteration % args.checkpoint_interval == 0 or iteration == num_iterations:
            ckpt_path = os.path.join(args.checkpoint_dir, run_name, f"ckpt_{iteration:06d}.pt")
            _save_checkpoint(ckpt_path, iteration, global_step, agent, optimizer, args)

    envs.close()
    writer.close()


if __name__ == "__main__":
    train()
