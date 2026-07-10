# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Bachelor thesis: testing and comparing exploration algorithms for playing and solving **Montezuma's Revenge** via the Arcade Learning Environment (ALE). Target environment: `ALE/MontezumaRevenge-v5`.

ALE documentation: https://ale.farama.org/

**Algorithm roadmap** (in planned order):
1. Count-based exploration — baseline to demonstrate limitations on hard-exploration games
2. RND (Random Network Distillation) — key algorithm of interest
3. PPO — standard policy gradient baseline

## Environment Setup

Python 3.13, `.venv` virtual environment. The VSCode interpreter is configured to `.venv/bin/python3`.

```bash
source .venv/bin/activate
```

`requirements.txt` at the repo root pins the key packages: `torch==2.8.0+cu124`, `gymnasium==1.3.0`, `ale-py==0.11.2`, `AutoROM==0.6.1`, `numpy==2.4.4`, `opencv-python-headless`, `tensorboard`, `wandb`, `pillow`. `agilerl` (2.6.1) is also installed in the venv but not listed in `requirements.txt` and not the primary framework going forward.

**Gymnasium 1.x vectorized env infos format:** gymnasium 1.x changed the infos format from gymnasium 0.x. Episode data is now in `infos["episode"]["r"][i]` / `infos["episode"]["l"][i]`, masked by `infos["_episode"][i]` (True when env `i` ended an episode). The old `infos["final_info"]` list-of-dicts pattern from CleanRL's original code does NOT work in gymnasium 1.x.

## ALE Environment Registration

ALE environments must be registered with gymnasium before use:

```python
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
env = gym.make("ALE/MontezumaRevenge-v5", render_mode="rgb_array")
```

For vectorized training (multiple parallel envs), use `gymnasium.vector.make` or AgileRL's helper:

```python
from agilerl.utils.utils import make_vect_envs
env = make_vect_envs("ALE/MontezumaRevenge-v5", num_envs=8)
```

## RL Framework

**CleanRL** is the target framework (replacing AgileRL) — but as a *style*, not a dependency: `src/agents/ppo.py`, `count_based.py`, and `rnd.py` are hand-written, self-contained single-file implementations following CleanRL's conventions. The `cleanrl` pip package itself is not installed and not imported anywhere.

AgileRL (with evolutionary HPO) remains installed but is secondary — relevant only if evolutionary hyperparameter search becomes part of the thesis scope later.

## JupyterHub Path

The repo is cloned to **`~/work/montezuma`** on JupyterHub (not `~/montezuma`). Files outside `~/work/` are deleted when the server stops. All notebook commands and paths use `~/work/montezuma`.

## Running Agents

Run from the **project root** with the venv active:

```bash
source .venv/bin/activate

# PPO baseline
python src/agents/ppo.py
python src/agents/ppo.py --total-timesteps 1000000 --num-envs 4

# Count-based exploration
python src/agents/count_based.py
python src/agents/count_based.py --exploration-coef 0.1 --hash-dim 128

# RND
python src/agents/rnd.py

# View TensorBoard logs
tensorboard --logdir runs
```

Key flags shared by all agents: `--seed`, `--num-envs`, `--total-timesteps`, `--capture-video`, `--track` (W&B), `--sync-envs`, `--clip-reward`, `--runs-dir`, `--videos-dir`, `--checkpoint-dir`/`--checkpoint-interval`, `--resume`. RND additionally has `--ent-coef`, `--anneal-lr`/`--no-anneal-lr`, `--obs-norm-init-steps`, `--int-gamma`, `--int-coef`, `--ext-coef`, `--update-proportion`. See `doc/running-agents.md` for the full flag reference and TensorBoard metrics table.

## Source Code Structure (`src/`)

```
src/
├── agents/
│   ├── base.py          # Shared: NatureCNN, layer_init, make_env, RoomTracker
│   ├── ppo.py           # PPO (CleanRL-style, standalone runnable)
│   ├── count_based.py   # PPO + SimHash count-based exploration bonus
│   └── rnd.py           # PPO + RND (Random Network Distillation)
```

Each agent file is a self-contained runnable script that imports shared utilities from `base.py`. The `sys.path.insert` at the top of each agent file makes them runnable from the project root without installing the package.

`base.py` provides:
- `NatureCNN` — Nature DQN CNN backbone `(N, 4, 84, 84) → (N, 512)`
- `make_env(env_id, idx, capture_video, run_name, ...)` — builds the standard Atari preprocessing stack; passes `frameskip=1` to `gym.make` so `AtariPreprocessing` handles frame-skipping without duplication. Also takes toggles for video/room-discovery recording and reward clipping (`videos_dir`, `video_episode_interval`, `record_room_discovery`, `clip_reward`) — see the function signature in `base.py` for the full parameter list
- `RoomTracker` — wrapper that reads room number from Atari RAM byte 3, adds `rooms_visited` to episode-end info
- `NewRoomRecorder` — wrapper that records video whenever an env reaches a new room high-water mark

## Experiment Tracking

**TensorBoard** for local tracking. CleanRL uses `torch.utils.tensorboard.SummaryWriter` natively — log to a `runs/` directory and view with:

```bash
tensorboard --logdir runs
```

Key metrics to log: episodic return, episode length, rooms explored, loss curves. RND additionally logs intrinsic-reward and predictor-loss metrics (`mean_intrinsic_rew`, `fwd_loss`, `explained_variance`, `approx_kl`, `clipfrac`, etc.) — see the TensorBoard metrics table in `doc/running-agents.md`.

**Weights & Biases (W&B)** is a cloud alternative worth considering once multiple algorithms are being compared. W&B's free tier adds cross-run comparison dashboards and automatic hyperparameter logging — useful when writing the results section. TensorBoard requires manually inspecting separate log directories; W&B shows everything in one view. Start with TensorBoard, migrate to W&B when running multi-algorithm comparisons.

## Video Recording

Use `gymnasium.wrappers.RecordVideo`. The environment must be initialized with `render_mode="rgb_array"`:

```python
from gymnasium.wrappers import RecordVideo

env = gym.make("ALE/MontezumaRevenge-v5", render_mode="rgb_array")
env = RecordVideo(
    env,
    video_folder="videos/",
    episode_trigger=lambda ep: ep % 50 == 0,
)
```

See `examples/recording_sample.py` for a working CartPole example of this pattern.

## Evaluation Metrics

Based on literature, report both:

1. **Rooms explored** (primary exploration metric) — number of distinct rooms the agent visits. Standard across all major papers; directly measures exploration breadth.
2. **Mean game score** — secondary performance metric.

### Literature benchmarks for comparison

| Algorithm | Rooms | Mean Score | Notes |
|-----------|-------|-----------|-------|
| Count-based (2016) | 15 | ~3,700 | Pseudo-count approach |
| RND (OpenAI, 2018) | 24 | ~10,000 | Intrinsic motivation |
| Go-Explore (Uber, 2019) | 37 / 238 | 43,000 / 650,000 | Without / with domain knowledge |

Sources: [RND paper](https://arxiv.org/abs/1810.12894), [Go-Explore paper](https://arxiv.org/abs/1901.10995), [Count-based paper](https://arxiv.org/abs/1703.01310).

## Repository Structure

- `src/` — main source code (agents + shared utilities)
- `examples/` — lightweight, standalone reference scripts; keep up to date with current ale_py/gymnasium API
- `doc/` — investigation write-ups and usage guides (e.g. `running-agents.md`, `throughput-investigation.md`)
- `slurm/` — SLURM job scripts for cluster training runs (`run_ppo.slurm`, `run_count_based.slurm`, `run_rnd.slurm`, `run_rnd_falsify.slurm`, `setup_hpc.sh`)
- `scripts/` — utility scripts, e.g. `profile_throughput.py` for measuring training SPS
- `.claude/skills/` — Claude Code skill definitions (see below)
- `cartpole-training/`, `_static/` — gitignored; training videos/static assets from the CartPole recording example, not relevant to the main project and may not exist in a fresh checkout

## Claude Code Skills

Project skills live in `.claude/skills/<skill-name>/`. Each skill follows this structure:

```
.claude/skills/<skill-name>/
├── SKILL.md           # Required: frontmatter + instructions
├── references/        # Docs loaded into context as needed
├── scripts/           # Executable scripts for deterministic tasks
└── assets/            # Templates, icons, etc.
```

**Available skills:**
- `skill-creator` — create, test, and iterate on new Claude Code skills
- `ale` — ALE environment setup, observation types, action spaces, preprocessing, stochasticity
- `gymnasium` — wrappers (AtariPreprocessing, FrameStack, RecordVideo, etc.), vectorized envs, spaces API

- `cleanrl` — PPO/RND single-file patterns, hyperparameters, TensorBoard metrics, macOS adaptation notes
