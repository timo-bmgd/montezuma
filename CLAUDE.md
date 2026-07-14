# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Bachelor thesis: testing and comparing exploration algorithms for playing and solving **Montezuma's Revenge** via the Arcade Learning Environment (ALE). Target environment: `ALE/MontezumaRevenge-v5`.

ALE documentation: https://ale.farama.org/

**Algorithm roadmap** (in planned order) — all three are implemented in `src/agents/`:
1. Count-based exploration (`count_based.py`, plus an AE-SimHash variant `count_based_ae.py`) — baseline to demonstrate limitations on hard-exploration games
2. RND (Random Network Distillation) (`rnd.py`) — key algorithm of interest
3. PPO (`ppo.py`) — standard policy gradient baseline

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

**CleanRL** is the target framework (replacing AgileRL) — but as a *style*, not a dependency: `src/agents/ppo.py`, `count_based.py`, `count_based_ae.py`, and `rnd.py` are hand-written, self-contained single-file implementations following CleanRL's conventions. The `cleanrl` pip package itself is not installed and not imported anywhere.

AgileRL (with evolutionary HPO) remains installed but is secondary — relevant only if evolutionary hyperparameter search becomes part of the thesis scope later.

## JupyterHub Path

The repo is cloned to **`~/work/montezuma`** on JupyterHub (not `~/montezuma`). Files outside `~/work/` are deleted when the server stops. All notebook commands and paths use `~/work/montezuma`.

Two notebooks drive the JupyterHub workflow:
- `jupyter-setup.ipynb` — one-time environment setup: clone/pull the repo, `pip install` Atari/RL deps (numpy pinned `<2` — see below), download ROMs, run a smoke test.
- `training-runs.ipynb` — quick-launch notebook for production training jobs. A `launch()` helper wraps `nohup` + a `gpu=` param that sets `CUDA_VISIBLE_DEVICES`, so multiple jobs (e.g. two RND seeds + a PPO baseline) can run in parallel across separate GPUs on a multi-GPU JupyterHub instance.

`--sync-envs` is required on JupyterHub for every run: `AsyncVectorEnv` uses Python multiprocessing, which is unreliable inside Jupyter/container environments (`RuntimeError: Numpy is not available`). `--sync-envs` falls back to the single-process `SyncVectorEnv`.

Production runs should pass a large `--checkpoint-interval` (e.g. `99999`) to effectively disable checkpointing — checkpoints otherwise fill the JupyterHub disk quota over long runs.

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

# Count-based exploration, AE-SimHash variant (currently non-functional -- see file docstring)
python src/agents/count_based_ae.py

# RND (Random Network Distillation)
python src/agents/rnd.py
python src/agents/rnd.py --total-timesteps 10000000 --num-envs 8

# View TensorBoard logs
tensorboard --logdir runs
```

Key flags shared by all agents: `--seed`, `--num-envs`, `--total-timesteps`, `--capture-video`, `--record-room-discovery`, `--video-episode-interval` (default 100 in `rnd.py`/`count_based.py`/`count_based_ae.py`, 50 in `ppo.py` — inconsistent, not yet reconciled), `--clip-reward`/`--no-clip-reward` (default on), `--track` (W&B), `--sync-envs`, `--runs-dir`/`--videos-dir`/`--checkpoint-dir`, `--checkpoint-interval`, `--resume <path>`, `--ent-coef`, `--anneal-lr`/`--no-anneal-lr`. `rnd.py` and `count_based.py` additionally have `--overlay-video`/`--overlay-episode-interval` (synced gameplay+dashboard videos with a bar-meter overlay of one calibrated metric — mutually exclusive with `--capture-video` for now). `rnd.py` additionally has `--int-gamma`, `--int-coef`, `--ext-coef`, `--update-proportion`, `--obs-norm-init-steps`; `count_based.py`/`count_based_ae.py` have `--exploration-coef`/`--hash-dim` (plus `--ae-*` flags for the AE variant). `count_based.py` also has `--hash-mode {index,pool}`/`--hash-pool-size` — the original `index` hash is degenerate-collapsed on Montezuma (one bucket takes ~50% of visits, measured by `scripts/simhash_occupancy_probe.py`); production count runs should use `--hash-mode pool`. See `doc/running-agents.md` for the fuller reference (not yet updated with every flag above — check the source file's `parse_args()` for the ground truth).

**Dual-signal passive riders** (see `doc/dual-signal-rider.md`): `count_based.py --passive-rnd` and `rnd.py --passive-simhash` run the *other* method's novelty signal as a passive observer on the same observation stream — computed, trained, and logged, but never added to the reward (bit-identical training with or without the rider; verified by `scripts/check_rider_noop.py`). Related flags on both agents: `--rider-seed` (+ per-agent `--rider-*` hyperparameters), `--step-log` (per-(step, env) dual-bonus/room/episode `.npz` shards under `{runs-dir}/{run_name}/step_log/`), `--artifact-interval N` (small analysis artefacts — RND predictor/target + running stats, bit-packed count tables, no optimizer states — under `{checkpoint-dir}/{run_name}/artifacts/`). All default off; behaviour without these flags is unchanged, except that checkpoints now also carry the SimHash count table (and rider state when a rider is on) so `--resume` no longer resets visit counts.

`--resume <path>` now derives `run_name` from the checkpoint's own path (`{checkpoint_dir}/{run_name}/ckpt_XXXXXX.pt`) in both `rnd.py` and `ppo.py`, instead of generating a fresh timestamped run on every resume — a resumed run continues writing into the *same* TensorBoard/W&B run. This matters because `run_name` itself can contain `/` (`env_id` is `ALE/MontezumaRevenge-v5`), so the fix is `Path(args.resume).resolve().parent.relative_to(Path(args.checkpoint_dir).resolve())`, not just `.parent.name` (which drops the `ALE/` segment). `--checkpoint-dir` must match between the original run and the `--resume` invocation for this to work.

`rnd.py` and `ppo.py` both have `--auto-stop`/`--no-auto-stop` (default on): stop training early if a collapse signature is sustained for `--auto-stop-patience` iterations, log a marker, checkpoint, and `sys.exit(42)` — see `doc/decisions.md` and `doc/hpc-onboarding.md` §7 for the failure mode this guards against and how to read the exit code. `rnd.py`'s variant checks entropy fraction + intrinsic-reward drop-from-peak + near-zero `approx_kl`/`clipfrac` together (`--auto-stop-entropy-frac`, `--auto-stop-intrinsic-drop`, `--auto-stop-kl-eps`, `--auto-stop-clipfrac-eps`), calibrated against the real numbers in `doc/10M-RND-run-failure-documentation.md`. `ppo.py`'s is simpler (no intrinsic term) and its default thresholds are provisional — PPO has no prior collapse incident on this codebase to calibrate against.

## Source Code Structure (`src/`)

```
src/
├── agents/
│   ├── base.py            # Shared: NatureCNN, layer_init, make_env, RoomTracker, NewRoomRecorder
│   ├── ppo.py             # PPO (CleanRL-style, standalone runnable)
│   ├── count_based.py     # PPO + SimHash count-based exploration bonus
│   ├── count_based_ae.py  # PPO + AE-SimHash (learned hash) -- currently non-functional, see file docstring
│   ├── rnd.py             # PPO + RND (raw curiosity signal, normalisation stats, dual value heads)
│   └── riders.py          # Passive riders (RNDRider/SimHashRider), StepLogger, small-artefact savers
```

Each agent file is a self-contained runnable script that imports shared utilities from `base.py`. The `sys.path.insert` at the top of each agent file makes them runnable from the project root without installing the package.

`base.py` provides:
- `NatureCNN` — Nature DQN CNN backbone `(N, 4, 84, 84) → (N, 512)`
- `make_env(env_id, idx, capture_video, run_name, videos_dir="videos", video_episode_interval=1, record_room_discovery=False, clip_reward=True, overlay_video=False)` — builds the standard Atari preprocessing stack; passes `frameskip=1` to `gym.make` so `AtariPreprocessing` handles frame-skipping without duplication. Wrapper stack: `ALE env → RoomTracker → AtariPreprocessing → FrameStackObservation → RecordEpisodeStatistics → [ClipReward] → [OverlayFrameProbe, OR RecordVideo optionally + NewRoomRecorder stacked on top]`. `overlay_video` and `capture_video` are mutually exclusive (enforced by each agent's `train()`, not by `make_env()` itself). `ClipReward` is applied after `RecordEpisodeStatistics` so logged episodic return is always true game score; only the training reward is clipped to `[-1, 1]` (matches the RND paper's preprocessing, Burda et al. 2018, Appendix A.3).
- `RoomTracker` — wrapper that reads room number from Atari RAM byte 3, adds `rooms_visited` to episode-end info
- `NewRoomRecorder` — wrapper that buffers frames per episode and writes an mp4 only when `rooms_visited` exceeds the previous best (new room high-water mark); see "Video Recording — Room Discovery Mode" below
- `OverlayFrameProbe` — exposes a uniform `overlay_render()` across all vector sub-envs so `envs.call("overlay_render")` is safe even though only env 0 renders; used by `--overlay-video` (see `src/agents/video_overlay.py`'s `EpisodeOverlayRecorder`)

## Experiment Tracking

**TensorBoard** for local tracking. CleanRL uses `torch.utils.tensorboard.SummaryWriter` natively — log to a `runs/` directory and view with:

```bash
tensorboard --logdir runs
```

Key metrics to log: episodic return, episode length, rooms explored, loss curves. RND additionally logs intrinsic-reward and predictor-loss metrics (`mean_intrinsic_rew`, `raw_intrinsic_rew_mean`, `obs_rms_std`, `reward_rms_std`, `fwd_loss`, `explained_variance`, `approx_kl`, `clipfrac`, etc. — see "Log Analysis" below for the full table).

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

## Video Recording — Room Discovery Mode

`NewRoomRecorder` in `base.py` buffers `rgb_array` frames in memory during each episode. On episode end it checks `info["rooms_visited"]`; if the count exceeds the stored high-water mark it writes an `.mp4` and updates the mark, otherwise discards the buffer. Requires `render_mode="rgb_array"` and `RoomTracker` in the wrapper stack.

Enable with `--capture-video --record-room-discovery`. Videos land in `videos/<run_name>/room_discovery/new_room_ep<N>_r<R>.mp4`.

Standard periodic recording uses `--capture-video --video-episode-interval N` (default 100). Supported by both `rnd.py` and `count_based.py`. `--record-room-discovery` stacks on top of periodic recording (both write to the same run's video folder) rather than replacing it — a single run with both flags produces `rl-video-episode-N.mp4` files and `room_discovery/new_room_ep*.mp4` files together.

## Log Analysis

**Text logs** (`logs/<name>.out`) — plain stdout/stderr. Contains per-episode return/length lines and per-iteration SPS prints. Readable directly.

**TensorBoard event files** (`runs/<run_name>/events.out.tfevents.*`) — binary protobuf. Parse with:

```python
from torch.utils.tensorboard.backend.event_file_loader import EventFileLoader
# or, easier:
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator("runs/<run_name>")
ea.Reload()
scalars = ea.Tags()["scalars"]          # list of available metric names
steps, vals = zip(*[(e.step, e.value) for e in ea.Scalars("charts/episodic_return")])
```

**Key metrics to check when diagnosing a failed RND run:**

| Metric | Healthy sign | Failure sign |
|--------|-------------|--------------|
| `charts/episodic_return` | Increasing trend | Flat at 0 forever |
| `losses/entropy` | Slow decay | Rapid collapse to ~0 |
| `charts/raw_intrinsic_rew_mean` | Non-zero, ~0.1–1.0 | Near zero throughout |
| `charts/obs_rms_std` | Converges to ~1.0 | Stays near 0 |
| `charts/reward_rms_std` | Non-zero | Near zero |
| `losses/explained_variance` | Grows toward 1.0 | Stays negative/zero |

**Production run data locations (local):**
- Logs: `logs/rnd_10m_s1.out`, `logs/rnd_10m_s2.out`, `logs/ppo_5m_s1.out`
- TensorBoard: `runs/ALE/MontezumaRevenge-v5__rnd__1__1781212716/`, `...__rnd__1__1781213250/`

## JupyterHub Launch Pattern

On JupyterHub, use `sys.executable` in `subprocess.Popen` — the kernel Python has all packages; the system Python does not. GPU assignment uses `CUDA_VISIBLE_DEVICES` env var:

```python
def launch(args, log_name, gpu=0):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    proc = subprocess.Popen(
        [sys.executable, *args],
        stdout=open(PROJECT / "logs" / log_name, "w"), stderr=subprocess.STDOUT,
        cwd=PROJECT, env=env,
    )
    return proc
```

The notebook `training-runs.ipynb` contains ready-to-run cells for all production runs.

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

Sources: [RND paper](https://arxiv.org/abs/1810.12894), [Go-Explore paper](https://arxiv.org/abs/1901.10995), [Count-based paper (Bellemare et al. 2016, pseudo-counts)](https://arxiv.org/abs/1606.01868).

Note: `src/agents/count_based.py`/`count_based_ae.py` in this repo implement a *different* count-based
method — Tang et al. 2017's SimHash/AE-SimHash approach ([arXiv:1611.04717](https://arxiv.org/abs/1611.04717)),
not Bellemare's pseudo-count density model above. The two are often confused; the benchmark row above
reflects Bellemare's numbers specifically.

## Repository Structure

- `src/` — main source code (agents + shared utilities; see Source Code Structure above)
- `examples/` — lightweight, standalone reference scripts; keep up to date with current ale_py/gymnasium API
- `doc/` — investigation write-ups and usage guides (`running-agents.md`, `throughput-investigation.md`, `decisions.md`, `pipeline-history.md`, `10M-RND-run-failure-documentation.md`, `rnd-vs-ppo-asymmetry-investigation.md`, `hpc-onboarding.md`, `dual-signal-rider.md`)
- `slurm/` — SLURM job scripts for cluster training runs: `run_rnd_smoke.slurm`/`run_ppo_smoke.slurm` (infrastructure smoke tests, minutes-scale, run these first — see `doc/hpc-onboarding.md`), `run_rnd.slurm`/`run_ppo.slurm` (production runs, hours-scale), `run_count_based.slurm` (production, count-based pipeline still under separate setup by the user), `run_count_rnd_rider.slurm` (production count-driven run with passive RND rider — the dual-signal analysis run, see `doc/dual-signal-rider.md`). `run_rnd_falsify.slurm` was removed 2026-07-14 — its ent_coef hypothesis was reverted (see `doc/decisions.md`) and it was repurposed into `run_rnd_smoke.slurm`.
- `scripts/` — utility scripts: `profile_throughput.py` (training SPS), `simhash_occupancy_probe.py` (hash bucket-occupancy degeneracy check), `check_rider_noop.py` (proves passive riders don't affect training)
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
