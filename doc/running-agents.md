# Running Agents & Viewing Results

## Prerequisites

Always activate the venv before running anything:

```bash
source .venv/bin/activate
```

---

## Running an Agent

All agents are run from the **project root**. Common flags apply to all of them:

```bash
# PPO baseline
python src/agents/ppo.py

# Count-based exploration
python src/agents/count_based.py

# RND (PPO + Random Network Distillation)
python src/agents/rnd.py
```

Useful flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--total-timesteps` | 10,000,000 | Total environment steps |
| `--num-envs` | 8 | Parallel environments (more = faster, more RAM) |
| `--seed` | 1 | Random seed |
| `--no-cuda` | — | Force CPU (automatic on macOS — no CUDA) |
| `--capture-video` | off | Record gameplay videos |
| `--track` | off | Log to Weights & Biases |

---

## Monitoring Training with TensorBoard

Start TensorBoard pointing at the `runs/` directory:

```bash
tensorboard --logdir runs
```

Then open **http://localhost:6006** in your browser. Logs appear in real time as the agent trains — no need to restart TensorBoard between runs.

Key metrics to watch:

| Metric | What it shows |
|--------|--------------|
| `charts/episodic_return` | Game score per episode |
| `charts/rooms_visited` | Exploration breadth (primary thesis metric) |
| `charts/SPS` | Training throughput (steps/second) |
| `charts/mean_intrinsic_rew` | RND curiosity signal (RND only) |
| `losses/fwd_loss` | RND predictor error — should decrease over time (RND only) |
| `losses/explained_variance` | Value function quality (higher = better) |

---

## Quick Training Run (~10 minutes) + Video

For a time-boxed experiment, calculate `--total-timesteps` from your expected SPS:

- **CPU (macOS):** ~130 SPS → 10 min ≈ 75,000 steps  
- **GPU:** ~1,000–3,000 SPS → 10 min ≈ 600,000–1,800,000 steps

Run with `--capture-video` to record gameplay:

```bash
python src/agents/rnd.py \
    --total-timesteps 75000 \
    --num-envs 4 \
    --capture-video
```

Videos are saved to `videos/<run-name>/`. Each episode from env 0 is recorded as a separate `.mp4` file. **The last file in the folder is the most-trained agent.**

Open TensorBoard in parallel to watch training progress:

```bash
# In a second terminal
tensorboard --logdir runs
```

### Finding the video after training

```bash
# List videos from the most recent run, newest last
ls -lt videos/ | head -5          # find the run folder
ls videos/<run-name>/*.mp4 | tail -1   # last recorded episode
```

Open it with:

```bash
open videos/<run-name>/<last-file>.mp4   # macOS
```

---

## RND-specific: Obs Normalisation Init

RND runs a short random-action phase before training starts to initialise the observation running statistics. Controlled by `--obs-norm-init-steps` (default: 50 iterations × num-steps × num-envs steps). For quick experiments, reduce it:

```bash
python src/agents/rnd.py --obs-norm-init-steps 5 --total-timesteps 75000
```

---

## Performance Expectations on CPU (macOS)

Measured via `scripts/profile_throughput.py` (see `doc/throughput-investigation.md`
for the full methodology and cluster-side numbers):

| Phase | num_envs=4 | Notes |
|-------|-----------|-------|
| Env-only (random actions) | ~4365 SPS | Emulation + wrapper stack, no NN |
| Full step, inference only | ~1772 SPS | + NatureCNN forward pass |
| Full step + backward | ~760 SPS | + backward/optimizer step every 128 steps |

The full training loop (forward + backward, matching `ppo.py`/`rnd.py`) runs at
roughly **~760 SPS** on CPU with 4 envs — NN compute dominates on CPU-only hardware,
unlike on a GPU where the bottleneck shifts back toward emulation/IPC overhead (see
`doc/throughput-investigation.md` §3). The default vectorization backend is
`AsyncVectorEnv` (one subprocess per env), not `SyncVectorEnv` — pass `--sync-envs`
to opt into the single-process alternative, which is easier to debug but doesn't
parallelize env stepping across CPU cores. For real training runs, use a GPU machine
or a cloud instance; local CPU runs are for quick smoke tests only.
