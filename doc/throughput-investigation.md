# Training throughput investigation

Summary of the investigation into whether training throughput (SPS) could be improved
before committing further GPU-hours to RND/count-based/PPO production runs, and what
was actually changed as a result.

## TL;DR

The standing "CPU/emulation-bound, not GPU-bound" hypothesis was based on reasoning
(flat 144-183 SPS across a full 10M-step RND run; a NatureCNN forward+backward is
cheap on an A100/V100 regardless of batch size), not measurement — no profiling had
ever actually been run in this repo. A code read of the SLURM scripts found a
concrete, previously-unverified contributor: every SLURM script requests
`--num-envs 32` but under-provisions CPUs for it, given every agent defaults to
`AsyncVectorEnv` (one subprocess per env). That's fixed (§2). A new profiling script
(§3) now lets that CPU-bound hypothesis be checked against real numbers instead of
assumed. Two engineering options (resolution downscaling, increased frame-skip) were
investigated and rejected (§4). EnvPool, the highest-upside option in principle, is
deferred as future work — it is not a drop-in given this repo's room-tracking
instrumentation (§5). PufferLib and CuLE are ruled out entirely (§6).

## 1. Confirmed: CPU/env-count oversubscription

Every agent (`ppo.py`, `count_based.py`, `rnd.py`) builds its vectorized env with:

```python
VecCls = gym.vector.SyncVectorEnv if args.sync_envs else gym.vector.AsyncVectorEnv
```

`AsyncVectorEnv` is the default — one OS subprocess per parallel env, each running
the full Python + ALE emulation + wrapper stack (`RoomTracker` → `AtariPreprocessing`
→ `FrameStackObservation` → `RecordEpisodeStatistics` → `ClipReward`) independently.
Before this investigation, every SLURM script requested `--num-envs 32` without
matching CPU allocation:

| Script | `--cpus-per-task` (before) | `--num-envs` | Ratio |
|---|---|---|---|
| `run_ppo.slurm` | 8 | 32 | 4x oversubscribed |
| `run_count_based.slurm` | 16 | 32 | 2x oversubscribed |
| `run_rnd.slurm` | 16 | 32 | 2x oversubscribed |
| `run_rnd_falsify.slurm` | 16 | 32 | 2x oversubscribed |

All four carried a comment admitting the CPU count was "a guess... confirm against
your partition's actual core count," never actually verified via `sinfo`.

## 2. Fix applied: SLURM CPU/env ratio

`--cpus-per-task` raised to **32** in all four scripts, matching `--num-envs` 1:1.
`--num-envs` was deliberately left unchanged: RND's own paper used `num_envs=128`
(this repo already runs 4x below paper scale), and `run_rnd.slurm`/
`run_rnd_falsify.slurm` exist specifically to validate an entropy-collapse fix
(`--ent-coef 0.01 --no-anneal-lr`, see the sibling failure-diagnosis doc) that depends
on per-update experience diversity — reducing `num_envs` would also change
`batch_size`/`num_iterations` and reopen already-tuned hyperparameter interactions
this close to the deadline.

The real core-per-GPU ratio for the target partition was not available at the time of
this fix (checked with the user — no `sinfo` output on hand, only a generic HPC
submission template with placeholder values). Each script's comment now says
explicitly: run `sinfo -o "%P %c %G"` before submitting; if the partition has fewer
than 32 cores/GPU, lower `--cpus-per-task` to match rather than assuming the request
will simply be rejected — SLURM queues an over-large request indefinitely in `PD`
state instead, which would silently burn walltime budget without ever running.

Also added to all four scripts, flagged as an **untested hypothesis** pending
profiling data:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

None of the scripts previously constrained PyTorch's CPU thread pool, which can
oversubscribe the same cores the `AsyncVectorEnv` workers are already contending for.

**Results (fill in after the next cluster run):**

| Run | SPS before | SPS after | Notes |
|---|---|---|---|
| `run_rnd_falsify.slurm` | ~144-183 (from prior 10M run, different script) | _TBD_ | |

## 3. `scripts/profile_throughput.py`

New standalone script, dependency-free beyond what's already in `requirements.txt`.
Reuses `make_env`/`NatureCNN`/`layer_init`/`RoomTracker` from `src/agents/base.py`
unmodified. Measures three phases against the same wrapper stack used in production:

1. **Env-only SPS** — random actions, no NN. Isolates emulation + wrapper + IPC cost.
2. **Full step, inference only** — env.step() + a NatureCNN-shaped forward pass.
3. **Full step + backward** — adds one backward/optimizer step every `--num-steps`
   env-steps (default 128, matching the agents), mirroring the real PPO update cadence.

If phase 1 SPS ≈ phase 3 SPS, that directly confirms the env/emulation-bound
hypothesis. It also runs a `RoomTracker.getRAM()` ablation (stubs the RAM read via a
runtime monkeypatch, comparing phase-1 SPS with and without it) and a `--cprofile`
mode (forces `--sync-envs`, since under `AsyncVectorEnv` the main process only sees
IPC-wait time, not what happens inside `env.step()` in the worker).

Usage:

```bash
python scripts/profile_throughput.py                                     # local sanity check
python scripts/profile_throughput.py --num-envs 32 --measure-steps 300   # cluster, real numbers
python scripts/profile_throughput.py --cprofile                          # function-level breakdown
```

**Results:**

| Environment | num_envs | device | [1] env-only | [1b] RoomTracker stubbed | [2] inference | [3] +backward |
|---|---|---|---|---|---|---|
| Local (macOS, CPU) | 4 | cpu | 4365.1 SPS | 4412.8 SPS (getRAM() tax ~1.1%) | 1771.8 SPS | 759.6 SPS |
| Cluster (A100) | 32 | cuda | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

The local run is **not** representative of the CPU-vs-GPU-bound question — on
CPU-only hardware the NN forward/backward is itself slow, so phase 3 dropping to 17%
of the phase-1 baseline just reflects NN cost dominating on a CPU, not an emulation
bottleneck. It does, however, confirm the script's methodology behaves sensibly (NN
cost should shrink to a small fraction of the total once run on the actual A100
GPUs) and that the `RoomTracker.getRAM()` tax is small (~1%) regardless. The cluster
run is the one that actually answers the CPU-bound question and should be done once,
after the SLURM CPU fix (§2), before committing to any 20h production job.

## 4. Rejected: resolution downscaling / increased frame-skip

Both were investigated and rejected without needing implementation:

- **Quarter-screen downscaling**: doesn't address a CPU/emulation-bound bottleneck —
  shrinking the CNN input doesn't reduce Python/IPC/emulator overhead, which is where
  the time actually goes. It would also destroy visual detail relevant to navigation
  and break comparability with every benchmark in `CLAUDE.md`'s literature table (all
  use 84x84).
- **Increased frame-skip** (e.g. 6-8 instead of 4): same reasoning — the bottleneck
  isn't primarily in frame-processing, so the throughput gain would be modest
  (~1.3-2x at best) while risking breaking Montezuma's room-1 rope-swing and gap-jump
  sequences, which require precise reflex timing that longer frame-skip is known to
  degrade (Braylan et al.; Metelli et al.).

## 5. Deferred: EnvPool

EnvPool is the highest-upside option in principle — CleanRL's own reference
benchmarks report ~5-15x SPS gains, and it has explicit `ALE/MontezumaRevenge-v5`
support. It is **not attempted in this pass**, for concrete reasons:

- `RoomTracker` and `NewRoomRecorder` (`src/agents/base.py:44-108`) are plain
  single-env `gym.Wrapper`s that read `self.unwrapped.ale.getRAM()[3]` and call
  `self.render()` directly, every step — this feeds `rooms_visited`, the thesis's
  **primary exploration metric**. EnvPool's batched C++ backend has no confirmed
  equivalent per-env RAM/frame hook; integrating it would mean either finding an
  EnvPool Atari info-dict equivalent or reimplementing room-tracking and
  video-capture against a batched interface. Real engineering risk, not a drop-in
  swap.
- Wheel compatibility is unconfirmed against either Python 3.13 (local dev venv) or
  3.11 (`module load python/3.11` on the cluster) — untested version split.
- Given the remaining thesis timeline, this is deferred rather than attempted.

## 6. Ruled out: PufferLib, CuLE

- **PufferLib**: only ~1.65x measured gain, and separately blocked outright for this
  repo by a hard `ale-py==0.9.0` pin with no Python 3.13 wheels.
- **CuLE**: dead project, no commits since 2022.

Neither is worth further consideration.
