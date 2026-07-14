# Dual-Signal Novelty Comparison: Passive Riders

**Added 2026-07-14.** Infrastructure for the thesis's core comparative analysis:
how do RND (prediction error) and SimHash counting (visit counts) *define*
novelty on the same states?

## Why a passive rider

Comparing the two novelty definitions requires both signals evaluated on
**identical observations**. Two separate runs cannot provide this — each agent
visits different states. Instead, one method *drives* exploration (its bonus is
added to the reward) while the other runs as a **passive rider** on the same
observation stream: its bonus is computed, trained (for RND), and logged per
step, but never added to any reward.

Implementation: `src/agents/riders.py` (`RNDRider`, `SimHashRider`,
`StepLogger`, artefact savers), wired into:

- `count_based.py --passive-rnd` — count-driven run, RND observing
- `rnd.py --passive-simhash` — RND-driven run, SimHash observing

## The no-op guarantee

A run with a rider enabled is **bit-identical** to the same run with the rider
disabled. This is enforced by RNG isolation (the rider initialises its network
under a forked torch CPU RNG and draws all training randomness — minibatch
permutations, update-proportion masks — from its own numpy `Generator`), and
by the rider having its own optimizer over its own parameters only.

Verified by `scripts/check_rider_noop.py`: after 3 iterations with seed 7,
agent weights, Adam optimizer state, `global_step`, and the full step-log
trajectory (rewards, active bonuses, room ids, episode ids, dones) are exactly
equal with and without `--passive-rnd`, while the rider itself produced a
live, non-constant signal. Re-run it after touching any of this code.

One observable (harmless) difference: the RND rider's obs-normalisation init
phase steps the envs with random actions before training (exactly like
`rnd.py`'s own init phase) and then re-resets with the run seed — so with
`--capture-video`, video episode *numbering* shifts, but the training stream
itself is unchanged (reseeding makes it deterministic).

## Hash mode: why `--hash-mode pool`

`scripts/simhash_occupancy_probe.py` (50k random-policy observations through
the real wrapper stack, seed 1) showed the original hash pipeline is
degenerate — but **collapsed**, not all-unique as `count_based.py`'s docstring
previously claimed:

| config | unique | singleton frac | mean n | max n | top-bucket share |
|---|---|---|---|---|---|
| `index` k=64 (old default) | 2,024 | 0.494 | 24.7 | 25,063 | **0.501** |
| **`pool16` k=64 (recommended)** | 3,127 | 0.388 | 16.0 | 1,492 | **0.030** |
| `pool16` k=128 | 22,223 | 0.713 | 2.2 | 425 | 0.009 |
| `pool16` k=32 | 379 | 0.198 | 131.9 | 4,919 | 0.098 |
| `pool8` k=64 | 797 | 0.236 | 62.7 | 5,889 | 0.118 |

Under `index` mode (stack-mean + 128 linspace-sampled pixels), a single bucket
absorbs *half of all visits* — the bonus signal is dominated by one mega-bucket.
`pool` mode (last stack frame, area-pooled to 16×16, Tang et al. 2017-style)
with k=64 gives healthy occupancy: no mega-bucket (top share 3%), meaningful
revisit counts (mean n=16), and a live singleton tail for genuinely new states.
The old `index` behaviour remains the default for backward compatibility;
production count runs should pass `--hash-mode pool --hash-pool-size 16`.

The projection matrix is seeded (`--seed` for the active counter,
`--rider-seed` for the rider). Passing the same seed + hash config in both
runs makes the hash function *identical* across runs, so count tables are
directly comparable. The RND rider cannot be made identical to an RND-driven
run's active RND (different weight draw), only architecturally equivalent —
treat it as a second instance of the same novelty-metric family.

## Per-step logging (`--step-log`)

`StepLogger` writes compressed `.npz` shards to
`{runs-dir}/{run_name}/step_log/steps_<first>_<last>.npz`, one row per
(rollout step, env):

| array | shape | meaning |
|---|---|---|
| `reward_ext` | (T, N) | extrinsic reward as trained on (post-clip) |
| `bonus_active` | (T, N) | driving method's raw bonus |
| `bonus_passive` | (T, N) | rider's raw bonus (NaN if no rider) |
| `room` | (T, N) | current room id (RAM byte 3), aligned with the bonus observation; −1 if unavailable |
| `episode_id` | (T, N) | per-env episode counter (restarts at 0 on `--resume`) |
| `done` | (T, N) | terminated ∨ truncated |
| `global_step` | (T,) | global step after each vector step |
| `norm_active` / `norm_passive` | (I,) | per-iteration RND normalisation divisor `sqrt(reward_rms.var)`; normalised bonus = raw / divisor. NaN for SimHash (unnormalised) |

Alignment: `room` and both bonuses describe the same `next_obs` returned by
that `envs.step()` call. Under gymnasium 1.x `NEXT_STEP` autoreset the row
following a `done` row is the reset observation (reward 0, action discarded by
the env); `RoomTracker` also reports `room` in reset infos so these rows have
valid room ids. Size: ~0.5 MB per 50-iteration shard at `num_envs=32`
(~25 MB per 10M steps).

## Small-artefact checkpointing (`--artifact-interval N`)

Saves to `{checkpoint-dir}/{run_name}/artifacts/` every N iterations plus at
the final iteration (and on rnd.py auto-stop). **No optimizer states, no
policy weights** — only what offline analysis needs to re-evaluate each bonus:

- `rnd_*.pt` / `rider_rnd_*.pt` — RND predictor + target state_dicts,
  obs/reward running stats (**≈15.7 MB** measured)
- `simhash_*.npz` / `rider_simhash_*.npz` — bit-packed count table, counts,
  projection matrix, hash config, β (**0.03–0.06 MB** measured at small scale;
  grows ~9 bytes compressed per unique bucket)

Full checkpoints (`--checkpoint-interval`) are unchanged, except they now also
carry the SimHash count table and (when a rider is enabled) the full rider
state, so `--resume` continues counts and rider training instead of resetting
them. `count_based.py` also gained the same `--resume` run_name recovery fix
as `rnd.py`/`ppo.py`.

## Launch commands

Count-driven run with passive RND rider (primary analysis run; SLURM wrapper:
`slurm/run_count_rnd_rider.slurm`):

```bash
python src/agents/count_based.py \
    --env-id ALE/MontezumaRevenge-v5 \
    --total-timesteps 50000000 --num-envs 32 --seed 1 \
    --hash-mode pool --hash-pool-size 16 --hash-dim 64 --exploration-coef 0.01 \
    --passive-rnd \
    --step-log \
    --artifact-interval 200 \
    --capture-video --record-room-discovery \
    --track --wandb-project montezuma-thesis \
    --runs-dir "$SCRATCH/runs" --videos-dir "$SCRATCH/videos" \
    --checkpoint-dir "$SCRATCH/checkpoints" --checkpoint-interval 100
```

RND-driven run with passive SimHash rider (optional, budget permitting —
sized like `slurm/run_rnd.slurm`'s modest 3M validation run). `--rider-seed 1`
deliberately matches the count run's `--seed 1` so the hash function is
identical across the two runs:

```bash
python src/agents/rnd.py \
    --env-id ALE/MontezumaRevenge-v5 \
    --total-timesteps 3000000 --num-envs 32 --seed 1 \
    --passive-simhash --rider-seed 1 \
    --rider-hash-mode pool --rider-hash-pool-size 16 --rider-hash-dim 64 \
    --step-log \
    --artifact-interval 50 \
    --capture-video --record-room-discovery \
    --track --wandb-project montezuma-thesis \
    --runs-dir "$SCRATCH/runs" --videos-dir "$SCRATCH/videos" \
    --checkpoint-dir "$SCRATCH/checkpoints" --checkpoint-interval 50
```

TensorBoard additions: `rider/*` scalars (raw/normalised bonus, fwd_loss or
unique_states, hash occupancy) and `charts/hash_*` occupancy metrics for the
active counter in `count_based.py` — watch `charts/hash_top_bucket_share`
(collapse) and `charts/hash_singleton_frac` (all-unique) against the sanity
criterion above.
