# The Noisy-TV Experiment

Main experiment of the thesis: does the noisy-TV problem manifest for RND on
`ALE/MontezumaRevenge-v5`?

## Hypothesis under test

Burda et al. 2018 ([arXiv:1810.12894](https://arxiv.org/abs/1810.12894)) argue
RND resists stochasticity traps because its target network is a *deterministic*
function of the observation — unlike next-frame-prediction curiosity, there is
no irreducibly unpredictable target. But that argument only says the target is
learnable *in principle*: a stochastic observation source still makes every
frame novel, and the predictor can only stop paying intrinsic reward for it by
learning to generalize the target function over the noise manifold, not by
memorizing frames. Whether a Nature-CNN-scale predictor actually does that at
84×84, before the policy gets captured, is an empirical question. This
experiment injects a synthetic "TV" (noise patch) into observations and
measures both:

- **Signal capture** — does intrinsic reward stay dominated by the patch
  (predictor never catches up on that region)?
- **Behavioral capture** — does the agent learn to *seek* the stochasticity
  (press the TV remote) instead of exploring the game?

## Design

`NoisyTVWrapper` (`src/agents/base.py`) sits between `AtariPreprocessing` and
`FrameStackObservation` and stamps a square patch of iid per-pixel uniform
noise into the processed 84×84 frame, once per agent step. It must sit there:
`AtariPreprocessing` reads the screen directly from the ALE object, so a
wrapper below it never influences the processed observations. Both the policy
(full 4-stack) and the RND predictor/target (newest frame) see the patch —
the trap is visible, hence learnable.

Modes (`--tv-mode`, on `rnd.py` and `ppo.py`):

| Mode | Action space | Patch | What it tests |
|---|---|---|---|
| `off` | Discrete(18) | none | baseline (wrapper not constructed; byte-identical to pre-TV code — verified by `scripts/check_noisy_tv.py --hash-off`) |
| `static` | Discrete(18) | resampled every `--tv-refresh-every` steps | signal degradation only — the patch is unavoidable, so no behavioral trap exists |
| `remote` | Discrete(19) | resampled when action 18 (NOOP-mapped) is chosen | **primary condition** — agent-controllable stochasticity, the paper's actual thought experiment |
| `sham-remote` | Discrete(19) | none | action-space-matched control: separates "extra action" effects from "the TV" |

`remote` is the primary thesis condition (methodology chapter: remote vs.
sham-remote vs. off, with static as the signal-degradation ablation), because
static cannot produce behavioral capture at all. Montezuma's minimal action
set is all 18 actions, so the remote is an *added* action rather than an
overloaded one; sham-remote controls for the action-space change. Note one
asymmetry worth a line in the methodology: sticky actions (0.25) operate on
the *executed* ALE action, but the resample triggers on the *chosen* action —
the TV remote is the only perfectly reliable action in the game, which if
anything makes the trap easier to learn (favorable to the hypothesis).

Default geometry: **the full HUD band — 12×84 at (0, 0)** of the 84×84 frame
(`--tv-size H W`, `--tv-position ROW COL`). Two empirical findings fixed this:

1. The playfield occupies rows ≥12 in every room, so the patch may not extend
   below row 11; the HUD band (score/lives, raw rows 0–30) is the only
   gameplay-irrelevant region. The strip covers the score/lives display —
   HUD, not gameplay (reward comes from the emulator, not pixels); the minor
   side effect is that score-change pixel novelty is masked in TV runs but
   not baselines, negligible at Montezuma's scoring frequency.
2. **Patch area is the only stimulus-strength lever.** RND's per-pixel
   obs_rms normalization tames *any* stationary noise distribution to ~unit
   variance, so amplitude cannot be raised; a checkpoint probe of a smoke run
   with a 12×12 patch showed fully re-randomizing it moves the target
   network's output by only ~7% per-dim and total prediction error by ~1% —
   far too weak to ever dominate the intrinsic signal. 12×84 = 1008 px (14%
   of the frame, 7× the area) is the largest safely placeable stimulus.

The patch is composited into `render()` output too, so it is visible in every
recorded video (in TV runs it covers the on-screen score display; logged
`episodic_return` is unaffected).

Everything else stays at `rnd.py`'s paper-matched defaults (`--int-coef 1.0`,
`--ext-coef 2.0`, `ent_coef 0.001`, sticky actions 0.25;
`--update-proportion` auto-resolves to `min(1, 32/num_envs)` since 2026-07-31 —
see `doc/decisions.md` and the caveat below).

## Metrics to read

- `charts/tv_action_frac` (remote/sham-remote only) — fraction of chosen
  actions that press the remote. Chance = 1/19 ≈ 0.053. Sustained ≫ chance in
  `remote` but not `sham-remote`/PPO = **behavioral capture**. The headline
  metric.
- `charts/tv_intrinsic_share` — occlusion diagnostic (every
  `--tv-diag-interval` iterations): 1 − occluded/full intrinsic reward when
  the patch region is mean-imputed. High in `remote`/`static`, ≈0 in
  `off`/`sham-remote` (those runs give the calibration floor — the imputed
  input is off-distribution for the predictor, so read the share as a trend,
  not an exact decomposition). Companions: `charts/intrinsic_rew_full_diag`,
  `charts/intrinsic_rew_occluded`.
- `charts/raw_intrinsic_rew_mean` — with a TV it should stay elevated instead
  of collapsing ~20x as in the historical baseline runs.
- **Known short-horizon behavior (pre-registered from the smoke run):** in
  `static` mode with per-step resampling, `tv_intrinsic_share` started at
  0.10–0.21 and decayed to ≈0 within ~100 iterations (~100k steps) — trained
  on fresh noise every step, the predictor learns noise-*invariance*.
  `remote` mode differs mechanically: the patch mostly repeats between
  presses, so memorizing current content beats invariance and each press
  should produce a fresh-noise error bump. Whether invariance also wins at
  10M-step scale (RND resists the trap) or capture emerges is exactly the
  experiment's question — read the share's *trajectory*, not one value.
- `charts/rooms_visited`, `charts/episodic_return` — the behavioral *cost* of
  capture: remote vs. off and vs. sham-remote deltas.
- Auto-stop bookkeeping: in `remote`/`static` the ≥10x intrinsic-drop conjunct
  of the collapse signature should never hold, so auto-stop is effectively
  neutralized and the run spends its full budget (exit 0) — that trajectory is
  the data. In `off`/`sham-remote` it behaves exactly like the baseline
  (exit 42 = the known collapse reappeared).

## Experiment matrix (10M steps / 8 envs, matching the historical baselines)

Launch via `slurm/run_rnd_tv.slurm` / `slurm/run_ppo_tv.slurm`
(`sbatch --export=ALL,SEED=1,TV_MODE=remote slurm/run_rnd_tv.slurm`) or the
matrix cell in `training-runs.ipynb`. Runs launched after 2026-07-31 are named
`{algo}_tv_{mode}_{RUN_TAG}` with `RUN_TAG` defaulting to `paper` (the
paper-faithful config), so they are distinguishable from the earlier batches
at a glance; export `RUN_TAG=""` for the legacy naming.

| # | Agent | TV_MODE | Seeds | Status | Purpose |
|---|---|---|---|---|---|
| 1 | rnd | off | 1, 2 | core | fresh baseline on current code (old `rnd_10m_s1/s2` predate code changes) |
| 2 | rnd | remote | 1, 2 | core | **primary**: behavioral + signal capture |
| 3 | rnd | sham-remote | 1, 2 | core | action-space-matched control |
| 4 | rnd | static | 1 (+2 if budget) | core-secondary | signal degradation without behavioral channel |
| 5 | ppo | remote | 1 | core | no-intrinsic control: capture should not occur |
| 6 | ppo | off | 1 | optional | fresh PPO reference |
| 7 | rnd | remote, `--tv-size 6/18` | 1 | optional | patch-size sensitivity |

Budget: 7 core runs × ~15–19 h (prior runs sustained SPS 144–183) ≈ 110–135
GPU-h.

### Run batch history

| batch | agent code | steps / envs | seeds | env config | update_prop. | run names |
|---|---|---|---|---|---|---|
| v1 (Jupyter, ~2026-07-20) | pre-fix | 10M / 21 | 1, 2 | legacy (30-min cap, no-ops 30) | 0.25 | `{algo}_tv_{mode}__{seed}` |
| v2 (HPC, ~2026-07-26) | pre-fix | 20M / 32 | 42, 43, 44 | legacy | 0.25 | `{algo}_tv_{mode}__{seed}` |
| SCHWERT (HPC, 2026-07-31→) | paper-faithful | 20M / 32 | 100, 200, 300 | paper (5-min cap, no no-ops) | auto → 1.0 | `rnd_tv_{mode}_SCHWERT__{seed}` |

SCHWERT re-runs the four RND conditions only; the six v2 PPO runs are carried
over as the no-intrinsic controls (see caveats below and `doc/decisions.md`).
Seeds are spaced >= num_envs so `seed+i` env seeds cannot collide across runs.

## Analysis checklist

1. `tv_action_frac` over training for #2 vs #3 vs #5 — the capture plot.
2. `tv_intrinsic_share` for #2/#4 vs the #1/#3 floor — the signal plot.
3. `rooms_visited` / `episodic_return` — #2 vs #1 and vs #3 (cost of capture;
   #3 vs #1 isolates the pure action-space effect).
4. `raw_intrinsic_rew_mean` — TV runs vs. the historical collapse curve.
5. Exit codes per run (`sacct --format=ExitCode`): 42 in off/sham = baseline
   collapse reproduced; 42 in remote/static = unexpected, investigate.
6. Spot-check videos: the patch flickers every step in static, changes only on
   presses in remote; in captured runs expect the avatar mostly idle while the
   patch flickers.

## Caveats to carry into the thesis write-up

- **The PPO control runs are reused from the pre-2026-07-31 environment
  configuration** (30-min episode cap, `noop_max=30`), while the final RND
  batch runs the paper-faithful config (5-min cap, no no-op starts). Verified
  benign: fewer than 0.2% of episodes in any of the six PPO runs exceeded the
  new 4,500-step cap (means 458--580 steps; worst run: 73 of 40,101 episodes),
  no-op starts affect only the first <=8 agent steps while sticky actions
  (p=0.25, present in both configs) dominate stochasticity, and the
  update-proportion fix does not apply to PPO (no predictor). The PPO arms
  serve only as the no-intrinsic `tv_action_frac` floor and descriptive
  exploration context; the primary behavioural null (sham-remote) is re-run
  under the final config. See `doc/decisions.md` 2026-07-31.
- **All pre-2026-07-31 runs trained the RND predictor ~4x slower than
  paper-equivalent.** The hard-coded `--update-proportion 0.25` is Burda et
  al.'s value *for 128 envs* (their rule pins the predictor's effective batch
  to a 32-env baseline, i.e. keep 32/num_envs of the experience); at our
  21/32 envs the paper-faithful value is 1.0. Slower predictor = slower
  novelty decay = favorable to capture — disclose when interpreting the v1/v2
  batches; fixed for subsequent runs (auto rule in `rnd.py`,
  `doc/decisions.md` 2026-07-31).

- The known gymnasium `NEXT_STEP` autoreset / GAE-masking bug
  (`doc/decisions.md`, 2026-07-13) is present in all conditions equally; the
  TV feature neither fixes nor worsens it.
- TV noise RNG state is not checkpointed: a `--resume` re-seeds the noise
  stream (distributionally identical iid noise; ALE state isn't checkpointed
  either). `--resume` refuses mismatched `--tv-*` flags.
- One free patch resample happens at every episode start (autoreset calls
  `reset()`), identical across TV conditions.
