# Hyperparameter & Methodology Decisions

Chronological log of non-obvious hyperparameter or methodology changes, with the
empirical reasoning behind them. Intended as a source for the thesis
methodology/discussion section — commit messages capture *what* changed, this
file captures *why*.

---

## 2026-07-29 — Episode conditions aligned to the RND paper (5-min cap, no no-op starts)

**Change:** `make_env` (`src/agents/base.py`) passes
`max_num_frames_per_episode=18_000` to `gym.make` (5 min of emulator time =
4,500 agent steps; ALE's v5 default was 108,000 = 30 min) and `noop_max=0`
(was 30). Applies to every agent, since all build envs through `make_env`.
Shipped in PR #23.

**Why:** Matches the RND paper's official code — Burda et al.'s `make_atari`
applies `StickyActionEnv` but no `NoopResetEnv` and uses
`max_episode_steps=4500` (x4 frames = 18,000); per Machado et al. 2018,
sticky actions (v5's `repeat_action_probability=0.25`) *replace* random
no-op resets as the stochasticity source. Side benefits: a behaviourally
captured (TV-watching) agent cycles episodes ~6x faster instead of idling
out the 30-min clock, and same-seed resets are now byte-identical (a
deterministic "first frame", used by `scripts/visualize_preprocessing.py`).

**How to apply / re-evaluate:** Runs before/after the change are
episode-length-comparable except in the far tail (<0.2% of episodes exceeded
the new cap in the v2 PPO runs — measured; see the PPO-reuse entry below).
Typical episodes (~450-580 steps) never hit either cap.

---

## 2026-07-31 — PPO control runs reused across the env-config change

**Change:** No new PPO runs for the final noisy-TV matrix. The six completed
v2 HPC runs (`ppo_tv_off` / `ppo_tv_remote`, seeds 42/43/44, 20M steps) are
carried over as the no-intrinsic controls, although they ran under the
pre-2026-07-31 environment configuration (30-min episode cap, `noop_max=30`)
while the final RND batch uses the paper-faithful config (5-min cap =
18,000 frames, no no-op starts). Only the four RND conditions are re-run
(seeds 100/200/300, spaced >= num_envs so `seed+i` env seeds cannot collide).

**Why (time-constrained, empirically defended):** (1) The episode-cap
mismatch is measured to be inert for these runs — fewer than 0.2% of episodes
in any of the six PPO runs exceeded the new 4,500-agent-step cap (per-run:
14/41,015 · 50/34,413 · 21/43,534 · 19/38,951 · 73/40,101 · 21/35,057;
mean episode lengths 458–580 steps). (2) No-op starts affect only the first
<=30 raw frames (<=8 agent steps); sticky actions (p=0.25), the dominant
stochasticity source (Machado et al. 2018), are present in both configs.
(3) The `--update-proportion` fix does not apply to `ppo.py` (no predictor),
so the PPO runs are unaffected by the substantive algorithmic change.
(4) The PPO arms serve only as the no-intrinsic `tv_action_frac` floor and
descriptive exploration context; the primary behavioural null for P2 is
sham-remote, which IS re-run under the final config.

**How to apply / re-evaluate:** State the config split explicitly in the
methodology (one table row: v1/v2 + PPO = old env config; final RND batch =
paper config). If reviewers or results demand it, the two PPO arms can be
re-run later under the final config with the same sbatch pattern
(`run_ppo_tv.slurm`, TV_MODE=off/remote) — nothing else depends on them.

---

## 2026-07-31 — RND `--update-proportion` default: 0.25 → auto (`min(1, 32/num_envs)`)

**Change:** `src/agents/rnd.py` `--update-proportion` default changed from a
hard-coded `0.25` to `None` = auto-resolve via the RND paper's scaling rule,
`min(1, 32/num_envs)` (`resolve_update_proportion()`). Explicit values still
pass through unchanged. The resolved value is printed at startup and lands in
the logged/checkpointed args.

**Why:** Burda et al. 2018 define this hyperparameter *relative to a 32-env
baseline*: they kept 25% of experience *because they ran 128 envs* (32/128), to
hold the predictor's effective batch — and thereby the intrinsic-reward decay
rate — constant across env counts. CleanRL (and our copy of it) hard-codes
0.25, which is only paper-correct at 128 envs. Every completed noisy-TV
production run (v1 Jupyter batch at 21 envs, v2 HPC batch at 32 envs,
`update_proportion=0.25` confirmed in the logged hyperparameters) therefore
trained the predictor at ~1/4 of the paper-equivalent experience rate. Slower
predictor ⇒ slower novelty decay ⇒ the memorisation gap `G` stays positive
longer ⇒ **favorable to noisy-TV capture** — a systematic bias, not noise, so
it must be fixed before further runs and disclosed as a caveat for the
completed ones.

**How to apply / re-evaluate:** New runs need no flag changes —
`run_rnd_tv.slurm` passes no `--update-proportion`, so 32-env runs now resolve
to 1.0. Do **not** resume pre-fix checkpoints under the new default (their
stored args say 0.25; pass `--update-proportion 0.25` explicitly if such a
resume is ever needed). When re-running the TV matrix, prefer run seeds spaced
`>= num_envs` apart (e.g. 100, 200, 300) so the `seed + i` env-seed derivation
cannot collide across runs (31/32 env seeds collided between the v2 runs
42/43/44 — inert, since trajectories are policy-driven, but avoidable).

---

## 2026-07-13 — RND `ent_coef` raised 0.001 → 0.01

**Change:** `src/agents/rnd.py` default `--ent-coef` raised from `0.001` to `0.01`
(10x), matching the value already used in `count_based.py` and `ppo.py`.

**Why:** Diagnosis of the 10M-step production runs (both seeds, `rnd_10m_s1`/`s2`,
run 2026-07-08) found a closed feedback loop: episodes were short (mean ~522
steps, 95.8% under 1000) because the agent stayed in room 1 the entire run
(`charts/rooms_visited` = 1.0 for all ~19k episodes, `episodic_return` = 0.0
throughout). The RND predictor only ever saw a narrow band of near-spawn states,
so it fit that distribution almost immediately (`raw_intrinsic_rew_mean` collapsed
~20x within the first 9% of the training budget). With extrinsic reward
permanently 0 and intrinsic reward now negligible, `ent_coef=0.001` was too weak
to hold entropy up against PPO's objective, so the policy prematurely converged
(`losses/entropy` 2.890 → 0.209, `approx_kl`/`clipfrac` → 0 well before the run
ended) and never explored further. Full diagnosis detail: see the
`project_rnd_results` memory (auto-memory, not in this repo).

**How to apply / re-evaluate:** This is a first attempt at counteracting the
collapse, not a proven fix — re-run the same TensorBoard diagnosis (see
CLAUDE.md § Log Analysis) on the next production run's `losses/entropy` and
`charts/raw_intrinsic_rew_mean` curves before assuming it worked. If entropy
still collapses early, the next lever to pull is addressing *why* episodes are
short in the first place (death-averse shaping, sticky-action settings, or
periodic RND predictor resets), not just raising `ent_coef` further.

**Confidence downgraded, 2026-07-13 (later same day):** see the entry below
("RND vs PPO asymmetry") — `ent_coef=0.001` turns out to be the RND paper's own
published value, not a weak/arbitrary one, which undercuts the framing above.
Treat this change as an untested hypothesis, not a fix, until the investigation
below concludes.

---

## 2026-07-13 — RND vs PPO asymmetry: the `ent_coef` diagnosis doesn't fully hold up

**Finding:** Challenged (by the user) on whether the entropy-collapse diagnosis
above was correct, re-examination surfaced two things the original diagnosis
missed:

1. **`ent_coef=0.001` is the RND paper's own published Atari hyperparameter**,
   not a weak or arbitrary value — confirmed independently in
   `doc/10M-RND-run-failure-documentation.md` (on the `worktree-rnd-10m-failure-doc`
   branch, §4: every "algorithm" hyperparameter in `rnd.py`, including
   `ent_coef`, is an exact match to Burda et al. 2018 / CleanRL's
   `ppo_rnd_envpool.py`). A diagnosis that boils down to "the paper's own value
   is too weak" needs a lot more evidence than a single collapsed run.
2. **PPO — no RND, no intrinsic reward, no entropy-collapse feedback loop —
   outperformed RND on the exact same codebase**, despite half the training
   budget:
   | Run | Episodes | Nonzero-return episodes | Max room | Max return |
   |---|---|---|---|---|
   | `ppo_5m_s1` (5M steps) | 9,146 | 9 | 2 | 400 |
   | `rnd_10m_s1` + `rnd_10m_s2` (10M steps each) | 19,106 + 17,171 | **0** | 1 | 0 |

   (Verified directly from `runs/ALE/jupyterlab/events.out.tfevents...1425.0`
   (PPO) vs `...1423.0`/`...1424.0` (RND s1/s2) — not from logs or memory.)
   RND performing *worse than a baseline with no exploration bonus at all* is
   not what "entropy decayed a bit too fast" predicts. Something in the
   RND-augmented path looks actively harmful, not just insufficiently tuned.

**Follow-up, 2026-07-14 — bump reverted:** `src/agents/rnd.py`'s `--ent-coef`
default restored to `0.001`. The confidence downgrade above already established
that `0.001` is the RND paper's own published Atari value (Burda et al. 2018),
not an under-tuned placeholder, and that RND underperforming a plain PPO
baseline can't be explained by `ent_coef` alone — so shipping a permanent 10x
deviation from the paper on the strength of one collapsed run was premature.
`ent_coef=0.01` remains a legitimate variable in the matched-budget ablation
proposed in `doc/rnd-vs-ppo-asymmetry-investigation.md` §4.3 (variants B/D) —
reverting the default doesn't retire the hypothesis, it just stops it from being
silently on-by-default in production runs (`slurm/run_rnd.slurm`) that don't
explicitly request it.

**Also found, independently of the above, and not present in either of the
other worktrees' RND write-ups:** gymnasium 1.3.0's vector envs default to
`AutoresetMode.NEXT_STEP` (confirmed empirically with a CartPole boundary
trace, not just read from docs). Under this mode, the observation at the
actual terminal step is the *true final frame* (not a reset), and the
*following* step silently discards whatever action was chosen and returns the
reset observation with `reward=0`. All three agents (`ppo.py`, `count_based.py`,
`rnd.py` — identical pattern, confirmed via grep) mask the GAE bootstrap by
checking `done_buf[t + 1]`, which is the correct check for the *old*
`SAME_STEP`/gymnasium-0.x convention CleanRL was originally written against.
Under the actual `NEXT_STEP` behavior, this fires one step too early: it
correctly cuts the bootstrap between the real terminal reward and the terminal
frame, but *not* between the terminal frame and the next episode's fresh
start — so value bootstraps across two unrelated episodes at every episode
boundary, and the (silently-discarded-by-the-env) action chosen on the
terminal frame is still used as a real policy-gradient sample. This bug is
identical in all three agent files, so it does not by itself explain the
PPO-vs-RND asymmetry above — but it's a real, verified defect worth fixing
regardless of whether it's the main cause.

**Status: unresolved.** Neither the entropy-collapse framing (this file, first
entry above) nor the compute-budget framing (`doc/10M-RND-run-failure-documentation.md`,
§4, §7 — this repo runs at ~8x fewer envs and ~50x fewer steps than the paper)
explains why RND specifically underperforms a PPO baseline running on the same
under-scale budget. See the investigation brief this finding produced:
`doc/rnd-vs-ppo-asymmetry-investigation.md`.

---

## 2026-07-13 — `count_based.py` recording flags added

**Change:** Added `--record-room-discovery` and `--video-episode-interval` to
`count_based.py`'s CLI, mirroring `rnd.py`. Previously `count_based.py` only
exposed `--capture-video`, which — via `make_env()`'s default
`video_episode_interval=1` — silently recorded *every single episode* of env 0
rather than a periodic sample, and had no way to enable room-discovery-only
recording at all.

**Why:** Both algorithms are meant to be compared under the same evaluation
protocol (rooms explored + mean score, per CLAUDE.md § Evaluation Metrics), so
recording behavior needs to be symmetric across agents. The gap was only caught
because count-based production runs were about to be submitted via `slurm/run_count_based.slurm`
with `--capture-video` and would have generated one video per episode for the
whole 10M-step run.

---

## 2026-07-13 — `--record-room-discovery` and periodic recording now stack

**Change:** `base.py`'s `make_env()` previously treated `record_room_discovery`
and periodic `--video-episode-interval` recording as either/or (an `if/else`)
— passing `--record-room-discovery` silently disabled periodic sampling.
Changed to stack both: `NewRoomRecorder` is added *in addition to* the periodic
`RecordVideo` wrapper when `--record-room-discovery` is set, not instead of it.

**Why:** Today's production runs need both a periodic visual sample of
training progress and the sparse "new room reached" highlight reel from a
single (expensive, slurm-queued) run — resubmitting a run just to get the
other recording mode would waste GPU-hours. Verified via local smoke test that
a single run now produces both `rl-video-episode-N.mp4` and
`room_discovery/new_room_ep*.mp4` files.

---

## 2026-07-14 — `--resume` run_name fix, and collapse auto-stop added

**Change 1 — `--resume` run_name fragmentation fixed:** `rnd.py`/`ppo.py`
previously computed `run_name = f"{env_id}__{exp_name}__{seed}__{int(time.time())}"`
unconditionally, so every `--resume <ckpt>` invocation started a *new*
timestamped TensorBoard/W&B/checkpoint directory instead of continuing the
original run. Now, when `--resume` is passed, `run_name` is recovered from
the checkpoint's own path instead: `checkpoints/<run_name>/ckpt_XXXXXX.pt`
makes `run_name` recoverable as the checkpoint's parent directory, relative
to `--checkpoint-dir`. Note this needed `Path(...).relative_to(checkpoint_dir)`,
not just `.parent.name` — `run_name` itself contains `/` (`env_id` is
`ALE/MontezumaRevenge-v5`), so `.parent.name` alone silently drops the `ALE/`
segment and reintroduces the exact same fragmentation bug under a different
name. Caught via local `--sync-envs` verification before this was believed
fixed — see commit history for the two-step fix.

**Why it matters for the HPC rollout:** every production script now supports
`--resume $SCRATCH/checkpoints/<run_name>/ckpt_XXXXXX.pt` (see
`slurm/run_rnd.slurm`, `slurm/run_ppo.slurm`) as the intended way to extend a
walltime-truncated or intentionally-modest run into a longer one without
starting over. Without this fix, every chained resume would have fragmented
into a new run directory, breaking exactly that workflow.

**Change 2 — collapse auto-stop added:** Both `rnd.py` and `ppo.py` now
support `--auto-stop`/`--no-auto-stop` (default **on**): if a collapse
signature is sustained for `--auto-stop-patience` consecutive iterations, the
run logs a marker, writes a checkpoint, and exits with code `42` (distinct
from a normal `0` completion or a crash's nonzero code).

- `rnd.py`'s signature combines three conditions, matched directly to the
  diagnosis in `doc/10M-RND-run-failure-documentation.md`: entropy fraction
  (`entropy / ln(action_space_n)`) below `--auto-stop-entropy-frac` (default
  0.15), `raw_intrinsic_rew_mean` down at least `--auto-stop-intrinsic-drop`
  (default 10x) from its running peak this run, and both `approx_kl` and
  `clipfrac` below near-zero epsilons (`--auto-stop-kl-eps`,
  `--auto-stop-clipfrac-eps`) — i.e. PPO updates have become no-ops. All
  three together, not any single metric alone, since each one individually
  can dip transiently without a real collapse in progress.
- `ppo.py`'s signature is simpler (entropy fraction + near-zero
  `approx_kl`/`clipfrac` only — no intrinsic-reward term, since PPO has none)
  and its default thresholds (`--auto-stop-entropy-frac 0.10`,
  `--auto-stop-patience 150`) are **provisional** — PPO has no prior collapse
  incident on this codebase to calibrate against, unlike RND's thresholds.
  Re-check after the first production `run_ppo.slurm` run whether they fired
  correctly (a real collapse) or too eagerly (should be loosened).

**Why:** the prior 10M-step RND runs burned ~91% of their walltime budget
after collapsing in the first ~9% — see
`doc/10M-RND-run-failure-documentation.md`. With `slurm/run_rnd.slurm` now
intentionally scoped as a modest, checkpoint-chainable validation run rather
than a single large speculative commitment (see that script's header), an
unattended collapse detector matters more, not less: a smaller run that
silently collapses and keeps going wastes proportionally the same fraction of
GPU-hours as the 10M-step failure did. Verified locally (`--sync-envs`,
forced trigger via extreme thresholds) that both agents write a final
checkpoint and exit `42` cleanly rather than crashing or hanging.

**How to apply / re-evaluate:** `sacct -j <id> --format=ExitCode` distinguishes
auto-stop (`42:0`) from normal completion (`0:0`) after the fact — see
`doc/hpc-onboarding.md` §5, §7. An auto-stop firing on `run_rnd.slurm` at
paper-default hyperparameters would confirm the collapse persists independent
of `ent_coef`, pointing back at the still-open `AutoresetMode.NEXT_STEP`
GAE-masking bug (documented above, explicitly out of scope for this round of
changes) as the next lever, not more compute.

---

## 2026-07-19 — Noisy-TV experiment feature (`--tv-mode`, the thesis's main experiment)

**Change:** Added `NoisyTVWrapper` to `base.py` and `--tv-*` flags to `rnd.py`
and `ppo.py` — a synthetic stochastic patch ("noisy TV") injected into
observations, with an action-triggered `remote` mode as the primary thesis
condition. Full design and experiment matrix: `doc/noisy-tv-experiment.md`.
Non-obvious calls made, and why:

- **Injection point is *forced*, not just chosen:** the wrapper sits between
  `AtariPreprocessing` and `FrameStackObservation` because `AtariPreprocessing`
  never consumes the wrapped env's observations — it calls
  `ale.getScreenGrayscale` directly into internal buffers, so a raw-frame
  observation wrapper below it has no effect on the processed obs. Injecting
  at 84×84 also gives exactly one noise sample per agent step, with no
  max-pool/INTER_AREA attenuation of the noise statistics.
- **`remote` over `static` as the primary condition:** a screen-fixed,
  unavoidable patch cannot produce *behavioral* capture — only signal
  degradation. The paper's thought experiment is about agent-controllable
  stochasticity, so the primary condition adds a 19th NOOP-mapped action that
  resamples the patch, with `sham-remote` (extra action, no patch) as the
  action-space-matched control. Montezuma's minimal set is all 18 actions, so
  there is no free action to overload; extending is cheap because nothing
  hardcodes 18 (actor head and auto-stop both read `single_action_space.n`).
- **Sticky-action asymmetry (methodology note):** ALE's 0.25 sticky actions
  apply to the *executed* action, but the patch resample triggers on the
  *chosen* action pre-mapping — the TV remote is the only perfectly reliable
  action in the game. Favorable to the hypothesis; disclose in the thesis.
- **NEXT_STEP autoreset interaction:** on the boundary step, vector envs call
  the sub-env's `reset()` and never `step()` (verified in gymnasium 1.3.0
  source), so the discarded boundary action can't resample the patch. Instead
  every episode start draws a fresh patch (wrapper `reset()`), identical
  across TV conditions. The known GAE-masking bug (2026-07-13 entry) is
  unaffected — the wrapper changes neither rewards nor done flags.
- **TV RNG is not checkpointed:** noise is iid per resample and the RNG is
  derived from the per-sub-env reset seed, so a `--resume` replays the start
  of the noise stream — distributionally identical, and ALE state isn't
  checkpointed either. Instead of RNG checkpointing, `--resume` now *aborts on
  mismatched `--tv-*` flags* (`check_tv_args_match` in `base.py`): `remote`
  and `sham-remote` load into identical network shapes, so a wrong-mode resume
  would otherwise silently swap the experiment mid-run.
- **Auto-stop is effectively neutralized in patched RND runs, by design:** a
  working TV keeps `raw_intrinsic_rew_mean` elevated, so the ≥10x
  intrinsic-drop conjunct never holds. That's correct behavior — it also means
  auto-stop cannot fire *spuriously* in a captured run (entropy collapsed onto
  the TV action, KL/clipfrac ≈ 0, but intrinsic still high). `off`/`sham-remote`
  runs keep the full baseline guard.
- **Occlusion diagnostic logs its own "full" value**
  (`charts/intrinsic_rew_full_diag`) rather than reusing
  `charts/raw_intrinsic_rew_mean`: the step-wise metric is computed with the
  *previous* iteration's `obs_rms`, the diagnostic with the post-update one —
  comparing across normalizers would confound `charts/tv_intrinsic_share`.
- **Byte-identical-when-off is verified by hash, not by training runs:**
  `scripts/check_noisy_tv.py --hash-off` hashes a 500-step fixed-action
  trajectory; it matched exactly between pre-TV `main` and the feature branch.
  Full training runs can't verify this because the obs-norm init loop's
  `envs.action_space.sample()` uses the space's own entropy-seeded RNG and was
  already nondeterministic across runs (pre-existing; a one-line
  `envs.action_space.seed(args.seed)` would fix it if full-run determinism is
  ever wanted — out of scope here).
- **Patch geometry was corrected by the smoke test: default is the full
  12×84 HUD strip, not a 12×12 corner.** The first smoke run (200k steps,
  12×12 patch at the top-right HUD corner) showed `tv_intrinsic_share` ≈ 0 —
  occluding the patch didn't change prediction error. A checkpoint probe
  found why, and it's mechanical, not a pipeline bug: the noise *was* in the
  RND input, correctly normalized (region std ≈ 1.01 post-obs_rms), but fully
  re-randomizing a 144-pixel patch (2% of the frame) moves the random target
  network's output by only ~7% per-dim ≈ ~1% of total prediction error.
  Since per-pixel obs_rms normalization standardizes *any* stationary noise
  distribution to ~unit variance, amplitude is not a usable lever — **area
  is the only stimulus-strength knob**, and the playfield boundary (rows ≥12
  in every room) caps safe height at 12 rows. Hence `--tv-size` became
  `H W` (default `12 84` at `0 0`): 1008 px, 7× the area, covering the
  score/lives display (HUD, not gameplay; score-change pixel novelty gets
  masked in TV runs but not baselines — negligible at Montezuma's scoring
  frequency, disclosed in `doc/noisy-tv-experiment.md`).
- **Pre-existing CPU-only logging quirk found while probing (not fixed
  here):** on CPU, `intr_buf.cpu().numpy()` returns a *view*, so the later
  in-place `intr_buf /= sqrt(reward_rms.var)` retroactively rewrites
  `curiosity_np` — `charts/raw_intrinsic_rew_mean` then logs the
  *normalized* value (observed: ~0.08 while the true raw error was ~65 with
  `sqrt(reward_rms.var)` ≈ 808). CUDA runs are unaffected (`.cpu()` copies),
  so all production/GPU numbers in prior docs remain valid; only local
  `--sync-envs` CPU smoke runs mislabel this one chart. Fix would be a
  one-line `.copy()`; left untouched in this round to keep the TV diff
  behavior-preserving (the hash check pins the off-path byte-identical).
- **Cosmetic:** the `--overlay-video` bar meter's `main_metric_range=(0, 0.3)`
  was calibrated against collapsing baselines; in TV runs normalized intrinsic
  sits near ~1 and the bar pins full. Left unchanged so baseline overlays stay
  comparable.

**How to apply / re-evaluate:** run the matrix in `doc/noisy-tv-experiment.md`
via `slurm/run_rnd_tv.slurm` / `run_ppo_tv.slurm`. Before any production
submission, `scripts/check_noisy_tv.py` must pass, and smoke expectations are
listed in that doc. The capture verdict rests on `charts/tv_action_frac`
(remote ≫ 1/19, sham ≈ 1/19) plus `charts/tv_intrinsic_share`.

---

## 2026-07-22 — NEXT_STEP autoreset GAE-masking bug fixed

**Change:** The `AutoresetMode.NEXT_STEP` GAE-masking defect flagged as "still-open"
in the 2026-07-13 "RND vs PPO asymmetry" entry (and referenced by the 2026-07-14
auto-stop note and the noisy-TV entry) is now fixed in all three agents
(`ppo.py`, `count_based.py`, `rnd.py`).

**Empirical confirmation (not just from the doc):** a CartPole boundary trace under the
repo's actual gymnasium 1.3.0 `SyncVectorEnv`, mimicking each agent's exact buffer
bookkeeping, reproduced the mechanism. At the genuine terminal step the env returns the
**true final frame** (`terminated=1`); the *following* step **discards** the sampled
action and returns the reset obs with `reward=0`, `terminated=0`. With
`done_buf[t]=next_done`, that fake step is **exactly** `done_buf[t]==1` — so `done_buf`
marks the final frame, leaving the actual reset frame unmarked one step later. CleanRL's
`nextnonterminal = 1 - done_buf[t+1]` then bootstraps V(final frame of ep N) → V(start of
ep N+1) at the fake step, and the discarded terminal-frame action is still fed to the
policy gradient.

**Fix (GAE/rollout masking only):**
- New shared helper `agents.base.compute_gae(...)` walls off every fake step
  (`done_buf[t]==1`): it carries no advantage and blocks advantage flow into earlier
  real steps. All three agents call it (rnd twice: `episodic=True` extrinsic,
  `episodic=False` intrinsic). For the extrinsic stream this is behaviour-preserving on
  every kept sample (the `nextnonterminal=0` firewall at the genuine terminal step
  already isolated the corruption to the fake step); for rnd's non-episodic intrinsic
  stream the wall is what actually stops the fake frame bridging two episodes' returns.
- New shared `agents.base.masked_mean(...)`; the update loops exclude fake steps from the
  policy/value losses and from `approx_kl`/`clipfrac` (advantage normalisation is over
  kept samples too). The RND predictor loss (`fwd_loss`) is **not** masked — the final
  frame is a real observation, valid for novelty training. (`entropy.mean()` used for the
  auto-stop `entropy_frac` diagnostic is left unmasked to preserve the calibrated
  thresholds; fake steps are rare, so the shift is negligible.)

**Why it is NOT expected to flip the RND-vs-PPO asymmetry:** the bug and the fix are
identical across all three agents, so both were distorted the same way. This makes the
matched-budget comparison *clean* — it is not a hypothesised cause of the asymmetry.

**Scope:** GAE + update-loop loss masking + two `base.py` helpers only. No hyperparameter
change; no env/wrapper change (`make_env`, the wrapper stack and `NEXT_STEP` are
untouched, so logging/video/room-tracking are unaffected — verified: `episodic_return` /
`rooms_visited` still log). The tempting one-liner alternative — switching the vector env
to `AutoresetMode.SAME_STEP` — was **rejected empirically**: under `SAME_STEP`,
`RecordEpisodeStatistics` returns episode data nested under `infos["final_info"]`
(+`infos["_final_info"]`) instead of top-level `infos["_episode"]` (the gymnasium-0.x
`final_info` pattern CLAUDE.md flags as broken), which would silently stop the agents'
episode logging.

**Verification:** `tests/test_gae_autoreset.py` — deterministic, no ROMs, self-runnable
(`python tests/test_gae_autoreset.py`) or via pytest. It pins the fake-step wall, the
terminal no-bootstrap, the intrinsic no-cross-episode-leak, the NEXT_STEP env signature,
and includes FAIL-before guards that assert the old inline recursion was buggy. Short CPU
smoke runs of all three agents (with real Montezuma episode boundaries) train without
NaN/shape errors and still log episodes.

---

## 2026-07-22 — Matched-budget comparison configured on the fixed code

**Change:** Configured (and pre-flighted, but not yet submitted — see below) one clean
matched-budget SLURM matrix to characterize, on the `NEXT_STEP` bug-fixed code, whether
(a) the early entropy/intrinsic collapse and (b) RND underperforming PPO survive the fix.
This is a fixed-budget characterization, **not** a paper-scale reproduction. Full package:
`doc/matched-budget-submission.md`.

**The matrix (8 cells), matched across agents:** `{PPO, RND, count-based} × {seed 1, 2}`
at each agent's standard `ent_coef` (PPO/count 0.01, RND 0.001), plus an **RND
`ent_coef=0.01` ablation** (`exp_name=rnd_ent01`, seeds 1,2) to isolate whether the
ent_coef mismatch — rather than the method — drives the RND<PPO gap.

**Chosen matched knobs, with reasoning:**
- **num_envs = 32** — already the scripts' value, in the requested 32–64 band, matches
  `--cpus-per-task=32` 1:1; 64 needs a confirmed ≥64-core/GPU partition + V100 memory
  headroom (not confirmable off-cluster), so 32 is the safe matched value (override
  `NUM_ENVS=64`).
- **total_timesteps = 3,000,000** — the cheap scale
  `doc/rnd-vs-ppo-asymmetry-investigation.md` §4.3 calls for (not the 10M/50M the scripts
  defaulted to, which that doc warns against); collapse historically appears <1M steps, so
  3M answers (a) with margin and gives a first read on (b), extendable via `--resume`.
- **anneal_lr = ON** — paper/CleanRL default, kept matched to avoid a confound; disabling
  it is a separate lever (failure doc §6), not this matrix.
- **auto-stop = ON** — collapsed runs exit 42 early (fast collapse yes/no, bounded cost).

**Implementation:** `slurm/run_{ppo,rnd,count_based}.slurm` gained behaviour-preserving
`--export` overrides (`NUM_ENVS`, `TOTAL_TIMESTEPS`, `ENT_COEF`, `EXP_NAME`, `ANNEAL_LR`,
`RESUME_FROM`); unset, each script is unchanged. `run_count_based.slurm`'s
`--gres=gpu:a100:1` corrected to generic `gpu:1` (an a100 request never schedules on a
V100 allocation). New `slurm/submit_matched_matrix.sh` fires the 8 cells with matched
`--export` values and logs job IDs + the commit hash.

**Commit / precondition:** the matrix runs the fix branch
`worktree-fix-next-step-gae-masking` (PR #14) — it is **not** on `main`, so the cluster
checkout must `git checkout` that branch and pass `tests/test_gae_autoreset.py` (7/7)
before submitting. The submit script records the exact commit hash it submits from.

**Status — submission pending cluster access.** All 8 cells' commands were pre-flighted on
the fixed code off-cluster (parse args, start training, exit 0, distinct run dirs). The
actual `sbatch` could **not** be run from the dev machine (no SLURM client, no cluster
route). Run `slurm/submit_matched_matrix.sh` on the login node after the STEP 0 checkout +
the STEP 2 `salloc` SPS smoke; site-specific placeholders (partition, real V100 gres type,
module names, `$SCRATCH`, walltime vs QOS cap, `WANDB_API_KEY`) must be confirmed there
first (`doc/hpc-onboarding.md` §1).

---

## 2026-07-25 — "Stuck in room 1" 20M/32-env runs: no regression, no bug found

**Decision:** Close the "the 20M/32-env noisy-TV runs are broken / regressed"
investigation **without a code fix**, because the premise does not survive its
own reference data. Full evidence in `doc/run-analysis.md` (TASK A) and
`doc/regression-findings.md` (B–F). Summary of what was concluded and why:

- **No regression.** The `analysis/HPC-Runs/` runs (32 env, 20M) reproduce the
  `analysis/Jupyter-Pod-Runs/` reference runs (21 env, 10M) curve-for-curve —
  entropy decays to ~0.3–0.6, `episodic_return`≈0, exploration intermittently
  reaches room 2 in `off` and never in `remote`/`sham`. The **entire**
  hyperparameter delta old→new is `num_envs` 21→32 and `total_timesteps` 10M→20M
  (extracted from the embedded `hyperparameters/text_summary` of each event
  file). The brief's expected deltas were wrong: env count was **21→32, not
  8→32**, and `update_proportion` was **unchanged at 0.25** (not newly broken).
- **Not the `NoisyTVWrapper` (A4, decisive).** No-patch arms (`off`,
  `sham-remote`) fail *identically* to patched arms (`remote`, `static`) in both
  run sets — `off` is actually the *best* arm and `sham` the worst, uncorrelated
  with patch presence. The wrapper was present in **both** run sets (added
  07-19), so "wrapper inserted between old and new" is false; the only code delta
  is the 07-22 NEXT_STEP GAE fix, which is behaviour-preserving and confirmed
  neutral by the identical curves.
- **Premise partly inaccurate.** The new `off` run **does** leave room 1 (room 2
  at ~4M, earlier than the old `off` s2 at ~6.4M), and a live random policy does
  **not** leave room 1 in 4000 steps — so `off` is not "below a random policy".
  Burda's reference numbers ("random finds the key every few 100k steps", "RND
  finds >half the rooms") are at ~2e9 frames; these runs are 8e7 frames (~25×
  fewer), where even Burda's RND is in the first rooms.
- **Metric verified sound (TASK D).** `RoomTracker` reads RAM byte 3 correctly
  and threads `rooms_visited` through the stack (live-tested); it registers 2 in
  the runs that reached room 2, so it is not stuck-at-1 blind. The agent is
  genuinely confined.
- **Implementation faithful (TASK E).** Intrinsic non-episodic (`episodic=False`),
  return-std (not reward-std) normalisation, obs-norm init by random rollout, and
  the GAE fix are all correct. The "~20× intrinsic collapse" is the predictor
  fitting the narrow room-1 distribution (a self-reinforcing symptom of
  confinement), not a defect.

**Root cause recorded:** a pre-existing, cross-algorithm (PPO is stuck too)
exploration weakness driven by under-scale vs Burda + early entropy decay at the
paper's `ent_coef=0.001`; **not** a bug and **not** introduced by any of the four
suspected changes.

**Actionable follow-up (evidence-gated, not applied):** the one real paper
mismatch is `update_proportion=0.25` at 32 envs (Burda used keep-prob 0.25 only
at 128 envs). Test `--update-proportion 1.0` in a ≤3M probe (V2 in
`regression-findings.md` § Verification plan) **before** changing the default or
spending any 20M budget — do not ship it as a fix on the strength of the argument
alone. Reference target: escape room 1 well before 10M (so 3M probes suffice).

**Hygiene actions for the next cluster runs** (surfaced by this analysis, not yet
done): preserve **checkpoints + videos** alongside the event files (enables the
post-hoc memorisation-gap probe F1 and direct video review A7 — both blocked here
by their absence), and record **`pip freeze` + the commit SHA** per run (would
close the execution-environment bisect arm mechanically instead of by inference).

**Scope note:** no source files changed for this diagnosis — only `CLAUDE.md`
(Thesis-context section) and the three docs above.
