# Hyperparameter & Methodology Decisions

Chronological log of non-obvious hyperparameter or methodology changes, with the
empirical reasoning behind them. Intended as a source for the thesis
methodology/discussion section — commit messages capture *what* changed, this
file captures *why*.

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
