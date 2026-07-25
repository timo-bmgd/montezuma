# Regression findings — room-1 confinement of the 20M/32-env noisy-TV runs

Companion to `doc/run-analysis.md` (TASK A). Read that first: it establishes,
from the run artifacts alone, that the "20M/32-env runs are stuck in room 1"
symptom is **not a regression and not the noisy-TV wrapper**. This document
records the root-cause conclusion (B, C), the metric verification (D), the
implementation audit (E), and the remaining implementation gaps (F).

---

## Root cause (single statement)

**There is no single introduced bug, and no regression.** The 32-env / 20M-step
RND runs reproduce, curve-for-curve, the behaviour of the 21-env / 10M-step
"reference" runs they are being compared against: entropy decays to ~0.3–0.6,
extrinsic return stays ~0, and exploration intermittently reaches room 2 in the
`off` arm and never in `remote`/`sham`. The room-1 confinement is a
**pre-existing, cross-algorithm (PPO included) exploration weakness**, whose
proximate driver is **under-scale relative to Burda et al.** (32 vs 128 envs;
8e7 vs ~2e9 frames) at which **RND's intrinsic signal provides essentially no
exploration lift over a random policy**: the new PPO-remote baseline keeps its
policy *near-uniform the whole run* (`entropy`≈2.89 of ln(19)=2.94, ent_coef=0.01,
no collapse) and is nonetheless *equally* confined (max room 2, return ~0) — so
the room-1 confinement is **not** caused by RND's entropy decay. RND *does* decay
to a near-deterministic room-1 policy (self-reinforcing: predictor fits the
narrow room-1 distribution → intrinsic reward flattens → weak pull to explore),
but that is the *endpoint*, not the cause — a high-entropy, near-random agent
reaches room 2 exactly as rarely. The one paper-config mismatch that could be
*making RND weaker than it needs to be* — `update_proportion=0.25` at 32 envs
instead of 1.0 — is real and worth fixing, but it was identical in the old runs,
so it is **not** the cause of any change in behaviour.

The reference-value alarm in the brief ("stuck at 20M ⇒ below a random agent ⇒
a bug") **does not hold** once the numbers are put on the same footing:
- Burda's "random agent finds the key every few hundred thousand steps / RND
  finds >half the rooms" are quoted at their scale (128 envs, ~2e9 frames). Our
  runs are 8e7 frames — ~25× shorter. At Burda's *own* early-frame budget, RND
  is still in the first rooms.
- The premise "the `off` baseline underperforms a random policy" is **false as
  stated**: the new `off` run reaches room 2 at ~4M (max return 400), and a live
  random policy here does *not* leave room 1 in thousands of steps (TASK D).
  RND-off is at least matching a random policy, not below it.

What *is* fair to say: RND on this codebase is **weak and high-variance** (one
`off` seed reaches room 2, another never does), and it does not convincingly
beat PPO — an open characterisation question that predates all four suspected
changes and is orthogonal to the noisy-TV experiment's validity.

---

## B — Candidate known-good commit

TASK A settled the regression question, so this is reported for completeness
rather than used to drive a bisect.

- **The "known-good" reference runs are the `analysis/Jupyter-Pod-Runs/` set**,
  produced **2026-07-20 / 07-21** (embedded run-name timestamps 1784548767 and
  1784631991). They ran on `main` as it stood then — after the noisy-TV feature
  merged (PRs #11/#12, commits `9c0e51a`/`ca9dbbd`, 07-19/07-20) but **before**
  the NEXT_STEP GAE-masking fix (`7e3d2d7`, merged via #16 `d60c671`, 07-24).
  Best single-commit estimate: **`ca9dbbd` … `dd09b4a`** (2026-07-20).
  Confidence: **high** on the date window (run-name epochs + commit dates);
  medium on the exact SHA (no commit hash is recorded in the event files — the
  `--track`/W&B path that would log it was disabled, `track=False`).
- **The new runs ran on `d60c671`** (07-24 merge) or a direct descendant — the
  first `main` state that contains *both* the wrapper and the GAE fix. The
  branch under analysis (`analysis/rnd-noisy-tv-set-v2`, HEAD `fffb2ca`) is 4
  commits ahead of `origin/main` and unpushed; those 4 commits only add analysis
  artifacts/docs, no agent-code change.

**Correction to the brief's "four things changed" list** (this matters for C):
| brief's claimed change | reality |
|------------------------|---------|
| env count 8 → 32 | **21 → 32** (old runs were never 8-env) |
| budget 10M → 20M | correct |
| jupyter pod → SLURM | correct |
| `NoisyTVWrapper` inserted | **false** — the wrapper is in *both* sets (added 07-19, before both). The only code delta is the NEXT_STEP GAE fix (07-22/24), documented as behaviour-preserving and confirmed neutral by the near-identical curves. |

---

## C — Bisect

**Not performed, by design.** The brief instructs: "If TASK A identifies the
cause, say so and stop — do not perform the bisect for completeness." TASK A
identified that there is **no regression to bisect**: the four candidate changes
do not separate a working run set from a broken one, because the old set is not
meaningfully more "working" than the new set. Running current code at 8 envs to
"see if it escapes room 1" would test a straw configuration (nothing ran at 8
envs). For the record, mapping each candidate to the evidence that retires it:

| candidate change | cheapest discriminating test | outcome / why retired without running it |
|------------------|------------------------------|------------------------------------------|
| **env count 21→32** | run current code at old env count | old runs *are* the 21-env datapoint (A6): same behaviour. Env count did not break it. |
| **`NoisyTVWrapper`** | run `off` on old commit | `off`/`sham` (no patch) fail identically to `remote` (patch) in **both** sets (A4); wrapper present in both; `check_noisy_tv.py --hash-off` passes. Retired. |
| **execution env (SLURM vs pod)** | diff `pip freeze` pod vs cluster | cannot diff (pod not reachable, no `pip freeze` captured). But curves are identical across the two environments, so any package delta is behaviour-neutral here. **Residual open item** — worth capturing `pip freeze` on the next cluster run to close formally. |
| **budget 10M→20M** | — | more budget cannot *cause* worse exploration; new 20M `off` reaches room 2 *earlier* than old 10M `off` s2. Retired. |

The one genuinely open *actionable* lever — `update_proportion=0.25` at 32 envs
— is **not** one of the four changes (it is unchanged old→new); it is carried
into the verification plan below as an improvement hypothesis, not a regression
fix.

---

## Minimal fix / next action

Because there is no regression, there is **no bug-fix diff to apply.** The
honest minimal action is a **one-line config change to test an improvement
hypothesis**, kept explicitly separate from any claim that it caused the symptom:

```diff
# src/agents/rnd.py  (or via CLI --update-proportion 1.0)
- p.add_argument("--update-proportion", type=float, default=0.25,
+ p.add_argument("--update-proportion", type=float, default=1.0,   # 32-env reference; 0.25 was Burda's 128-env value
```

Rationale: at 32 envs the predictor's effective batch should match Burda's
32-env reference (keep-prob 1.0); `0.25` quarters it and yields a noisier, less
informative intrinsic signal — a plausible contributor to weak RND. **Do not
ship this as a default** without the probe below confirming it, per the brief's
"don't change paper-matched hyperparameters as a fix without evidence"
constraint. (`0.25` is not itself paper-matched *at 32 envs* — that is the
argument for testing `1.0` — but the change is still gated on evidence.)

---

## Verification plan

All runs are **probes (≤3M steps)**, not full 20M runs, and must run on the
cluster/GPU (the dev box has no GPU and its shell resolves a mismatched
`gymnasium`; only `./.venv/bin/python` imports ALE cleanly, and CPU training is
far too slow for a 3M-step probe). Reference target from the brief: **the agent
should escape room 1 well before 10M steps** — so 3M is a valid read.

| # | probe | setting | confirmation criterion |
|---|-------|---------|------------------------|
| V1 | **reproduce** | current code, `rnd off`, 32 env, 3M, seed 42 | matches the 20M `off` shape in the first 3M (entropy→~0.6, intermittent room-2). Confirms the analysis harness. |
| V2 | **update_proportion** | as V1 but `--update-proportion 1.0`, seeds 42+1 | **improvement** = higher/again-rising `raw_intrinsic_rew_mean` late, later entropy decay, and room-2 reached earlier/more often than V1. Null = no change ⇒ `0.25` was not the lever. |
| V3 | **seed variance** | current code, `rnd off`, 3M, seeds 1/2/3 | quantifies how much of "reaches room 2 or not" is seed noise (A5 shows it is large). Sets the descriptive-only bar P4 already assumes. |
| V4 | **pure-intrinsic sanity** | `rnd off`, `--ext-coef 0`, 3M | Burda's `ext_coef=0` "finds >half the rooms" claim is the one that should hold *earliest*; if pure-intrinsic RND also cannot leave room 1 by 3M, the weakness is scale/optimization, not the ext/int mixing. |

A result that would count as "fixed / explained": V2 shows a clear,
seed-robust improvement (root cause = under-trained predictor from
`update_proportion`), **or** V1/V3/V4 all reproduce the weak-but-not-broken
behaviour with large seed variance and no config fixes it at 3M (root cause =
under-scale + entropy dynamics; the correct response is *more envs/frames or an
exploration-schedule change*, not a code fix — and, per the thesis's
characterisation-only scope, is reported descriptively rather than "solved").

---

## D — Metric verification (is the agent stuck, or the metric blind?)

**Verdict: the agent is genuinely stuck; the `rooms_visited` metric is sound.**

1. **Code audit** (`src/agents/base.py:45–66`). `RoomTracker` reads
   `self.unwrapped.ale.getRAM()[3]` after every inner step and reports
   `len(unique rooms)` in `info["rooms_visited"]`. RAM byte **3** is the
   canonical Montezuma's Revenge room register (matches the widely-used
   Go-Explore / ALE RAM annotations). `RoomTracker` sits *below*
   `AtariPreprocessing`, so it reads RAM on all 4 skipped frames per agent step
   and the room set is monotone within an episode — if anything, over- rather
   than under-counting.
2. **Live end-to-end test** (this session, real ALE via `./.venv/bin/python`):
   ```
   reset info keys: ['lives','episode_frame_number','frame_number','seeds','rooms_visited']
   4000-step random rollout → distinct rooms_visited values: [1]
                             → distinct RAM[3] room ids:     [1]
   ```
   `rooms_visited` is present at reset (=1) and every step, threads through the
   full wrapper stack, and a random policy provably never leaves room 1 in 4000
   steps. So "return 0 in room 1" is real, not a short episode / clipped reward.
3. **Positive control from the logs.** `rooms_visited=2` *does* register in the
   runs where an agent reached room 2 (new `off`, old `off` s2, PPO). A metric
   that were stuck-at-1 could not produce those 2s. So the metric is not blind;
   the four arms that read 1 forever really stayed in room 1.
4. **Return cross-check.** `episodic_return` is logged from
   `RecordEpisodeStatistics`, which is stacked *below* `ClipReward`
   (`base.py:380–382`), so it is the **true unclipped game score**. Its zeros are
   genuine (not an artifact of reward clipping to ±1).

The only unverifiable-here item is a *scripted trajectory that provably leaves
room 1* (would require a hand-authored action sequence into room 2); the
positive control in (3) substitutes for it, since real runs already demonstrate
the metric registering room 2.

---

## E — Targeted implementation checks (against Burda et al. 2018)

Each item, with file:line and pass/fail. **Net: the RND implementation is
faithful to Burda/CleanRL; no inverted-sign or wrong-denominator defect.**

| # | check | location | verdict |
|---|-------|----------|---------|
| E1 | **Intrinsic returns non-episodic** (int value head not masked at episode end) | `rnd.py:511–512` calls `compute_gae(..., episodic=False)`; `base.py:437,440` keep `nextnonterminal=1.0` for the intrinsic stream | ✅ **PASS** — non-episodic. Caveat below. |
| E2 | **Autoreset / GAE (NEXT_STEP) masking** | `base.py:397–448` (`compute_gae`) + `rnd.py:529,563` (`b_keep`/`mb_keep`) + `masked_mean` | ✅ **PASS / already fixed** (see quantification) |
| E3 | **Obs-norm initialised by random rollout before optimisation** | `rnd.py:406–418` (50 iters × 128 steps random policy → `obs_rms.update`) | ✅ **PASS** — `obs_rms_std`≈10.7 (no-patch), non-trivial |
| E4 | **Reward-norm denominator = std of intrinsic RETURNS, not rewards** | `rnd.py:486–496` — `RewardForwardFilter` builds discounted returns, then `reward_rms.update_from_moments(...); intr_buf /= sqrt(reward_rms.var)` | ✅ **PASS** — matches CleanRL `ppo_rnd_envpool.py` exactly |
| E5 | `gamma_E=0.999, gamma_I=0.99, ext_coef=2.0, int_coef=1.0, ent_coef=0.001`, sticky 0.25 | `rnd.py:93–115`; sticky = ALE v5 default (not overridden in `make_env`) | ✅ **PASS** — all match |
| E6 | **Policy entropy over training** | Fig 1 top-left; `losses/entropy` | ⚠️ real early decay to ~0.3–0.6, *identical old↔new*, at paper `ent_coef`. **Not the cause of confinement** — the PPO-remote control keeps entropy ~2.89 (near-uniform) and is equally stuck. RND's endpoint, not why it fails. |
| E7 | **The ~20× intrinsic-reward "collapse"** | `charts/raw_intrinsic_rew_mean` | ⚠️ **explained, not pathological**: predictor fitting the narrow room-1 distribution (see A5). Symptom, self-reinforcing; not the auto-stop failure. |

**E1 caveat (worth a thesis footnote, not a fix).** The intrinsic GAE is
non-episodic in the sense that matters (`nextnonterminal=1`), but the shared
`compute_gae` *additionally* walls off every NEXT_STEP fake step via
`lastgaelam *= (1 - done_buf[t])` (`base.py:446`). For the intrinsic stream this
also **cuts the λ-advantage flow across the episode boundary** (the death step),
so a pre-terminal step's intrinsic advantage bootstraps one step onto
V_int(final frame) rather than accumulating discounted intrinsic reward from the
next episode through λ. Pure Burda RND keeps the flow uncut. The effect is a
*more conservative* (quasi-episodic-at-boundaries) intrinsic advantage — a
second-order deviation, the value head still carries the cross-boundary signal,
and it is **identical old↔new**, so it is not the regression. Flagged for
awareness; changing it is out of scope (and would be a methodology change, not a
bug-fix).

**E2 quantification (the brief asked to quantify, not just classify).** The
NEXT_STEP GAE-masking bug documented in `doc/decisions.md` (2026-07-13) was
**fixed 2026-07-22** (`7e3d2d7`, `compute_gae` + `masked_mean`). Under the old
inline masking (`nextnonterminal = 1 - done_buf[t+1]`), at every episode boundary
one fake step (the discarded terminal-frame action, `reward=0`) bootstrapped
V(final frame of ep N) → V(start of ep N+1), and that discarded action was fed to
the policy gradient. Magnitude: boundaries occur ~once per ~300–1000 env-steps
per env vs a 128-step rollout, so ≈ 10–40% of rollouts contain one corrupted
step out of 128 (< ~0.3% of samples), affecting **all conditions equally**. The
fix walls those steps out of GAE and out of the PPO/value losses (`b_keep`),
leaving the RND predictor loss unmasked (final frames are real observations).
The old reference runs (07-20/21) predate the fix; the new runs include it; the
curves are nonetheless near-identical — empirically confirming the fix's "small,
uniform, behaviour-preserving" characterisation. So E2 is **not** a live cause.

**Bonus (from `decisions.md`, verified against code, not a production concern):**
on **CPU**, `intr_buf.cpu().numpy()` (`rnd.py:487`) returns a *view*, so the later
in-place `intr_buf /= sqrt(reward_rms.var)` (`rnd.py:496`) retroactively rewrites
`curiosity_np`, mislabelling `charts/raw_intrinsic_rew_mean` as the *normalised*
value. **All production runs are CUDA**, where `.cpu()` copies — so every number
in this analysis is unaffected. Only local `--sync-envs` CPU smokes would mislog
that one chart. One-line `.copy()` fix if ever wanted; out of scope.

---

## F — Implementation gaps vs the experimental plan (report only; do not implement)

The thesis plan (CLAUDE.md § Thesis context: H1, P1–P4, the 5 conditions)
requires machinery the codebase does not yet have. **None implemented — listed
for approval.**

| # | required by | gap | notes |
|---|-------------|-----|-------|
| F1 | H1 / P3 | **`charts/tv_memorisation_gap` probe** measuring `G = E[err(freshly sampled patch)] − E[err(currently displayed patch)]` on the *same* underlying frames | Not present. The existing `tv_intrinsic_share` (occlusion diagnostic, `rnd.py:545–555`) measures a *different* quantity (occluded-vs-full error, i.e. how much of the signal the patch region contributes), **not** the fresh-vs-displayed gap H1 is defined on. `G` is the actual test of H1 and is **computable post-hoc from checkpoints** — but **no checkpoints were saved into `analysis/`**, so it cannot be back-computed from the existing runs; either re-copy the cluster checkpoints or fold the probe into the next runs. |
| F2 | P3 / stimulus design | **Offline stimulus-calibration script**: measures Δ prediction error per candidate patch design against a reference scale (the err-gap between well-visited and rarely-visited *rooms*) | Not present. `scripts/check_noisy_tv.py` only verifies determinism/geometry; `analyze_runs.py` only overlays finished runs. The 07-19 `decisions.md` note ("area is the only stimulus lever; a 12×12 patch ≈ 1% of error") was found by an *ad-hoc checkpoint probe*, not a reusable script — that probe is exactly what F2 should become. |
| F3 | conditions list | **`frozen` mode** (18 actions, patch present, never resampled — HUD-occlusion control) | Not a named mode. Expressible *approximately* as `static` with a very large `--tv-refresh-every` (patch drawn once at `reset`, then effectively never resampled within an episode), **but** `check_tv_geometry` requires `refresh_every ≥ 1` and there is no `∞` sentinel, and `static` still redraws a *new* patch at each episode reset (frozen should keep one fixed patch). Needs either a real `frozen` mode or a `refresh_every=∞` path + a "draw once, keep" flag. |
| F4 | P3 sweep | **`--tv-refresh-every` sweep for T ∈ {1, 64, ∞}** | `T=1` and `T=64` already work (`static` mode). **`T=∞` is not expressible** (int ≥ 1). Blocks the "frozen end" of the P3 non-monotonicity sweep; tied to F3. |
| F5 | P4 / inference | Seed replication | Every arm is **one seed** (`off` new is seed 42; old arms seed 1, one `off` seed 2). A5 shows room-2-or-not is largely seed noise. P4 already restricts these to descriptive use; still, ≥3 seeds/arm are needed even for the descriptive `rooms_visited`/`return` reporting to be honest. |

Additionally, two hygiene items surfaced (not gaps in the plan, but they cost
this analysis): **(a) checkpoints and videos were not preserved** with the event
files — preserving them (they are already written at `checkpoint_interval=100`)
would have enabled F1 post-hoc and A7 directly; **(b) no `pip freeze` / commit
SHA is recorded per run** (W&B `--track` is off) — logging both would close the
"execution-environment" bisect arm (C) mechanically instead of by inference.

---

## TL;DR for the supervisor

1. The 20M/32-env runs are **not broken and not a regression** — they match the
   21-env/10M "reference" runs curve-for-curve, and the new `off` arm actually
   reaches room 2 *earlier* than the old one.
2. The **noisy-TV wrapper is exonerated** (no-patch arms fail identically to
   patched arms; wrapper was in both run sets; hash check passes).
3. The **metric is sound and the agent is genuinely in room 1** (live check +
   positive control).
4. The RND **implementation is faithful to Burda** (intrinsic non-episodic,
   return-std normalisation, obs-norm init, GAE fix all correct).
5. What remains is a **real, pre-existing, cross-algorithm exploration weakness**
   at ~25× under-scale, plus one worth-testing config mismatch
   (`update_proportion=0.25` at 32 envs). Confirm with the ≤3M probes above
   before spending any 20M budget.
6. Several **plan-critical pieces don't exist yet** (memorisation-gap probe,
   stimulus-calibration script, `frozen`/`T=∞` support, multi-seed) — flagged
   in F for approval, not implemented.
