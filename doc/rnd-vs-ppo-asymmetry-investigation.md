# Investigation brief: why does RND underperform PPO on the same codebase?

Status: **open**. This file is a self-contained brief for whoever picks this up next
(a fresh Claude Code session, a human, or both) — it assumes no memory of how this was
found. Do not start from scratch: read §1 first, most of the groundwork already exists
across three parallel investigations that don't fully agree with each other.

## 0. The question

The production RND runs (`rnd_10m_s1`, `rnd_10m_s2`, 10M steps each) never left room 1
and never scored a single point across 19,106 + 17,171 episodes combined. The
production PPO run (`ppo_5m_s1`, **half** the step budget, no RND, no intrinsic
reward, no entropy-collapse mechanism at all) reached **room 2** and scored up to
**400** on 9 separate episodes out of 9,146.

**RND performed strictly worse than a baseline with no exploration bonus, on less
than double the steps.** Two prior diagnoses (see §1) explain *why RND's own metrics
look unhealthy* (entropy collapse, intrinsic reward collapse, sub-paper-scale
hyperparameters), but neither one explains, or even directly examines, why RND is
worse than PPO specifically. That comparison is the actual gap. Find out why, and
confirm it empirically — don't stop at a plausible story.

## 1. What already exists — read these first, do not re-derive them

This repo has several parallel git worktrees (`git worktree list` from the repo root)
independently investigating pieces of this. Check current state before assuming
anything below is still accurate (branches move):

- **`worktree-rnd-10m-failure-doc`** (branch, commit `68eb731` at time of writing) —
  `doc/10M-RND-run-failure-documentation.md`. Thorough. Confirms every RND
  hyperparameter (including `ent_coef=0.001`) is an exact match to the paper /
  CleanRL's `ppo_rnd_envpool.py`, except `num_envs` (8 vs paper's 128) and
  `total_timesteps` (10M vs paper's ~492M). Frames the collapse as compute-budget
  compression (this run gets ~4,882 iterations vs. the paper's ~30,000 — the
  entropy/LR-anneal schedule is squeezed into ~6x fewer updates). Does **not**
  compare against PPO's own performance anywhere in the doc — this is the gap.
- **`worktree-rnd-retry-prep`** (commit `cb34663`) —
  `slurm/run_rnd_falsify.slurm`: a cheap 3M-step, `--ent-coef 0.01 --no-anneal-lr`
  falsification test, meant to be run *before* committing to a full production
  budget. Has explicit pass/fail criteria in its header comment. Not yet run (no
  results recorded as of this writing). Also fixed a real but unrelated SLURM
  CPU/env-count oversubscription bug (`doc/throughput-investigation.md`) — worth
  having in place regardless of this investigation's outcome.
- **`main`** (this investigation) — `doc/decisions.md`, entries dated 2026-07-13.
  Documents: (a) an `ent_coef` 0.001→0.01 change now flagged **low-confidence**
  (0.001 turned out to be the paper's own value, not a weak one — same finding as
  the bullet above, reached independently); (b) the PPO-vs-RND numeric comparison
  in §2 below; (c) a confirmed, empirically-verified gymnasium 1.x autoreset bug,
  §3 below, not present in either other worktree's writeup.

None of these three fully explains the PPO-vs-RND gap. Reconcile them, don't pick one
and ignore the others — e.g. the compute-budget framing and the autoreset bug are not
mutually exclusive, and the falsify-slurm test doesn't isolate the autoreset bug at
all (it only varies `ent_coef`/`anneal_lr`).

## 2. The core data point (verify this first — it should be trivially reproducible)

```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# PPO: runs/ALE/jupyterlab/events.out.tfevents.1783201501.jupyter-s0584385.1425.0
# RND s1: ...1423.0   RND s2: ...1424.0
# Point EventAccumulator at ONE specific file, never the shared directory —
# loading multiple runs together interleaves scalars under the same tag names.
```

| Run | Steps | Episodes | Nonzero-return episodes | Max `rooms_visited` | Max `episodic_return` |
|---|---|---|---|---|---|
| `ppo_5m_s1` | 5M | 9,146 | 9 (8×100, 1×400) | 2 | 400 |
| `rnd_10m_s1` | 10M | 19,106 | 0 | 1 | 0 |
| `rnd_10m_s2` | 10M | 17,171 | 0 | 1 | 0 |

Also cross-check `logs/ppo_5m_s1.out` directly (`grep -oE "return=[0-9.]+" | sort | uniq -c`)
— it should match the TB data (9 nonzero episodes). This was checked from both the
log and the TB file independently before writing this doc; it should reproduce.

## 3. Confirmed bug: gymnasium 1.x `AutoresetMode.NEXT_STEP` vs. the GAE masking code

Verify this yourself before trusting it — here's the exact repro used:

```python
import gymnasium as gym
import numpy as np
envs = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1")])
obs, info = envs.reset(seed=0)
for step in range(200):
    obs, reward, terminated, truncated, info = envs.step(np.array([0]))
    print(step, obs[0][0], reward[0], terminated[0], truncated[0])
    if terminated[0] or truncated[0]:
        for extra in range(3):
            obs, reward, terminated, truncated, info = envs.step(np.array([0]))
            print(" +", extra, obs[0][0], reward[0], terminated[0], truncated[0])
        break
```

Result: the terminal step returns the **true final observation** (cart mid-fall) with
a real reward. The **next** step silently discards the given action, returns
`reward=0.0`, `terminated=False`, and the fresh reset observation. This is
`gymnasium.vector.AutoresetMode.NEXT_STEP` — confirmed as the default for both
`SyncVectorEnv` and `AsyncVectorEnv` in the installed gymnasium 1.3.0
(`inspect.signature` shows `autoreset_mode: ... = AutoresetMode.NEXT_STEP`).

All three agents (`src/agents/ppo.py`, `count_based.py`, `rnd.py` — grep for
`nonterminal` / `done_buf\[t` to find the exact lines, identical pattern in all
three) mask the GAE value bootstrap by checking `done_buf[t + 1]`. That's the
correct check for the **`SAME_STEP`** convention (gymnasium 0.x, and what CleanRL's
original `ppo_atari.py` was written against): under `SAME_STEP`, the terminal step
itself returns the reset observation, so every loop index corresponds to a real,
causally-effective transition, and `done_buf[t+1]` correctly marks exactly where two
episodes' data meet.

Under the **actual** `NEXT_STEP` behavior running here, the boundary between two
unrelated episodes' observations is one index later than the code's mask accounts
for: `done_buf[t+1]=True` correctly cuts the bootstrap between the real terminal
reward and the terminal frame, but the code never cuts the bootstrap between the
terminal frame (index `t+1`) and the next episode's fresh-start observation (index
`t+2`) — `done_buf[t+2]` is `False` there, so the mask never fires. Consequence,
once per episode boundary (i.e. very frequently — mean episode length in the RND
runs was ~522-573 steps):
1. Value gets bootstrapped from `t+1` (terminal frame of episode A) across into
   `t+2`'s value (a state from unrelated episode B) as if they were sequential.
2. The action chosen at `t+1` (based on the terminal/death frame) is used as a real
   policy-gradient sample, even though the environment silently discarded it and
   just reset instead of executing it.

**This bug is identical in all three agent files** — it does not, by itself, explain
why RND is uniquely worse than PPO. Confirming/ruling this out is part of the task
below, not a foregone conclusion.

**Candidate fix, low-risk, worth trying first:** both `SyncVectorEnv` and
`AsyncVectorEnv` accept an `autoreset_mode` constructor argument
(`gymnasium.vector.AutoresetMode.SAME_STEP` restores the exact semantics the
existing GAE code assumes) — this may be a one-line fix in `base.py`'s vector-env
construction rather than a rewrite of the GAE indexing math. Verify this actually
produces `SAME_STEP`-style output (re-run the CartPole repro above with
`autoreset_mode=AutoresetMode.SAME_STEP` passed to `SyncVectorEnv`) before trusting
it, and confirm it doesn't have some other interaction with `RecordEpisodeStatistics`
/ `AtariPreprocessing` / the `infos["_episode"]` masking pattern documented in
`CLAUDE.md`'s gymnasium-1.x section before shipping it.

## 4. The task

1. **Reconcile, don't duplicate**, the three existing analyses (§1). Produce one
   coherent account, not a fourth competing doc.
2. **Explain the PPO-vs-RND asymmetry specifically.** Candidates to check, roughly in
   order of how directly they'd explain *RND being worse than no-intrinsic-reward*:
   - Does fixing the `NEXT_STEP`/`SAME_STEP` bug (§3) alone close the gap? It affects
     both agents equally in principle, but RND's *combined* advantage
     (`b_adv = int_adv*int_coef + ext_adv*ext_coef`) uses the same
     (potentially-corrupted) `ext_adv` as PPO **plus** an intrinsic stream — check
     whether the corrupted terminal-frame sample gets amplified rather than just
     inherited, e.g. via the intrinsic reward computed on the immediately-following
     reset observation, or via how `update_proportion`'s random predictor-training
     mask interacts with it.
   - Is there something specific to how `obs_rms` / `reward_rms` normalization
     interacts with the short, frequent-death episode pattern that PPO doesn't have
     (PPO has no equivalent running-normalization state at all)?
   - Re-examine the RND network/loss code (`RNDModel`, `RewardForwardFilter`,
     `_normalize_obs`, the combined-advantage weighting) line by line against the
     paper — a second, skeptical pass, specifically looking for anything that would
     make intrinsic-driven exploration *actively worse than none*, not just weaker
     than hoped. (A first such pass already happened in this session and found
     nothing beyond §3 — don't assume it was exhaustive.)
   - Consider whether the compute-budget framing (`worktree-rnd-10m-failure-doc`
     §4/§7) combined with RND's *additional* learning burden (predictor network,
     dual value heads, more loss terms per update) means RND simply needs
     proportionally more updates than PPO to reach the same place — i.e. maybe
     there's no bug at all beyond §3, and RND is just more sample-hungry per
     update in a way that a 2x-larger budget doesn't compensate for. If this is the
     conclusion, say so explicitly rather than defaulting to "there must be a bug."
3. **Confirm empirically — do not conclude from code-reading alone.** Design a
   matched-budget ablation, run it (locally or via SLURM), and report actual numbers:
   - Use a small, cheap budget for all variants (e.g. 2-3M steps — matching the scale
     of `worktree-rnd-retry-prep`'s `run_rnd_falsify.slurm`), not the full 10M/100M
     production scale, until something actually works.
   - Run a **fresh PPO baseline at the same small budget** on current `main` code
     (not the old 5M production run — code has changed since: reward clipping,
     obs-norm fix, etc.) to have a true apples-to-apples comparison, not just old
     production data from a different code version.
   - Suggested matrix (adjust as needed, but isolate one variable at a time):
     | Variant | `ent_coef` | autoreset fix | Expected if §3 is the cause | Expected if §4.2's "just needs more updates" theory is right |
     |---|---|---|---|---|
     | A | 0.001 (paper value) | no | reproduces original collapse | reproduces original collapse |
     | B | 0.01 (current `main` default) | no | still collapses (bug independent of `ent_coef`) | may partially help |
     | C | 0.001 | yes | recovers toward PPO-level performance | still underperforms PPO |
     | D | 0.01 | yes | best of both | best of both, still may lag PPO |
   - Pass/fail should be stated in terms of §2's numbers: does the variant reach at
     least room 2 and/or score above 0 within the matched budget, comparably to (or
     better than) the fresh PPO baseline run at the same budget?
4. **Write the conclusion back to `doc/decisions.md`**, replacing the "Status:
   unresolved" line in the "RND vs PPO asymmetry" entry with the actual answer, and
   update this file's header (`Status: open` → resolved/superseded) accordingly. If a
   code fix results, log it there too (what changed, why, what confirmed it worked —
   this repo's established convention, see the rest of `doc/decisions.md`).

## 5. Don't do this

- Don't re-run the full 10M/100M-step production `slurm/run_rnd.slurm` /
  `run_count_based.slurm` until this is resolved — that's exactly the expensive
  mistake this investigation exists to prevent repeating.
- Don't treat "the paper needed 50x more compute" as a full explanation for the
  PPO-vs-RND gap specifically — it may be part of the story, but it doesn't by
  itself explain why RND is worse than a *simpler* algorithm with less compute.
- Don't fix `ent_coef` or the autoreset bug and call it done without the matched-
  budget empirical comparison in §4.3. Both prior diagnoses in this repo were
  plausible-sounding and wrong (or at least unconfirmed) — this file exists because
  code-reading alone already produced one confident-but-shaky conclusion this week.
