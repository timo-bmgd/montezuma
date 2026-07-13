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
