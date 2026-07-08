# 10M-step RND run failure: diagnosis

Summary of the investigation into why the RND production runs (`rnd_10m_s1`, `rnd_10m_s2`,
10M steps each, run on JupyterHub ~2026-07-04 to 2026-07-08) never left room 1 of
Montezuma's Revenge.

## TL;DR

Both seeds converge to the same dead end: the policy's entropy and the RND intrinsic
reward both collapse within the first **~9% of training** (well under 1M of the 10M
steps), after which PPO's updates become no-ops and the run spends the remaining ~91%
of its budget doing nothing new. `rooms_visited` never exceeds 1 in either run. This
is not simply "not enough training time" — the run actively stalls early — but the
10M-step budget itself is also ~40-50x smaller than what the original paper used to
get its headline results, so it was always a long shot at this scale.

## 1. Initial data-quality problem: wrong TensorBoard file

The TensorBoard directory originally pointed to locally
(`runs/ALE/MontezumaRevenge-v5__rnd__1__1781212716/`) turned out **not** to be the
production run's data:

- It stops at global_step 20,480, while the text log (`logs/rnd_10m_s1.out`) reaches
  step 9,995,936.
- It was written on host `MacBook-Air.local`, while the log shows `Using device: cuda`
  (a GPU box).
- It's missing tags added in later commits (`charts/episodic_return`,
  `charts/rooms_visited`, `charts/obs_rms_std`, `charts/reward_rms_std`) — it predates
  commits `c1ccbad` (thesis logging), `6d02c4d` (`NewRoomRecorder`), and `3fee53a`
  (reward clipping + obs-norm init fix).

Conclusion: it's a short local dev smoke test that happened to share a run-name
timestamp bucket, not the JupyterHub production run. **Do not use it.**

The real event files were retrieved from JupyterHub and placed in
`runs/ALE/jupyterlab/`. They were matched to the correct log files by PID (the
`cuda_dnn.cc` line printed by each process at startup):

| Event file (PID) | Log file |
|---|---|
| `events.out.tfevents.1783201501.jupyter-s0584385.1423.0` | `logs/rnd_10m_s1.out` |
| `events.out.tfevents.1783201501.jupyter-s0584385.1424.0` | `logs/rnd_10m_s2.out` |
| `events.out.tfevents.1783201501.jupyter-s0584385.1425.0` | `logs/ppo_5m_s1.out` |

Note: point `EventAccumulator` at the **specific file**, not the shared
`runs/ALE/jupyterlab/` directory — loading all three at once interleaves scalars from
different runs under the same tag names and gives corrupted results.

## 2. What the text logs show (before real TB data was available)

From `logs/rnd_10m_s1.out` (19,106 episodes logged over 10M steps):

- **`return=0.0` for every single logged episode.** Zero raw game score, the entire run.
- Episode lengths are heavily skewed short (`terminal_on_life_loss=False`, so each
  episode is a full 6-life game, not a single life):

  | Length bucket | Episodes | % |
  |---|---|---|
  | < 1000 steps | 18,299 | 95.8% |
  | 1000–4999 | 721 | 3.8% |
  | 5000–19999 | 16 | 0.1% |
  | 20000–26999 | 65 | 0.3% |
  | 27000 (truncation cap) | 5 | 0.0% |

  Mean length 522 steps. This is a "dies fast, repeatedly, never scores" signature,
  not "survives fine but wanders without finding reward."
- No NaNs, errors, or crashes anywhere in the 24,004-line log. SPS stable
  (144–183) for all 4,882 iterations — the process ran to completion as designed.

`logs/rnd_10m_s2.out` shows the same pattern: 0/17,171 episodes with nonzero return,
92.6% of episodes under 1000 steps, mean length 573.

## 3. Confirmed diagnosis from the real TensorBoard data

Once the correct event files (`runs/ALE/jupyterlab/...1423.0` for s1,
`...1424.0` for s2) were available, both runs' iteration counts (4,882) and episode
counts (19,106 / 17,171) matched the text logs exactly, confirming they're the right
files.

### `rooms_visited` — confirmed room-1 ceiling

`charts/rooms_visited` = **1.0 for every single logged episode**, in both s1
(19,106/19,106) and s2 (17,171/17,171). Previously this was only inferred from video
evidence (`NewRoomRecorder` only ever produced one video, the first episode); this
directly confirms it from the metric itself.

### Entropy collapse — real and severe, in both seeds

`losses/entropy` starts at **2.890 = ln(18)**, i.e. the agent starts as a literal
uniform-random policy over the 18 actions, then collapses:

| step | s1 entropy | s2 entropy |
|---|---|---|
| 2,048 | 2.890 (100%) | 2.890 (100%) |
| 909,312 | 1.305 (45%) | 2.196 (76%) |
| 1,818,624 | 0.690 (24%) | 1.055 (37%) |
| 2,727,936 | 0.575 (20%) | 0.992 (34%) |
| ~9,998,000 (end) | 0.209 (**7%**) | 0.169 (**6%**) |

s2's collapse is a little more gradual early on, but both seeds land at roughly the
same place: ~6-7% of starting entropy by the end of training.

### Intrinsic reward collapse — in lockstep with entropy, both seeds

`charts/raw_intrinsic_rew_mean` (pre-normalization RND predictor MSE — the raw
novelty signal):

| step | s1 | s2 |
|---|---|---|
| 2,048 | 196.5 | 146.3 |
| 909,312 | 9.7 (**20x drop**) | 7.1 (**20x drop**) |
| 9,998,336 (end) | 6.8 (flat/noisy for remaining ~91%) | 3.4 (flat/noisy for remaining ~91%) |

`charts/mean_intrinsic_rew` (normalized): s1 0.217 → 0.009-0.015; s2 0.228 → 0.010-0.014.
Same ~20x collapse, same timing (within the first ~9% of the 10M-step budget), in
both independently-seeded runs.

### PPO effectively freezes

`losses/approx_kl` and `losses/clipfrac` both decay to **~0.0 by the final logged
iteration** in both s1 and s2. PPO's policy updates become no-ops well before the run
ends — the policy has stopped meaningfully changing for a large fraction of the total
budget.

### What's healthy (ruled out as causes)

- `charts/obs_rms_std`: stable 10.5–11.0 throughout, both runs. The pre-fix
  obs-normalization init bug (see §5) is confirmed present in the code that produced
  this run, but the data shows normalization itself was **not** meaningfully
  corrupted by it. Downgraded from the ranked-causes list.
- `charts/reward_rms_std`: non-zero throughout, smooth decline (s1: 905→460, s2:
  642→348) — tracking a genuinely shrinking intrinsic signal, not itself broken.
- `charts/int_value_mean` (~1.0–1.5) is internally consistent with
  `mean_intrinsic_rew`≈0.01 under the non-episodic `int_gamma=0.99` horizon
  (0.01/(1−0.99)≈1.0) — the intrinsic value head correctly fit a signal that had
  already vanished, it's not the value head that's broken.

### Root-cause chain (high confidence, closed feedback loop, reproduced in both seeds)

Episodes are short (agent dies fast, repeatedly) → the RND predictor only ever sees a
narrow band of near-spawn states → it fits that narrow distribution almost
immediately (within ~9% of the 10M-step budget) → intrinsic reward collapses ~20x
before the agent could plausibly have found room 1's exit (the rope/ladder/chasm
sequence) → with extrinsic reward permanently 0 and intrinsic reward now negligible,
nothing counteracts natural PPO entropy decay → policy prematurely converges and
effectively freezes (`approx_kl`/`clipfrac`→0, LR also anneals to 0) → episodes stay
short → loop repeats for the remaining ~90% of training with no further exploration
pressure.

## 4. Hyperparameter provenance

Checked every hyperparameter in `src/agents/rnd.py` against the RND paper (Burda et
al. 2018, Table 5 / Appendix A.4) and CleanRL's reference implementation
(`ppo_rnd_envpool.py`):

| Hyperparameter | This repo | Paper / CleanRL | Match? |
|---|---|---|---|
| `lr` | 1e-4 | 1e-4 | matches |
| `gamma` (extrinsic) | 0.999 | 0.999 | matches |
| `int_gamma` (intrinsic) | 0.99 | 0.99 | matches |
| `gae_lambda` | 0.95 | 0.95 | matches |
| `num_minibatches` | 4 | 4 | matches |
| `update_epochs` | 4 | 4 | matches |
| `clip_coef` | 0.1 | 0.1 | matches |
| `ent_coef` | 0.001 | 0.001 | matches |
| `vf_coef` | 0.5 | 0.5 | matches |
| `int_coef` / `ext_coef` | 1.0 / 2.0 | 1 / 2 | matches |
| `update_proportion` | 0.25 | 0.25 | matches |
| `obs_norm_init_steps` | 50 | 50 | matches |
| `num_steps` (rollout length) | 128 | 128 | matches |
| **`num_envs`** | 8 default (~16 in this run) | **128** | **8x smaller** |
| **`total_timesteps`** | 10,000,000 | **~492M agent-steps (1.97B frames)** | **~50x smaller** |

Nearly every "algorithm" hyperparameter — including `ent_coef=0.001`, which looked
suspiciously low in isolation — is an exact, faithful copy of the paper/CleanRL
defaults. It was not an arbitrary or mistaken choice.

The only two deviations are `num_envs` and `total_timesteps`, and both look like
deliberate compute-scale-downs rather than oversights: `rnd.py`'s own module
docstring documents the paper-scale invocation as the intended "full scale" example:

```
python src/agents/rnd.py --total-timesteps 2000000000 --num-envs 128
```

The production run just used the (much smaller) argparse defaults instead of that
documented command.

Sources: [RND paper (arXiv 1810.12894)](https://arxiv.org/abs/1810.12894), [CleanRL
`ppo_rnd_envpool.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py),
[CleanRL RND docs](https://docs.cleanrl.dev/rl-algorithms/ppo-rnd/).

## 5. Confirmed code state at run time (minor contributor, ruled mostly out)

`logs/rnd_10m_s1.out` starts 2026-07-04 21:44:58, ~1h49m *before* commit `3fee53a`
("add reward clipping and fix RND obs-norm init buffering bug") landed at 23:34:02.
Since a running process can't pick up new code, this production run used the pre-fix
`rnd.py`:

- No `--clip-reward` (irrelevant here since extrinsic reward was always 0 anyway).
- The obs-norm init loop's flush condition was
  `len(frames_buf) == num_envs * num_steps` instead of `== num_steps`, silently
  dropping part of the initial calibration frames.

Per §3, `obs_rms_std` looks healthy throughout both real runs (stable 10.5–11.0), so
this bug is confirmed present in the code but **not** a meaningful contributor to the
observed failure — `obs_rms` continues to update online every iteration regardless of
init quality, and the data shows no sign of corrupted normalization.

## 6. Would changing hyperparameters help?

- **Raise `ent_coef`** (e.g. 10-20x, to ~0.01-0.02). This is the most direct,
  well-targeted fix for the entropy collapse. `ent_coef=0.001` was calibrated for the
  paper's ~30,000-iteration schedule (492M steps / batch_size); this run only gets
  4,882 iterations (10M / ~2048) — about 6x fewer. LR annealing decays to 0 by
  iteration 4,882 instead of iteration 30,000, compressing whatever entropy-decay
  dynamics were tuned for the longer schedule into a much shorter window.
- **Increase `num_envs`** toward the paper's 128 (or as far as GPU memory allows,
  e.g. 32–64 on a V100). This isn't just "more data" — the paper's own scaling
  section states: *"agents trained with larger batches of experience collected from
  more parallel environments obtain higher mean returns after similar numbers of
  updates."* More parallel envs also directly addresses the predictor-overfitting
  mechanism in §3's root-cause chain: more simultaneous, independently-seeded
  rollouts feed the predictor and policy at every update, slowing predictor
  saturation and raising the odds that at least one parallel agent finds progress
  that then reinforces through the shared policy weights.
- **Reconsider `--anneal-lr`.** Given the schedule is already compressed ~6x relative
  to the reference, LR (and effective exploration capacity) is fully exhausted well
  within the current run. Disabling it or adding a floor would leave more update
  magnitude available in the back half of training.
- **Increasing `total_timesteps` alone, without the above, likely will not help.**
  `approx_kl`/`clipfrac` are already ~0.0 by the final logged iteration in both s1
  and s2 — the policy is already inert well before 10M steps. More steps under the
  same config ≈ more steps of a frozen policy.

## 7. Realistic expectations at this training scale

No, 10M steps should not be expected to produce meaningful room progress, on two
independent grounds:

- **Compute-budget gap.** The paper's own smaller-scale ablation (32 parallel envs —
  still 2x more than this repo's ~16) needed **1.6 billion frames (~400M
  post-frameskip steps)** to reach a mean return of ~7,570. That's roughly **40x**
  this run's 10M-step budget. Even a correctly-tuned run at this repo's scale would
  plausibly still be room-1-bound at 10M steps purely from the budget gap — the paper
  frames higher env-count and more frames as necessary for its headline
  24-room/~10,000-score numbers, not optional extras.
- **This run isn't "slow," it's stalled.** Per §3, the policy stops meaningfully
  updating well before the 10M-step mark. It is not accurately described as "given
  more time it would keep improving at the same rate" — fixing the entropy/intrinsic
  collapse (§6) is a prerequisite before more steps would be useful at all.

**Implication for thesis scope:** either scale up meaningfully (more envs + a budget
in the hundreds-of-millions-of-steps range, which may exceed what's feasible on the
current V100 allocation/timeline), or reframe the RND result around a fixed, feasible
compute budget — e.g. showing the trend/comparison between count-based, RND, and PPO
at an affordable budget rather than chasing the paper's absolute numbers. Worth
deciding explicitly and documenting as a known scope constraint rather than
discovering it late.

## 8. Data locations (for future reference)

**Logs (local):**
- `logs/rnd_10m_s1.out`, `logs/rnd_10m_s2.out`, `logs/ppo_5m_s1.out`

**TensorBoard — real data (use this):**
- `runs/ALE/jupyterlab/events.out.tfevents.1783201501.jupyter-s0584385.1423.0` (rnd s1)
- `runs/ALE/jupyterlab/events.out.tfevents.1783201501.jupyter-s0584385.1424.0` (rnd s2)
- `runs/ALE/jupyterlab/events.out.tfevents.1783201501.jupyter-s0584385.1425.0` (ppo s1)

**TensorBoard — stale, do not use:**
- `runs/ALE/MontezumaRevenge-v5__rnd__1__1781212716/`
- `runs/ALE/MontezumaRevenge-v5__rnd__1__1781213250/`

## 9. Open items / next steps

- Draft a next-run config with `ent_coef` raised, `num_envs` increased, and
  `anneal_lr` reconsidered (§6), sized to what's actually feasible on the available
  V100 allocation.
- Re-run and re-diagnose with the same method (`EventAccumulator` pointed at the
  specific event file, cross-checked against the text log's episode/iteration counts)
  to confirm the collapse signature is gone before scaling up further.
- Decide and document the thesis's compute-budget framing (§7) so expectations for
  the RND results section are calibrated before more runs are spent.
