# Results brief → next-steps decision (RND noisy-TV batch)

**You are a research-planning/writing assistant for a bachelor thesis.** Topic: comparing
exploration algorithms (count-based → RND → PPO) on Montezuma's Revenge (ALE
`ALE/MontezumaRevenge-v5`). Primary metric = **distinct rooms explored**; secondary =
game score. Literature targets: count-based ~15 rooms / ~3,700; **RND (the key algorithm
of interest) ~24 rooms / ~10,000**; Go-Explore 37+. Below is the latest experiment batch
and its diagnosis. **Your job: decide and articulate the next steps** — which experiments
to prioritize, how to frame the RND result in the thesis, and whether the noisy-TV
ablation is salvageable at feasible compute. Weigh the open tensions in §5; don't just
pick the first option.

## 1. The batch

6 completed runs, each the full **10M timesteps**, `num_envs=21`, all hyperparameters an
exact match to the RND paper / CleanRL **except** `num_envs` (paper: 128) and
`total_timesteps` (paper: ~492M steps / 1.97B frames). It's an RND **noisy-TV ablation**
(a stochastic "TV" patch injected into the observation — the canonical RND failure mode)
plus a PPO baseline and a 2nd RND seed:

## 2. Headline metrics (the important ones)

| run | algo · tv_mode | max rooms | reached room 2 | max score | nonzero-score eps | final entropy | final TV-share of intrinsic |
|-----|----------------|----------:|---------------:|----------:|------------------:|--------------:|----------------------------:|
| ppo_tv_off s1        | PPO · off         | **2** | 2 / 18,888  | 400 | 12  | **2.763** | — (no intrinsic) |
| rnd_tv_off s1        | RND · off         | 1     | 0 / 19,267  | 0   | 0   | 0.503 | -0.09 |
| rnd_tv_off s2        | RND · off         | **2** | 6 / 18,244  | 400 | 420 | 0.611 | -0.20 |
| rnd_tv_remote s1     | RND · remote      | 1     | 0 / 20,640  | 100 | 4   | 0.762 |  0.12 |
| rnd_tv_sham-remote s1| RND · sham-remote | 1     | 0 / 18,802  | 0   | 0   | 0.415 | -0.06 |
| rnd_tv_static s1     | RND · static      | 1     | 0 / 22,225  | 0   | 0   | **0.282** |  0.20 |

Context numbers: starting entropy = ln(18) = 2.89 (uniform-random over 18 actions);
mean episode length 450–550 steps across all runs (agent dies fast every episode);
entropy first drops below 1.0 between 1.35M–4.04M steps in every RND run.

## 3. Four findings

1. **Exploration failed across the board.** 4 of 6 runs never left room 1; the two that
   did (PPO, rnd_off s2) reached only room 2, in a handful of episodes (2 and 6 of ~18k).
   Nothing is remotely near RND's literature target of ~24 rooms.
2. **RND provided no benefit over PPO — it was worse.** PPO (no intrinsic reward, and it
   is the tv_off baseline) matched or beat every RND run on both rooms and score, at the
   *same* budget. This replicates a previously documented, still-**open** PPO>RND
   asymmetry on this codebase.
3. **Entropy collapse in every RND run** (final 0.28–0.76 vs PPO's 2.76 ≈ near-uniform).
   The `static` condition (strongest TV distraction) collapsed hardest (0.28).
4. **The noisy-TV effect is real but currently un-measurable.** `static`/`remote` TV
   modes capture a positive share of the intrinsic reward (0.20 / 0.12 vs ~0 for
   off/sham-remote) — the TV *does* siphon RND's curiosity. But because RND fails to
   explore **even with the TV off**, the ablation cannot yet isolate a TV *effect on
   exploration*: the independent variable works, the dependent variable is floored at
   ~1 room in every condition.

## 4. Diagnosis (from `doc/10M-RND-run-failure-documentation.md` and `doc/rnd-vs-ppo-asymmetry-investigation.md`)

- **Collapse loop (high confidence, reproduced across seeds):** short episodes (die fast,
  extrinsic reward ~always 0) → RND predictor overfits the narrow band of near-spawn
  states within the first ~9% of training → raw intrinsic reward drops ~20x → with
  extrinsic reward permanently 0, nothing opposes PPO's natural entropy decay → policy
  freezes (`approx_kl`/`clipfrac` → 0, LR anneals to 0) → episodes stay short. Closed loop.
- **Compute-budget gap:** 10M steps is ~40–50x below the paper's scale, and `num_envs=21`
  vs 128. Even a *correctly* tuned run would plausibly still be room-1-bound at this
  budget — but this run is worse than that: it actively **stalls early**, it isn't merely
  slow.
- **Confirmed gymnasium 1.x `NEXT_STEP` autoreset bug:** corrupts the GAE value bootstrap
  at every episode boundary (present identically in all three agents). Not yet confirmed
  as the cause of the RND<PPO gap — it affects both agents in principle, but may be
  amplified in RND via the combined intrinsic+extrinsic advantage. A `SAME_STEP` fix is
  proposed but unverified.

## 5. Open tensions to weigh (do not resolve these by fiat — reason about them)

- **What is the actual result?** "RND collapses / is undertrained at feasible compute" (a
  methods/negative finding) vs the *intended* "noisy TV distracts RND exploration." Only
  the former is currently supported by data.
- **Undertraining vs collapse vs autoreset bug** — not mutually exclusive. The docs
  prescribe a cheap **matched-budget ablation (2–3M steps)** varying `ent_coef`
  (0.001↔0.01), the autoreset fix (on/off), and `num_envs`, with a **fresh same-budget
  PPO baseline**, *before* spending more compute. This has not been run.
- **Thesis-scope decision (explicitly flagged, undecided):** chase paper-scale numbers
  (hundreds of millions of steps + more parallel envs — may exceed the V100 allocation /
  thesis timeline) **vs** reframe the RND section around a *fixed feasible budget*,
  reporting the count-based/RND/PPO comparison and the collapse phenomenon as the finding.
- **Salvaging noisy-TV:** the ablation only becomes informative once RND explores with the
  TV off. Is it worth fixing RND first, or is "TV captures intrinsic reward share" (§3.4)
  already a reportable micro-result on its own?

## 6. Deliverable

Given the above, produce a prioritized next-steps plan: (a) the *minimum* experiment(s)
needed to disambiguate §5's tensions, sized to feasible compute; (b) a recommended thesis
framing for the RND results section under each plausible outcome; (c) an explicit
recommendation on the noisy-TV ablation's fate. State assumptions and stopping criteria.

_Artifacts: run data + overlaid figures in `analysis/` (`figures/`, `summary.csv`,
`README.md`); full diagnoses in the two `doc/` files cited in §4._
