# Thesis Writing Guide — Does the Noisy-TV Problem Manifest for RND on Montezuma's Revenge?

> Companion to `doc/noisy-tv-experiment.md` (the runbook) and `doc/decisions.md`
> (the dated methodology log). This is the **thesis-writing guide**: chapter-by-chapter
> scaffold, the full theoretical argument, a condensed runbook, a results-interpretation
> decision tree, threats to validity, the figure/table list, and an annotated
> bibliography. Thesis is **noisy-TV-centric** (RND is the subject; count-based and PPO
> are baselines/controls, not a co-equal comparison), **English**, **full theory**.
>
> **Repo-state note (2026-07-25):** the production scale in `slurm/run_rnd_tv.slurm` is
> now **10M steps / `num_envs=32`** (A100 node, 32 cores/GPU), overridable via
> `NUM_ENVS` / `TOTAL_TIMESTEPS`. `doc/noisy-tv-experiment.md`'s header still reads
> "8 envs / SPS 144–183" and its final caveat still calls the `NEXT_STEP` GAE bug
> "open" — **both are stale**: the GAE bug was **fixed** 2026-07-22 (`decisions.md`;
> `agents.base.compute_gae` / `masked_mean`; `tests/test_gae_autoreset.py` 7/7), so all
> noisy-TV runs on current code use correct GAE. This guide uses the corrected numbers.

---

## 0. The thesis in one paragraph (paste-and-refine into your abstract)

Random Network Distillation (RND; Burda et al. 2018) is widely held to be robust to
the *noisy-TV problem* — the failure mode in which an intrinsically motivated agent
becomes transfixed by a source of irreducible stochasticity instead of exploring —
because its novelty signal is the prediction error against a **fixed, deterministic**
random target network, so there is no stochastic target to chase. This thesis tests
that claim empirically and adversarially on `ALE/MontezumaRevenge-v5`, a canonical
hard-exploration benchmark. I introduce a controllable synthetic "television": a patch
of per-pixel uniform noise injected into the agent's observation, in a variant where the
agent can *choose* to re-randomize it via a dedicated action (the "remote"). I measure
whether RND's intrinsic reward becomes dominated by the patch (*signal capture*) and
whether the policy learns to seek it at the expense of game exploration (*behavioral
capture*), against an action-space-matched control and a no-intrinsic-reward (PPO)
control. [Result sentence — fill in after runs.] The finding [confirms / qualifies /
refutes] the robustness claim under a compute budget (32 parallel envs, 10M frames —
~200× fewer frames than the original paper) typical of a single-GPU academic setting.

**Research question (RQ).** Under a controllable stochastic distractor, does RND's
deterministic-target novelty signal remain uncaptured — in its intrinsic reward and in
the resulting policy — on Montezuma's Revenge at academic scale (32 envs, 10M frames)?

**Contribution.** (1) A minimal, controllable noisy-TV apparatus for ALE that isolates
the phenomenon with matched controls; (2) a two-axis measurement (signal vs behavioral
capture) with an occlusion-based decomposition of intrinsic reward; (3) an empirical
verdict on the RND robustness claim in the under-scale regime where it is actually
deployed by most practitioners, not the ~1000×-frame regime of the original paper.

---

## PART A — CHAPTER-BY-CHAPTER SCAFFOLD

For each chapter: its job, what to write, which results/figures land there, and the
traps to avoid.

### A.1 Introduction (~3–5 pp)
- **Job:** motivate exploration in sparse-reward RL; name Montezuma's Revenge as the
  emblematic hard-exploration game; introduce intrinsic motivation and the noisy-TV
  failure mode as its Achilles' heel; state the RND robustness claim; pose the RQ.
- **Write:** the funnel — sparse reward → intrinsic motivation → the noisy-TV objection →
  RND's proposed immunity → "but is it immune *at the scale people actually run it*?" →
  RQ and contributions. Close with a one-paragraph roadmap of the chapters.
- **Figure:** one screenshot of Montezuma with the injected patch (from a recorded video),
  captioned to show the reader what the "TV" literally is. This single image does a lot
  of expository work.
- **Trap:** do not overclaim novelty of the *phenomenon* (noisy-TV is well known). Your
  novelty is the **controllable, matched-control test of RND specifically at academic
  scale**. Say that plainly.

### A.2 Background & Related Work (~6–10 pp)
- **Job:** give the reader the machinery (PPO, GAE, RND) and situate the noisy-TV problem
  in the intrinsic-motivation literature.
- **Write, in this order:**
  1. **RL preliminaries:** MDP, policy gradients, PPO (Schulman et al. 2017), GAE
     (Schulman et al. 2015). Keep terse — cite, don't re-derive.
  2. **Exploration & intrinsic motivation:** count-based/pseudo-counts (Bellemare et al.
     2016; Tang et al. 2017 — the SimHash method your `count_based.py` implements),
     prediction-error curiosity / ICM (Pathak et al. 2017), RND (Burda et al. 2018),
     Go-Explore (Ecoffet et al. 2019/2021) as the SOTA context. Use the benchmark table
     (§D.1).
  3. **The noisy-TV problem:** conceptual origin in Schmidhuber's curiosity work
     (unpredictable *and* uncontrollable stimuli); the empirical demonstration for
     forward-dynamics curiosity in Burda et al. 2018b ("Large-Scale Study of
     Curiosity-Driven Learning", the Unity-maze TV); ICM's inverse-dynamics features and
     Savinov et al. 2018 (episodic curiosity) as prior *mitigations*. Then: RND's claim
     that a deterministic target sidesteps it. **This is the paragraph your whole thesis
     interrogates — write it carefully and quote the RND paper's exact wording.**
  4. **ALE methodology:** sticky actions and the recommended evaluation protocol (Machado
     et al. 2018); why `repeat_action_probability=0.25` matters for your design.
- **Trap:** distinguish the two Burda-2018 papers (RND = arXiv:1810.12894; Large-Scale
  Study = arXiv:1808.04355). Reviewers notice conflation. Also: your repo's count-based
  method is Tang-2017 SimHash, **not** Bellemare-2016 pseudo-counts — the benchmark table
  row is Bellemare's; say so (this exact caveat is already in `CLAUDE.md`).

### A.3 The Noisy-TV Apparatus (Method) (~5–8 pp)
- **Job:** specify the intervention precisely enough to reproduce, and justify every
  design choice. This is where your engineering rigor becomes thesis credit.
- **Write:** pull directly from `doc/noisy-tv-experiment.md` and `doc/decisions.md`
  (2026-07-19 entry). Cover, each with its rationale:
  1. **Where the noise is injected** (between `AtariPreprocessing` and
     `FrameStackObservation`, on the processed 84×84 frame). Justify: `AtariPreprocessing`
     reads the ALE screen directly, so a wrapper below it is invisible; injecting above
     it gives exactly one noise sample per agent step with no frame-skip/max-pool/resize
     attenuation of the noise statistics. Both the policy (4-frame stack) and the RND
     predictor/target (newest frame) see the patch → the trap is *visible and learnable*.
  2. **The four modes** (`off`, `static`, `remote`, `sham-remote`) as a factorial that
     separates two orthogonal factors: *has-patch* and *has-remote-action*. Present the
     mode table (§C.1). Emphasize `remote` as primary and *why* `static` cannot show
     behavioral capture.
  3. **Patch geometry** — the 12×84 HUD strip and the **empirical reason it is that shape,
     not a small square**: (a) the playfield starts at row 12 in every room, so the patch
     must live in the HUD band; (b) RND's per-pixel observation normalization standardizes
     *any* stationary noise to ~unit variance, so **amplitude is not a lever — area is**;
     a checkpoint probe showed a 12×12 patch perturbs the target network's output by only
     ~7%/dim and total prediction error by ~1%. Report the probe as a mini-result (it is
     a genuine finding about RND's obs-normalization). Disclose that the strip masks the
     on-screen score pixels — HUD, not gameplay; logged `episodic_return` comes from the
     emulator and is unaffected.
  4. **Noise model & reproducibility** — iid per-pixel uniform, per-sub-env RNG salted
     from the run seed, persistent across autoresets; one free resample at each episode
     start. State reproducibility guarantees and the `check_noisy_tv.py --hash-off`
     byte-identical result (feature off ⇒ pre-existing pipeline unchanged).
  5. **The remote action & sticky actions** — the added 19th action maps to NOOP in the
     game and re-randomizes the patch. **Disclose the asymmetry your mentor asked you to
     own:** sticky actions (0.25) act on the *executed* action, but the resample triggers
     on the *chosen* action, so the remote is the one perfectly reliable action in the
     game — which, if anything, makes the trap *easier* to learn, biasing toward finding
     capture, not away. This is a favorable-to-rejection bias; naming it strengthens the
     paper.
- **Trap:** a reviewer will ask "why is masking the score not a confound?" Pre-empt it:
  reward is read from RAM/emulator, not pixels; the only pixel-novelty lost is the score
  digits changing, which at Montezuma's scoring frequency is negligible and is *equal
  across all patched conditions*.

### A.4 Experimental Setup (~3–4 pp)
- **Job:** hyperparameters, hardware, the run matrix, seeds, evaluation protocol, metrics.
- **Write:**
  - **Hyperparameters** as a table (§C.3), flagged as **paper-matched** (Burda et al.
    2018 / CleanRL `ppo_rnd_envpool`): `int_coef 1.0`, `ext_coef 2.0`, `int_gamma 0.99`,
    `gamma 0.999`, `ent_coef 0.001`, `lr 1e-4`, `num_steps 128`, `update_proportion 0.25`,
    `obs_norm_init_steps 50`. **Scale:** `num_envs 32`, `total_timesteps 10M`
    (batch = 32×128 = 4096). State the **scale gap** explicitly: the paper used ~128
    envs and ~1.6–2×10⁹ frames; you run 32 envs × 10⁷ frames — ~4× fewer envs and ~200×
    fewer frames. This is not a weakness to hide; it is *the regime your RQ is about*
    (what practitioners actually run).
  - **Correct GAE.** State that the runs use the `NEXT_STEP`-autoreset GAE fix
    (`agents.base.compute_gae`, verified by `tests/test_gae_autoreset.py`), so value
    bootstrapping never bridges two episodes — a correctness precondition, not a variable.
    (This *removes* a confound that distorted the historical baselines; see §E.3.)
  - **The run matrix** (§C.2), seeds {1, 2} on the core RND conditions.
  - **Evaluation protocol:** sticky actions on (Machado et al. 2018); rooms-visited
    (from RAM byte 3 via `RoomTracker`) as the primary exploration metric and mean
    episodic return (true game score, logged before reward clipping) as secondary; the
    two capture metrics as the primary *dependent variables*.
  - **Hardware & throughput:** A100-80GB, ~1320 SPS at `num_envs=32` (measured
    2026-07-24) ⇒ ~2.1 h per 10M-step run; chained-resume workflow via `RESUME_FROM`.
    Note auto-stop (exit 42) semantics.
- **Trap:** report `episodic_return` as *true score* and state that reward clipping to
  [−1,1] applies only to the training signal (matches RND paper preprocessing).

### A.5 Results (~6–10 pp) — write after the runs finish
- **Job:** present, don't interpret (interpretation goes in Discussion; a light "as
  expected / unexpectedly" is fine).
- **Structure by dependent variable, not by run:**
  1. **Behavioral capture:** `tv_action_frac` vs training step, remote (#2) vs
     sham-remote (#3) vs PPO-remote (#5), with the 1/19 chance line. Your money figure.
  2. **Signal capture:** `tv_intrinsic_share` (and its `full`/`occluded` components) for
     remote/static vs the off/sham calibration floor.
  3. **Exploration outcome:** `rooms_visited` and `episodic_return`, remote vs off vs
     sham. **Check the fresh `off` baseline (#1) first (see §E.2):** on the fixed 32-env
     code it may actually explore (unlike the buggy-8-env historical runs), which would
     restore dynamic range to this axis; if it still collapses, lean on the capture
     metrics, which don't require the baseline to explore.
  4. **Intrinsic-reward trajectory:** `raw_intrinsic_rew_mean` for TV vs the off baseline.
  5. **The 12×84 geometry probe** as a standalone result (RND obs-norm ⇒ area-not-
     amplitude).
- **Report** per-seed curves + mean; state exit codes (42 vs 0) per run.

### A.6 Discussion (~4–6 pp)
- **Job:** answer the RQ, map the observed pattern onto the outcome taxonomy (§D.2),
  reconcile with the paper's claim, and handle confounds honestly.
- **Write:** which of Outcomes A–D you observed and what it implies about the
  deterministic-target argument; the **static-vs-remote regime distinction** (§B.4) as
  the interpretive key; the interaction with any residual under-scale weakness of the
  baseline (§E.2). If you find **signal capture without behavioral capture** (the most
  likely outcome given the weak-stimulus probe), that is a *nuanced, defensible, and
  publishable* result: RND's intrinsic reward is polluted by controllable stochasticity
  but the pollution is too weak to steer the policy at this scale.
- **Trap:** do not let "RND resisted the TV" and "the agent under-explored for unrelated
  reasons" be confused. Separate *immunity to the TV* from *general exploration failure*.

### A.7 Conclusion & Future Work (~2 pp)
- **Job:** one-paragraph restatement of the verdict; limitations in one paragraph;
  concrete future work.
- **Future work worth naming:** larger compute (close the ~200× frame gap toward the
  paper); **amplitude via a non-stationary noise distribution that obs-norm cannot
  flatten** (the real way past the area-only ceiling, §B.5); other games; a "TV that
  shows structured images" (closer to the original Unity-maze TV) rather than white
  noise; measuring capture as a function of extrinsic-reward density; and the noisy-TV
  test applied to the count-based / AE-SimHash agents already in the repo.

---

## PART B — FULL THEORETICAL DEVELOPMENT (the intellectual core)

Put a condensed version of this in Background/Method and the full argument in Discussion.

### B.1 RND, formally
Let `x` be a (normalized) observation. A fixed, randomly initialized **target**
`f: x ↦ ℝ^k` is frozen at init. A **predictor** `f̂_θ` is trained by regression to match
it. The intrinsic reward is the prediction error
```
    r_i(x) = ½ · ‖ f̂_θ(x) − f(x) ‖²
```
(your `rnd.py` computes exactly this — `_rnd_error`, half-summed squared error over the
512-dim heads, on the **newest** frame of the stack, obs-normalized and clipped to ±5).
The predictor is trained on a `update_proportion=0.25` random subset of each batch. Novel
states have high error → high intrinsic reward; as the predictor fits a region, its
reward decays. (The GAE/autoreset fix affects how *advantages* are computed from these
rewards, not the reward definition itself.)

### B.2 Why the deterministic target is claimed to defeat the noisy-TV
Prediction-error curiosity for **forward dynamics** predicts the *next* observation
`x'` from `(x, a)`. If the environment (or a TV) injects irreducible noise into `x'`,
the target is a random variable and the expected error has an irreducible floor
`≥ Var[x' | x, a]` — the agent is paid forever for standing in front of the noise. RND's
target is `f(x)`, a **deterministic function of the current observation**: for any *fixed*
`x`, `f(x)` is a constant vector, so a sufficiently expressive predictor can drive
`r_i(x) → 0` by regression. "There is no stochastic target to chase" — hence the claimed
immunity.

### B.3 The subtlety this thesis exploits: generalization vs memorization over the noise manifold
The deterministic-target argument is airtight for a *finite or recurring* state set. A
noise patch breaks that premise: the set of observations containing the patch is
effectively **infinite and (almost surely) never-repeating**. The predictor can only
make `r_i → 0` on that set by **generalizing** `f` across the noise manifold, i.e.
learning that the noise pixels do not change the "meaning" of the state — not by
memorizing individual frames. Two sub-cases:

- **If the target `f` is (near-)invariant to the patch pixels** — `f(x_noise) ≈
  f(x_no-noise)` — the predictor generalizes trivially and intrinsic reward on the patch
  region vanishes. **RND resists.** Your 12×12 probe measured exactly this sensitivity and
  found it *small* (~7%/dim; ~1% of total error): a random-init Nature-CNN target is only
  weakly sensitive to a small high-frequency patch. This is *why amplitude fails as a
  lever* and why you enlarged the patch to a 12×84 strip (area is the only knob that
  raises `f`'s sensitivity to the patch).
- **If `f` is sensitive to the patch** — each fresh noise sample yields a target vector
  the predictor has not seen — error persists → intrinsic reward persists → **capture is
  possible.**

So the empirical question reduces to: **is a random-init CNN target sufficiently sensitive
to a controllable noise strip that the predictor cannot generalize it away before the
policy learns to exploit it?** — a question the original robustness claim never quantified.

### B.4 Two regimes: `static` (per-step) vs `remote` (on-demand) — the interpretive key
This distinction is the theoretical heart of the design and the most citable idea in the
thesis.

- **`static`, resampled every step.** The predictor is trained on a *fresh iid noise
  sample every step*, so the empirical training distribution of patches **is** the full
  noise distribution. By SGD it converges to `f̂(x) ≈ E_noise[f(x)]` on the patch region —
  i.e. it learns **noise-invariance** — and `r_i → 0`. This is the "RND resists"
  outcome *by construction*, and your smoke run confirmed it: `tv_intrinsic_share` fell
  from 0.10–0.21 to ≈0 within ~100k steps. Static therefore mostly bounds the *signal-
  degradation* question and provides a within-run demonstration that the predictor *can*
  learn invariance when forced to.
- **`remote`, resampled only on press.** Between presses the patch **repeats**, so
  memorizing the *current* patch (low error) beats learning invariance; but **each press
  produces a novel patch → an error spike → a positive intrinsic reward**. An agent that
  presses is *rewarded for pressing*. This creates a candidate **positive-feedback loop**:
  press → intrinsic reward → policy-gradient reinforces pressing → press more. Behavioral
  capture, if it occurs, lives here. The open question is whether this loop's per-press
  reward (bounded by the *weak* target sensitivity of §B.3) is large enough, after
  reward normalization and against Montezuma's own (sparse) extrinsic gradient, to
  actually steer the policy.

**Prediction from theory + probe:** signal pollution in `remote` is real but weak;
whether it crosses the threshold for behavioral capture is genuinely uncertain and is
exactly what the experiment decides. The most likely outcome is **Outcome C** (§D.2):
measurable signal capture, little or no behavioral capture — RND *partially* vulnerable.

### B.5 The observation-normalization interaction (why "area, not amplitude")
RND normalizes observations with a running per-pixel mean/variance (`obs_rms`), clipped
to ±5, initialized from a random-action rollout that (in your design) **runs with the TV
active**, so the patch pixels are normalized to ~unit variance regardless of the raw
uniform amplitude (std ≈ 73.6). Consequence: you **cannot** make the TV "louder" by
scaling pixel intensity — normalization flattens it. The only levers that survive
normalization are (a) **spatial area** (more patch pixels → more of `f`'s input perturbed)
and (b) **temporal structure** (non-stationary noise whose statistics `obs_rms` cannot
track — future work, §A.7). This is a real, reportable property of RND, not just an
implementation detail; give it a subsection.

---

## PART C — CONDENSED RUNBOOK (reference while running/writing)

### C.1 Conditions (the factorial)
| Mode | Action space | Patch present? | Remote action? | Tests |
|---|---|---|---|---|
| `off` | Discrete(18) | no | no | baseline; byte-identical to pre-TV code |
| `static` | Discrete(18) | yes, per-step | no | signal degradation (no behavioral channel) |
| `remote` | Discrete(19) | yes, on press | yes | **primary** — signal + behavioral capture |
| `sham-remote` | Discrete(19) | no | yes | action-space-matched control |

Two orthogonal factors: *has-patch* (off/sham vs static/remote) and *has-remote-action*
(off/static vs remote/sham). `remote − sham` isolates the TV's effect holding the action
space fixed; `sham − off` isolates the pure cost of a larger action space.

### C.2 Run matrix (10M steps, 32 envs)
| # | Agent | Mode | Seeds | Priority | Purpose |
|---|---|---|---|---|---|
| 1 | RND | off | 1, 2 | core | fresh baseline on fixed current code |
| 2 | RND | remote | 1, 2 | **core (primary)** | behavioral + signal capture |
| 3 | RND | sham-remote | 1, 2 | core | action-space-matched control |
| 4 | RND | static | 1 (+2 if budget) | core-secondary | signal degradation only |
| 5 | PPO | remote | 1 | core | no-intrinsic control (capture should NOT occur) |
| 6 | PPO | off | 1 | optional | fresh PPO reference |
| 7 | RND | remote, `--tv-size 6 6` / `18 84` | 1 | optional | patch-area sensitivity |

At ~1320 SPS a 10M-step run is ~2.1 h; the ~8 core runs are ~15–20 GPU-h total. **Cheap
pilot:** capture/collapse signatures historically appear <1M steps, so a 3M-step
`TOTAL_TIMESTEPS=3000000` first pass on #1/#2/#3 answers the qualitative question in <1 h
before committing to full 10M runs. Launch:
`sbatch --export=ALL,SEED=1,TV_MODE=remote slurm/run_rnd_tv.slurm` (from repo root), or
the `TV_JOBS` cell in `training-runs.ipynb`. Run `python scripts/check_noisy_tv.py` once
before the first launch.

### C.3 Hyperparameters (all paper-matched; only scale differs)
`env=ALE/MontezumaRevenge-v5`, sticky actions `p=0.25`, 4-frame stack, 84×84 grayscale,
`frame_skip=4`, `terminal_on_life_loss=False`. `lr 1e-4` (annealed), `num_steps 128`,
`num_minibatches 4`, `update_epochs 4`, `clip_coef 0.1`, `ent_coef 0.001`, `vf_coef 0.5`,
`gamma 0.999` (ext, episodic), `int_gamma 0.99` (int, non-episodic), `gae_lambda 0.95`,
`int_coef 1.0`, `ext_coef 2.0`, `update_proportion 0.25`, `obs_norm_init_steps 50`.
**Scale:** `num_envs 32`, `total_timesteps 10M`, batch 4096. GAE via the fixed
`compute_gae`. Scale vs Burda et al. 2018: ~32 vs ~128 envs, 10⁷ vs ~10⁹ frames.

### C.4 Metrics glossary (exact TensorBoard tags)
| Tag | Meaning | Healthy / null | Capture signature |
|---|---|---|---|
| `charts/tv_action_frac` | fraction of chosen actions = remote (remote/sham only) | ≈ 1/19 ≈ 0.053 | ≫ 0.053 and **rising** (remote only) |
| `charts/tv_intrinsic_share` | (full − occluded)/full intrinsic reward; patch's share | ≈ 0 (off/sham floor) | elevated & sustained (remote/static) |
| `charts/intrinsic_rew_full_diag` / `..._occluded` | components of the above | — | full ≫ occluded |
| `charts/raw_intrinsic_rew_mean` | mean pre-normalization RND error | non-zero; may decay | stays elevated (TV feeds it) |
| `charts/rooms_visited` | distinct rooms (RAM byte 3) | rises past 1 | remote < off (cost of capture) |
| `charts/episodic_return` | true game score | > 0 eventually | remote < off |
| `losses/entropy` | policy entropy | slow decay | collapse onto action 18 (remote) |
| `charts/collapse_streak` / exit code | auto-stop bookkeeping | 42 = collapse detected | 42 in remote/static = unexpected |

Chance line for `tv_action_frac` = **1/19 ≈ 0.0526**. Read `tv_intrinsic_share` as a
**trajectory/trend**, not a point value (the occluded input is off-distribution for the
predictor, so the decomposition is approximate — say this in the caption).

---

## PART D — REFERENCE TABLES

### D.1 Literature benchmarks (for Related Work; from `CLAUDE.md`)
| Method | Rooms | Mean score | Note |
|---|---|---|---|
| Count-based pseudo-counts (Bellemare et al. 2016) | 15 | ~3,700 | *your `count_based.py` is Tang-2017 SimHash, a different method — say so* |
| RND (Burda et al. 2018) | ~24 | ~10,000 | the subject of this thesis; achieved at ~200× your frame budget |
| Go-Explore (Ecoffet et al. 2019/2021) | 37 / 238 | 43,000 / 650,000 | without / with domain knowledge; SOTA context |
| **Historical RND (10M, 8 env, buggy GAE)** | **1** | **0** | `rnd_10m_s1/s2` collapsed — an under-scale, pre-fix data point (see §E.2/E.4) |
| **Historical PPO (5M, 8 env, buggy GAE)** | **2** | **400** | `ppo_5m_s1` — the (now-superseded) RND<PPO asymmetry |

The historical rows are **pre-fix** (buggy GAE, 8 envs). Treat them as motivating context
for the collapse concern, not as the baseline your TV runs are compared against — that is
the *fresh* `off` run (#1) on fixed 32-env code.

### D.2 Outcome taxonomy (fill the observed row; drives the Discussion)
| Outcome | `tv_action_frac` (remote) | `tv_intrinsic_share` (remote) | rooms/return vs off | Interpretation |
|---|---|---|---|---|
| **A — RND resists** | ≈ chance | → 0 | no worse | deterministic target + generalization defeat even a controllable trap at this scale; robustness claim upheld |
| **B — Behavioral capture** | ≫ chance, rising | elevated | worse | RND is **not** immune; controllable stochasticity captures the policy |
| **C — Signal-only capture** (likeliest per probe) | ≈ chance | elevated | ≈ unchanged | intrinsic reward polluted but too weak to steer policy; RND *partially* vulnerable |
| **D — Static degrades, remote doesn't** | ≈ chance | remote→0, static transient | no worse | predictor generalizes even the per-step case; supports §B.4 invariance argument |

---

## PART E — THREATS TO VALIDITY / LIMITATIONS (write this chapter honestly — it earns marks)

1. **External validity — one game, one architecture.** Montezuma + Nature-CNN only.
   Don't generalize "RND is/isn't immune" beyond "…on this game at this scale." Name it.
2. **Baseline exploration strength (the axis-range caveat).** Historical RND runs — on
   **buggy-GAE, 8-env** code — never left room 1 and scored 0; PPO at half the budget
   reached room 2 / 400. On the **fixed-GAE, 32-env** code your fresh `off` baseline (#1)
   may explore more, restoring dynamic range to the rooms/return axis — but this is *not
   yet established*. **Check #1 first.** If it still collapses, the exploration-cost axis
   is underpowered (you cannot lose exploration the baseline never had), so lean the
   verdict on the **capture metrics** (`tv_action_frac`, `tv_intrinsic_share`), which are
   meaningful regardless of baseline exploration, and report exploration deltas as
   secondary.
3. **A prior correctness bug — now fixed (strength, disclose as such).** The gymnasium
   `NEXT_STEP` autoreset vs GAE done-masking defect (`doc/decisions.md`, fixed 2026-07-22
   via `agents.base.compute_gae` + update-loop `masked_mean`, verified by
   `tests/test_gae_autoreset.py`) distorted the *historical* baselines. Your noisy-TV runs
   use the **corrected** code, so value bootstrapping never bridges episodes — this
   *removes* a confound rather than holding one constant. State it explicitly and cite the
   test; it is a methodological plus, not a caveat. (Precondition: run the matrix on
   fixed-code `main`/the merged branch, not an old checkout.)
4. **Historical RND-vs-PPO asymmetry — out of scope.** `doc/rnd-vs-ppo-asymmetry-
   investigation.md` (RND underperforming plain PPO on the old code) is now **superseded
   background**, separately characterized on the fixed code
   (`doc/matched-budget-submission.md`). It is *context on baseline health*, not a result
   of this study; mention once, don't build on it.
5. **Amplitude ceiling from obs-normalization (§B.5).** You cannot make the TV arbitrarily
   salient; area is the only stationary lever and the playfield boundary caps it at the
   12×84 strip. A truly overpowering distractor would need non-stationary noise (future
   work). State that a "capture did not occur" verdict is conditional on stimulus strength
   ≤ what a normalization-surviving stationary patch can deliver.
6. **Sticky-action asymmetry (favorable bias).** The remote is the only non-sticky
   (perfectly reliable) action → biases *toward* capture. Your design is therefore a
   *conservative* test of the robustness claim: if capture still doesn't occur, the claim
   is well supported; if it does, the reliable-action advantage is a partial explanation.
7. **Small seed count.** 2 seeds on core RND conditions; report per-seed curves, avoid
   significance claims you can't support, treat trends qualitatively.
8. **Local vs GPU logging quirk (only if you cite any local `--sync-envs` numbers).** On
   CPU, `charts/raw_intrinsic_rew_mean` aliases the normalized buffer (documented in
   `doc/decisions.md`); GPU/production numbers are unaffected. Use production (GPU) runs
   for any reported figure.

---

## PART F — FIGURES & TABLES TO PRODUCE (with source runs)
1. **Fig. Apparatus** — Montezuma frame with the 12×84 patch (from a `remote` recorded
   video). *Intro/Method.*
2. **Fig. Behavioral capture** — `tv_action_frac` vs step: remote(#2, both seeds), sham
   (#3), PPO-remote(#5); chance line at 0.053. *Results, headline.*
3. **Fig. Signal capture** — `tv_intrinsic_share` vs step: remote(#2), static(#4) vs
   off(#1)/sham(#3) floor. *Results.*
4. **Fig. Intrinsic trajectory** — `raw_intrinsic_rew_mean`: TV runs vs the off baseline.
   *Results/Discussion.*
5. **Fig. Exploration** — `rooms_visited` and/or `episodic_return`: remote vs off vs
   sham. *Results (flag underpower per §E.2 if the baseline under-explores).*
6. **Fig. Geometry probe** — target-network output sensitivity / intrinsic-error share vs
   patch area (12×12 → 12×84, run #7). *Method or Results — the RND obs-norm finding.*
7. **Table. Hyperparameters** (§C.3). **Table. Run matrix** (§C.2). **Table. Outcome
   taxonomy with observed row** (§D.2). **Table. Final per-condition summary** (rooms,
   score, `tv_action_frac` final, `tv_intrinsic_share` final, exit code, per seed).

The notebook cells (status table + 2×2 plot grid over TV runs) produce raw material for
Figs 2–5 directly from the event files.

---

## PART G — ANNOTATED BIBLIOGRAPHY (core citations)
- **Burda, Edwards, Storkey, Klimov (2018/2019), "Exploration by Random Network
  Distillation", arXiv:1810.12894.** *The* subject. Cite for the RND method, the
  deterministic-target argument, and the Montezuma numbers. Quote its noisy-TV claim.
- **Burda, Edwards, Pathak, Storkey, Darrell, Efros (2018), "Large-Scale Study of
  Curiosity-Driven Learning", arXiv:1808.04355.** The empirical noisy-TV demonstration
  (Unity maze + TV) for prediction-error curiosity. Your motivating prior evidence.
- **Pathak, Agrawal, Efros, Darrell (2017), "Curiosity-driven Exploration by
  Self-supervised Prediction" (ICM), arXiv:1705.05363.** Inverse-dynamics features as a
  *mitigation* of uncontrollable-distractor capture — the pre-RND state of the art.
- **Schmidhuber (1991; 2010), artificial curiosity / formal theory of creativity.**
  Conceptual origin of "predictable-but-uncontrollable" vs "learnable" novelty — the
  intellectual root of the noisy-TV intuition.
- **Savinov et al. (2018), "Episodic Curiosity through Reachability", arXiv:1810.02274.**
  Another noisy-TV-motivated method; related-work breadth.
- **Bellemare, Srinivasan, Ostrovski, Schaul, Saxton, Munos (2016), "Unifying Count-Based
  Exploration and Intrinsic Motivation", arXiv:1606.01868.** Pseudo-counts; the 15-rooms
  benchmark row.
- **Tang et al. (2017), "#Exploration: A Study of Count-Based Exploration for Deep RL",
  arXiv:1611.04717.** SimHash — the method your `count_based.py` actually implements.
- **Ecoffet, Huang, Lehman, Stanley, Clune (2019/2021), "Go-Explore", arXiv:1901.10995
  / Nature.** SOTA on Montezuma; upper-bound context.
- **Machado, Bellemare, Talvitie, Veness, Hausknecht, Bowling (2018), "Revisiting the
  Arcade Learning Environment", arXiv:1709.06009.** Sticky actions + evaluation protocol.
  Cite for `repeat_action_probability=0.25` and rooms/score reporting.
- **Bellemare, Naddaf, Veness, Bowling (2013), "The Arcade Learning Environment", JAIR.**
  The benchmark itself.
- **Schulman et al. (2017), "Proximal Policy Optimization", arXiv:1707.06347;** **Schulman
  et al. (2015), "GAE", arXiv:1506.02438.** The RL backbone (GAE is exactly what your
  `compute_gae` implements, and where the autoreset masking matters).
- **Huang et al. (2022), "CleanRL", JMLR.** The single-file implementation style your
  agents follow (methodological reproducibility note).

---

## PART H — PRE-SUBMISSION DISCLOSURE CHECKLIST (things a reviewer will look for)
- [ ] Scale gap vs the RND paper stated numerically (~32 vs 128 envs; ~200× fewer
      frames), framed as the regime of interest.
- [ ] Runs are on the **GAE-fixed** code; cite `tests/test_gae_autoreset.py`.
- [ ] Sticky-action asymmetry disclosed as a conservative (capture-favoring) bias.
- [ ] Patch geometry rationale (area-not-amplitude; obs-norm) explained, with the probe.
- [ ] Score-masking non-confound argued (reward from emulator, not pixels).
- [ ] Baseline-strength caveat: `off` (#1) checked first; verdict leans on capture metrics
      if the baseline under-explores.
- [ ] RND-vs-PPO asymmetry cited once as superseded background, not built upon.
- [ ] `static` = per-step invariance regime vs `remote` = on-demand memorization regime
      explained as the interpretive key.
- [ ] Count-based method = SimHash (Tang 2017), not pseudo-counts (Bellemare 2016).
- [ ] Two Burda-2018 papers cited distinctly.
- [ ] Reproducibility: seeds, `--hash-off` byte-identical off-path, RNG salting, exact
      hyperparameters, code/commit reference.
- [ ] Mentor's delegated decision (remote-as-primary) owned in the methodology, with its
      consequences (needs a free action; the sticky-action asymmetry) followed through.

---

*Source material in the repo:* `doc/noisy-tv-experiment.md` (runbook — note its header
scale/GAE caveat lags this guide), `doc/decisions.md` (dated methodology log incl. the
2026-07-19 noisy-TV entry, the 2026-07-19 geometry-probe finding, and the 2026-07-22
GAE-fix entry), `doc/matched-budget-submission.md` + `doc/rnd-vs-ppo-asymmetry-
investigation.md` + `doc/10M-RND-run-failure-documentation.md` (baseline-collapse
context, superseded), `src/agents/rnd.py` / `base.py` (ground-truth defaults, the
wrapper, `compute_gae`), `scripts/check_noisy_tv.py` + `tests/test_gae_autoreset.py`
(validation), `CLAUDE.md` (benchmarks, current-focus framing, conventions).
