# ⚠️ IMPORTANT — framing realizations for the thesis (Results & Discussion)

**Read this before writing chapters 4 (Results) and 5 (Discussion).** These are
interpretation decisions and corrections that came out of re-reading
`chapter/01_Introduction.tex` + `chapter/03_Methodology.tex` against the actual
run data (`analysis/HPC-Runs` seed 42, `analysis/Jupyter-Pod-Runs` seed 1/2).
Everything here is checked against the thesis's *own* definitions (§ references
are to `03_Methodology.tex`). Companion data doc: `doc/noisy-tv-results.md`.

Four things are load-bearing and easy to get wrong:

---

## ⚠️ 1. The result is **self-limiting capture** (outcome 3), NOT resistance

The methodology (`\label{HypothesisH1}`, §3.5) defines exactly three outcomes by
the **shape of the `tv_action_frac` trajectory**:

- **resistance** — no rise
- **sustained capture** — rise that persists
- **self-limiting capture** — *rise that decays* ("frequent remote presses raise
  the effective refresh rate toward the per-step regime, which closes `G`, which
  removes the incentive to press")

The data is the **third** case. In `remote`, `tv_action_frac` **rises to ~0.33
around 4–5M steps, then decays** to ~0.01 — and this reproduces in *both* remote
seeds (seed 42: 0.33→0.009; Jupyter seed 1: 0.23→0.010). `sham-remote` never
rises (stays ~0). A transient rise that decays, well above the `sham` null, is
the textbook self-limiting signature.

> **Correction of an earlier analysis note:** an intermediate draft described P2
> as "no sustained excess over sham → null upheld," which reads as *resistance*
> (outcome 1). That collapses three outcomes into two. The correct reading is
> **self-limiting capture** — arguably the more interesting result, and the one
> the methodology explicitly warns is "easy to miss if only the start of training
> is looked at."

**To claim it as *demonstrated* rather than *inferred*:** the self-limiting story
is a trajectory claim (the memorisation gap `G` open early → closed late). The
end-of-training checkpoint alone can't show that. Run `probe_patch_response.py`
on an **early (~4–5M, during the spike)** and a **late (~10M)** `remote`
checkpoint and compare `content_sensitivity`/`G`: higher early, lower late =
mechanism demonstrated. (This is the "early/late probe" in the TODOs.)

---

## ⚠️ 2. P1 as *worded* is FALSIFIED by the data — but H1 is not

P1 (§3.5) states: *"`tv_intrinsic_share` is elevated early **and decays over
training** as the predictor generalises over the patch. Falsifier: a flat,
non-decaying share."*

**The data meets P1's falsifier.** `tv_intrinsic_share` is elevated **and does
not decay — it *rises* and plateaus**: `remote` climbs to ~+0.14, `static` to
~+0.20, both flat/rising to the end, vs the ~−0.05 `off`/`sham` artifact floor.
So the literal prediction "the share decays" is not what happened.

**This does not sink H1 — it exposes that P1 conflates two senses of "generalise":**
1. *Drive the raw patch error to zero* (learn to predict the noise). The predictor
   **cannot** do this — per-pixel white noise is unpredictable — so the patch
   remains a persistent error source and the share stays high. As the predictor
   learns the *rest* of the frame (room 1), the patch becomes a **larger share**
   of the shrinking remaining error → the share *rises*. That is itself a
   signal-side finding: **the TV increasingly dominates the curiosity budget.**
2. *Become content-invariant* (the conditional-mean solution, §3.4 `CondMean`):
   give the same error to any patch content, so `G = err(fresh) − err(displayed)
   → 0`. The predictor **can** do this, and it is what H1's capture criterion
   (`G > 0` sustained) actually hinges on.

The occlusion share (P1) measures sense 1; the memorisation gap (H1) measures
sense 2. **The data can falsify P1 while supporting H1** — a rising share (can't
zero the noise) *with* `G ≈ 0` (content-invariant) is exactly what produces
self-limiting capture: persistent signal, no exploitable gap.

**Action for the thesis (your call, LaTeX):** either (a) reword P1 so its
falsifier is about the *gap* (content-sensitivity), not the raw occlusion share,
or (b) keep P1 and report the non-decaying share as a *falsification-with-
refinement* — "the raw share does not decay (P1 falsified as stated), but the
content-sensitivity probe shows the gap is closed, which is the mechanism P1 was
proxying for." **Do not report the rising share as if it confirmed P1.**

**Blocker for settling this:** whether `G ≈ 0` is real (content-invariant) or the
predictor genuinely isn't generalising must be decided by the **content-sensitivity
probe** (`probe_patch_response.py`). Until that runs, this stays "P1 falsified;
H1 pending the gap probe." Note the thesis already flags the direct
`tv_memorisation_gap` probe as **[TODO, unimplemented]** — the probe script now
exists (`scripts/probe_patch_response.py`); it just needs to be run on the
cluster checkpoints.

---

## ⚠️ 3. Keeping Montezuma central — the curiosity-diversion argument (for Discussion)

**The concern (valid):** the H1 *mechanism* — predictor generalises over a noise
patch — is environment-agnostic. With the agent stuck in room 1, Montezuma risks
reading as mere wallpaper behind the TV. The research question and contribution,
though, are Montezuma-specific *by construction* (§3.1: "Montezuma's Revenge is
the game RND was built to solve, so it is where the immunity claim is worth
testing cleanly").

**The defensible framing that keeps Montezuma essential *without* needing the
agent to explore far:** reframe the occlusion result (P1) as **curiosity
diversion**. In Montezuma the extrinsic reward is all zeros until the first key
(Background §`ExplorationProblem`: the first reward "requires a specific sequence
of jumps and a ladder climb before any points appear"), so **until that first
key, intrinsic reward is effectively the agent's only compass toward it.** A TV
that captures a *persistent, non-decaying, ~14–20% (and rising) share* of that
intrinsic reward is therefore diverting exactly the signal the agent depends on
to make its first progress — in the one game where that dependence is maximal.
That is *why* a noisy TV is dangerous here specifically and not in a dense-reward
game.

Precise wording that survives scrutiny (both reward streams stay live per §3.1,
`c_ext=2.0`, `c_int=1.0` — so do **not** say the experiment turns extrinsic off):

> *"Before the first key, the extrinsic stream is all zeros; intrinsic reward is
> the agent's only compass toward that first reward. The noisy TV is measured to
> capture a persistent, non-decaying share of exactly that signal."*

**Is "a difference in curiosity points toward better/worse handling" the argument?**
Refined: the clean, defensible claim is about how curiosity is **allocated** (TV
region vs game content) — that is the *mechanism* by which the TV would impair
exploration. The downstream *gameplay* consequence we can only observe partially
(key-grabs, rare room-2, §4 below) and must report **descriptively** — resources
limited our ability to see the full behavioural effect. So: *allocation is the
clean result; gameplay consequence is the resource-limited descriptive layer.*

**Honest scope limit (state it):** we cannot claim "the TV stops the agent
solving Montezuma," because the `off` baseline does not solve it either at this
budget. The claim is a *quantified, persistent diversion of the curiosity budget*
plus its *mechanism (self-limiting via a closing gap)* — not a demonstrated
behavioural collapse.

---

## ⚠️ 4. The first-key milestone (Timo's idea) — a finer, Montezuma-specific exploration read

**Motivation:** binary "left room 1" is too coarse and too noisy to carry a
result here. "Grabbed the first key" is a milestone the agent can plausibly reach
without full backtracking, and it is a genuinely Montezuma-specific behaviour.
**It is derivable from the already-logged `episodic_return`** (no new metric) —
so it fits P4's descriptive-only framing.

**Scoring structure (verified empirically):** across every run, the *only*
non-zero returns that ever occur are **exactly 100 and 400** — never 200/300. So:
- **return = 100** → grabbed the first key, still in room 1;
- **return = 400** → key **plus** progress beyond room 1 (the +300 requires
  leaving);
- the two milestones are **cleanly separable**, and "grab the key" is strictly
  the easier one.

![key-grab milestones](figures/regression/fig4_key_milestones.png)

Episodes reaching each milestone (single seed per arm — **descriptive only**):

| arm (seed, config) | grabbed key (≥100) | progress (≥400) | room≥2 eps | first key @ |
|---|---:|---:|---:|---|
| rnd `off` s2 (Jup) | **420** | 9 | 6 | ~5.5M |
| ppo `remote` s42 | 33 | 4 | 2 | ~0.3M |
| ppo `off` s1 (Jup) | 12 | 2 | 2 | ~0.78M |
| rnd `off` s42 | 9 | 1 | 1 | ~4.1M |
| rnd `remote` s1 (Jup) | 4 | 0 | 0 | ~4.1M |
| rnd `sham` s42 | 1 | 0 | 0 | ~0.05M |
| rnd `off` s1 (Jup) | 0 | 0 | 0 | — |
| rnd `remote` s42 | **0** | 0 | 0 | — |
| rnd `sham` s1 (Jup) | 0 | 0 | 0 | — |
| rnd `static` s1 (Jup) | 0 | 0 | 0 | — |

**What it shows (all descriptive, single-seed, read with the confounds below):**
1. **The first key IS learnable by RND** — `rnd off` seed 2 grabbed it **420
   times**, onset ~5.5M. So "grab the key" is a real, reachable milestone, not a
   fluke. (Your instinct to treat it as a big step is right.)
2. **Backtracking to leave is the bottleneck, not grabbing the key** — key-grabs
   ≫ progress everywhere (off s2: 411×100 vs 9×400; ppo-remote: 29×100 vs 4×400).
   This directly supports your hypothesis that the agent can reach the key but
   mostly cannot convert it into leaving the room (likely under-trained
   backtracking). Report this explicitly.
3. **Only the no-TV, 18-action baseline (`off` s2) ever *learned* the key**;
   every TV / extra-action arm (`remote`, `sham`, `static`) reached it 0–4 times.
   Suggestive that the patch and/or the added action impair first-key learning.
4. **PPO grabs the key more than most RND arms** (ppo-remote 33 vs rnd-remote 0)
   because PPO keeps a near-uniform (high-entropy) policy and keeps stumbling,
   while RND's entropy collapses and it stops trying. Worth a sentence in
   Discussion; ties to the pre-existing "RND underperforms PPO at this scale"
   thread.

**Confounds / caveats (must accompany the above):** one seed per arm (except
`off`); `off` itself is 0 in seed 1 and 420 in seed 2 (huge seed variance); the
"TV arms never learn the key" signal is confounded — `sham` (no patch, +1 action)
also fails, so it cannot be cleanly attributed to the TV vs the action-space
change. **Budget mismatch:** the table mixes 10M Jupyter runs (seed 1/2) with 20M
HPC runs (seed 42), and a 20M run has ~2× the episodes in which to stumble on a
key, so raw *counts* are not comparable across the two sets. The striking case is
`rnd off` s2 (Jupyter, **10M**) with **420** grabs vs `rnd off` s42 (HPC, **20M**)
with only **9** — the shorter run scored far more, so this is seed variance, not a
budget effect. The only apples-to-apples slice is the seed-42 HPC set (all 20M, one
config): `off` 9, `remote` 0, `sham` 1 — still n=1 and still swamped by the 0-vs-420
seed swing. A fairer cross-budget comparison would normalise to a *rate* (grabs per
episode or per M steps) rather than a raw count. So this is a **descriptive**
finding (P4), consistent with the TV and/or the added action impairing first-key
learning, *not* an inferential claim.

---

## Seed situation (realization, not a blocker)

- You have a **de-facto second seed** for `off`/`remote`/`sham` (the Jupyter
  runs), but they **do not compare completely** — they differ in three ways:
  `num_envs` 21 vs 32, budget 10M vs 20M, and they predate the NEXT_STEP GAE fix
  (Jupyter runs carry the old masking bug). **Do not pool the numbers.** Use them
  as a *reproduction-under-perturbation*: the qualitative P1/P2 effects repeat
  across a seed change *and* a config change, which for a descriptive study is a
  stronger robustness statement than a same-config replicate.
- **`static` has only ONE run ever** (seed 1, Jupyter, with the GAE bug) — the
  biggest single-seed gap. First check whether a seed-42 `static` run exists on
  `$SCRATCH` (you mentioned 10M seed-42 checkpoints for all four categories — if
  so, just pull its event file); otherwise it is the top candidate for a re-run.
- A background second-seed 32-env replication *broadens* the result but does not
  gate the writeup; P4/§3.12 already restrict these metrics to descriptive use
  (n=1–2 seeds, no significance test).

## Alignment with the thesis's own open items (from the methodology read)

The methodology already flags three **[TODO]s** that match findings here:
`tv_memorisation_gap` / direct `G` probe (now built: `scripts/probe_patch_response.py`),
the dedicated `frozen` wrapper mode (approximated by `static` at a large
`--tv-refresh-every`; a ~10-line change would make it a real mode), and the
offline calibration script (procedure described, not yet coded). So the gaps
flagged in `doc/regression-findings.md` §F are the thesis's own known TODOs, not
new surprises.
