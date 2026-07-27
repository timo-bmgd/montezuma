# TODO / next steps — noisy-TV thesis

Durable version of the running to-do list (was living in chat). Split into the
**core** items that the thesis result needs, and one **optional** extra to add
only if time allows. See `doc/thesis-framing-notes.md` for *why* each matters and
`doc/noisy-tv-results.md` §2 for the exact probe command.

---

## CORE — needed for the result (all cheap; no big new training)

- [ ] **1. Static seed check.** Look on `$SCRATCH/boomgaarden/montezuma/checkpoints`
  (and `.../runs`) for a **seed-42 `static`** run. You said 10M seed-42
  checkpoints exist for all four categories — if a `static__42` run is there,
  just **copy its TensorBoard event file** into `analysis/` (no re-run). If it
  genuinely slipped, run it (`run_rnd_tv.slurm` with `TV_MODE=static`). This is
  the biggest single-seed gap — `static` currently has only seed 1 (Jupyter, old
  GAE bug).

- [ ] **2. Run the patch-response probe sweep (H1 mechanism + self-limiting demo).**
  Now a one-command, cluster-native pipeline — no local download, non-blocking
  CPU jobs (they neither need a GPU nor compete with training for one):
  ```bash
  bash slurm/submit_probe_sweep.sh          # 12 CPU jobs: 3 seeds × 4 RND arms
  #   each traces ONE run's FULL checkpoint trajectory -> analysis/probe/probe_*.csv
  #   (SEEDS="42", ARMS="remote static", NUM_FRAMES=2048, DRY_RUN=1 all overridable)
  python scripts/plot_probe_trajectory.py --indir analysis/probe   # figures + summary
  ```
  The **dense trajectory** (all ~49 checkpoints/run) subsumes the old two-point
  early/late idea: watch `content_sensitivity` **elevated early, decaying late**
  for `remote`/`static` (= memorisation gap closing = self-limiting *demonstrated*,
  not inferred), read against the `off`/`sham` floor. `summary_probe.md` reports the
  peak→final `content_sens_drop` per arm; `patch_contribution` is the predictor-level
  P1 read. PPO arms are excluded (no predictor). Underlying primitive:
  `scripts/probe_patch_response.py` (now emits `iteration`/`global_step` columns);
  see `noisy-tv-results.md` §2. **Repeatable:** re-run after seed 44 lands — it just
  re-globs whatever checkpoints exist.

- [x] **3. Early/late probe** — *folded into #2.* The full-trajectory sweep
  demonstrates the gap closing across all checkpoints directly, so no separate
  two-checkpoint early/late run is needed. (`probe_patch_response.py` loads
  checkpoints; `analyze_runs.py` overlays event files — different tools.)

- [ ] **4. Decide the P1 wording (LaTeX).** The data falsifies P1 *as worded*
  (share is elevated **and rising**, meeting P1's "flat, non-decaying" falsifier),
  but not H1. Either reword P1 around the *gap* (content-sensitivity), or report
  it as falsification-with-refinement. See `thesis-framing-notes.md` §2.

- [ ] **5. (Optional robustness) Second-seed 20M matrix.** Background job; matches
  and broadens the seed-42 result. Does **not** gate the writeup (P4/§3.12 keep
  these metrics descriptive at n=1–2 seeds). Do only if time.

---

## OPTIONAL — add ONLY if time allows (the "Full path": P3 dose sweep)

**This is an optional extra, not required for the core result.** It substantiates
the thesis's "refresh rate = adjustable dose" contribution (§3.6 / Intro §128)
and tests P3 directly by drawing the **non-monotonic G(T) curve** — the "hump"
(small gap at every-step refresh, maximal at an intermediate interval, small
again at a frozen patch). Without it, P3 rests on the occlusion proxy and the
dose contribution stays asserted rather than demonstrated. The thesis already
flags its two prerequisites as `[TODO]` (§3.9 `frozen` mode; §3.10
`tv_memorisation_gap` probe).

**What it involves:**

- [ ] **A. Code change (~10 lines; needs a decision to implement).** Add a
  `frozen` end (`T = ∞`) to `NoisyTVWrapper` (`src/agents/base.py`): let
  `--tv-refresh-every 0` (or a `--tv-mode frozen`) mean "draw one patch at each
  episode reset, never resample within the episode," and relax
  `check_tv_geometry` (currently requires `refresh_every ≥ 1`). Add an assertion
  to `scripts/check_noisy_tv.py`. Decide once: per-episode redraw (natural, no
  RNG checkpointing) vs one patch for the whole run.
  - Also expose `--tv-refresh-every` in the SLURM launcher (one-line
    `REFRESH_EVERY` passthrough in `run_rnd_probe.slurm`), or just call
    `rnd.py` directly with the flag.

- [ ] **B. Runs — `static` mode, one seed, short (~5–8M each).** `static` has no
  behavioural channel, so it isolates the *dose→gap* relationship cleanly; room-1
  confinement is fine (`G` is a predictor property, not an exploration one).
  - `T = 1`: already have it (existing `static` checkpoint).
  - `T = 64`: `--tv-mode static --tv-refresh-every 64`, ~5M.
  - `T = ∞` (frozen): `--tv-mode static --tv-refresh-every 0` *(after step A)*, ~5M.

- [ ] **C. Probe + plot.** Run `probe_patch_response.py` on each run's checkpoint;
  plot `content_sensitivity` / `G` vs `T`; check for the predicted hump.
  - **Probe refinement likely needed:** the current probe measures the gap using
    *eval-time* fresh patches (fine at `T=1`). To measure the *true* gap at
    `T>1`, where a specific patch persists and can be memorised, extend the probe
    to evaluate the predictor on the **actual persisted patch** (regenerated from
    the seeded RNG) vs a fresh one. Otherwise read `content_sensitivity` across
    `T` as the proxy.

**Cost:** ~2 short runs (~2h each) + the code change + the probe ≈ ~1 day.
**Caveats:** the frozen endpoint's "small G" is the *prediction* P3 makes — the
sweep *tests* it, don't assume it; single seed; the probe refinement above.
