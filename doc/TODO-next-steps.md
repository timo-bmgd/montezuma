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

- [ ] **2. Run the checkpoint probe (H1 mechanism).** On the cluster, run
  `scripts/probe_patch_response.py` on the ~10M checkpoint of each category
  (command in `noisy-tv-results.md` §2). Send back the CSV / printed rows. Gives
  `patch_contribution`, `content_sensitivity`, `G_proxy` per arm.

- [ ] **3. Early/late probe (demonstrates self-limiting capture).** Run
  `probe_patch_response.py` on **two** checkpoints of the *same* `remote` run:
  - early ≈ during the `tv_action_frac` spike (~4–5M → iteration ~1000–1200 →
    `ckpt_001000.pt`),
  - late ≈ ~10M (iteration ~2441 → `ckpt_002441.pt`).

  Compare `content_sensitivity`/`G_proxy`: **higher early, lower late** = the
  memorisation gap closed over training = self-limiting capture *demonstrated*
  (not just inferred). NOTE: this is `probe_patch_response.py` (loads a
  checkpoint), **not** `analyze_runs.py` (which overlays event files).

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
