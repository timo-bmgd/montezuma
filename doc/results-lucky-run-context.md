# Context notes for results-chapter section: "The lucky remote run" (rnd_tv_remote, seed 43)

Working material for writing the results section by hand — episode-level anatomy of the
first acquisition burst, direct answers to the how-did-it-look questions, the zoomed
figure, and every number's provenance. The companion *discussion*-chapter draft (the
luck/consolidation interpretation) is `doc/discussion-lucky-run.md`; this file sticks to
observables, as befits a results chapter.

Run: `analysis/HPC-Runs/runs/ALE/MontezumaRevenge-v5__rnd_tv_remote__43__1785112914`
(20M steps, 32 envs, 3.0 h wall clock on kiwinode01).
Figure: `doc/figures/lucky-run-burst_s43.{pdf,png}` (`scripts/plot_lucky_run_burst.py`).
Full-run overview figure: `doc/figures/lucky-run-rnd_tv_remote_s43.{pdf,png}`.
All 568 multi-room episodes as CSV: `doc/lucky-run-multiroom-episodes.csv`
(columns: record index, global step, rooms, return, length, gap to previous multi-room
episode in steps and in episodes).

## Q1 — Was there one specific super-lucky first episode that got all 4 rooms?

Yes, with a two-step precursor structure. The first 4-room episode is a single,
identifiable event: **global step 8,028,288** (episode record 15189), return 400, length
**5,758 steps** — 1.2 h into the 3-hour job. It did not come out of nowhere: it was the
run's 5th multi-room episode, preceded by two isolated 2-room episodes millions of steps
earlier (4.61M, 5.64M) and two more at 7.70M and 7.90M that opened the burst. But no
episode ever stopped at 3 rooms — in this run (and the whole batch) an episode either
dies by the second room or carries straight through to rooms 3 and 4. The first time the
agent survived past room 2's hazard, it saw all four rooms at once.

The single most spectacular episode of the entire batch follows late in the burst: the
**5-room episode at step 8,969,984** — 17,005 steps long (≈ 19 minutes of game time, vs.
a run-median episode of ~440 steps), 500 points (it collects a second key deeper in,
consistent with the key visible in room 4 in the recorded video). It was never repeated;
no other episode in any of the 30 runs reached 5 rooms or scored 500 except this one and
154 late episodes of the sham-remote seed-44 run (which scored 500 on a 2-room route).

## Q2 — Which episode is the recorded video?

**Global step 8,630,624** (episode record 15487): 4 rooms, return 400, length **4,376
steps** — identified exactly, not estimated: the mp4 contains 4,377 frames (one per agent
step plus the reset frame), and 4,376 is the unique matching episode length in the burst.
It is the 7th four-room episode, sitting in the densest cluster of the whole run: three
4-room episodes within 24k steps (8,606,880 / 8,621,024 / 8,630,624). Env 0's
`NewRoomRecorder` counter labels it `ep00521`; the YouTube upload is
<https://youtube.com/shorts/4pNE5FtkZec>.

## Q3 — What distances do the attempts have from one another?

Gap to the previous multi-room episode, through the burst (full table in the CSV):

| step | rooms | length | gap (steps) | gap (episodes) |
|---:|:--:|---:|---:|---:|
| 4,611,520 | 2 | 1,387 | — | — |
| 5,642,592 | 2 | 1,392 | 1,031,072 | 1,666 |
| 7,701,152 | 2 | 2,818 | 2,058,560 | 3,273 |
| 7,900,512 | 2 | 1,537 | 199,360 | 163 |
| **8,028,288** | **4** | 5,758 | 127,776 | 71 |
| 8,103,456 | 2 | 4,389 | 75,168 | 44 |
| 8,222,720 | 2 | 4,802 | 119,264 | 51 |
| 8,328,352 | 2 | 5,162 | 105,632 | 64 |
| 8,429,408 | 4 | 1,863 | 101,056 | 66 |
| 8,524,864 | 4 | 5,785 | 95,456 | 29 |
| 8,577,760 | 4 | 5,199 | 52,896 | 18 |
| 8,581,472 | 2 | 8,022 | 3,712 | 2 |
| 8,606,880 | 4 | 5,545 | 25,408 | 9 |
| 8,621,024 | 4 | 4,338 | 14,144 | 6 |
| **8,630,624** | **4** | 4,376 | 9,600 | 9 |
| 8,722,016 | 4 | 1,432 | 91,392 | 35 |
| 8,723,808 | 2 | 2,338 | 1,792 | 2 |
| 8,729,856 | 4 | 2,547 | 6,048 | 6 |
| 8,791,264 | 4 | 2,724 | 61,408 | 27 |
| **8,969,984** | **5** | 17,005 | 178,720 | 187 |
| 9,046,112 | 4 | 2,967 | 76,128 | 91 |
| 9,160,800 | 2 | 1,270 | 114,688 | 132 |
| 9,191,648 | 4 | 1,150 | 30,848 | 33 |
| 9,206,176 | 2 | 1,315 | 14,528 | 10 |
| 9,214,048 | 2 | 1,701 | 7,872 | 9 |
| 9,221,568 | 2 | 1,418 | 7,520 | 7 |
| 9,228,256 | 4 | 1,076 | 6,688 | 6 |
| 9,228,768 | 4 | 1,381 | 512 | 1 |
| 9,230,656 | 2 | 1,459 | 1,888 | 2 |
| 9,278,528 | 4 | 3,939 | 47,872 | 45 |
| 9,284,192 | 4 | 7,110 | 5,664 | 8 |
| 9,302,336 | 2 | 1,219 | 18,144 | 19 |
| 9,438,752 | 4 | 1,526 | 136,416 | 193 |
| 9,483,168 | 2 | 961 | 44,416 | 75 |
| 9,496,000 | 2 | 902 | 12,832 | 16 |
| 9,530,592 | 2 | 1,141 | 34,592 | 66 |

Readings:

- **Gaps collapse by four orders of magnitude**: millions of steps between the isolated
  precursors, ~100k in the early burst, down to 512 steps at the peak. The 512-step /
  1-episode gap (9,228,256 → 9,228,768) means two different parallel environments
  finished 4-room episodes almost simultaneously — by the peak, the behaviour lived in
  the shared policy, not in one env on a streak.
- **The route gets ~4× faster inside the burst.** Early 4-room episodes take 5.2–5.8k
  steps (wandering into the rooms); the late ones take 1,076–1,526 steps (direct
  execution). That within-burst speed-up is policy improvement made visible at the
  episode level.
- **After 9,438,752 the 4-room episodes stop**; three shortening 2-room episodes follow,
  then a 550k-step drought to the next multi-room episode (10,087,168) — the onset of the
  near-extinction phase covered in the discussion draft.

## The funnel (how leaky each stage is)

Burst window 7.6M–9.6M, all 32 envs: **1,751 episodes → 188 collected the key but died
in room 1 (return 100) → 48 additionally opened a door but still died inside room 1
(return 400, rooms = 1) → 34 walked through (rooms ≥ 2) → 18 reached 4+ rooms.** Even at
the burst's peak, an episode had a ~2% chance of leaving room 1. For contrast, the
preceding 3.6M steps (5,855 episodes): 61 keys, 3 doors, 2 exits.

## What the statistics do during the burst (figure panels, top to bottom)

- **Raw intrinsic reward** (`charts/raw_intrinsic_rew_mean`, batch mean per iteration):
  baseline ~15 with sharp spikes to 40+ aligned with the multi-room episodes — each
  excursion pays out a novelty spike as unseen rooms hit the RND predictor. The spikes
  shrink over the burst and are gone after ~9.3M: the novelty of rooms 2–4 was consumed
  by the very visits it rewarded.
- **Remote-press fraction** (`charts/tv_action_frac`): below/at chance (1/19 ≈ 0.053)
  until ~7.7M, climbs with the burst, peaks at **0.15 ≈ 3× chance around 8.7M** —
  exactly the densest 4-room cluster — and falls back below chance by ~9.3M. Elevated
  remote-pressing and genuine discovery co-occur in the same episodes, so PPO's
  credit assignment reinforces both together; the co-movement is an observation from a
  single run, and the causal reading belongs in the discussion, not here.
  `charts/tv_intrinsic_share` over the window sits at ~0.14–0.16.
- **Extrinsic value estimate** (`charts/ext_value_mean`): first sustained lift of the
  run starts at ~7.6M, before most 4-room episodes — driven by the newly frequent
  key/door scores (the 188 + 48 near-miss episodes), not by the exits themselves.
  Oscillates 0.05–0.25 through the window; does not stabilize yet (that happens at ~16.5M).
- **Policy entropy**: ~1.0 nats throughout the window — the burst happens in a policy
  that is neither fresh (2.94 = uniform) nor collapsed (0.44 at run end). Dip to ~0.7
  around 8.4–8.6M, rebound to ~1.15 at 8.7M. `losses/approx_kl` runs 0.003–0.004
  (updates alive); learning rate at 8.6M is 5.7e-5 (57% of initial).

## Suggested figure caption

> Episode-level view of the first acquisition burst of `rnd_tv_remote` seed 43
> (7.0M–9.8M of 20M steps). Top: every episode that left room 1 as a stem (light: 2
> rooms, dark: 4+), with episodes that scored but died inside room 1 as gray ticks;
> annotated are the first 4-room episode (step 8.03M), the recorded video episode
> (8.63M, footnote X), and the singular 5-room episode (8.97M; 17,005 steps, 500
> points). Below, per-iteration training statistics over the same window: raw intrinsic
> reward (novelty spikes aligned with the excursions), remote-press fraction (peaking at
> ≈ 3× chance at the burst's core), extrinsic value estimate, and policy entropy.

## Reproduction

```bash
source .venv/bin/activate
python scripts/plot_lucky_run_burst.py \
    --run-dir analysis/HPC-Runs/runs/ALE/MontezumaRevenge-v5__rnd_tv_remote__43__1785112914 \
    --out-dir doc/figures
```

Episode records join `charts/rooms_visited`, `charts/episodic_return`, and
`charts/episodic_length` on their global step (each episode end logs all three at the
same step). Six of the 568 multi-room records log return 0 (likely RAM room-byte flicker
at death); none fall in the burst window. Episode indices in the tables are positions in
the merged 32-env stream; per-env indices are not logged — the video episode was matched
via its frame count instead.
