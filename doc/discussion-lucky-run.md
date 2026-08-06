# Discussion-chapter section draft: the "lucky" run (rnd_tv_remote, seed 43)

Drafted 2026-08-06 from the HPC noisy-TV batch (`analysis/HPC-Runs/`, 30 runs × 20M steps).
Intended placement: discussion chapter, after the main noisy-TV results — roughly one page
plus one figure. The section answers "why didn't the agent that reached room 4 learn the
behaviour?" — and the honest answer, which the data forces, is that it *did* learn it,
extremely slowly, nearly lost it twice, and ran out of budget mid-consolidation. That
correction is the point of the section: it shows concretely how much luck sits between a
working implementation and the first visible progress on Montezuma's Revenge.

Figure: `doc/figures/lucky-run-rnd_tv_remote_s43.{pdf,png}` (regenerate with
`scripts/plot_lucky_run.py`, see §Reproduction).

---

## Draft section text

### What it takes to leave the first room: anatomy of the batch's one successful run

Across the thirty 20M-step runs of the main batch, leaving the first room was an anomaly.
Of roughly 1.06 million episodes played in total, 2,329 (0.2%) visited a second room, and
99% of those came from just two runs; the other twenty-eight runs produced eighteen such
episodes between them. The clearest case is `rnd_tv_remote` seed 43, the only run in the
batch to reach a fourth room (once, briefly, a fifth). It is worth walking through this
run in detail, because it shows what the aggregate numbers hide: how much luck is needed
before learning can even begin, and how fragile the result is once it appears.[^video]

[^video]: A recorded four-room episode from this run — the agent collects the key,
opens the right-hand door and descends two further rooms, with the next key already in
view when it dies — is available at <https://youtube.com/shorts/4pNE5FtkZec>.

The run needed 4.4 million environment steps, roughly 9,700 episodes, before it collected
its first reward of any kind. The key in room 1 sits at the end of a long, precise action
sequence — down the ladders, across the rope, past the patrolling skull, and all the way
back up — which a policy whose entropy is still above 1 nat executes only by accident,
and sticky actions (25% repeat probability) perturb even a correct sequence. Tellingly,
the first episode to score the full 400 points still died inside room 1: the door reward
is granted on opening, and the agent did not survive to walk through. The first actual
room transition came 200,000 steps later.

Finding the reward once is not the bottleneck, however; keeping it is. After reward
clipping, a successful episode contributes two +1 increments among the 4,096 transitions
of a rollout batch, which PPO uses for four epochs and then discards — on-policy learning
has no replay buffer in which a lucky trajectory could be banked. The mechanism that
should bridge this gap, RND's intrinsic reward, does make the newly seen rooms attractive,
but by construction the bonus depletes with every repeat visit as the predictor catches
up, whereas the injected noise patch, refreshed every frame, never depletes: its share of
the total intrinsic reward climbed steadily to about a fifth by the end of the run.
During the first cluster of successes the remote-pressing fraction also ran at roughly
twice chance level (0.095 vs. 1/19 ≈ 0.053) — mild behavioural capture coinciding with
the discovery phase — though with a single run this remains an observation, not a causal
claim. The extrinsic critic, meanwhile, had estimated the value of the average state at
essentially zero for 14 million steps.

Figure X shows the resulting dynamics. A first cluster of successes appears around
8–9M steps: sixteen episodes leave room 1, eleven of them reaching a fourth room (the
recorded episode above is from this phase). The behaviour then all but disappears — the
five million steps from 11M to 16M contain nine exits in total — before a second
acquisition, from about 16.5M steps, finally sticks: in the final million steps 24% of
episodes leave room 1 and 15% reach four or more rooms, the critic's mean value estimate
rises by an order of magnitude, and the explained variance of the extrinsic value head
turns positive. The budget ends in the middle of this consolidation, with the success
rate still climbing, the learning rate annealed to a few percent of its initial value,
and policy entropy at 0.44 nats.

Two readings matter for this thesis. First, at this compute scale, progress on
Montezuma's Revenge is not a learning curve but a rare-event search followed by a race:
a run must stumble upon the key sequence at all — twenty-eight of thirty never
meaningfully did — and must then re-encounter it often enough, before the novelty bonus
depletes and the entropy and learning-rate schedules wind down, to hand the behaviour
over to the extrinsic critic. Seed 43 won that lottery near the middle of its budget and
still needed almost all of the remaining 12M steps for the hand-over. This is the failure
structure Ecoffet et al. (2021) call *detachment* and *derailment*, and it motivates
Go-Explore's explicit archive of visited states — precisely the banking mechanism that
on-policy intrinsic motivation lacks. Second, the spread across seeds is a warning about
evaluation: the three seeds of this identical configuration finished with maximum room
counts of 1, 5 and 1. At this budget the gap between the best and worst run of a
condition is dominated by whether the lottery hit, not by the condition itself, which is
why the reference RND results rest on budgets around 500M steps with 128 parallel
environments (Burda et al., 2019), and why results here are reported across seeds rather
than as single curves.

A final observation underlines how narrow the learned competence is. Of the 568
multi-room episodes, 226 visited exactly two rooms and 341 exactly four — not one visited
exactly three. The agent either dies at the second room's hazard or clears it and coasts
through the next two rooms: what was learned is a single stereotyped route, not a search
strategy. Depth of penetration increased; breadth of exploration did not.

### Suggested figure caption

> Timeline of `rnd_tv_remote` seed 43 over its 20M-step budget (500k-step bins). From
> top: share of episodes leaving room 1 (light: ≥ 2 rooms, dark: ≥ 4 rooms); mean
> episodic game score; mean extrinsic value estimate; policy entropy. Dashed lines mark
> the first reward (4.4M steps), the first four-room episodes (8.0M; the recorded episode
> of footnote X is from this phase), and the onset of the final re-acquisition (≈ 16.5M).
> The behaviour almost vanishes between 11M and 16M steps and consolidates only in the
> last fifth of the run; the budget ends mid-consolidation.

---

## Supporting numbers (source data for every claim above)

All from `analysis/HPC-Runs/runs/ALE/MontezumaRevenge-v5__rnd_tv_remote__43__1785112914`
(TensorBoard event file; 35,292 episode-end records; run completed the full 20M steps,
`collapse_streak` peaked at 51 < patience 100, so auto-stop never fired).

| Fact | Value |
|---|---|
| Batch size | 30 runs (RND off/remote/sham-remote/static × seeds 42/43/44 + SCHWERT 100/200/300; PPO off/remote × 42/43/44), 20M steps each, `num_envs=32` |
| Batch total episodes / episodes with ≥ 2 rooms | 1,061,997 / 2,329 (0.22%) |
| ...of which from remote s43 + sham-remote s44 | 568 + 1,743 = 2,311 (99.2%); remaining 28 runs: 18 episodes, max 3 per run |
| First nonzero return (s43) | step 4,404,832, return 400, `rooms_visited=1` (died before using the door) |
| First room transition | step 4,611,520 (~200k steps later) |
| Episodes before first reward | ≈ 9,700 (buckets 0–4M: 8,991 episodes, + ~680 in 4–4.4M) |
| First burst | 8–9M: 16/608 episodes ≥ 2 rooms, 11 ≥ 4 rooms; single 5-room episode at 8,969,984 (return 500) |
| Near-extinction | exits per 1M-step bucket, 11M→16M: 1, 0, 7, 1, 0 |
| Final million steps | 227/944 episodes ≥ 2 rooms (24%), 143 ≥ 4 rooms (15%) |
| Rooms-visited distribution | {1: 34,724; 2: 226; 4: 341; 5: 1} — zero 3-room episodes |
| Return distribution | {0: 33,148; 100: 855; 400: 1,288; 500: 1} |
| `ext_value_mean` (2M-step buckets) | ≤ 0.085 through 16M (0.015 in 14–16M), then 0.28 / 0.35 |
| `explained_variance` | −0.69 in 14–16M, +0.39 / +0.67 in 16–20M |
| Entropy | 2.64 (first 1M bucket) → 0.46 (last bucket; final scalar 0.444); uniform = ln 19 ≈ 2.94 |
| `approx_kl` | ~0.003–0.004 mid-run → 0.0004 in the last million |
| Learning rate | 1e-4 annealed linearly to 0 at 20M (18–20M bucket mean 5.0e-6) |
| `tv_action_frac` | 0.095 in the 8–9M burst (chance 1/19 ≈ 0.053); 0.025 in the final buckets (below chance) |
| `tv_intrinsic_share` | ≈ 0.01 (first 1M) → 0.19–0.24 (final buckets) |
| `raw_intrinsic_rew_mean` | 29.9 (first 1M) → ~15 baseline; local bumps to 18.5 (burst) and 16–17 (re-acquisition) |
| Sham-remote s44 (the only other consolidator) | 1,743 exits, all on a 2-room route ({2: 1,742; 4: 1}); onset ~12M, ~88% exit rate late in the run, mean return ≈ 357 in the last 3M |
| Recorded video | env-0 episode 521 (`new_room_ep00521_r04.mp4`) = the episode at step 8,630,624 (4 rooms, return 400, length 4,376 — identified by frame count, see `doc/results-lucky-run-context.md`); <https://youtube.com/shorts/4pNE5FtkZec> |

Cross-checks worth keeping in mind when editing: reward clipping means key = +1, door = +1
for the learner (true scores 100/300 appear only in logging, which records pre-clip
returns); episode counts are across all 32 envs, while the video's "episode 521" is env
0's own counter; 6 of the 568 multi-room episodes logged return 0 (likely RAM room-byte
flicker at death) — they do not affect any conclusion.

## Reproduction

```bash
source .venv/bin/activate
# figure
python scripts/plot_lucky_run.py \
    --run-dir analysis/HPC-Runs/runs/ALE/MontezumaRevenge-v5__rnd_tv_remote__43__1785112914 \
    --out-dir doc/figures
# batch-wide overlays / summary table (generic pipeline)
python scripts/analyze_runs.py --logdir analysis/HPC-Runs/runs/ALE
```

References: Burda et al., *Exploration by Random Network Distillation*, 2019
(arXiv:1810.12894); Ecoffet et al., *First return, then explore*, Nature 2021 (Go-Explore;
arXiv:1901.10995).
