# Matched-budget comparison — submission package

One clean, matched-budget characterization of the three exploration methods on the
**bug-fixed** code (the gymnasium `NEXT_STEP` GAE-masking fix, PR #14). Purpose — answer,
empirically and cheaply:

- **(a)** Does the early entropy / intrinsic-reward collapse (documented in
  `doc/10M-RND-run-failure-documentation.md`) survive the bug fix?
- **(b)** Does RND underperforming plain PPO (documented in
  `doc/rnd-vs-ppo-asymmetry-investigation.md`) survive the bug fix?

This is a **fixed-budget characterization, NOT** a reproduction of the paper's 24-room
result (~492M steps / 128 envs is out of compute scope).

> **Submission status:** the matrix is fully configured, and every cell's command has been
> pre-flighted on the fixed code (see STEP 2). The actual `sbatch` submission (STEP 3) must
> be run **on the cluster login node** — it could not be executed from the dev machine
> (no SLURM client / no cluster route there). Run `slurm/submit_matched_matrix.sh` on the
> login node to fire the matrix and capture the job IDs.

---

## STEP 0 — Precondition: run the FIXED code

The cluster runs whatever commit is checked out in its repo copy (`setup_hpc.sh` builds
the venv in place; it does not check out a branch). The fix lives on branch
**`worktree-fix-next-step-gae-masking`** (PR #14) — it is **not** on `main`. On the login
node, before submitting:

```bash
cd ~/workspace                          # home checkout (repo is small; keep it off /scratch)
git clone git@github.com:timo-bmgd/montezuma.git   # skip if already cloned
cd montezuma
git fetch origin
git checkout worktree-fix-next-step-gae-masking
git rev-parse HEAD                      # RECORD this hash — it is the commit the matrix runs

# build the dedicated conda env on /scratch (one-time, ~6 GB, a few minutes):
bash slurm/setup_hpc.sh
conda activate /scratch/$USER/conda-envs/montezuma

# verify the fix is actually present in the checked-out code (no ROMs / GPU needed):
python tests/test_gae_autoreset.py       # must print "7/7 passed"
grep -q "def compute_gae" src/agents/base.py && echo "compute_gae present"
```

Do not submit if `tests/test_gae_autoreset.py` does not pass — that means the checkout is
running the old masking. `submit_matched_matrix.sh` prints the short commit hash it submits
from, so the job-ID log records exactly which code each job ran.

---

## STEP 1 — The matrix (8 cells)

Matched across every agent: **identical `num_envs`, `total_timesteps`, seed set, and
`anneal_lr`**. The only intended difference between agents is the exploration method, so
any RND-vs-PPO gap is attributable to method, not budget.

| Cell | Script | `exp_name` | `ent_coef` | Seeds | Purpose |
|---|---|---|---|---|---|
| PPO | `run_ppo.slurm` | `ppo` | **0.01** (ppo.py std) | 1, 2 | no-intrinsic baseline |
| RND | `run_rnd.slurm` | `rnd` | **0.001** (paper) | 1, 2 | primary RND |
| count-based | `run_count_based.slurm` | `count_based` | **0.01** (std) | 1, 2 | SimHash exploration |
| RND ablation | `run_rnd.slurm` | `rnd_ent01` | **0.01** | 1, 2 | isolates whether the ent_coef mismatch, not the method, drives RND<PPO |

Matched knobs and the reasoning (STEP 1 requirements):

1. **num_envs = 32.** All three scripts already used 32; it sits in the requested 32–64
   band and matches `--cpus-per-task=32` 1:1 (`AsyncVectorEnv` spawns one subprocess per
   env — fewer CPUs than envs silently taxes throughput; more CPUs than the partition has
   per GPU queues the job in `PD` forever, per `doc/hpc-onboarding.md` §1). Going to 64
   needs a **confirmed** ≥64-core-per-GPU partition *and* GPU-memory headroom on the V100 —
   neither confirmable off-cluster — so 32 is the safe matched value. Override with
   `NUM_ENVS=64` once `sinfo -o "%P %c %G"` confirms the ratio.
2. **total_timesteps = 3,000,000 (3M).** Deliberately the cheap scale
   `doc/rnd-vs-ppo-asymmetry-investigation.md` §4.3 calls for (and what `run_rnd.slurm` was
   already sized at), **not** 10M/50M — that doc explicitly warns against re-running the
   expensive budget before something works. The prior collapse manifested within **<1M**
   steps, so 3M answers **(a)** with margin and gives a fast first read on **(b)**; if (b)
   is ambiguous at 3M, extend the *same* runs with the `--resume` chaining workflow (below)
   rather than restarting. Expected SPS: the prior ~16-env runs logged **144–183 SPS**
   (`doc/10M-RND-run-failure-documentation.md` §2); at 32 envs on a V100 assume the same
   order → 3M steps ≈ **4.5–6 h** worst case, **~2.5 h** if SPS is higher. The `06:00:00`
   matched walltime covers the pessimistic case; **confirm real SPS with the salloc smoke
   (STEP 2) before trusting it**, and auto-stop + `--resume` de-risk the estimate.
3. **anneal_lr = ON.** The paper/CleanRL default (the agents default `--anneal-lr` True).
   Kept ON and matched across agents to avoid adding a confound; disabling it is a separate
   lever (`doc/10M-RND-run-failure-documentation.md` §6), not part of this clean matrix.
   Overridable per-run with `ANNEAL_LR=0` for a future ablation.
4. **Seeds ≥ 2 per agent** (1 and 2). Drop the ablation to one seed (`ABLATION_SEEDS=1`)
   if the V100 queue is tight.
5. **auto-stop ON** (default). A collapsed run stops early and exits **42**, giving a fast
   collapse yes/no and not burning the walltime — so collapsed RND cells are cheap.
6. **ent_coef:** each agent at its standard value for the primaries (the mismatch is
   itself a documented confound), plus the RND `ent_coef=0.01` ablation to test whether
   matching RND's ent_coef to PPO's closes the gap — i.e. whether the gap is the method or
   the entropy-coef.
7. **Env stack / noisy-TV untouched** — out of scope for this matrix.

Implementation: the three production scripts now take behaviour-preserving `--export`
overrides (`NUM_ENVS`, `TOTAL_TIMESTEPS`, `ENT_COEF`, `EXP_NAME`, `ANNEAL_LR`,
`RESUME_FROM`); unset, each script behaves exactly as before. `run_count_based.slurm`'s
`--gres=gpu:a100:1` was corrected to the generic `gpu:1` (an a100 request never schedules
on a V100 allocation).

### Cluster specifics — CONFIRMED for HTW KI-Werkstatt (`kiwihead01`, 2026-07-23)
- **Partition `Debug_node`** (the only one; default). `MaxTime=UNLIMITED`,
  `DefaultTime=1 day` → no walltime cap; a 3M run fits one job, no resume-chaining needed.
- **`--gres=gpu:1`**, untyped (nodes advertise `gpu:4`, no v100/a100 qualifier).
- **32 cores per GPU** (128 CPU ÷ 4 GPU) → `--cpus-per-task=32` + `NUM_ENVS=32` is a clean
  1:1 fit on one GPU. 64 is not viable here.
- **No `python`/`cuda` modules** — Python is a dedicated conda env
  (`/scratch/$USER/conda-envs/montezuma`, built by `slurm/setup_hpc.sh`); CUDA comes from
  the torch cu124 wheel. The scripts activate conda, not `module load`.
- **Scratch**: `/scratch/$USER` (40 T, ~32 T free) → output lands in
  `/scratch/$USER/montezuma/{runs,checkpoints,videos}` (the scripts' fallback resolves this).

Still to set before submitting: **`WANDB_API_KEY`** for `--track` (`doc/hpc-onboarding.md`
§3 — a missing key hangs the job; or drop `--track`), and the **real SPS** from the STEP 2
salloc smoke (to right-size `WALLTIME`).

### Shared-cluster etiquette (cluster rules)
Only **8 GPUs exist cluster-wide** (2 nodes × `gpu:4`), shared with other KI-Werkstatt
users. The full matrix is 8 one-GPU jobs = the entire cluster — **do not submit all 8 at
once.** Stage it (below), stay reachable in Slack `#ki-werkstatt-hpc` while jobs run, and
expect admins may cancel jobs on short notice to coordinate access. Never hold a GPU you
are not using.

---

## STEP 2 — Pre-flight (done on the fixed code; re-run the salloc smoke on-cluster)

Off-cluster, every cell's exact command was replicated at a tiny budget
(`--num-envs 2 --num-steps 32 --total-timesteps 256 --sync-envs --no-cuda`) on the fixed
code. All four configs (PPO ent 0.01, RND ent 0.001, RND ent 0.01, count-based ent 0.01)
**parsed their args, started training, ran iterations, and exited 0**, producing distinct
run dirs (`ppo`, `rnd`, `rnd_ent01`, `count_based` — the ablation does not collide with the
primary RND runs). The GAE-fix regression test passes 7/7.

Before the mass submission, run the on-cluster equivalent inside a short `salloc`
(`doc/hpc-onboarding.md` §4) to confirm GPU visibility + real SPS:

```bash
salloc --partition=Debug_node --gres=gpu:1 --cpus-per-task=32 --mem=32G --time=00:20:00
conda activate /scratch/$USER/conda-envs/montezuma
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python src/agents/rnd.py --total-timesteps 40000 --num-envs 32 --obs-norm-init-steps 2   # note the SPS
exit
```

Only proceed to STEP 3 once this is clean. Note the shared-cluster etiquette below.

---

## STEP 3 — Submit the matrix

On the login node, from the repo root, with the fixed commit checked out (STEP 0).
**Stage it** — 8 GPUs are shared cluster-wide, so don't grab them all at once:

```bash
mkdir -p slurm-logs
DRY_RUN=1 bash slurm/submit_matched_matrix.sh                 # preview all 8 cells

# Stage 1 — validate one real cell end-to-end (env + GPU + SPS + output paths):
sbatch --export=ALL,SEED=1,NUM_ENVS=32,TOTAL_TIMESTEPS=3000000,ENT_COEF=0.001,EXP_NAME=rnd,ANNEAL_LR=1 slurm/run_rnd.slurm

# Stage 2 — once that's healthy, the 3 primary seed-1 cells (PPO, RND, count):
SEEDS=1 ABLATION_SEEDS="" bash slurm/submit_matched_matrix.sh

# Stage 3 — the seed-2 primaries + the ent_coef ablation, as GPUs free up / Slack allows:
SEEDS=2 bash slurm/submit_matched_matrix.sh                   # seed-2 primaries + ablation seeds 1,2
```

`submit_matched_matrix.sh` already defaults `PARTITION=Debug_node`, `GRES=gpu:1`,
`WALLTIME=06:00:00`; override inline if the salloc smoke says so, e.g.
`WALLTIME=08:00:00 bash slurm/submit_matched_matrix.sh`. Job IDs + the commit hash land in
`slurm-logs/matched_matrix_<stamp>.tsv`.

Extend any run to a larger matched budget without restarting (resume-chaining):
```bash
sbatch --export=ALL,SEED=1,TOTAL_TIMESTEPS=3000000,RESUME_FROM=/scratch/$USER/montezuma/checkpoints/<run_name>/ckpt_XXXXXX.pt slurm/run_rnd.slurm
```

---

## STEP 4 — Monitor, locate output, read the result

**Monitor:**
```bash
squeue -u $USER                                                          # queued/running
tail -f slurm-logs/<job-name>_<job-id>.out                               # live stdout
sacct -j <job-id> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode   # during & after
seff <job-id>                                                            # CPU/mem efficiency (post-hoc)
```

**Where output lands** (`doc/10M-RND-run-failure-documentation.md` §8,
`doc/hpc-onboarding.md` §2): under `$SCRATCH/montezuma/` —
`runs/<run_name>/` (TensorBoard events), `checkpoints/<run_name>/ckpt_*.pt`,
`videos/<run_name>/` (periodic + `room_discovery/`); SLURM stdout/err in
`slurm-logs/<job-name>_<job-id>.{out,err}`. `run_name` is
`ALE/MontezumaRevenge-v5__<exp_name>__<seed>__<timestamp>`.

**Reading the outcome:**
- **Exit code** (`sacct ... ExitCode`): **`42:0` = auto-stop fired = the collapse
  signature recurred** (answers **(a)**: collapse survived the fix). **`0:0`** = ran the
  full budget without the sustained collapse signature. Any other nonzero = crash (check
  `slurm-logs/*.err`).
- **Collapse diagnostic** — load the run's TB event file
  (`EventAccumulator`, per CLAUDE.md § Log Analysis) and check
  `losses/entropy` (healthy: slow decay from ln(18)=2.89; collapse: rapid fall to ~0) and
  `charts/raw_intrinsic_rew_mean` (healthy: non-zero, ~0.1–1; collapse: ~20× drop within
  the first ~9% of the budget). `charts/collapse_streak` shows how close auto-stop came.
- **(b) RND vs PPO** — compare `charts/rooms_visited` (max > 1?) and
  `charts/episodic_return` (any nonzero?) across the PPO, RND, and count-based cells at the
  same budget. Prior evidence to beat (`asymmetry` doc §2): PPO reached room 2 / score 400
  at 5M; RND stayed at room 1 / score 0 at 10M. Because the bug **and** its fix were
  identical across all three agents, the fix is **not** expected to flip the ordering by
  itself — it makes the comparison clean.
- **Ablation** — if `rnd_ent01` behaves like `rnd` (still collapses / still < PPO), the
  ent_coef mismatch is **not** the driver; if it closes the gap, it is.
