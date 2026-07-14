# HPC / SLURM Onboarding

A from-scratch walkthrough for running this repo's training jobs on an HPC
cluster via SLURM, written for a first-time SLURM/HPC user. If you already know
SLURM, skip to §5 (Submitting, monitoring, canceling) and §6 (Smoke-test
checklist).

Companion scripts: `slurm/run_rnd_smoke.slurm`, `slurm/run_ppo_smoke.slurm`
(infrastructure sanity check, minutes), `slurm/run_rnd.slurm`,
`slurm/run_ppo.slurm` (production runs, hours). Run the smoke tests first —
always.

---

## 1. What to find out first

Before touching any `.slurm` file, you need three things from the cluster:
what hardware you actually have, what your account is allowed to use, and
what software is available via the module system. All of the SLURM scripts in
this repo currently carry **placeholder resource requests** — they were
written without real cluster numbers and say so in their header comments.
Don't submit them as-is; adjust based on what you find here.

### Hardware

```bash
# One-line summary of every partition: name, node count, state, CPUs, GPUs
sinfo -o "%P %D %t %c %G"

# Full detail for the partition you'll actually use (fill in the name from above)
scontrol show partition <partition-name>

# Full detail for a specific node in that partition
scontrol show node <node-name>
```

What you're looking for:
- **Partition name** — this repo's scripts assume `--partition=gpu`; confirm
  that's the real name on your cluster (`sinfo -o "%P"` lists all of them; the
  one with GPUs is usually but not always literally named `gpu`).
- **GPU type and count per node** — the `%G` column shows something like
  `gpu:a100:4`. This repo's scripts request `--gres=gpu:a100:1`; if your
  cluster's GPU type differs (V100, H100, etc.) or the generic-resource name
  is spelled differently, you'll need to change that flag or drop the type
  and just request `--gres=gpu:1`.
- **CPUs per node/GPU** — this is the one that actually broke throughput
  before (see `doc/throughput-investigation.md` §1-2): every script defaults
  `AsyncVectorEnv`, which spawns **one OS subprocess per `--num-envs`**. If
  `--cpus-per-task` is set lower than `--num-envs`, the env workers queue for
  CPU time instead of running in parallel — a silent throughput tax, not a
  crash. Conversely, if you request more CPUs than the partition actually has
  per GPU, SLURM does **not** reject the job; it queues it indefinitely in
  `PD` (pending) state, burning wall-clock time until someone notices. Always
  check the real ratio before submitting:
  ```bash
  sinfo -o "%P %c %G"
  ```

### Account limits (fair-share / QOS)

```bash
# Your account's usage quality-of-service (walltime caps, priority, etc.)
sacctmgr show qos format=Name,MaxWall,MaxTRESPerUser,Priority

# Your specific account/association — what partitions and resources you're allowed
sacctmgr show association where user=$USER format=Account,Partition,QOS,GrpTRES
```

If your allocation has a walltime cap shorter than a script's `--time=`
request, `sbatch` will reject the job outright with a clear error (this one
fails fast, unlike the CPU-oversubscription case above). Note the cap now —
it directly affects the production-run resume-chaining strategy in §7.

### Storage quota

Cluster-specific — check your site's documentation or the message-of-the-day
banner shown at SSH login (`cat /etc/motd` if you missed it). You're looking
for the size and purge policy of your **scratch** space specifically (see
§2). There's no universal SLURM command for this — it varies by filesystem
(Lustre, GPFS, etc.) and by site.

### Module system

```bash
module avail python     # confirm a python/3.11-compatible module exists
module avail cuda       # confirm a cuda/12.4-compatible module exists
module spider python    # more detail on exact versions if `avail` is ambiguous
```

Every script in `slurm/` currently has:
```bash
module load python/3.11
module load cuda/12.4
```
These are placeholders carried over from a generic HPC template — nobody has
confirmed they match your cluster's actual module names yet. If `module load
python/3.11` fails, `module avail python` will show you the real name to
substitute (e.g. `python/3.11.4` or `Python/3.11.4-GCCcore-12.3.0` depending
on the site's naming convention).

---

## 2. Filesystem conventions

HPC clusters standardly split storage into tiers with very different
properties. This repo's scripts already assume the common two-tier pattern:

- **Home** (`$HOME`) — small quota (often single-digit GB), permanent,
  usually backed up. Keep the repo checkout and your Python venv here. Do
  **not** point `--runs-dir`/`--videos-dir`/`--checkpoint-dir` here — a
  10M-step RND run's checkpoints alone can be hundreds of MB to low GB, and
  video recording adds more.
- **Scratch** (commonly `$SCRATCH`, sometimes `/scratch/$USER` or
  site-specific) — large quota (TBs), **temporary**: most sites purge files
  after some inactivity window (30-90 days is common, but check your site's
  policy — this is exactly what to look for in the storage-quota step above).
  This is where training output goes. Every script in `slurm/` already
  targets:
  ```bash
  SCRATCH="${SCRATCH:-/scratch/$USER}/montezuma"
  mkdir -p "$SCRATCH"/{runs,checkpoints,slurm-logs}
  ```
  If your cluster doesn't export a `$SCRATCH` environment variable, the
  fallback `/scratch/$USER` may or may not be correct — confirm the real path
  (often documented alongside the storage-quota info in §1, or discoverable
  via `df -h` for mounted filesystems you have write access to).
- **Project space** (naming varies) — some clusters offer a third tier for
  a research group's shared, non-purged storage. Not assumed by any script
  here. If your cluster has one and your thesis videos/checkpoints matter
  long-term, copy the final artifacts there before scratch purges them —
  don't rely on scratch as permanent storage for anything you'd be upset to
  lose.

Videos in particular are worth watching: `--record-room-discovery` writes an
mp4 every time the room high-water mark increases (cheap, sparse), but
periodic `--capture-video --video-episode-interval N` recording adds one file
every N episodes for the whole run — check scratch usage periodically on long
production runs (`du -sh $SCRATCH/montezuma/videos`).

---

## 3. Weights & Biases (W&B) signup

The smoke-test and production scripts pass `--track`, which calls
`wandb.init(...)` in both `rnd.py` and `ppo.py`. You need an account and an
API key before submitting anything with `--track`.

**Cost: free**, no card required, for what this repo needs. This codebase
only syncs TensorBoard scalars (`sync_tensorboard=True` — see
`train()` in either agent file), not model checkpoints or large media
artifacts, so W&B's storage limits are a non-issue regardless of tier:
- **Free personal tier** (wandb.ai signup) — 100GB storage included, unlimited
  scalar logging. This is enough on its own.
- **Free academic Pro tier** (wandb.ai/academic_application, needs a `.edu` or
  equivalent university affiliation) — more generous limits, worth applying
  for since it's free, but not required to get started today.

**Sign up:** go to wandb.ai, create an account (email or SSO).

**Compute nodes don't have a browser**, so the login flow differs from a
laptop:
1. On your laptop/any browser, log in to wandb.ai and go to
   `wandb.ai/authorize` to copy your API key.
2. On the cluster, either:
   - Run `wandb login` interactively (on a login node, not inside a batch
     job) and paste the key when prompted — it's cached in `~/.netrc` for
     all future jobs, or
   - Export it directly before submitting: `export WANDB_API_KEY=<your key>`
     (add this to your shell profile on the cluster, e.g. `~/.bashrc`, so
     every `sbatch` job's environment inherits it automatically).

Do this once, before your first `sbatch` with `--track` — a job that hits
`wandb.init()` without valid credentials will hang waiting for interactive
login input inside a non-interactive batch job, which just wastes the
allocation.

---

## 4. First hands-on step: `salloc`, before any `.slurm` file

Don't debug a batch script blind. Before wrapping anything in `sbatch`, grab
a short interactive allocation and run the training command directly — this
is the fastest possible iteration loop for confirming imports, GPU
visibility, and paths all work, with immediate feedback instead of a queued
job's output file.

```bash
# Adjust partition/gres/cpus-per-task to your real numbers from §1
salloc --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=8G --time=00:15:00
```

Once the allocation drops you into a shell on a compute node:

```bash
module load python/3.11   # or whatever §1 confirmed is correct
module load cuda/12.4
source /path/to/repo/.venv/bin/activate    # or wherever your venv lives on the cluster
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python src/agents/rnd.py --total-timesteps 20000 --num-envs 4 --sync-envs
```

If this works — GPU detected, a few iterations print SPS numbers without
crashing — you've confirmed the environment before ever touching a `.slurm`
file. `exit` to release the allocation when done (or let the `--time` limit
expire it automatically).

---

## 5. Submitting, monitoring, canceling

```bash
# Dry-run syntax check without actually queuing — catches typos in #SBATCH directives
sbatch --test-only slurm/run_rnd_smoke.slurm

# Actually submit
mkdir -p slurm-logs
sbatch --export=ALL,SEED=1 slurm/run_rnd_smoke.slurm

# Your queued/running jobs
squeue -u $USER

# Detailed info on a specific job — only works while the job is still in
# SLURM's memory, roughly 30 minutes after completion
scontrol show job <job-id>

# Same info, works both during and long after the job finishes — this is the
# one to use once scontrol starts returning "Invalid job id"
sacct -j <job-id> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode

# Live-tail the job's stdout while it runs
tail -f slurm-logs/<job-name>_<job-id>.out

# Cancel a running or queued job
scancel <job-id>

# Post-hoc CPU/memory efficiency report (after the job finishes)
seff <job-id>
```

`seff` does **not** report GPU utilization — it's CPU/memory only. For GPU
usage, either watch `nvidia-smi` during a `salloc` session, or check whether
your site has a GPU-aware monitoring tool (varies by cluster — ask your HPC
admins or check site docs if this matters for your write-up).

**Exit codes worth knowing about specifically for this repo:** both `rnd.py`
and `ppo.py` now exit with code **42** if the collapse auto-stop fires (see
§6 below and `doc/decisions.md`) — that's a deliberate stop, not a crash.
`sacct`'s `ExitCode` column will show `42:0` for this case, distinguishable
from a normal `0:0` completion or a Python traceback's nonzero code.

---

## 6. Reading smoke-test results

The smoke tests (`slurm/run_rnd_smoke.slurm`, `slurm/run_ppo_smoke.slurm`)
exist to check **infrastructure correctness**, not training quality — do
videos generate, does TensorBoard get all the expected tags, does
checkpoint/resume work, does W&B sync. A few thousand steps is enough to
check all of that; it says nothing about whether the algorithm is learning
well (that's what the production runs and the open
`doc/rnd-vs-ppo-asymmetry-investigation.md` are for).

Checklist (also embedded as comments in each smoke script):

- [ ] **Job completes without error.** `sacct -j <id> --format=State,ExitCode`
  shows `COMPLETED` / `0:0`.
- [ ] **No NaNs or crashes in the log.** `slurm-logs/<name>_<id>.out` shows
  steady `SPS=` prints every iteration, no Python tracebacks, no `nan` in any
  printed metric.
- [ ] **Videos exist and play.** Look for at least one
  `videos/<run_name>/room_discovery/new_room_ep*.mp4` (episode 1 always
  qualifies as a "new" room, so this should always produce at least one file)
  and at least one periodic `videos/<run_name>/rl-video-episode-*.mp4`. Copy
  them off the cluster and confirm they actually play (a zero-byte or
  corrupt mp4 is a real failure mode if the job was killed mid-write).
- [ ] **TensorBoard has all expected tags with non-degenerate values.**
  ```python
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  ea = EventAccumulator("runs/<run_name>")
  ea.Reload()
  print(sorted(ea.Tags()["scalars"]))
  ```
  For `rnd.py`, expect (per CLAUDE.md § Log Analysis and the metrics this
  repo already logs): `charts/episodic_return`, `charts/episodic_length`,
  `charts/rooms_visited`, `charts/learning_rate`, `charts/SPS`,
  `charts/mean_intrinsic_rew`, `charts/raw_intrinsic_rew_mean`,
  `charts/raw_intrinsic_rew_std`, `charts/obs_rms_std`,
  `charts/reward_rms_std`, `charts/ext_value_mean`, `charts/int_value_mean`,
  `charts/collapse_streak`, `losses/value_loss`, `losses/ext_v_loss`,
  `losses/int_v_loss`, `losses/policy_loss`, `losses/entropy`,
  `losses/approx_kl`, `losses/clipfrac`, `losses/fwd_loss`,
  `losses/explained_variance`. `ppo.py` logs the same `losses/*` and
  `charts/learning_rate`/`charts/SPS`/`charts/episodic_*`/`charts/rooms_visited`/
  `charts/collapse_streak` tags, minus the RND-specific ones. "Non-degenerate"
  means: not NaN, not flat-zero for the whole run (a `raw_intrinsic_rew_mean`
  stuck at exactly 0 the entire smoke test would indicate a wiring bug, not
  just early-training noise, since it should never be exactly zero even at
  step 1).
- [ ] **Checkpoints written at least twice, and `--resume` works.**
  ```bash
  ls checkpoints/<run_name>/
  ```
  should show 2+ `ckpt_*.pt` files given the smoke script's short
  `--checkpoint-interval`. Then submit a second smoke job pointing
  `--resume` at one of them and confirm in the log that it prints `Resumed
  from ... at iteration N` and that the new run writes into the **same**
  `runs/<run_name>/` and `checkpoints/<run_name>/` directories rather than a
  fresh timestamped one (this was a real bug, fixed as part of this same
  round of changes — see `doc/decisions.md`).
- [ ] **W&B run visible.** Check the `montezuma-thesis-smoke` project on
  wandb.ai; confirm the scalar curves match what TensorBoard shows locally.

If any box fails, the most likely culprits: wrong module names (§1) causing
an import failure before training even starts; `$SCRATCH` path not resolving
(§2) causing writes to fail or land somewhere unexpected; missing/expired
`WANDB_API_KEY` (§3) causing `--track` to hang; or a real code bug in the
recording/checkpoint path — check the traceback in `slurm-logs/*.err` first.

---

## 7. Iterating toward production

Once both smoke tests pass, resource sizing shifts from "conservative
placeholder" to "real numbers from §1":

- Set `--cpus-per-task` to genuinely match `--num-envs` 1:1 (or whatever the
  real core-per-GPU ratio on your partition supports — see §1's throughput
  note).
- Set `--gres=gpu:<real-type>:1` once you know what's actually available,
  rather than the generic `--gres=gpu:1` the smoke scripts use.
- Check your QOS walltime cap (§1) against the production scripts'
  `--time=` request — if your cap is shorter, you're already set up to
  handle this via resume-chaining (next point), so it's not a blocker, just
  something to plan around.

**Start small, extend via `--resume`, rather than one large speculative
commitment.** `slurm/run_rnd.slurm` in particular is intentionally sized as
a modest, checkpoint-chainable first attempt (~3M steps) rather than the
paper's full ~492M-step budget — see the script's own header comment for why.
If it completes cleanly (no auto-stop trigger, i.e. exit code `0:0` not
`42:0`) and shows real progress (`charts/rooms_visited` > 1 and/or nonzero
`charts/episodic_return`), that's the signal to keep going: submit another
`sbatch` job with `--resume $SCRATCH/checkpoints/<run_name>/ckpt_XXXXXX.pt`
pointing at the latest checkpoint, extending the same run rather than
starting over. This is exactly what the resume-fix in this round of changes
was for — before it, every `--resume` invocation silently started a
fresh timestamped run directory instead of continuing the original one.

If the **auto-stop fires instead** (exit code `42:0`, and a clear
`AUTO-STOP: collapse signature sustained...` line in the log) — that's not a
crash, it's the collapse-detection logic working as intended (see
`doc/decisions.md` and `doc/10M-RND-run-failure-documentation.md` for the
failure mode it's guarding against). It means the same collapse from the
prior 10M-step runs is reappearing even under paper-default hyperparameters,
and the next lever to pull is **not** more compute — see
`doc/rnd-vs-ppo-asymmetry-investigation.md`'s open questions (in particular
the `AutoresetMode.NEXT_STEP` GAE-masking bug documented in `doc/decisions.md`,
explicitly out of scope for the current round of changes) before committing
more GPU-hours to the same hyperparameters.
