# Analysis set — RND noisy-TV ablation (6 completed 10M runs)

The 6 most recent **completed** production runs, copied from `runs/` on 2026-07-22.
Each ran the full `total_timesteps=10_000_000` (reached step 9,999,360 — none tripped
early auto-stop). Shared config: `num_envs=21`, `num_steps=128`, `lr=1e-4`,
`gamma=0.999`, `int_gamma=0.99`, `int_coef=1.0`, `ext_coef=2.0`, `ent_coef=0.001`,
`clip_reward=True`, `anneal_lr=True`.

These form a **noisy-TV ablation** — the classic RND failure mode where a stochastic
"TV" patch in the observation inflates the prediction-error bonus and can capture the
agent. TV: `tv_size=[12,84]` patch at `tv_position=[0,0]`, `tv_refresh_every=1`.

## Runs

| dir | algo | tv_mode | seed | batch (ts) |
|-----|------|---------|------|------------|
| `MontezumaRevenge-v5__ppo_tv_off__1__1784548767`        | PPO | off         | 1 | 1784548767 |
| `MontezumaRevenge-v5__rnd_tv_off__1__1784548768`        | RND | off         | 1 | 1784548767 |
| `MontezumaRevenge-v5__rnd_tv_off__2__1784548767`        | RND | off         | 2 | 1784548767 |
| `MontezumaRevenge-v5__rnd_tv_remote__1__1784631991`     | RND | remote      | 1 | 1784631991 |
| `MontezumaRevenge-v5__rnd_tv_sham-remote__1__1784631991`| RND | sham-remote | 1 | 1784631991 |
| `MontezumaRevenge-v5__rnd_tv_static__1__1784631991`     | RND | static      | 1 | 1784631991 |

## Analysis pipeline

`scripts/analyze_runs.py` loads every run under a `--logdir`, draws overlaid
comparison charts (all runs on shared axes) into `figures/`, and writes a
derived-events table to `summary.md` / `summary.csv`:

    source .venv/bin/activate
    python scripts/analyze_runs.py --logdir analysis     # regenerates figures/ + summary.*

Figures produced (in `figures/`): `_overview.png` (6-panel), plus one chart per metric —
`rooms_visited` (with ▼ markers at first exit from room 1), `episodic_return`,
`episodic_length`, `raw_intrinsic_rew_mean`, `mean_intrinsic_rew`, `tv_intrinsic_share`,
`tv_action_frac`, `entropy`, `explained_variance`, `approx_kl`.

### Interactive overlay (TensorBoard)

    tensorboard --logdir analysis        # all 6 runs overlaid on each scalar

## Derived-events summary

| run | tv_mode | max_rooms | left room 1 | final return (l20) | max return | peak intrinsic | final entropy | final TV share |
|-----|---------|----------:|------------:|-------------------:|-----------:|---------------:|--------------:|---------------:|
| ppo_tv_off s1         | off         | 2 | 3.99M | 0  | 400 | —     | 2.763 | —      |
| rnd_tv_off s1         | off         | 1 | never | 0  | 0   | 192.8 | 0.503 | -0.091 |
| rnd_tv_off s2         | off         | 2 | 6.44M | 10 | 400 | 145.3 | 0.611 | -0.198 |
| rnd_tv_remote s1      | remote      | 1 | never | 0  | 100 | 257.9 | 0.762 |  0.117 |
| rnd_tv_sham-remote s1 | sham-remote | 1 | never | 0  | 0   | 219.0 | 0.415 | -0.063 |
| rnd_tv_static s1      | static      | 1 | never | 0  | 0   | 234.4 | 0.282 |  0.203 |

Read-outs:
- **Exploration is weak across the board** — only the PPO baseline (~4.0M steps) and one
  RND `off` seed (~6.4M) ever left room 1; the other four never did.
- **Noisy-TV signature**: `static` and `remote` TV modes drive a positive, growing
  **TV share of intrinsic reward** (~0.20 / ~0.12), i.e. the curiosity bonus is being
  spent on the TV patch, whereas `off`/`sham-remote` stay near zero. Their raw intrinsic
  reward also stays elevated late in training.
- **Entropy collapse** in every RND run (final 0.28–0.76 vs PPO's 2.76), lowest for
  `static` — consistent with the TV pinning the policy.

`summary.csv` has the machine-readable version (also columns for `peak_intrinsic_step`
and `final_expl_var_l20`).
