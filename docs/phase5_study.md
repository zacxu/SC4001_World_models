# Phase 5: train, evaluate and compare B/D

Run these **Bash commands on the Linux GPU server** after syncing the updated
code and tests. Preserve the existing P5.1/P5.2 registrations and saved model,
replay and normalization files. The completed P5.2 candidate is reused below.
If its path differs, replace the `--matrix` argument with the path in
`artifacts/phase52/latest_cohort.txt`.

The experiment has **four fresh runs**: B/D x seeds 0/1. B trains for 20 local
PPO cycles; D adds 25 teacher cycles, 128 trajectories and 100 distillation
updates before its 20 local cycles. World models stay at 4,500 updates. No new
real training data is collected.

**Stop if a command fails.** Verification uses CPU fixtures; training and final
native evaluation occur only when you explicitly run their steps below.

## 1. Verify and freeze the completed workflow

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
P5_CHECK="artifacts/phase5/check_$(date +%Y%m%d_%H%M%S)"
P5_ROOT="artifacts/phase5_runs/cohort_$(date +%Y%m%d_%H%M%S)"
python -m scripts.phase5 verify --output-dir "$P5_CHECK"
```

Expected: `complete_p535455_implementation`, with zero failures/errors/skips.

After reviewing prior test exposure, use the following declaration **only if
the fixed 20-episode test bank has not been used for development or evaluation**:

```bash
python -m scripts.phase5 freeze \
  --matrix artifacts/phase52_runs/paired_20261112_074725/cohort_manifest.json \
  --verification-report "$P5_CHECK/implementation_checks.json" \
  --output-dir "$P5_ROOT" --confirm-test-bank-unused &&
P5_COHORT="$P5_ROOT/cohort_manifest.json" &&
printf '%s\n' "$P5_COHORT" > artifacts/phase5/latest_cohort.txt
```

Expected: `cohort_frozen`, `execution_ready: true`. This creates fresh prospective
run slots; it does not promote earlier development runs or start training.
**Freeze once.** In a new terminal, restore its paths with:

```bash
P5_COHORT=$(cat artifacts/phase5/latest_cohort.txt)
P5_ROOT=$(dirname "$P5_COHORT")
```

## 2. Prepare all four runs before starting training

Choose physical GPU IDs on the left. Each selected GPU becomes logical
`cuda:0`. These four commands initialize runs without optimizer updates:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.phase5 prepare --cohort "$P5_COHORT" --variant B --seed 0 --device cuda:0 &&
CUDA_VISIBLE_DEVICES=1 python -m scripts.phase5 prepare --cohort "$P5_COHORT" --variant D --seed 0 --device cuda:0 &&
CUDA_VISIBLE_DEVICES=2 python -m scripts.phase5 prepare --cohort "$P5_COHORT" --variant B --seed 1 --device cuda:0 &&
CUDA_VISIBLE_DEVICES=3 python -m scripts.phase5 prepare --cohort "$P5_COHORT" --variant D --seed 1 --device cuda:0
```

If fewer GPUs are available, reuse an available GPU ID during preparation and
schedule training on available devices. Every slot must be prepared first.
Prepare each slot once; after an error, rerun only the unfinished preparation.

## 3. Train in tmux

Open the session from an interactive SSH terminal:

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  /usr/bin/tmux -L wm-p5-system -f /dev/null new-session -A -s wm-p5
```

Create additional windows with **Ctrl+B, C**. In each window run this block,
setting its three assignments from the table:

| Window | `GPU` | `VARIANT` | `SEED` |
|---|---:|---|---:|
| B seed 0 | 0 | B | 0 |
| D seed 0 | 1 | D | 0 |
| B seed 1 | 2 | B | 1 |
| D seed 1 | 3 | D | 1 |

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
P5_COHORT=$(cat artifacts/phase5/latest_cohort.txt)
P5_ROOT=$(dirname "$P5_COHORT")
GPU=0 VARIANT=B SEED=0
CUDA_VISIBLE_DEVICES="$GPU" python -m scripts.phase5 run \
  --run-dir "$P5_ROOT/runs/$VARIANT/seed_$SEED" --device cuda:0
```

With four available GPUs these are independent parallel jobs. Use one writer
per run directory. Detach with **Ctrl+B, D** and reuse the tmux command to return.
Keep valid poor outcomes and failed/missing runs visible. Allow at most two
infrastructure retries with the same seed/configuration and committed state;
do not retry numerical failures or replace seeds.

## 4. Check progress or switch GPUs safely

In another activated project terminal, restore the paths from Step 1 and run:

```bash
P5_COHORT=$(cat artifacts/phase5/latest_cohort.txt)
P5_ROOT=$(dirname "$P5_COHORT")
python -m scripts.phase5 status --cohort "$P5_COHORT"
python -m scripts.phase53 ledger --cohort "$P5_COHORT" --output-dir "$P5_ROOT/ledger"
```

Each final run must show `training_complete: true` and 20 local cycles. D also
needs 25 teacher cycles, 128 trajectories and 100 distillation updates.

To pause one run, set `VARIANT`/`SEED` and request a checkpointed stop:

```bash
VARIANT=B SEED=0
python -m scripts.phase5 stop --run-dir "$P5_ROOT/runs/$VARIANT/seed_$SEED"
```

Wait for its training process to exit. Then reuse the **Step 3 run command** with
another `GPU` value. It resumes saved optimizer/RNG state; do not prepare again.
Keep the same Python/CUDA runtime, numerical settings and a compatible GPU.
If a progress audit reports that training advanced during its read, rerun it.

## 5. Evaluate the final actors, then analyze the paired results

Wait until **all four runs finish**. Use the same Python environment and **CPU
for all four evaluations**, preserving paired action RNG streams. In a new
SSH/tmux terminal, activate the existing environment and restore the run paths:

```bash
cd /home/zixuan/projects/WM1 &&
source scripts/gpu_server_env.sh &&
P5_COHORT=$(cat artifacts/phase5/latest_cohort.txt) &&
P5_ROOT=$(dirname "$P5_COHORT")
```

```bash
for SEED in 0 1; do
  for VARIANT in B D; do
    python -m scripts.phase54 evaluate \
      --run-dir "$P5_ROOT/runs/$VARIANT/seed_$SEED" \
      --output-dir "$P5_ROOT/evaluation/$VARIANT/seed_$SEED" \
      --device cpu --resume || break 2
  done
done
```

This runs 20 fixed held-out episodes per actor, 80 total. Reusing these exact
commands preserves completed episodes after interruption. Keep every valid poor
outcome; do not select a new checkpoint, bank or output directory from results.

If all four evaluations already completed, continue directly with this block.
It also works from a new terminal. A `python: command not found` error means
the current shell cannot find the project interpreter; activate the existing
environment below instead of switching to the system `python3`.

```bash
cd /home/zixuan/projects/WM1 &&
source scripts/gpu_server_env.sh &&
P5_COHORT=$(cat artifacts/phase5/latest_cohort.txt) &&
P5_ROOT=$(dirname "$P5_COHORT") &&
python -m scripts.phase55 analyze --cohort "$P5_COHORT" \
  --evaluation "$P5_ROOT/evaluation/B/seed_0/evaluation.json" \
  --evaluation "$P5_ROOT/evaluation/D/seed_0/evaluation.json" \
  --evaluation "$P5_ROOT/evaluation/B/seed_1/evaluation.json" \
  --evaluation "$P5_ROOT/evaluation/D/seed_1/evaluation.json" \
  --output-dir "$P5_ROOT/analysis_$(date +%Y%m%d_%H%M%S)" &&
python -m scripts.phase53 ledger --cohort "$P5_COHORT" --output-dir "$P5_ROOT/ledger"
```

The analysis creates `paired_analysis.json` plus arm, paired-difference,
per-layout and episode CSV tables. `complete` means all required evidence was
authenticated, regardless of whether D improves over B. `needs_review` leaves
missing or invalid evidence explicit and never fills it with zeros. Report the
two seed differences and descriptive mean/range; no statistical superiority
claim is established by this exploratory study.

## 6. Finish P5.6-P5.8 on this server

Once all training and 80 test episodes are complete, follow the
[Phase 5 completion guide](phase5_completion.md). It supplies CPU-only server
commands for secondary diagnostics, resource accounting, two endpoint figures
and the final acceptance gate. It also regenerates the paired analysis and final
ledger from saved records if the preceding analysis command was missed. Keep the
same frozen cohort; no additional training or evaluation episodes are required.
