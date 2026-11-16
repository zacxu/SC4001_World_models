# P3.5 completion and P3.6 validation

## Current saved result

`B_week_check_fp32_v1/relay/seed_0` has completed its training budget:
**80/80 PPO cycles, 1,280 actor and critic optimizer steps each,
100,000 real joint transitions and 5,000 cumulative model updates**.

The saved controller is `stopped`, with phase `local`: it stopped after the
last PPO cycle, before the separate final actor-export step. Therefore
`evaluation/real_000100000` is absent. That missing directory caused the
P3.6 manifest-receipt error. No additional training is needed to export it.

The earlier 50k diagnostic achieved 0/8 successes for both actors. Preserve
[that result](../artifacts/phase36_runs/relay_week_seed0_20261110_120844/phase3_acceptance.json);
the final actor's real performance still needs validation.

## 1. Open the server environment

After transferring the updated scripts and support files, use an interactive
SSH terminal. Open or reattach tmux:

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  /usr/bin/tmux -L wm-p36-system -f /dev/null new-session -A -s wm-first-b
```

Inside tmux, choose an available GPU. Physical GPU `3` becomes logical `cuda:0`:

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
export CUDA_VISIBLE_DEVICES=3
CASE=artifacts/phase35_runs/B_week_check_fp32_v1/relay/seed_0
EXPORT="$CASE/evaluation/real_000100000_saved_v1"
```

## 2. Export the saved final actor without training

```bash
python -m scripts.phase35_export --run-dir "$CASE" --output-dir "$EXPORT"
```

This CPU-only command verifies the latest committed checkpoint, original run
identity and completed training counters, then publishes actor-only weights,
manifest, receipt and `export_provenance.json`. It never resumes an optimizer,
collects data or changes the original run's status/checkpoints.

Use a fresh output directory. If this export has already been created, keep it
and proceed to step 3. If a previous publication was interrupted, preserve that
directory and choose a new suffix such as `saved_v2`.

The original status can still read `stopped`; the separate export report records
that its training budget was complete. Do not repeat `prepare` or `recover`.
Ordinary training resume requires the original source/runtime; this export
command supports the newer P4.1 checkout because it only projects saved weights.

## 3. Validate the untrained and final actors

Use fresh output directories and the same **eight validation episodes**, bank
seed **3600**, for both actors. Rerun the baseline with the current evaluator:
old cached reports were produced before the P4.1 source changes.

```bash
OUT="artifacts/phase36_runs/relay_week_seed0_100k_$(date +%Y%m%d_%H%M%S)"

python -m scripts.phase36 evaluate \
  --actor-export "$CASE/evaluation/real_000000000" \
  --output-dir "$OUT/baseline" --episodes 8 --bank-seed 3600 --device cuda:0 &&
python -m scripts.phase36 evaluate \
  --actor-export "$EXPORT" \
  --output-dir "$OUT/trained" --episodes 8 --bank-seed 3600 --device cuda:0
```

Detach with **Ctrl+B, then D**. Each evaluation loads only the local actor;
it performs no policy/model updates and never adds evaluation data to replay.
Reports are saved as `$OUT/baseline/evaluation.json` and
`$OUT/trained/evaluation.json`. Compare success rate, mean return and latch
progress. These are development validation episodes, separate from the final
study's planned test bank. Disclose the earlier 50k inspection as well.

## 4. Interpret validation and acceptance separately

A completed evaluation establishes measured behavior for that actor and bank.
Improvement over the paired untrained actor supports a limited feasibility
claim. Zero success or no latch progress remains a reason to review the learned
model/policy before a larger campaign; it is not a receipt problem.

The historical `phase36 accept` gate requires the original training source and
all supplied implementation, mechanics and evaluation reports to match the
checkout. **The newer P4.1 checkout does not meet that source requirement.**
Reusing the old `$PREV` reports or editing their hashes will not make it valid.
The commands above produce fresh diagnostic evidence without claiming formal
P3.6 acceptance. Use the archived original checkout for that historical gate;
preserve its source and reports if it is still available.
