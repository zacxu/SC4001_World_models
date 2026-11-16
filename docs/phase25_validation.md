# P2.5: screen the frozen pilot checkpoints

**Current result:** the [2026-11-09 screen](../artifacts/phase25_runs/shortened_20261109_100750/campaign/report.json)
is complete. Both Relay cases are ready for exploratory Phase 3; Cover keeps the
overall campaign at `needs_review`. The one-week study uses Relay only. Reuse
this evidence when selection/data are unchanged, then run the
[bounded B check](phase36_acceptance.md) with `relay_week_v1`. The commands below
remain the reproducible screening procedure; another identical screen is not
required merely because later policy budgets changed.

Use the four checkpoints frozen in [P2.4](phase24_training.md).
They may have different saved update counts; the campaign records those counts
and original targets. **Do not wait for 5,000 updates.** Keep the earlier
3,750/4,000-update reports as historical results.

The screen evaluates each shared B/D model once, using up to 64 windows and the
same sampling seed. The default runs four workers in parallel on **four GPUs,
one worker per GPU**; a two-GPU alternative is below.
It performs no model training or new simulator collection.

## 1. Start tmux and activate the server environment

Complete the [new server setup](../linux_gpu_server_setup.md), including the
checkpoint inspection. From an SSH shell, start a persistent terminal:

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
tmux new -s wm-p25
```

Inside tmux, choose the available **physical GPU IDs** in the last line:

```bash
export WM_ROOT=/home/zixuan/projects/WM1
cd "$WM_ROOT"
source scripts/gpu_server_env.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

For example, selecting physical GPUs `1,3,4,6` still gives the Python workers
logical GPUs `0,1,2,3`. Detach with **Ctrl+B, then D**; reconnect using
`tmux attach -t wm-p25`. In a new SSH shell, change to the project and source
`scripts/gpu_server_env.sh` before reconnecting. No queue submission is needed.

## 2. Optional timing check

Skip this if you have already measured runtime. For a new measurement, use the
campaign command in step 3 with `--windows 8` and a fresh output directory.
This times eight windows **per model**, 32 total. Such a small sample cannot
establish readiness. No fixed runtime is guaranteed.

## 3. Screen the four selected checkpoints

Run this entire block in the activated tmux terminal:

```bash
cd /home/zixuan/projects/WM1
P25_RUN="artifacts/phase25_runs/shortened_$(date +%Y%m%d_%H%M%S)"
python -m scripts.phase25 campaign \
  --selection artifacts/phase24_selection/shortened_v1/selection.json \
  --output-dir "$P25_RUN/campaign" \
  --gpus 0,1,2,3 --workers-per-gpu 1 --windows 64 --seed 0
```

For **two GPUs**, expose two physical IDs and use
`--gpus 0,1 --workers-per-gpu 2`. This still starts all four workers in parallel;
check GPU memory if sharing a device. Set the mask before starting Python.

Required CPU implementation checks run automatically before inference. All four
selected checkpoint/receipt pairs are verified; newer saves or validation scores
cannot change the selection. Use a fresh output directory for every attempt.

If you see `Permission denied: '/campaign'`, rerun the complete block so
`P25_RUN` is assigned. Per-worker logs are in
`$P25_RUN/campaign/worker_logs/`. Check these logs if a worker fails or runs out
of GPU memory. For sequential local CPU evaluation, replace both GPU flags with
`--device cpu`; retain the same selection, replay and normalization files.

## 4. Read the result

Read `$P25_RUN/campaign/report.json` and `pilot_model_selection.json`.
Both need matching SHA receipts. They record the frozen selection identity,
actual per-model update counts, shortened-budget reason and readiness:

- `ready_for_exploratory_phase3`: supports a small integration pilot, followed
  by the [real-policy acceptance check](phase36_acceptance.md).
- `needs_review`: inspect the listed event, success, termination or image
  alarms before expanding policy training.
- `failed`: resolve the reported failure before proceeding.

The earlier 4,000-update screen was `needs_review`; a shortened budget does not
clear its warnings. Every compact screen keeps `P2.5_quality_acceptance: false`:
diagnostic episodes use training geometry, so formal held-out-geometry acceptance
remains separate.

Metrics remain unchanged: up to 32 uniform and 32 event-targeted windows,
complete-prefix reconstruction, autonomous recorded-action rollouts at horizons
1/5/10/15, RGB versus persistence, reward and event/termination predictions.
Missing rare-event positives remain unvalidated. Acceptance thresholds are unchanged.

Use the **original P2.4 `selection.json`** for Phase 3's
`--initial-selection`; P2.5's `pilot_model_selection.json` records evaluation
evidence and is not a replacement input. Back up both reports and all checkpoint
pairs in durable storage.
