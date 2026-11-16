# Finish P5.6-P5.8 on the server

Run these **Bash commands on your existing Linux server**, after all four Phase 5
learners and their 80 final test episodes have finished. These steps read saved
results on CPU. They do not train models or run more episodes. No GPU is needed.

Sync the new **`study_reporting/` directory** and updated documentation to
`/home/zixuan/projects/WM1`. Preserve the frozen training code, existing
registrations, checkpoints, inputs, evaluation directories and SHA receipts.
The new package sits outside the frozen cohort's source set so the completed
experiment keeps its original identity. Run from this project root.

## 1. Activate the environment and verify the reporting tools

In your server SSH terminal or existing tmux session:

```bash
cd /home/zixuan/projects/WM1 &&
source scripts/gpu_server_env.sh &&
python -m pip install -r study_reporting/requirements.txt &&
P5_COHORT=$(cat artifacts/phase5/latest_cohort.txt) &&
P5_ROOT=$(dirname "$P5_COHORT") &&
P5_FINISH_CHECK="artifacts/phase565758/check_$(date +%Y%m%d_%H%M%S)" &&
python -m study_reporting.phase5 verify --output-dir "$P5_FINISH_CHECK"
```

Expected: `complete_p565758_implementation`, zero failures/errors/skips.
The extra dependency is [Matplotlib 3.11.2](https://pypi.org/project/matplotlib/3.11.2/)
for saved SVG/PNG figures; the existing
training requirements stay unchanged. Verification uses synthetic CPU fixtures
and mocked graphics, without reading your scientific results.

## 2. Generate the final evidence and acceptance report

In the **same terminal**, after Step 1 succeeds:

```bash
P5_FINISH="$P5_ROOT/completion_$(date +%Y%m%d_%H%M%S)"
python -m study_reporting.phase5 complete \
  --cohort "$P5_COHORT" \
  --verification-report "$P5_FINISH_CHECK/implementation_checks.json" \
  --output-dir "$P5_FINISH"
```

This automatically regenerates P5.5's paired analysis and a final cumulative
ledger from the existing records, then produces:

| Output within `$P5_FINISH` | Purpose |
|---|---|
| `endpoint_analysis/` | Per-arm, paired, per-layout and episode tables |
| `secondary_evidence.json` | Success, return, latch, termination, successful completion time and saved PPO/distillation diagnostics |
| `resource_summary.json`, `resources.csv` | Recorded work, unit timings, runtime and retained storage |
| `training_diagnostics.csv` | Saved final-cycle PPO and before/after distillation measurements |
| `figures/endpoint_results.svg` and `.png` | All four final endpoints, with both seeds visible |
| `figures/paired_effects_compute.svg` and `.png` | Paired differences and D's additional compute |
| `deviations.json` | Scope limitations and unrecorded measurements |
| `phase5_acceptance.json` | Final execution/evidence decision |

Expected: `status: passed`, `phase5_acceptance: true`. This confirms complete,
authenticated execution, including zero or negative results. It does **not**
establish successful transfer or statistical superiority. No-success completion
times stay null with denominator zero. Unrecorded GPU occupancy, CPU time,
queue wait and historical memory/storage peaks remain unknown; recorded unit
elapsed times are not GPU occupancy. PPO diagnostics cover saved final committed
cycles; an authenticated full learning curve is not reconstructed from an
unbound metrics log.

## 3. Recheck the saved report

Continue in the same terminal:

```bash
python -m study_reporting.phase5 inspect \
  --report "$P5_FINISH/phase5_acceptance.json"
```

After `passed`, retain the complete directory and follow the
[P6.1-P6.2 server guide](phase6_release.md). Keep all upstream evidence on the
server; the new stages read saved traces and package the completed study.

If a command fails or reports `needs_review`, read its reasons and preserve that
output. Fix the missing evidence or dependency and use a **fresh completion
directory** for another attempt; do not repeat training or native evaluation to
improve the scores. Do not freeze or prepare a new cohort. The gate authenticates
the original four runs, budgets, pairing, test bank, receipts and recorded retry
history; it cannot repair missing checkpoints or infer unlogged physical work.
