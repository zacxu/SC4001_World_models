# P4.5: D engineering acceptance

**Let P4.4 finish training.** You can run Step 1 in a separate terminal now.
Step 2 requires the completed campaign's receipt-bound `completed_student.json`.
Neither command trains the research models or runs real-environment episodes.

Sync the new `src/world_models_acceptance/`, `scripts/phase45.py` and updated
`tests/` files. Preserve the existing P4.1-P4.4 production files, stage
directories, checkpoints and receipts so the running job can continue.

## 1. Verify the implementation on CPU

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
P45_VERIFY="artifacts/phase45/check_$(date +%Y%m%d_%H%M%S)"
python -m scripts.phase45 verify --output-dir "$P45_VERIFY" &&
  printf '%s\n' "$P45_VERIFY" > artifacts/phase45/latest_verification.txt
```

This runs locality, target/gradient isolation, pairing, recovery and deployment
fixtures, including a tiny synthetic D smoke check. A passing implementation
report establishes tested mechanics; it does not accept your trained campaign.
The successful verification directory is saved for Step 2, including if you
open a new terminal. If verification fails, stop and inspect its report.

## 2. Accept the completed D run

Run after P4.4 is complete. Set `CASE` to the actual integrated run directory;
use the adopted directory if you registered completed standalone P4.1-P4.3 work.
Keep all referenced stage directories and original frozen input paths available.

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
CASE=artifacts/phase44_runs/D_four_day_v1/relay/seed_1
P45_RUN="artifacts/phase45_runs/accept_seed1_$(date +%Y%m%d_%H%M%S)"

if [ -s artifacts/phase45/latest_verification.txt ]; then
  P45_VERIFY=$(cat artifacts/phase45/latest_verification.txt)
  python -m scripts.phase45 accept --run-dir "$CASE" \
    --verification-report "$P45_VERIFY/implementation_checks.json" \
    --output-dir "$P45_RUN" --device cpu
else
  printf '%s\n' 'Run Step 1 successfully first.'
fi
```

For the existing adopted seed-0 run, use
`CASE=artifacts/phase44_runs/D_adopted_v1/relay/seed_0` and change the output name
to `accept_seed0_...`. Use a fresh output directory for every audit. If you use a
different project path, replace the `cd` path in both steps.

If Step 1 already passed using the earlier guide, save its actual directory
to `artifacts/phase45/latest_verification.txt`. You can also reuse the shipped
`artifacts/phase45/implementation_v1/` directory if it is synced with all three
files (`implementation_checks.json`, `implementation_checks.sha256`, `tests.txt`):

```bash
python - <<'PY'
from world_models_acceptance.evidence import load_valid_verification
load_valid_verification('artifacts/phase45/implementation_v1/implementation_checks.json')
from pathlib import Path
Path('artifacts/phase45/latest_verification.txt').write_text(
    'artifacts/phase45/implementation_v1\n', encoding='utf-8')
print('Current implementation evidence verified; continue with Step 2.')
PY
```

This rechecks the current source, report receipt and test log. If it fails,
run Step 1 again. Preserve failed acceptance outputs and retry into a fresh
directory; no P4.4 retraining is needed for a verification-path error.

The audit checks completed stage lineage, saved targets, immutable model/data,
paired initialization, optimizer/RNG counters, budgets and local deployment.
It writes **`phase4_acceptance.json`** and its SHA receipt; the terminal shows a
short decision and report path. A nonzero exit or `needs_review` means inspect
its reasons before proceeding. Acceptance supports **`--device cpu` only**.
Exact GPU execution and teacher-target regeneration remain unmeasured. Rerun
Step 1 after any verification-source change.

**A pass is engineering acceptance only.** It neither proves successful transfer
nor promotes development/adopted runs into a fresh comparison cohort. P5 still
requires the matched reduced B run, a frozen cohort and the paired final
real-environment evaluation. Historical P4.4 acceptance flags remain false; the
new receipt-bound P4.5 report records the separate decision. D is allowed to
perform worse than B.
