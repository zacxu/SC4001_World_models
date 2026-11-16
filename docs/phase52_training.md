# P5.2: prepare the paired B/D policy training runs

**The training code is complete; research training is pending.** Steps 1-4 below
verify and register the inputs without training. They use CPU only.

**If you already completed these steps, continue with the
[complete Phase 5 guide](phase5_study.md).** P5.3-P5.5 are now implemented.
That guide reuses your unchanged P5.2 candidate, runs combined verification and
freezes a separate prospective cohort before the four fresh training runs.
Do not repeat this candidate setup solely because endpoint files were added.

The planned experiment has four fresh runs: B/D x seeds 0/1.

| Method | Training per seed |
|---|---|
| B | 20 local PPO cycles |
| D | 25 teacher cycles, 128 distillation trajectories, 100 supervised updates, then 20 local PPO cycles |

Both methods reuse each seed's frozen 4,500-update world model and 50,000 real
joint transitions. No additional P2.4 training is required.

Run the steps **in order, in the same Linux terminal**. Stop if any command fails.

## 1. Open the project environment

Sync the updated code and tests first. Keep your existing checkpoints, replay,
normalization and original four-case selection, including its Cover checkpoints.

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
```

## 2. Verify P5.2

```bash
P52_CHECK="artifacts/phase52/check_$(date +%Y%m%d_%H%M%S)"
python -m scripts.phase52 verify --output-dir "$P52_CHECK"
```

Expected: **`complete_p52_implementation`**, with no failures, errors or skips.
The checks use small synthetic fixtures, including optimizer updates.

## 3. Create an updated P5.1 registration

P5.2 added source files, so create a fresh registration and preserve the old one.

```bash
P51_CHECK="artifacts/phase51/check_$(date +%Y%m%d_%H%M%S)"
P51_RUN="artifacts/phase51_runs/protocol_$(date +%Y%m%d_%H%M%S)"
python -m scripts.phase51 verify --output-dir "$P51_CHECK" &&
python -m scripts.phase51 register --output-dir "$P51_RUN" \
  --initial-selection artifacts/phase24_selection/shortened_v1/selection.json \
  --verification-report "$P51_CHECK/implementation_checks.json"
```

Expected: **`protocol_registered`**. Registration reads and hashes saved inputs.

## 4. Register and inspect the four P5.2 run slots

```bash
P52_RUN="artifacts/phase52_runs/paired_$(date +%Y%m%d_%H%M%S)"
python -m scripts.phase52 register --output-dir "$P52_RUN" \
  --registration "$P51_RUN/protocol_registration.json" \
  --verification-report "$P52_CHECK/implementation_checks.json" &&
python -m scripts.phase52 inspect --cohort "$P52_RUN/cohort_manifest.json" &&
printf '%s\n' "$P52_RUN/cohort_manifest.json" > artifacts/phase52/latest_cohort.txt
```

Expected: **`training_matrix_registered`**, with:

```text
training_tools_ready: true
execution_ready: false
cohort_frozen: false
```

This reserves `runs/B/seed_0`, `runs/D/seed_0`, `runs/B/seed_1` and
`runs/D/seed_1` under `$P52_RUN`. It does not start training.

## 5. Continue with the complete Phase 5 workflow

Follow [phase5_study.md](phase5_study.md) for the combined verification, prospective
freeze, training, accounting, final evaluation and analysis. The original P5.2
candidate keeps `execution_ready: false`; the new cohort has its own identity
and fresh run directories. No old receipt or development run is relabelled.

Modifications to the original registered sources require fresh parent
verification/registration. The additive endpoint migration verifies every old
source byte and binds all new implementation sources separately.
