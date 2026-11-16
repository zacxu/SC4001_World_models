# P5.1: freeze the exploratory protocol

This step records the Relay-only B/D protocol, two paired seeds, fixed inputs,
20 final test episodes per actor and descriptive endpoint analysis. It does
**not** train policies, run test episodes or make the comparison ready to run.

Sync `src/world_models_study/`, `scripts/phase51.py`,
`configs/studies/relay_endpoint_protocol_v1.yaml` and updated `tests/`.
Preserve existing source files, run directories and
receipts. Run these Bash commands on the Linux server that holds both seeds'
replay, normalization and selected P2.4 checkpoints.
Retain the original four-case selection and its referenced Cover checkpoints;
the historical selection validator still checks them. Registration reads and
hashes existing files on CPU and does not construct or optimize models.

## 1. Verify the static implementation

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
P51_VERIFY="artifacts/phase51/check_$(date +%Y%m%d_%H%M%S)"
if python -m scripts.phase51 verify --output-dir "$P51_VERIFY"; then
    printf '%s\n' "$P51_VERIFY" > artifacts/phase51/latest_verification.txt
fi
python -m scripts.phase51 plan
```

Verification uses static CPU fixtures, without optimizer updates or native
episodes. Continue only after `complete_p51_implementation`.

## 2. Register the protocol and frozen inputs

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
if [ -s artifacts/phase51/latest_verification.txt ]; then
    P51_VERIFY=$(cat artifacts/phase51/latest_verification.txt)
    P51_RUN="artifacts/phase51_runs/protocol_$(date +%Y%m%d_%H%M%S)"
    if python -m scripts.phase51 register --output-dir "$P51_RUN" \
      --initial-selection artifacts/phase24_selection/shortened_v1/selection.json \
      --verification-report "$P51_VERIFY/implementation_checks.json"; then
        printf '%s\n' "$P51_RUN/protocol_registration.json" > artifacts/phase51/latest_registration.txt
    fi
else
    printf '%s\n' 'Complete Step 1 successfully before registering.'
fi
```

Defaults use `data/initial_replay/pilot`, `artifacts/phase23/normalization` and
`configs` under this project. Supply `--data-root`, `--statistics-root` or
`--config-root` only if those stores are elsewhere. Keep the recorded paths and
their receipt-bound files available. Each registration needs a fresh directory.

## 3. Inspect the saved registration

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
if [ -s artifacts/phase51/latest_registration.txt ]; then
    python -m scripts.phase51 inspect \
      --registration "$(cat artifacts/phase51/latest_registration.txt)"
else
    printf '%s\n' 'Complete Step 2 successfully before inspecting.'
fi
```

A successful registration reports **`protocol_frozen: true`** and
**`candidate_only: true`**, with **`execution_ready: false`** and
**`cohort_frozen: false`**. P5.2 now supplies the
[matched training tools](phase52_training.md). The implemented
[complete Phase 5 workflow](phase5_study.md) adds final testing, paired analysis
and a separate prospective freeze. Use that workflow after P5.2 registration.
The preserved P5.1 candidate schema records its original pending
capabilities; newer implementation readiness is recorded separately. No training
command is supplied by P5.1.

Existing completed/adopted D runs retain their development chronology; this
registration cannot retrospectively promote them into a fresh cohort. Preserve
previous P4.5 reports. Adding tests changes the current verification source set,
so rerun P4.5 verification if you later perform another P4.5 audit. Protocol
registration does not establish learned quality or B-versus-D efficacy.
