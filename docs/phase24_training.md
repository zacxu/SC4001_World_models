# P2.4: fit, validate, and refresh on the GPU server

The P2.4 implementation is complete and checked with CPU fixtures, including
small optimizer/checkpoint tests. Research fitting, native validation-bank
collection, GPU execution, and the P2.5 prediction-quality gate remain pending
until you run them on the server. Implementation checks do not establish that a
trained world model has usable autonomous rollouts.

The requested default is **four independent workers running in parallel on one
GPU, logical GPU 0**: relay/seed 0, relay/seed 1, cover/seed 0, and cover/seed 1.
Each worker fits one shared initial B/D model. This produces four physical fits,
with two bindings per completed fit. The launcher starts all four workers before
waiting for any worker to finish. It does not perform distributed training.

These steps assume the existing Linux environment and transferred pilot replay
already passed the [setup through P2.3](../linux_gpu_server_setup_through_p23.md).
Keep that environment. No dependency reinstall is required for these new Python
modules. Run commands from the project root in your assigned GPU allocation.

## 1. Transfer the new source and activate the existing environment

Publish the updated source commit to your existing GitHub repository using your
normal Git workflow. Include `scripts/phase24.py`, the new training modules and
tests, this guide, and the updated project documentation and `.gitignore`.
Keep `.gitattributes` included. Source publication does not transfer ignored
replay, checkpoints, validation banks, or run directories.

On the server, inspect local changes before updating. Stop if this checkout has
uncommitted changes that need to be preserved; resolve those before pulling.

```bash
cd "$HOME/projects/world-models"
git status --short
git pull --ff-only
source .venv/bin/activate
python -m pip check
```

Use the actual existing project directory if yours differs. If you previously
used a Conda environment directly, activate that existing environment instead of
the `.venv`; do not create another environment for this step.

Preserve `data/initial_replay/pilot/`, `artifacts/phase23/normalization/`, their
receipts, and the four source files marked `-text` in `.gitattributes`. Do not
convert line endings, edit historical manifests, regenerate frozen statistics,
or replace SHA receipts to make a check pass. The four stores contain 200,000
joint transitions physically; B/D share each task/seed store and normalizer.

Set process thread limits and the deterministic CUDA workspace setting:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m scripts.phase24 --help
python -m scripts.phase24 launch --help
nvidia-smi
python - <<'PY'
import sys
import torch
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
assert torch.cuda.is_available(), "Activate the existing GPU environment and GPU allocation."
print("Visible GPUs:", torch.cuda.device_count())
print("Logical GPU 0:", torch.cuda.get_device_name(0))
assert torch.cuda.is_bf16_supported(), "The default fitting precision requires bfloat16 support."
assert torch.ones(8, device="cuda:0").sum().item() == 8
PY
df -h .
```

Check your server's storage quota as well as filesystem free space. At the
default interval each case retains its update-zero checkpoint and checkpoints
every 250 updates, including update 5,000. Allow space for all four cases,
optimizer states, logs, validation evidence, and backups. Measure actual sizes
after the small run below; free space alone does not establish quota headroom.

`--gpus` names GPUs visible to the current allocation. With a scheduler-provided
`CUDA_VISIBLE_DEVICES`, logical GPU 0 can refer to a different physical GPU.
The launcher preserves that mapping and gives each child one visible GPU.
Each worker therefore reports `cuda:0`, including workers sharing one GPU.

## 2. Run implementation checks and inspect the launch plan

These commands do not start a research fit or simulator collection. `verify`
runs CPU unit fixtures, including tiny optimizer/checkpoint tests; it writes a
new implementation-verification report. Use a new check directory each time.

```bash
set -e
P24_CHECK="artifacts/server_p24_check_$(date +%Y%m%d_%H%M%S)"
python -m scripts.phase24 verify --output-dir "$P24_CHECK"
python -m scripts.phase24 plan \
  --output-dir artifacts/phase24_runs/pilot_v1 \
  --gpus 0 --workers-per-gpu 4 \
  --report "$P24_CHECK/initial_launch_plan.json"
```

Expected verification status: `complete_p24_implementation`. Inspect skips:
CUDA-dependent tests can be skipped in the implementation suite and do not
replace the server's actual GPU run. The plan must list four jobs, all mapped
to GPU 0, `training_executed: false`, and zero optimizer updates executed by the
plan. Planning reads the sealed replay, paired bindings, and frozen statistics.

The approved initial defaults are:

| Setting | Value per worker |
|---|---:|
| Initial optimizer updates | 5,000 |
| Effective sequence batch | 32 |
| Maximum window | 64 decisions: 16 burn-in + 48 loss steps |
| Microbatch | 4 sequences |
| Precision | bfloat16 computation; float32 probability/loss/optimizer state |
| Optimizer | AdamW, learning rate 0.0001, weight decay 0.0001 |
| Global gradient clipping | 100 |
| Checkpoint and fitting-data diagnostic interval | 250 updates |

`python -m scripts.phase24 <command> --help` and the checked-in implementation
are the source of truth for available arguments. Record any pilot overrides.

## 3. Run the four small overfit checks in parallel

Run this on the server yourself, using a fresh directory. All four cases run
concurrently on GPU 0. The fixed batch uses fitting episodes, includes task
milestones and ordinary windows, and is kept separate from the full fit.

```bash
python -m scripts.phase24 launch \
  --stage overfit \
  --output-dir artifacts/phase24_runs/overfit_v1 \
  --gpus 0 --workers-per-gpu 4 \
  --effective-batch 4 --microbatch 1 \
  --burn-in 4 --loss-steps 4 --updates 50 \
  --checkpoint-every 10 --validate-every 10 \
  --validation-windows 4 --log-every 1
python -m scripts.phase24 status --output-dir artifacts/phase24_runs/overfit_v1
```

Each case should reach `complete_fit_stage` with 50 stage updates. Inspect its
`events.jsonl` for `fixed_overfit_fixture`, `metrics.jsonl` for loss/gradient and
label support, and `result.json` for the completed checkpoint. This summary
prints the observed fitting trend without changing any artifacts:

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path("artifacts/phase24_runs/overfit_v1")
for family in ("relay", "cover"):
    for seed in (0, 1):
        case = root / family / f"seed_{seed}"
        result = json.loads((case / "result.json").read_text())
        rows = [json.loads(line) for line in (case / "metrics.jsonl").read_text().splitlines()]
        fit = [row for row in rows if row.get("event") == "fit"]
        first = sum(row["total"] for row in fit[:5]) / min(5, len(fit))
        last = sum(row["total"] for row in fit[-5:]) / min(5, len(fit))
        print(family, seed, result["status"], "updates", result["stage_updates"], "first/last mean loss", first, last)
        names = ("event/latch_activations",) if family == "relay" else ("event/console_1_completions", "event/console_2_completions")
        for name in names:
            print(" ", name, "last global diagnostics:", fit[-1].get("classification", {}).get(name))
PY
```

Before scaling, require finite losses and gradients, a clear downward loss
trend on the fixed batch, and positive support for the relevant latch/console
labels. Inspect their predictions separately: decreasing background-image loss
does not demonstrate milestone reproduction. If support is missing or event
predictions remain uninformative, stop and inspect the selected windows and
target alignment. Fifty updates are a configured debugging budget, not a
validated learning requirement or a guarantee of success. Preserve failures and
document any pilot adjustment applied to both B/D methods.

A successful small overfit run checks the fitting path. Held-out prediction
quality and autonomous event reproduction still belong to P2.5. Never use the
overfit checkpoint as the shared initial research checkpoint.

## 4. Start the full four-worker initial fit

Use a persistent foreground terminal so that disconnecting the SSH client does
not close the launcher. If `tmux` is available, start a session:

```bash
tmux new-session -s wm-p24
```

Inside that session, return to the project and activate the same environment:

```bash
cd "$HOME/projects/world-models"
source .venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m scripts.phase24 launch \
  --stage initial \
  --output-dir artifacts/phase24_runs/pilot_v1 \
  --gpus 0 --workers-per-gpu 4 \
  --microbatch 4
```

Run this command yourself when the small-run gate passes. It starts the four
full fits concurrently and leaves the launcher in the foreground. The omitted
`--updates` argument selects the configured 5,000 updates per case. Keep the
effective batch at 32 and the default 16/48 context/loss split. The small-run
overrides from step 3 do not carry into this new command or directory.

Detach with **Ctrl-B, then D**. Reattach with `tmux attach -t wm-p24`. If your
server uses another persistent job mechanism, run the same foreground command
there within your assigned GPU allocation.

From a second terminal in the project environment:

```bash
python -m scripts.phase24 status --output-dir artifacts/phase24_runs/pilot_v1
cat artifacts/phase24_runs/pilot_v1/launch_status.json
tail -n 20 artifacts/phase24_runs/pilot_v1/worker_logs/relay_seed_0.log
nvidia-smi
```

Worker logs are `worker_logs/{relay,cover}_seed_{0,1}.log`. Per-case
`status.json`, `events.jsonl`, `metrics.jsonl`, `run_manifest.json`, and
`result.json` describe progress, invocation history, configuration, timings,
losses, gradients and checkpoint provenance. GPU memory and time are measured
on this server; the local implementation tests did not establish them.

Expected completed checkpoint paths:

```text
artifacts/phase24_runs/pilot_v1/relay/seed_0/checkpoints/update_00005000.pt
artifacts/phase24_runs/pilot_v1/relay/seed_1/checkpoints/update_00005000.pt
artifacts/phase24_runs/pilot_v1/cover/seed_0/checkpoints/update_00005000.pt
artifacts/phase24_runs/pilot_v1/cover/seed_1/checkpoints/update_00005000.pt
```

Each checkpoint has a sibling `.sha256`. Each case's `bindings/B.json` and
`bindings/D.json` reference the same completed initial checkpoint and frozen
normalizer. A completed optimizer schedule has status `complete_fit_stage`;
the launcher reports `complete_fit_stages`. Both retain P2.5 quality acceptance
as false.

Other parallel mappings are supported, for example `--gpus 0,1
--workers-per-gpu 2` or `--gpus 0,1,2,3 --workers-per-gpu 1`. These still run
four independent workers. The default remains four workers sharing GPU 0.
Inspect the plan before changing allocation or worker mapping.

## 5. Resume an interrupted full fit

Preserve the original directory and logs. Re-run the exact step-4 launch
arguments with `--resume` added:

```bash
python -m scripts.phase24 launch \
  --stage initial \
  --output-dir artifacts/phase24_runs/pilot_v1 \
  --gpus 0 --workers-per-gpu 4 \
  --microbatch 4 \
  --resume
```

Resume restores the latest committed optimizer-boundary checkpoint for each
case, including model/optimizer state, absolute update count, replay sampler
state and sampling/global RNG state. It finishes the remaining part of the
original 5,000-update target. Already completed cases do not repeat their
optimizer schedule.

Settings, source identities, replay order/digests, normalization, precision and
device mapping must match the recorded run. A changed microbatch is a changed
run configuration: use a fresh run directory and document the adjustment.
Do not silently change it under `--resume`. Exact continuation is scoped to
the supported pinned runtime; the run records runtime/device information.

Incomplete checkpoint files are preserved under `checkpoints/uncommitted/`
during guarded recovery. Missing receipts and changed digests fail closed.
Work since the last committed checkpoint can be repeated after a crash; its
earlier logs remain visible and count toward physical compute. Fitting adds no
new environment interactions. If no committed checkpoint is available,
preserve the failed run and use a fresh output directory.

## 6. Collect a separate validation-geometry bank

After the initial fits, collect validation data yourself on the server. The
native simulator needs the existing Linux ACC installation; fitting itself did
not need to start it. Restore its environment variables in this terminal:

```bash
export WORLD_MODELS_ACC="$PWD/.tools/acc-linux/acc"
export IMAGEIO_FFMPEG_EXE="$(command -v ffmpeg)"
test -x "$WORLD_MODELS_ACC"
set -e
for task in relay cover; do
  for seed in 0 1; do
    python -m scripts.phase24 collect-validation \
      --task "$task" --seed "$seed" --episodes-per-template 4 \
      --output-dir "data/model_validation/pilot/$task/seed_$seed"
  done
done
```

This loop runs native collection cases sequentially; the training launcher
above remains parallel. Each case uses both held-out validation templates and
alternates random/scripted episodes. The bank has purpose `model_validation`,
is ineligible for fitting/imagination, and is distinct from the 10% hashed
model-diagnostic partition of training geometry.

Preserve each entire bank: manifest and receipts, episode NPZ/SHA pairs,
`validation_interactions.jsonl`, compiled scenarios, and native runtime journals.
Bank loading verifies typed/index/reward/event consistency and native
dispatch/ACK evidence. Validation collection has real simulator cost, recorded
separately; those transitions do not enter learning replay or normalization.
The bank directory is ignored by Git and needs its own verified backup.

Never overwrite an existing bank to retry collection. A failed attempt retains
its error record and charged interactions. Inspect the cause and use a fresh
bank directory for a new campaign; update the diagnostic paths accordingly.

## 7. Evaluate held-out validation and compare burn-in lengths

Use fresh diagnostic output paths. These commands load completed checkpoints,
perform no optimizer updates, and collect no new simulator interactions:

```bash
set -e
FIT_ROOT=artifacts/phase24_runs/pilot_v1
BANK_ROOT=data/model_validation/pilot
DIAG_ROOT=artifacts/phase24_runs/pilot_v1/validation_v1
test ! -e "$DIAG_ROOT"
mkdir -p "$DIAG_ROOT"
for task in relay cover; do
  for seed in 0 1; do
    python -m scripts.phase24 validate \
      --checkpoint "$FIT_ROOT/$task/seed_$seed/checkpoints/update_00005000.pt" \
      --validation-bank "$BANK_ROOT/$task/seed_$seed" \
      --device cuda:0 --microbatch 1 --validation-windows 32 \
      --output "$DIAG_ROOT/${task}_seed_${seed}_validation.json"
    python -m scripts.phase24 burn-in \
      --checkpoint "$FIT_ROOT/$task/seed_$seed/checkpoints/update_00005000.pt" \
      --validation-bank "$BANK_ROOT/$task/seed_$seed" \
      --device cuda:0 --microbatch 1 --validation-windows 32 \
      --output "$DIAG_ROOT/${task}_seed_${seed}_burn_in.json"
  done
done
```

Expect `geometry_split: validation`, exact checkpoint/bank identities, selected
windows, zero optimizer updates, and zero new interactions in each report.
The burn-in report compares 16, 32 and 64 context steps with a full-prefix
reference using the same loss anchors and target transitions. Available context
near episode reset can be shorter than the requested burn-in. The loss horizon
stays fixed; increasing context does not shorten its target suffix.

Truncated context initializes reset memory at the window's **absolute** episode
clock, reconstructs posterior context without gradients and detaches that
boundary. It is an explicit approximation to full-prefix history. Latent
sampling makes these comparisons stochastic diagnostics; suffix draws are not
coupled across context lengths. Do not treat one comparison as a definitive
context-selection result. Inspect support and event-specific behavior and
document any common B/D pilot choice before freezing the protocol.

During fitting, periodic `model_diagnostic` metrics use the held-out episode
partition of **training geometry** and do not select validation configurations.
`validate` without `--validation-bank` also reports that training-geometry
diagnostic scope. Burn-in selection requires the separate validation bank.
Neither posterior-conditioned validation loss nor this context comparison
passes P2.5: persistence/constant-frequency comparisons, autonomous fixed-action
rollouts, rare-event calibration and future-leakage checks remain required.

## 8. Refresh a branch after future policy collection

The refresh fitting API is implemented. Later real local-policy collection
belongs to subsequent phases and is not implemented by this P2.4 CLI. Do not run
the examples below until genuine branch replay exists with its counted training
ledger and valid sealed manifests.

Each refresh must extend the parent's ordered replay cohort by exactly 25,000
new real joint transitions. New shards must record `training_variant: B` or
`training_variant: D` matching the requested branch, share the task/seed/schema,
use training geometry, and have unique episode/content identities. Never use a
validation bank or synthetic PPO images as refresh data.

Example first B refresh, using an illustrative future replay path:

```bash
python -m scripts.phase24 refresh \
  --task relay --seed 0 --variant B \
  --parent-checkpoint artifacts/phase24_runs/pilot_v1/relay/seed_0/checkpoints/update_00005000.pt \
  --replay data/online/pilot/B/relay/seed_0/block_0001/replay \
  --output-dir artifacts/phase24_runs/refresh_v1/B/relay/seed_0/block_0001 \
  --device cuda:0 --microbatch 4 --updates 500
```

The first completed refresh has absolute model-update count 5,500 and checkpoint
`checkpoints/update_00005500.pt`. For the next block, use that completed parent,
repeat `--replay` for **all** online shards oldest first, append the new shard,
and select a new output directory:

```bash
python -m scripts.phase24 refresh \
  --task relay --seed 0 --variant B \
  --parent-checkpoint artifacts/phase24_runs/refresh_v1/B/relay/seed_0/block_0001/checkpoints/update_00005500.pt \
  --replay data/online/pilot/B/relay/seed_0/block_0001/replay \
  --replay data/online/pilot/B/relay/seed_0/block_0002/replay \
  --output-dir artifacts/phase24_runs/refresh_v1/B/relay/seed_0/block_0002 \
  --device cuda:0 --microbatch 4 --updates 500
```

The initial replay is supplied through the normal `--data-dir` default; do not
repeat it as an online `--replay` shard. The parent must be a completed initial
or same-branch refresh checkpoint. Keep B/D outputs and later replay separate.
The initial checkpoint and normalizer are shared; later branch model weights
can diverge as their real data differ.

Refresh sampling uses a 50/50 mixture of uniform valid fitting anchors from all
ordered replay and from the latest 100,000 physical transitions. Diagnostic
transitions occupy their real chronology positions but cannot be sampled for
fitting. This is a mixture probability, not a promise of exactly 16 anchors
from each component in every batch of 32.

The initial fitting-only normalizer stays frozen. Checkpoints bind its identity,
ordered replay hashes, configuration/source hashes, optimizer state, update
schedule and RNG state. A refresh has its own recorded branch/stage stream;
resuming that refresh restores its recorded state. Use the same refresh command
with `--resume` only for an interrupted existing refresh directory.

## Stop conditions and handoff

Stop on a missing GPU allocation, unsupported precision, OOM, insufficient disk
space/quota, a missing bank or receipt, changed data/source/configuration digest,
nonfinite loss/gradient, inconsistent label support, or a failed worker. Keep
the corresponding logs and checkpoints. For OOM, choose a smaller microbatch
in a **fresh run** while retaining effective batch 32; aggregate demand comes
from four simultaneous workers. Do not reduce the real budget or effective
batch silently. Re-plan and record any other allocation change.

Gradient accumulation releases each completed microbatch graph and uses global
component counts plus per-field auxiliary denominators over the full effective
batch. It preserves the declared loss reductions despite short windows and
unequal validity masks. No fitting/refresh command recollects replay or charges
new real interactions.

P2.3 found only **five non-deadline failure transitions** in fitting replay and
sparse latch/console events. Report their support and behavior separately from
dominant ordinary frames and continue outcomes. A finite decreasing aggregate
loss is insufficient evidence that these rare events were learned. Missing
event evidence is a reason to investigate coverage/alignment and the P2.5 gates,
not to relabel absent events as successful predictions.

Before handing off, retain the source commit, frozen initial replay and
normalizers, all four run directories and paired bindings, validation banks,
diagnostic reports and separate simulator accounting. Verify an independent
backup/restore rather than relying on ignored local files or Git alone.
Record the actual server/GPU, elapsed time, memory peaks, completion/failure
status and any pilot deviations. Phase 3 policy learning should wait for the
P2.5 model-quality gate; completing P2.4's optimizer schedule does not open it.
