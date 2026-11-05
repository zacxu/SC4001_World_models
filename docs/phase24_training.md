# P2.4: continue old checkpoints on ASPIRE2A

Complete [ASPIRE2A setup through P2.3](../linux_gpu_server_setup_through_p23.md)
and finish synchronizing the **stopped old run** first. These Bash commands
continue its four models in parallel, **one worker per allocated A100 GPU**.
Saved optimizer updates count toward the original total; training does not
restart from zero. Keep the source snapshot until the continuation is complete.

ASPIRE2A uses [PBS batch jobs](https://nsccsg.github.io/quickstart/submit-a-batch-job/).
Once submitted, training continues after SSH disconnects; tmux is optional for
the login terminal. Run all training and GPU checks inside the allocation.

## 1. Create the batch script on the login node

Use project **`11704278`**, then enter the transferred project:

```bash
export PROJECT_ID=11704278
export WM_ROOT=/home/users/create/smrksun/scratch/WM1
cd "$WM_ROOT"
```

Repeat this block after reconnecting. `PROJECT_ID` is your PBS allocation ID,
not the directory name or username.

Create the script below. Set `SOURCE` to the run folder you actually transferred
(`pilot_v1` or `pilot_4gpu_v1`, for example). `RUN` must be a fresh destination
for the first import. Preserve both paths for subsequent submissions.

```bash
cat > p24_aspire2a.pbs <<'PBS'
#!/bin/bash -l
#PBS -N wm-p24
#PBS -q normal
#PBS -l select=1:ngpus=4
#PBS -l walltime=24:00:00
#PBS -j oe

set -e
export WM_ROOT=/home/users/create/smrksun/scratch/WM1
cd "$WM_ROOT"
mkdir -p artifacts/pbs
exec > "artifacts/pbs/p24_${PBS_JOBID}.log" 2>&1
module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WM_ROOT/.envs/wm312"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source scripts/aspire2a_cuda.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1

SOURCE=artifacts/phase24_runs/pilot_v1
RUN=artifacts/phase24_runs/aspire2a_v1

# PBS assigns physical GPUs. Leave its CUDA_VISIBLE_DEVICES unchanged.
python - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 4, "Expected four allocated GPUs"
for index in range(4):
    with torch.cuda.device(index):
        assert torch.cuda.is_bf16_supported(), "bfloat16 GPU support required"
        x = torch.ones((16, 16), device=f"cuda:{index}", dtype=torch.bfloat16)
        assert (x @ x).float().sum().item() == 4096
PY

if [[ "${P24_RESUME:-0}" == 0 ]]; then
    python -m scripts.phase24 verify \
      --output-dir "artifacts/aspire2a_p24_check_${PBS_JOBID}"
    python -m scripts.phase24_import prepare \
      --source-dir "$SOURCE" --output-dir "$RUN" --gpus 0,1,2,3
fi

exec python -m scripts.phase24_import resume --output-dir "$RUN"
PBS
```

This requests **one node with four GPUs**, with CPU/RAM assigned by PBS.
The [supplied guide](../ASPIRE2A-General-Quickstart-Guide.pdf), pages 3 and 8,
identifies these as A100 40 GB GPUs. Use `normal`; switch to `ai` only if your
project has access. Leave `CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER` as
provided by the cluster. Logical indices `0,1,2,3` map to the assigned GPUs;
each worker sees its own GPU as `cuda:0`.

Install the CUDA compatibility libraries once in setup Step 4 before submitting
this job. Its activation line above is required for the reported R570 driver;
recreate the PBS script if you generated it before this update.

The import preserves weights, Adam state, sampler, random streams and each
model's latest committed update. It checks source/configuration/data identities
and compatible software, and records path/GPU/runtime changes in
`RUN/import.json` with a SHA receipt. H200-to-A100 numerical equivalence is
unmeasured. Original batch settings and the original **5,000-update target**
remain in force; A100 memory fit and speed must be checked on the cluster.

## 2. Submit and disconnect

From the same project directory on the login node:

```bash
P24_JOB=$(qsub -P "$PROJECT_ID" p24_aspire2a.pbs)
printf '%s\n' "$P24_JOB" | tee p24_job.id
qstat "$P24_JOB"
```

`Q` means queued; `R` means running. You can now disconnect.
The 24-hour walltime is an allocation limit, not an estimated training duration.
Controller output is in `artifacts/pbs/p24_JOBID.log`; worker errors/logs are
under `artifacts/phase24_runs/aspire2a_v1/worker_logs/`.
If verification/import fails, read the controller log before resubmitting.

Check the training job with:
```bash
qstat 26122087.pbs101
```

## 3. Check optimizer updates

From a login terminal, print each worker's **completed optimizer updates / total
target**:

```bash
cd /home/users/create/smrksun/scratch/WM1
python3 - <<'PY'
import json
from pathlib import Path
root = Path("artifacts/phase24_runs/aspire2a_v1")
for task in ("relay", "cover"):
    for seed in (0, 1):
        path = root / task / f"seed_{seed}" / "status.json"
        if not path.is_file():
            print(f"{task}/seed_{seed}: status file not created yet")
            continue
        status = json.loads(path.read_text())
        progress = f"{status['model_updates']}/{status['target_stage_updates']}"
        print(f"{task}/seed_{seed}: {progress} optimizer updates")
PY
```

`status file not created yet` (`--/--` in the older command) means there is no
progress file at that path. Check `qstat -u "$USER"`: when `wm-p24` is `Q`, it
has not started. Once `R`, it verifies and imports the checkpoints before
training; successful import creates the status files with the saved counts.
If they remain missing, inspect the controller log from Step 2. Keep the
existing queued job; do not submit a duplicate.

`model_updates` includes the imported checkpoint's saved updates. For example,
1,500 saved updates plus 100 new updates displays `1600/5000 optimizer updates`.
The target is read from the run's status rather than hardcoded.

Change the root if you chose another `RUN`. `import.json` records the imported
count for each case in `cases[].model_updates`; compare it with the current
count if progress looks too low. Updates after the old server's last saved
checkpoint repeat and are not added twice.

## 4. Stop and continue in another PBS job

To stop the submitted job, on the login node:

```bash
qdel "$(cat p24_job.id)"
qstat -u "$USER"
```

Wait until the old job has exited before submitting another. Checkpoints save
at update 0, every **250 updates**, and completion; both `.pt` and `.sha256`
files are required. A stop or walltime expiry does not force a save, so work
since the last committed checkpoint repeats.

To continue the **ASPIRE2A run**, submit the same script with import skipped:

```bash
P24_JOB=$(qsub -P "$PROJECT_ID" -v P24_RESUME=1 p24_aspire2a.pbs)
printf '%s\n' "$P24_JOB" | tee p24_job.id
```

Keep the code, environment, paths, batch settings, update target and per-worker
GPU-mask mapping fixed. Another physical A100 node is allowed when these match.
If PBS supplies different mask tokens, resume will reject the allocation:
import the latest ASPIRE2A run into a **new destination** in that allocation,
using it as `SOURCE`, rather than editing manifests or SHA receipts.
Keep the working run under `WM1`. Scratch is temporary: copy checkpoint/receipt
pairs and import reports to your PC or other durable storage regularly.

After fitting, use the [compact P2.5 engineering screen](phase25_validation.md)
before expanding Phase 3 experiments.
