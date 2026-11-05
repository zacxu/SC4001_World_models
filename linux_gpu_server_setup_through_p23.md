# ASPIRE2A setup through P2.3

Use these **Bash commands on ASPIRE2A**, then continue your existing P2.4
checkpoints with [the training guide](docs/phase24_training.md).
The supplied [quickstart PDF](ASPIRE2A-General-Quickstart-Guide.pdf), especially
pages 2, 3, 7, 8 and 11, describes RHEL 8, PBS and A100 40 GB GPUs.
Current [NSCC specifications](https://nsccsg.github.io/systems/aspire2a/system-specifications/)
and [job guidance](https://nsccsg.github.io/quickstart/submit-a-batch-job/)
were also checked for this update.

**Login nodes:** transfer files, edit scripts and submit jobs.
**Allocated compute nodes:** install/build software, run checks and train.
Production training uses PBS batch jobs, which continue after SSH disconnects.

## 1. Create WM1 in your scratch directory

You are already connected. Run on the login node:

```bash
export WM_ROOT=/home/users/create/smrksun/scratch/WM1
mkdir -p "$WM_ROOT"
cd "$WM_ROOT"
myprojects
myquota
```

Use this same `WM_ROOT` for every job and after reconnecting. Put the source,
environment, replay and runs directly inside `WM1`, without an extra
`MAPPO_wm` subfolder. Your PBS project ID is **`11704278`**; the submission
commands below use this allocation.

Scratch is temporary and subject to purge. Keep backups of replay, frozen
statistics and checkpoint/receipt pairs on your PC or other durable storage.
See [NSCC storage details](https://nsccsg.github.io/systems/aspire2a/system-specifications/#storage).

## 2. Upload your files with WinSCP

**Do this after Step 1 creates `WM1`, before requesting a compute session in
Step 3.** You can upload now; environment installation comes after the transfer.

On the **old server**, press **Ctrl+C** in the training terminal and wait for
the launcher and all workers to exit. This preserves the last committed
checkpoints; it does not force a new save.

If the old server's latest checkpoints are not on your PC yet, download the
stopped run first. Your local code checkout alone may not contain them.

In WinSCP:

1. Open your ASPIRE2A SFTP connection using your approved access route.
2. In the **right-hand remote pane**, open this destination:

   ```text
   /home/users/create/smrksun/scratch/WM1/
   ```

3. In the **left-hand local pane**, open your project snapshot. Upload its
   **contents directly into `WM1`**, using **binary transfer mode**.

The resulting layout should look like this (using `pilot_v1` as the example
old run):

```text
WM1/
  pyproject.toml
  src/
  scripts/
  tests/
  configs/
  docs/
    research_contract.md
    interface_contract.md
    phase0_status.md
    runbook.md
  data/initial_replay/pilot/
  artifacts/phase23/normalization/
  artifacts/phase24_runs/pilot_v1/
```

Transfer all the required contents below, retaining their relative paths:

| Transfer | Required contents |
|---|---|
| Project source | Complete `src/`, `scripts/`, `tests/`, `configs/`, `scenarios/`, requirements, `pyproject.toml` and `.gitattributes` |
| Documentation | Entire `docs/` folder, including `research_contract.md`, `interface_contract.md`, `phase0_status.md` and `runbook.md`; Phase 0 requires these files |
| Initial replay | Entire `data/initial_replay/pilot/`: four task/seed folders, including replay, bindings and provenance |
| Frozen P2.3 statistics | Entire `artifacts/phase23/normalization/`, including generation receipts |
| Existing P2.4 run | Entire actual run folder, such as `artifacts/phase24_runs/pilot_v1/`, including `launch_plan.json`, every case manifest, checkpoints and SHA receipts |

Recreate the environment on ASPIRE2A; exclude old `.venv/`, `.envs/`, package
caches and platform-specific tool binaries. Keep the old server's source bytes
and configurations exactly as used by its checkpoints. Retain `.gitattributes`;
avoid line-ending conversion and a checkout of a different fitting revision.

Also transfer the new `scripts/phase24_import.py` and its dependency
`scripts/phase24_relocate.py` from this checkout, plus
`scripts/aspire2a_cuda.sh` for CUDA compatibility on ASPIRE2A. These work
alongside the existing fitting code. Transfer the corresponding tests if
running the full suite.

Keep the old run folder as the import source. The new ASPIRE2A continuation
will use a **different output folder**. Finish synchronization before importing;
do not synchronize old results over that new folder later.

## 3. Request a compute session for setup

**Continue only after the WinSCP upload in Step 2 has finished.** From
`WM_ROOT` on the login node:

```bash
export PROJECT_ID=11704278
qsub -I -P "$PROJECT_ID" -q normal \
  -l select=1:ngpus=1 -l walltime=02:00:00
```

Wait for the allocated compute shell, then set the working path in that shell:

```bash
export WM_ROOT=/home/users/create/smrksun/scratch/WM1
cd "$WM_ROOT"
module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS="$WM_ROOT/.cache/conda"
export PIP_CACHE_DIR="$WM_ROOT/.cache/pip"
nvidia-smi
```

If the previous command reported `qsub: cannot send environment with the job`,
retry the submission above without `-v WM_ROOT`. That option requires `WM_ROOT`
to be exported in the submitting terminal; a new SSH session may not have it.
Setting the path inside the compute shell avoids that dependency.

PBS chooses the GPU and assigns CPU/RAM automatically. Keep its
`CUDA_VISIBLE_DEVICES` assignment. See [interactive jobs](https://nsccsg.github.io/quickstart/launch-an-interactive-job/).

Your allocated A100 node reported driver **570.124.06**, which exposes CUDA
driver API **12.8**. The project's PyTorch **2.15.0+cu132** uses CUDA **13.2**,
so it needs the compatibility-library step below on this node.
NVIDIA lists `cuda-compat-13-2` as compatible with A100-class data-center GPUs
on supported R570 drivers. The helper installs these libraries under `WM1`
and activates them for your processes; it does not change the host driver.
[NVIDIA forward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html).

Keep PyTorch 2.15 when continuing these checkpoints. Its published CUDA builds
use CUDA 13.x, and the checkpoint importer checks the PyTorch release.
[PyTorch packaging notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432).

## 4. Recreate the environment

Run once in the allocated compute shell:

```bash
conda create -y -p "$WM_ROOT/.envs/wm312" -c conda-forge \
  python=3.12 pip ffmpeg openal-soft sdl2 libgl
conda activate "$WM_ROOT/.envs/wm312"
python -m pip install --upgrade pip
python -m pip install -r requirements-linux-gpu.txt
python -m pip install --no-build-isolation -e .
python -m pip check
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

For an already-created environment, activate it and skip creation/install.
The import step checks Python major/minor, NumPy and the PyTorch release against
the old checkpoint. A Python patch, CUDA build or GPU change is recorded as a
runtime migration; numerical equivalence across H200 and A100 is unmeasured.

If package downloads are blocked, use NSCC's supported network route or transfer
Linux CPython 3.12/x86-64 wheels and ACC from a connected Linux machine.
Retain the versions in the requirements. Do not copy a Windows environment.

**If the environment is already installed, start here.** Upload
`scripts/aspire2a_cuda.sh` from this checkout to `WM1/scripts/` using WinSCP.
In the allocated compute shell, with `wm312` active, run:

```bash
source scripts/aspire2a_cuda.sh --install
```

This downloads NVIDIA's approximately 92 MiB compatibility archive, checks its
pinned SHA256, and activates its libraries. The archive and checksum come from
[NVIDIA's CUDA 13.2.2 manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.2.2.json).
In later compute sessions, activate the environment and run
`source scripts/aspire2a_cuda.sh` before starting Python; no download is needed.
The P2.4/P2.5 guides include this activation.

Confirm CUDA and bfloat16 on the allocated GPU:

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA allocation/driver check failed"
assert torch.cuda.is_bf16_supported(), "P2.4 requires bfloat16 support"
x = torch.ones((16, 16), device="cuda:0", dtype=torch.bfloat16)
assert (x @ x).float().sum().item() == 4096
print(torch.cuda.get_device_name(0))
PY
```

Continue only after this succeeds. `nvidia-smi` can still report the host's
CUDA 12.8 support. If the fresh Python check reports error 803/804 or unsupported
driver features, retain the output and ask NSCC about the compatibility setup.
The test and checkpoint import still need to pass on the allocated hardware.

## 5. Install Linux ACC for scenario checks

On these x86-64 nodes, install ACC **1.60**, including its include files:

```bash
mkdir -p .tools/acc-linux
curl --fail --location \
  https://github.com/ZDoom/acc/releases/download/1.60/acc-1.60-linux64.zip \
  -o .tools/acc-1.60-linux64.zip
python -m zipfile -e .tools/acc-1.60-linux64.zip .tools/acc-linux
chmod +x .tools/acc-linux/acc
export WORLD_MODELS_ACC="$WM_ROOT/.tools/acc-linux/acc"
export IMAGEIO_FFMPEG_EXE="$(command -v ffmpeg)"
```

If a simulator shared library is still unavailable, retain the error and ask
NSCC for the supported library/module. Scenario compilation is needed for the
full simulator checks and later collection; P2.4 fitting itself reads replay.

## 6. Verify the transferred project through P2.3

Run these checks in the compute session with the environment active. Proceed
only after each command succeeds:

```bash
CHECK_DIR="artifacts/aspire2a_check_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$CHECK_DIR"
module list > "$CHECK_DIR/modules.txt" 2>&1
nvidia-smi > "$CHECK_DIR/nvidia-smi.txt"
python -m pip freeze > "$CHECK_DIR/pip-freeze.txt"
python -m scripts.phase0 check --output "$CHECK_DIR/phase0_acceptance.json"
```

Expect `"status": "passed"`. If Phase 0 reports `failed`, display its actual
failed package pins, missing documents and unit-test output before continuing:

```bash
python - "$CHECK_DIR/phase0_acceptance.json" <<'PY'
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
print("Package mismatches:", {name: item for name, item in report["bootstrap_lock"].items() if not item["matches"]})
print("Missing documents:", [name for name, exists in report["documents"].items() if not exists])
tests = report["tests"]
print("Test exit code:", tests["exit_code"])
if tests["exit_code"]:
    print(tests["stdout"])
    print(tests["stderr"])
PY
```

After Phase 0 passes, run the full suite:

```bash
python -m unittest discover -s tests -q
```

Verify all 200,000 joint transitions, paired B/D bindings and frozen statistics:

```bash
python - <<'PY'
from pathlib import Path
from world_models.data.replay import ReplayDataset
from world_models.training.normalization import NormalizationStats
for family in ("relay", "cover"):
    for seed in (0, 1):
        case = Path("data/initial_replay/pilot") / family / f"seed_{seed}"
        data = ReplayDataset(case / "replay", verify=True)
        assert data.manifest["transition_count"] == 50000
        for variant in ("B", "D"):
            bound = ReplayDataset.from_binding(case / "bindings" / f"{variant}.json", variant=variant, verify=False)
            assert bound.path == data.path and bound.manifest_sha256 == data.manifest_sha256
        stats = NormalizationStats.load(Path("artifacts/phase23/normalization") / family / f"seed_{seed}.json")
        stats.validate_for_dataset(data)
        print("Verified:", family, seed)
PY

python -m scripts.phase22 verify --output-dir "$CHECK_DIR/phase22"
python -m scripts.phase23 verify \
  --output-dir "$CHECK_DIR/phase23" \
  --statistics-dir artifacts/phase23/normalization
```

Expect **`complete_p22`** and **`complete_p23`**. Inspect simulator-test skips;
retain failures in this new check directory. Preserve the transferred
normalizers and receipts rather than regenerating them to bypass a mismatch.

When finished, release the interactive allocation:

```bash
exit
```

Back on the login node, continue with [P2.4 checkpoint import and PBS training](docs/phase24_training.md).
You can disconnect after batch submission. Keep the original server snapshot
and back up new ASPIRE2A checkpoints outside scratch.
