# GitHub + pilot-data transfer to an Ubuntu GPU server

Use **GitHub for the runnable project and small research artifacts**. Transfer
**only `data/initial_replay/pilot/` separately**. Recreate the Linux environment
and tools, then verify through **P2.3**. P2.4 training remains a separate step.

| Through GitHub | Separate data transfer | Reinstall on Ubuntu |
|---|---|---|
| Source, tests, configs, scenario sources, plans/docs, requirements, selected reports, frozen normalization files | Entire `data/initial_replay/pilot/` folder: about **7.91 GiB** | Python environment, CUDA-enabled PyTorch, Linux ACC |

The repository contents are about **16 MiB**, with no file near GitHub's 100 MiB
limit. The updated `.gitignore` excludes data, venvs, tools, transfer archives,
and generated outputs. It keeps `src/world_models/data/` and the frozen P2.3
normalizers included. The original Windows folder can remain unchanged.

This recreates the runnable project and initial datasets. Ignored historical
videos, screenshots, and long diagnostic traces remain in your Windows archive;
they are not needed for these setup checks.

## 1. Upload the latest project files to GitHub

Create an **empty GitHub repository**. In Windows PowerShell:

```powershell
Set-Location -LiteralPath 'D:\world models'
if (-not (Test-Path -LiteralPath '.git')) { git init -b main }
git add --all
git status --short
```

Check that source/configuration files, this guide, `requirements-linux-gpu.txt`,
`.gitignore`, `.gitattributes`, and `artifacts/phase23/normalization/` are included.
`data/`, `.venv/`, `.tools/`, and transfer ZIPs should not appear as added files.
Do not force-add ignored data.

Then commit and inspect the repository address:

```powershell
git commit -m "Project ready through P2.3"
git remote -v
```

If there is **no `origin` remote**, add yours, replacing the placeholder URL:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

If an origin already exists, confirm it points to the repository you want to
use. Once the address is correct, upload:

```powershell
git push -u origin HEAD
```

Keep `.gitattributes` in the same commit. It preserves the exact bytes of
published evidence and the four source files referenced by frozen statistics,
so their SHA receipts remain valid after cloning on Linux. Do not run line-ending
conversion tools on those files or rewrite old receipts.

## 2. Put only the pilot data on the external disk

Create **`pilot.zip`** from this folder itself:

```text
D:\world models\data\initial_replay\pilot
```

Keep every file within all four complete case folders, including replay,
bindings, ledgers, collection reports, runtime archives, and recovery provenance:

```text
pilot/
  relay/seed_0/
  relay/seed_1/
  cover/seed_0/
  cover/seed_1/
```

Each case contains 50,000 joint transitions, totaling **200,000**. Copy this ZIP
to the external disk. Use **NTFS or exFAT**, because the ZIP will exceed 4 GB.
There is no need to copy the Windows `.venv`, compiler, or whole project folder.

## 3. Clone the project and restore the data on Ubuntu

The server needs Internet access for GitHub and package downloads, a compatible
NVIDIA driver, and enough local storage for the data and new environment.
If you lack sudo access, ask the administrator to provide these system packages:

```bash
sudo apt-get update
sudo apt-get install -y git unzip curl libopenal1 libsdl2-2.0-0 libgl1 ffmpeg
mkdir -p "$HOME/projects"
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git "$HOME/projects/world-models"
cd "$HOME/projects/world-models"
```

Connect the external disk. Replace the ZIP path below with its actual location:

```bash
mkdir -p data/initial_replay
test ! -e data/initial_replay/pilot
unzip "/path/to/pilot.zip" -d data/initial_replay
```

The resulting paths must be `data/initial_replay/pilot/relay/seed_0/replay/`,
and the equivalent seed/task paths above. The guard stops accidental merging
with an existing dataset. Keep the restored data and frozen normalizers unchanged.

## 4. Create the Linux environment and install dependencies

Use **Python 3.12**. On Ubuntu 24.04, install it if needed:

```bash
sudo apt-get install -y python3.12 python3.12-venv
```

On older Ubuntu, ask for Python 3.12 or use an existing Conda installation:

```bash
conda create -n wm312 python=3.12 -y
conda activate wm312
```

Once Python 3.12 is available, run from the cloned project folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-linux-gpu.txt
python -m pip install --no-build-isolation -e .
python -m pip check
```

The GPU recipe keeps the project package pins and selects **torch 2.15.0+cu132**.
Do not install `requirements-phase2-cpu.lock` on this server: it selects CPU-only
torch. CUDA runtime dependencies come with the wheel installation.

Check the driver and GPU access. Driver **595.71.05 or newer** is the conservative
baseline for this CUDA 13.2 recipe; ask the administrator about compatibility if
the server's driver is older. On a cluster, use your assigned GPU session.
[CUDA driver reference](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-toolkit-release-notes/index.html)

```bash
nvidia-smi
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
assert torch.cuda.is_available(), "Check the GPU allocation, driver, and torch installation."
for index in range(torch.cuda.device_count()):
    assert torch.ones(8, device=f"cuda:{index}").sum().item() == 8
    print(index, torch.cuda.get_device_name(index))
PY
```

## 5. Install the Linux scenario compiler

Download Linux ACC **1.60** and extract the whole archive, including its `.acs`
include files. Existing scenario sources are already in the repository.

```bash
case "$(uname -m)" in
  x86_64) acc_archive='acc-1.60-linux64.zip' ;;
  aarch64) acc_archive='acc-1.60-linuxarm64.zip' ;;
  *) echo 'Unsupported CPU architecture'; exit 1 ;;
esac
mkdir -p .tools/acc-linux
curl --fail --location "https://github.com/ZDoom/acc/releases/download/1.60/$acc_archive" -o ".tools/$acc_archive"
unzip -o -q ".tools/$acc_archive" -d .tools/acc-linux
chmod +x .tools/acc-linux/acc
export WORLD_MODELS_ACC="$PWD/.tools/acc-linux/acc"
export IMAGEIO_FFMPEG_EXE="$(command -v ffmpeg)"
```

ViZDoom's Linux package supplies the engine and bundled base assets. Generated
maps are rebuilt from the scenario sources; the Windows tool binaries are not
needed on this server.

## 6. Verify the restored project through P2.3

Run from the project folder with `.venv` active:

```bash
set -e
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
CHECK_DIR="artifacts/server_check_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$CHECK_DIR"

python -m scripts.phase0 check --output "$CHECK_DIR/phase0_acceptance.json"
python -m unittest discover -s tests -q
```

Check the full replay data, B/D bindings, and frozen normalization metadata:

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

The tests should pass; inspect any skipped scenario-compilation tests. The last
two commands should report **`complete_p22`** and **`complete_p23`**. They run CPU
implementation checks, including gradients, without optimizer updates. The
earlier GPU check separately confirms CUDA access.

Stop on errors and retain their output. Existing Windows report paths are
historical; do not replace them with Linux paths or edit receipts to bypass a
failure. New reports go into `CHECK_DIR`, leaving the original reports intact.

For a new terminal, return to the cloned folder and run:

```bash
source .venv/bin/activate
export WORLD_MODELS_ACC="$PWD/.tools/acc-linux/acc"
export IMAGEIO_FFMPEG_EXE="$(command -v ffmpeg)"
```

Keep the original Windows research archive. The Linux reconstruction uses the
GitHub checkout plus the transferred pilot folder. P2.4 fitting is now
implemented; after these setup checks continue with the
[GPU training runbook](docs/phase24_training.md). Its default is four parallel
workers sharing one GPU. Research fitting and P2.5 quality acceptance remain
separate user-run server work.
