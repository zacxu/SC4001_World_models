# P2.4: Linux training

Run these Bash commands on `/home/zixuan/projects/MAPPO_wm`.
Training uses **four parallel workers on one GPU**, with **5,000 optimizer
updates per worker** for the full run.

## 1. Start tmux and choose the GPU

```bash
cd /home/zixuan/projects/MAPPO_wm
tmux new-session -A -s wm-p24
```

If this session already has training running, skip to step 4.

Inside tmux:

```bash
cd /home/zixuan/projects/MAPPO_wm
source .venv/bin/activate
set +e
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0  # Change this to your GPU number.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Use your existing Conda environment instead if applicable. If a scheduler
already assigns GPU variables, keep its assignment. The selected GPU appears
as `cuda:0`; all four workers share it.

## 2. Verify once before training

```bash
P24_CHECK="artifacts/server_p24_check_$(date +%Y%m%d_%H%M%S)"
python -m scripts.phase24 verify --output-dir "$P24_CHECK"
```

Continue only after `complete_p24_implementation`. On failure, read
`tail -n 100 "$P24_CHECK/tests.txt"`. Transfer the complete `tests/` folder if
verification reports missing test modules.

## 3. Run the short check, then full training

Short check: **50 updates per worker**, in a fresh output directory.

```bash
python -m scripts.phase24 launch --stage overfit \
  --output-dir artifacts/phase24_runs/overfit_v1 \
  --effective-batch 4 --microbatch 1 --burn-in 4 --loss-steps 4 \
  --updates 50 --checkpoint-every 10 --validate-every 10 \
  --validation-windows 4 --log-every 1
```

After all four checks finish successfully and fitting loss decreases, start
full training in the same tmux session:

```bash
python -m scripts.phase24 launch --output-dir artifacts/phase24_runs/pilot_v1
```

Detach with **Ctrl+B, then D**. Training continues after you disconnect from SSH.
To reconnect:

```bash
tmux attach-session -t wm-p24
```

## 4. Check optimizer updates

From another server terminal, print only the latest reported optimizer update
count for each worker (`--` means no report yet):

```bash
cd /home/zixuan/projects/MAPPO_wm
python3 - <<'PY'
import json
from pathlib import Path
root = Path("artifacts/phase24_runs/pilot_v1")
for task in ("relay", "cover"):
    for seed in (0, 1):
        path = root / task / f"seed_{seed}" / "status.json"
        updates = json.loads(path.read_text())["model_updates"] if path.is_file() else "--"
        print(f"{task}/seed_{seed}: {updates}")
PY
```

For the short check, change `pilot_v1` to `overfit_v1`. Rerun this command for
updated counts.

## 5. Stop and resume

Checkpoints save at **update 0, every 250 updates, and completion**; the short
check saves every **10 updates**. Each case stores `.pt` and matching `.sha256`
files under its `checkpoints/` directory.

To move a run into tmux or stop it temporarily:

1. Press **Ctrl+C** in the original training terminal and wait for all workers to stop.
2. Enter tmux using step 1, keeping the **same environment, GPU, code, and options**.
3. Repeat the original launch command with `--resume`. For the default full run:

```bash
python -m scripts.phase24 launch --output-dir artifacts/phase24_runs/pilot_v1 --resume
```

For the short check, repeat its full command with `--resume` added.
**Ctrl+C does not force a save**: updates after the last saved checkpoint repeat.
Resume requires a saved checkpoint for each started case; verification alone
creates no training checkpoints.
