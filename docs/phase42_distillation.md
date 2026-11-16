# P4.2: one-time local student distillation

Run this stage **after P4.1 finishes all 25 teacher cycles** and publishes
`frozen_teacher.json`. It generates **128 teacher trajectories** of up to 15
steps, then performs **100 student optimizer updates**, each using 32 full
sequences from the fixed buffer. Teacher, world model and student critic stay
frozen. Before/after KL and entropy are training diagnostics; real performance
remains unmeasured. P4.3 local PPO does not start here.

**While your teacher is still running, keep its server source checkout unchanged.**
New `src/world_models` files change P4.1's resume identity. Implement or verify
P4.2 in a separate checkout now, then synchronize after the teacher's completed
handoff is saved. P4.2 accepts additive modules only when every source file
recorded by that teacher remains byte-identical. Do not edit its old receipts.

Keep both the **initial committed checkpoint** (`state_00000001_*`) and the
completed teacher checkpoint. P4.2 verifies their saved weights with P4.1's
original device label, then transfers them to your chosen device. It does not
recreate the initial weights from a seed across different PyTorch builds.

## 1. Verify and preview

On the Linux server, after transferring the complete P4.2 implementation:

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
python -m scripts.phase42 verify --output-dir "artifacts/phase42/check_$(date +%Y%m%d_%H%M%S)"
python -m scripts.phase42 plan --seed 0
```

These commands use CPU fixtures and inspect the fixed budget. They do not
generate research trajectories or train the research student.

## 2. Start distillation from the completed teacher

Open the working system tmux session:

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  /usr/bin/tmux -L wm-p42-system -f /dev/null new-session -A -s wm-distill-0
```

Inside tmux, replace `2` with your available GPU. Use a fresh `CASE` directory:

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
export CUDA_VISIBLE_DEVICES=2
TEACHER=artifacts/phase41_runs/teacher_four_day_v1/relay/seed_0
CASE=artifacts/phase42_runs/distill_four_day_v1/relay/seed_0

python -m scripts.phase42 prepare --teacher-run "$TEACHER" \
  --run-dir "$CASE" --device cuda:0 &&
python -m scripts.phase42 run --run-dir "$CASE" --device cuda:0
```

Preparation validates the teacher and paired initialization without generating
the buffer or optimizing. Only `run` performs those workloads. Detach with
**Ctrl+B, then D**. For seed 1, use its completed teacher and a separate session
and run directory; independent seeds can use separate available GPUs.

## 3. Progress and safe pause

From another terminal:

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
CASE=artifacts/phase42_runs/distill_four_day_v1/relay/seed_0
python -m scripts.phase42 status --run-dir "$CASE"
```

Check `trajectories_completed` out of **128**, then
`distillation_optimizer_steps` out of **100**. Before/after full-buffer
diagnostics take additional time. To pause:

```bash
python -m scripts.phase42 stop --run-dir "$CASE"
```

Wait for `stopped` and for the old worker to exit before starting another.
Each completed trajectory and supervised update is checkpointed. An interrupted
unfinished unit may be repeated; committed progress and optimizer/RNG state are
restored. Keep the complete run directory, buffer, checkpoints and receipts.

## 4. Resume

Inside tmux, activate the same environment and choose your available GPU:

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
export CUDA_VISIBLE_DEVICES=3
python -m scripts.phase42 run \
  --run-dir artifacts/phase42_runs/distill_four_day_v1/relay/seed_0 \
  --device cuda:0
```

Keep the saved sources, precision and compatible runtime/hardware. Do not repeat
`prepare`. Resume continues toward 128 trajectories and 100 total updates.
`--max-units N` optionally pauses after N additional trajectories/optimizer
updates combined. Before/after diagnostics do not count as units, and the
100-update cap remains fixed.

Preserve `warm_started_student.json`, its receipt, the bound checkpoint and
buffer diagnostics for P4.3/P4.4. Local PPO must start with fresh optimizer
state and the original paired critic; the
distillation optimizer and teacher trajectories are not student PPO state/data.
A finished P4.2 run does not establish D's real-environment quality or complete
the full D pipeline.

## If preparation reported an initial identity change

The previous importer compared CPU weight hashes with P4.1's CUDA hashes. The
hash includes the device name, so identical saved values could fail that check.
The corrected importer verifies the original CUDA-labelled identities without
changing the teacher, its manifests or receipts. Student initialization also
records hashes after moving to the selected device.

Transfer the updated `src/world_models/distillation/` directory,
`scripts/phase42.py` and `tests/` to the server, then repeat **Step 2**. This error
occurs before preparation creates the P4.2 run directory. **Do not rerun P4.1
training or delete its checkpoints.** If the initial checkpoint was omitted
from the transfer, copy its complete folder, including `state.pt`, `manifest.json`
and `manifest.sha256`.
