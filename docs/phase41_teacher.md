# P4.1: globally informed teacher

P4.1 implements the teacher network, full-prefix imagination, recurrent PPO and
a resumable **25-cycle teacher stage** for Relay seeds 0 and 1. Each teacher uses
its selected **4,500-update world model and 50k initial replay**, frozen throughout.
It performs **400 teacher actor and 400 teacher critic optimizer updates**.
It collects no real training data and does not train a student or refresh the model.
Teacher quality and GPU runtime still require your training run.

Shared world tokens enter only the teacher's current action head through
four-head, width-256 cross-attention. Its CNN/GRU memory uses only its own RGB
and executed previous action. The local student interface is unchanged.

**Before synchronizing code:** finish the existing B development run and its
P3.6 evaluation, or keep that server checkout separate. New source files change
its resume identity. P4.2 now has a separate [distillation stage](phase42_distillation.md);
P4.3-P4.5, the reduced B controller and the final test pipeline still need
implementation. Freeze those sources before preparing the final B/D
cohort. A standalone teacher trained earlier is development evidence; later
integration must explicitly validate its provenance before any reuse.

## 1. Check implementation and preview the budget

On the Linux GPU server, in the complete transferred project:

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
python -m scripts.phase41 verify --output-dir "artifacts/phase41/check_$(date +%Y%m%d_%H%M%S)"
python -m scripts.phase41 plan --seed 0
```

Verification uses CPU synthetic fixtures and tiny optimizer tests. Neither
command starts research training. The fixed teacher registration is in
`src/world_models/teacher/protocol.py`; `relay_week_v1` remains the older schedule.

## 2. Start a fresh teacher when ready

Open the working system tmux session:

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  /usr/bin/tmux -L wm-p41-system -f /dev/null new-session -A -s wm-teacher-0
```

Inside tmux, choose an available GPU by replacing `2` below. It becomes `cuda:0`
inside the process. Use a fresh directory for preparation:

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
export CUDA_VISIBLE_DEVICES=2
CASE=artifacts/phase41_runs/teacher_four_day_v1/relay/seed_0

python -m scripts.phase41 prepare --seed 0 --run-dir "$CASE" \
  --initial-selection artifacts/phase24_selection/shortened_v1/selection.json \
  --device cuda:0 &&
python -m scripts.phase41 run --run-dir "$CASE" --device cuda:0
```

`prepare` validates the frozen selection and initializes the teacher without
optimization. Only `run` trains. It stops at **25 total cycles**, not 25 additional
cycles on every resume. Detach with **Ctrl+B, then D**. For seed 1, use a separate
session, `--seed 1` and a directory ending in `seed_1`. Independent seeds may run
on separate available GPUs; two workers must never share a run directory.

## 3. Progress and safe pause

From another terminal:

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
CASE=artifacts/phase41_runs/teacher_four_day_v1/relay/seed_0
python -m scripts.phase41 status --run-dir "$CASE"
```

Look for teacher cycles out of **25**, and teacher actor/critic optimizer steps
out of **400 each**. The model counter stays at 4,500. A checkpoint is committed
after every complete cycle. To pause:

```bash
python -m scripts.phase41 stop --run-dir "$CASE"
```

Wait for `stopped` and the old worker to exit. A stop request finishes the
current PPO cycle before saving. A forced kill can lose that unfinished cycle;
resume uses the latest receipt-verified checkpoint, not an ahead-of-save status.

## 4. Resume, including on another matching GPU

Inside tmux, activate the same environment, choose the available GPU, and run:

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
export CUDA_VISIBLE_DEVICES=3
python -m scripts.phase41 run \
  --run-dir artifacts/phase41_runs/teacher_four_day_v1/relay/seed_0 \
  --device cuda:0
```

Keep the saved source, precision, logical device and compatible runtime/hardware
identity. Do not repeat `prepare`. `--max-cycles N`, when supplied, pauses after
N additional complete cycles within the existing 25-cycle cap.

Completion means the teacher stage is saved, not that it learned successful
cooperation. `metrics.jsonl` records imagined reward, entropy and gradient
diagnostics; `attempts.jsonl` records completed or interrupted cycle attempts.
Preserve these with `run_manifest.json`, receipts, checkpoints,
`frozen_teacher.json` and the frozen model selection for later distillation.
Do not deploy the teacher as a local actor or describe this as a completed D run.

After P4.1 finishes and `frozen_teacher.json` plus its receipt are saved, follow
the [P4.2 guide](phase42_distillation.md). Its importer checks the completed
teacher's original source bytes and inputs while permitting the newly added
distillation package. Do not synchronize these new source files into a running
P4.1 checkout: its ordinary training resume still requires its original sources.
