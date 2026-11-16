# P4.3: 20 unguided local student PPO cycles

Run after P4.2 has completed **128 trajectories and 100 distillation updates**
and saved `warm_started_student.json`. P4.3 loads that student actor and its
original paired critic, creates **fresh actor and critic PPO optimizers**, then
runs **20 cycles**: 256 starts, horizon 15, four epochs and four minibatches per
cycle. The target is **320 actor and 320 critic optimizer steps**.

The student uses its own local observations and executed action histories.
Teacher queries and teacher KL are zero throughout; the world model stays
frozen at the selected 4,500 updates. No real collection or model refresh runs.

## 1. Transfer and check

Keep completed P4.1/P4.2 checkpoints, handoffs and receipts intact. Synchronize
the P4.3 source files and tests to the server **after P4.2 finishes**. Additions
to production sources change an unfinished P4.1/P4.2 run's resume identity.

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
python -m scripts.phase43 verify --output-dir "artifacts/phase43/check_$(date +%Y%m%d_%H%M%S)"
python -m scripts.phase43 plan --seed 0
```

Verification uses CPU fixtures and small synthetic optimizer tests. It does
not train the research student or measure its GPU speed or real performance.

## 2. Start training in tmux

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  /usr/bin/tmux -L wm-p43-system -f /dev/null new-session -A -s wm-refine-0
```

Inside tmux, select an available GPU and use a **fresh** `CASE` directory:

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
export CUDA_VISIBLE_DEVICES=2
DISTILL=artifacts/phase42_runs/distill_four_day_v1/relay/seed_0
CASE=artifacts/phase43_runs/refine_four_day_v1/relay/seed_0

python -m scripts.phase43 prepare --distillation-run "$DISTILL" \
  --run-dir "$CASE" --device cuda:0 &&
python -m scripts.phase43 run --run-dir "$CASE" --device cuda:0
```

`prepare` validates the completed handoff and initializes without imagination or
optimization. Only `run` trains. Detach with **Ctrl+B, then D**.
Preparation and resume audit upstream checkpoints and all buffer shards before
the first cycle, so allow time for that initial read.

For seed 1, use its own completed P4.2 run, a separate `seed_1` run directory
and tmux session. Never reuse seed 0's warm-start as seed 1. Independent seeds
may use separate GPUs; one refinement run uses one GPU.

## 3. Progress and safe pause

From another terminal:

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
CASE=artifacts/phase43_runs/refine_four_day_v1/relay/seed_0
python -m scripts.phase43 status --run-dir "$CASE"
```

Check `local_cycles_completed / target_local_cycles` (**20**) and
`actor_optimizer_steps / target_actor_optimizer_steps` and the corresponding
critic counters (**320** each). A completed cycle adds 16 updates to each.
`status.json` is a progress snapshot; committed checkpoints govern recovery.

```bash
python -m scripts.phase43 stop --run-dir "$CASE"
```

Wait for `stopped` and for the old worker to exit. A checkpoint is saved after
each cycle. If forcibly interrupted, an unfinished cycle may repeat; committed
cycles, optimizer and RNG state are restored. Keep the complete run directory.

## 4. Resume on an available GPU

Inside tmux, activate the same environment and choose the GPU:

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
export CUDA_VISIBLE_DEVICES=3
python -m scripts.phase43 run \
  --run-dir artifacts/phase43_runs/refine_four_day_v1/relay/seed_0 \
  --device cuda:0
```

Do not repeat `prepare` or P4.2. Resume restores the **P4.3 PPO optimizers**;
fresh optimizer initialization occurs only when first preparing this stage.
Keep sources and compatible runtime/hardware unchanged while resuming.
`--max-cycles N` pauses after N additional cycles within the fixed total of 20.

Completion publishes checkpoint-bound `refined_student.json` and an actor-only
`actor_export/` for the later evaluation pipeline. Keep both and their receipts.
P4.3 does not run native evaluation, establish transfer benefit, or complete
P4.4/P4.5, the matching reduced B controller, or the P5 comparison.
