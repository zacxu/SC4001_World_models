# P3.5: historical B development training

This guide retains the earlier `relay_week_v1` development schedule. The active
four-day comparison is defined in the [research plan](../detailed_research_implementation_plan.md).
The seed-0 development run's training budget is now complete; use
[saved-actor export and validation](phase36_acceptance.md) next.

Use `--profile pilot --study-profile relay_week_v1` when preparing a new run.
This records the amended budget without changing the historical configuration
or P2.4 checkpoint files:

| Per run | Budget |
|---|---|
| Real joint transitions | 100,000, including the existing 50,000 |
| Local PPO | 40 initial cycles, then 20 cycles after each of two online blocks |
| Each online block | 25,000 real transitions and 250 model optimizer updates |
| Total local optimization | 80 cycles; 1,280 actor and 1,280 critic updates |

Both selected Relay models start at 4,500 model updates. Completing the two
online refreshes gives 5,000 cumulative model updates; those extra updates use
new replay and do not complete the original initial-fit schedule.

## 1. Continue the existing development check

The recovered `B_week_check_fp32_v1/relay/seed_0` run completed **80/80 cycles**,
100,000 real transitions and 5,000 cumulative model updates. It stopped before
the separate final-export unit. Follow [the export guide](phase36_acceptance.md#2-export-the-saved-final-actor-without-training)
to publish the saved actor without further optimization and run validation.
The earlier 50k evaluation achieved 0/8 successes for both actors; acceptance
remains `needs_review` until new compatible evidence is assessed. Do not prepare
or recover this run again. P3.1-P3.4 are implemented components, not separate jobs.

An optional read-only budget preview in the activated Linux environment is:

```bash
python -m scripts.phase35 plan --task relay --seed 0 --profile pilot \
  --study-profile relay_week_v1 \
  --initial-selection artifacts/phase24_selection/shortened_v1/selection.json
```

For optional CPU implementation checks in local Windows PowerShell, use a fresh
output directory:

```powershell
.\.venv\Scripts\python.exe -m scripts.phase35 verify --output-dir artifacts/phase35_checks/relay_week_v1
```

## 2. Launch the two B study runs after the protocol is ready

**Wait until the bounded check is reviewed, D and the study evaluator are
implemented, and the source/configuration is frozen.** Prepare fresh study runs;
the first development check is separate. Adding Phase 4 modules changes the
source identity and prevents exact resume of an earlier run. Never change an
existing pilot/final run into the amended budget.

From your server's SSH shell:

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
tmux new -s wm-b
```

Inside tmux, activate Python and choose two available physical GPUs:

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
test -x "$WORLD_MODELS_ACC"
export CUDA_VISIBLE_DEVICES=0,1
```

The following starts **two parallel Relay workers**, one per selected GPU. D's
two workers can later use the other two GPUs through its Phase 4 launcher; the
commands here run B only.
Both B workers can reach native collection together. Use this full parallel
launch after measuring support for two simultaneous two-player environments;
the current runner does not serialize collectors across separate jobs.

```bash
SELECTION=artifacts/phase24_selection/shortened_v1/selection.json
RUN=artifacts/phase35_runs/B_relay_week_v1
mkdir -p "$RUN/worker_logs"
PIDS=()
for SEED in 0 1; do
  (
    set -e
    CASE="$RUN/relay/seed_$SEED"
    if [ ! -f "$CASE/run_manifest.json" ]; then
      python -m scripts.phase35 prepare --task relay --seed "$SEED" \
        --profile pilot --study-profile relay_week_v1 \
        --run-dir "$CASE" --device "cuda:$SEED" \
        --initial-selection "$SELECTION"
    fi
    exec python -m scripts.phase35 run --run-dir "$CASE" --device "cuda:$SEED"
  ) >> "$RUN/worker_logs/relay_seed_$SEED.log" 2>&1 &
  PIDS+=("$!")
done
FAILED=0
for PID in "${PIDS[@]}"; do
  wait "$PID" || FAILED=1
done
printf 'Workers finished; failure flag: %s\n' "$FAILED"
```

`cuda:0` and `cuda:1` refer to the first and second GPUs in your mask. Detach
with **Ctrl+B, then D**. In a new SSH shell, change to the project, source
`scripts/gpu_server_env.sh`, then run `tmux attach -t wm-b`.

## 3. Check progress, stop and resume

In another activated server shell:

```bash
cd /home/zixuan/projects/WM1
RUN=artifacts/phase35_runs/B_relay_week_v1
for SEED in 0 1; do
  python -m scripts.phase35 status --run-dir "$RUN/relay/seed_$SEED"
done
```

`local_cycles_completed` counts PPO cycles; `actor_optimizer_steps` and
`critic_optimizer_steps` count optimizer updates. `model_updates` includes the
selected initial count, while `real_joint_transitions` includes the initial
50,000. Per-update diagnostics are in each run's `metrics.jsonl`.

Request a safe stop for both workers:

```bash
for SEED in 0 1; do
  python -m scripts.phase35 stop --run-dir "$RUN/relay/seed_$SEED"
done
```

A stop finishes the current real collection block, model update or PPO cycle and
commits a checkpoint. Once both workers are stopped, rerun the launch block with
the **same source, runtime, GPU mapping and paths**. Existing manifests resume
their recorded budget. Checkpoints also occur every ten PPO cycles, every 250
model updates and at stage boundaries.

A forced kill during real collection leaves a counted partial block and refuses
automatic recollection; preserve its ledger for reconciliation. Interrupted
optimizer work restarts from the last committed state. Keep the checkpoints,
replay, selected P2.4 checkpoint/receipt pairs and run manifests together.

`evaluation_checkpoints.json` lists local actor exports. Exports do not themselves
run evaluation or establish policy acceptance. The historical `pilot` and
`final` profiles remain available when no study amendment is supplied; they are
not the active one-week study.
