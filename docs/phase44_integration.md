# P4.4: integrated fixed-data D schedule

The controller connects **25 teacher cycles → 128 buffer trajectories → 100
distillation updates → 20 unguided local PPO cycles → final actor export**.
Each seed uses its selected frozen 4,500-update model and existing 50,000 real
transitions. There is no new real collection, model refresh or local teacher KL.

**Let your current P4.3 run finish. It does not need to restart.** P4.4 adds a
separate `world_models_campaign` package and `scripts/phase44.py`; existing
P4.1–P4.3 production files stay unchanged. Keep the running checkout's existing
files intact. Preserve all completed stage directories, checkpoints and receipts.

## 1. Check the implementation

Transfer `src/world_models_campaign/`, `scripts/phase44.py`,
`configs/studies/relay_four_day_v1.yaml`, `tests/test_campaign_*.py` and
`tests/test_phase44_cli.py`. Then run in a server terminal:

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
python -m scripts.phase44 verify --output-dir "artifacts/phase44/check_$(date +%Y%m%d_%H%M%S)"
python -m scripts.phase44 plan --seed 0
```

Verification uses CPU fixtures, including small synthetic optimizer tests; it
does not train the research models or evaluate real performance.

## 2. Register your existing completed P4.3 run

Only after its status is `complete` and `refined_student.json` is saved:

```bash
export CUDA_VISIBLE_DEVICES=2
python -m scripts.phase44 adopt \
  --refinement-run artifacts/phase43_runs/refine_four_day_v1/relay/seed_0 \
  --run-dir artifacts/phase44_runs/D_adopted_v1/relay/seed_0 \
  --device cuda:0
```

Use a **fresh** destination. Adoption validates the complete P4.1–P4.3 lineage
and fixed budgets, records its historical source/runtime identities and publishes
the integrated completion/export. It takes no optimizer steps and does not
repeat training. Historical runs are marked `adopted_development`; adoption
does not relabel them as a fresh, prospectively frozen comparison cohort.

## 3. Start a fresh integrated run when required

Use this route for a fresh seed, rather than repeating your existing run.
Do not run the individual P4.1–P4.3 commands alongside this controller for the
same run. Open tmux:

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  /usr/bin/tmux -L wm-p44-system -f /dev/null new-session -A -s wm-d-1
```

Inside tmux, select an available physical GPU and a fresh directory:

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
export CUDA_VISIBLE_DEVICES=3
CASE=artifacts/phase44_runs/D_four_day_v1/relay/seed_1

python -m scripts.phase44 prepare --seed 1 --run-dir "$CASE" \
  --initial-selection artifacts/phase24_selection/shortened_v1/selection.json \
  --device cuda:0 &&
python -m scripts.phase44 run --run-dir "$CASE" --device cuda:0
```

`prepare` validates inputs and saves initialization without training. Only `run`
trains. Detach with **Ctrl+B, then D**. Each independent seed may use a separate
GPU; one integrated run uses one GPU. For a fresh seed 0, change both the seed
argument and directory. Match the intended cohort's data, initialization and
source registration before comparing B and D. Fresh integrated runs also retain
`fresh_cohort_eligible: false` until the P4.5/P5 checks and prospective cohort
freeze are completed. The matched reduced B controller remains separate work.

## 4. Progress, safe pause and resume

For the fresh seed-1 run above, open another terminal:

```bash
cd /home/zixuan/projects/WM1
source scripts/gpu_server_env.sh
CASE=artifacts/phase44_runs/D_four_day_v1/relay/seed_1
python -m scripts.phase44 status --run-dir "$CASE"
```

Progress reports the current `phase` and counters: teacher cycles **25**, buffer
trajectories **128**, distillation updates **100**, local cycles **20** and local
actor/critic optimizer steps **320 each**. Teacher actor/critic each receive
**400** optimizer steps. Buffer generation and distillation are separate stages.

To pause that run:

```bash
python -m scripts.phase44 stop --run-dir "$CASE"
```

Wait for `stopped` and for the old worker to exit. The controller checkpoints
after each teacher/local cycle, trajectory and supervised update. Committed
stage work and optimizer/RNG state are retained; an interrupted unfinished unit
may repeat and its physical work stays in the journal.

Inside tmux, choose an available GPU and resume:

```bash
export CUDA_VISIBLE_DEVICES=1
python -m scripts.phase44 run --run-dir "$CASE" --device cuda:0
```

Keep the same environment, logical device and registered sources. Do not repeat
`prepare`. `--max-units N` pauses after N additional accounted units; there are
273 learning/generation units and one final export unit in the full schedule.
Keep the complete integrated run and referenced stage directories.

Completion provides the final local actor and receipt-bound accounting without
loading a teacher at deployment. P4.5 acceptance, the matched reduced B run and
P5 real-environment evaluation remain separate; training completion does not
establish that D outperforms B.
