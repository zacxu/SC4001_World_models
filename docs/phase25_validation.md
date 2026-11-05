# P2.5: compact engineering screen

This checks whether the shared world model is usable for a **small exploratory
Phase 3 pilot**. It is not an additional B/D experiment or formal model-quality
acceptance. Keep both relay and cover, seeds 0 and 1: four shared models, each
evaluated once. No research training or simulator collection occurs here.

Use the [existing P2.4 environment](phase24_training.md), source, fitting replay
and frozen normalization in `/home/users/create/smrksun/scratch/WM1` on ASPIRE2A.
Run evaluation in a PBS compute allocation; preserve its GPU mask/order and
keep `--device cuda:0` for the allocated GPU.

P2.4 can keep training. Sync the new P2.5 code/tests and documentation while
preserving the server's existing P2.4 source and configs. Local checkpoint
copies may be stale. Screen on a free GPU, or wait for the training allocation.

## 1. Verify the evaluation implementation

On the login node, request one GPU using project **`11704278`**:

```bash
export PROJECT_ID=11704278
export WM_ROOT=/home/users/create/smrksun/scratch/WM1
cd "$WM_ROOT"
qsub -I -P "$PROJECT_ID" -q normal \
  -l select=1:ngpus=1 -l walltime=02:00:00
```

In the allocated compute shell, activate the environment and use a fresh
evidence directory. Keep this shell open through the evaluation:

```bash
export WM_ROOT=/home/users/create/smrksun/scratch/WM1
cd "$WM_ROOT"
module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WM_ROOT/.envs/wm312"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source scripts/aspire2a_cuda.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
set +e
P25_VERIFY="artifacts/server_p25_verify_$(date +%Y%m%d_%H%M%S)"
python -m scripts.phase25 verify --output-dir "$P25_VERIFY"
```

Continue after successful verification. It writes `implementation_checks.json`,
its SHA receipt and `tests.txt`. Reuse existing structural/overfit evidence where
available; synthetic tests do not validate trained-checkpoint event behavior.

## 2. Time eight windows on one checkpoint

Choose a common checkpoint update before inspecting scores. The example uses
3,000 for all four models; use the update you actually preserved. With P2.4's
default cadence, updates 3,000 and 4,000 both have numbered checkpoint pairs.

```bash
P25_RUN="artifacts/phase25_runs/screen_$(date +%Y%m%d_%H%M%S)"
python -m scripts.phase25 evaluate \
  --checkpoint artifacts/phase24_runs/aspire2a_v1/relay/seed_0/checkpoints/update_00003000.pt \
  --output-dir "$P25_RUN/timing_relay_seed_0" \
  --device cuda:0 --windows 8 --seed 0 \
  --verification-report "$P25_VERIFY/implementation_checks.json"
```

Inspect timing, event support and predictions before scaling. Time includes
complete prefixes and I/O; extrapolation to four models is approximate. No
half-day runtime is guaranteed. Eight timing windows do not establish readiness.

## 3. Screen the four shared checkpoints

Use the same sampling seed and the same explicit checkpoint update. The campaign
does not automatically choose a model's best checkpoint:

```bash
python -m scripts.phase25 campaign \
  --run-root artifacts/phase24_runs/aspire2a_v1 --checkpoint-update 3000 \
  --output-dir "$P25_RUN/campaign" \
  --device cuda:0 --windows 64 --seed 0 \
  --verification-report "$P25_VERIFY/implementation_checks.json"
```

Four cases run sequentially to bound GPU memory. Use `--windows 8` for a smaller
first campaign; the cap is 64 per model. Nondefault paths use `--data-dir` and
`--statistics-dir`. Change `aspire2a_v1` if your P2.4 continuation uses another
run folder. Exit the compute shell with `exit` when finished. Copy the reports
and selected checkpoint pairs to durable storage; scratch is temporary.

Screening an intermediate checkpoint does not finalize P2.4, change its update
target or B/D bindings, or make it eligible for refresh.

## What the screen measures

- Up to 32 uniform anchors plus up to 32 distinct event-targeted anchors from
  existing **diagnostic episodes**. Report the actual sample and positive support.
- Complete real prefixes followed by one prior-only, recorded-action rollout
  up to 15 steps; report horizons 1, 5, 10 and 15 from that same rollout.
- RGB versus persistence, learned outcomes and milestones versus fitting-only
  frequencies, and reward versus their task-rule baseline with exact deadlines.
  Stage accuracy is also reported; player-state error is not part of this screen.
- Separate uniform and event-targeted results, unique episodes and distinct
  positive events. Overlapping windows are not independent trials. Model
  termination errors and unavailable terminal RGB need explicit validity rules.

Prefix caches must match the exact model, replay/episode, prefix and sampling
identity. No future image or privileged target may enter an imagined step.
Early model termination cannot be hidden by discarding rows or forcing continue.
If held-out death positives are absent, death prediction remains **unvalidated**.

Evaluation and campaign directories contain `report.json` and its SHA receipt.
A completed campaign also writes `pilot_model_selection.json` and its receipt:
all four shared checkpoint identities plus the campaign readiness, including
`needs_review` when appropriate. This is separate pilot evidence; it does not
rewrite P2.4's completion status or B/D bindings. Require the final campaign
`report.json` and both SHA receipts before using the selection; an interrupted
campaign can leave a selection file without a completed report.
Read `pilot_readiness.status`: `ready_for_exploratory_phase3`, `needs_review` or
`failed`, with its reasons, unchecked classes and support. A ready screen permits
a small Phase 3 integration pilot followed by real local-policy checks, not a
B/D efficacy claim. Agree on the same supported horizon for both arms; no
configuration changes automatically.

Provisional alarms are fixed in `compact.PILOT_THRESHOLDS` and documented in the
[P2.5 plan](../detailed_research_implementation_plan.md#p25-model-validation-and-acceptance-gate).
All four models must satisfy the support, coverage, baseline-error and event
checks. Missing predictions are penalized. An eight-window timing sample cannot
satisfy the uniform-support requirement. These are engineering alarms, not
calibrated research acceptance thresholds.

Every report keeps `P2.5_quality_acceptance: false`. These diagnostic episodes
use **training geometry**, despite being excluded from fitting. Formal held-out
geometry evidence and untouched final tests remain separate requirements.

Defer new collection, burn-in sweeps, videos, ablations and broad tuning.
Preserve reports, identities, timing and available overfit evidence. No P2.5
server runtime or learned-quality result has been measured locally.
