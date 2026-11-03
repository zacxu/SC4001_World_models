# P2.3: model objectives and loss reductions

P2.3 is complete. The objective, replay/output alignment, fitting-only continuous
normalization, and loss/gradient/class diagnostics are executable. The model
remains **untrained**: verification performs forward/backward passes and checks
that weights do not change. P2.4 optimizer updates and P2.5 predictive-quality
acceptance remain pending.

The [acceptance report](../artifacts/phase23/phase23_acceptance.json) records
source hashes, objective tests, exact normalization identities, full fitting
label profiles, and real-data loss/gradient checks. The
[regression report](../artifacts/phase23/regression_tests.json) covers the full
current unit suite. The [results review](../artifacts/phase23/README.md) separates
measured implementation behavior from remaining research risks.

Measured validation: **62 P2.3 tests passed**. The full suite ran 389 tests with
**388 passed and one skipped** because Windows symlink privileges were
unavailable. Real forward/backward checks covered 20 windows from 14 fitting
episodes, including six positive milestone windows. No model weights changed.

## Objective and precise denominators

The approved objective and default weights are unchanged:

```text
L_WM = 100 L_RGB + L_r + L_d + L_aux + 0.5 L_dyn + 0.1 L_rep
```

| Term | Calculation | Reduction |
|---|---|---|
| RGB | Squared error for floating [0,1] RGB | Sum over valid scalar channel/pixel entries divided by their count, across both views, time, and batch |
| Reward | Huber(delta=1) on the bounded milestone component | Sum over valid transition bonus targets divided by count |
| Outcome | Cross-entropy of continue/success/non-deadline-failure logits | Sum over valid transition outcome targets divided by count |
| Auxiliary continuous | Huber(delta=1), standardized **targets** and heads predicting standardized values | Mean within each field, then equal mean over fields with valid support |
| Auxiliary binary | BCE with logits for binary states and event presence | Same per-field mean rule |
| Auxiliary stage | Cross-entropy with three relay/four cover classes, including terminal success stage | Same per-field mean rule |
| Dynamics KL | `max(1, sum_32 KL(stopgrad(q) || p))` | Floor each sample after summing all 32 categorical variables, then mean over valid samples |
| Representation KL | `max(1, sum_32 KL(q || stopgrad(p)))` | Same floor/reduction; gradients through q only |

Both KL distributions use the P2.2 mixed probabilities `(1-.01)*softmax(logits)
+ .01/32`. Category sums and probability/loss arithmetic use float32 under
bfloat16 autocast. Entropy and unfloored KL are logged separately. Entropy
diagnostics report both summed joint entropy and nats per categorical variable.

Each primitive returns a `LossTerm(numerator, count)`. Its mean is
`numerator/max(count,1)`. Empty selections have a graph-connected zero, no
free-nat charge, and zero support. Invalid entries are selected away **before**
MSE, Huber, BCE, cross-entropy, normalization, or KL; multiplying a NaN loss by a
zero mask is never used. A nonfinite **valid** prediction or label raises.

Missing fields contribute neither a fabricated zero target nor a dilution term
to `L_aux`: divide by the number of fields with support in that reduced batch.
Every supported declared field has the same weight, regardless of its number
of available examples. Reports include each field's count and mean. No event
weights, sampling ratios, reward scales, or model dimensions changed in P2.3.

For later microbatching, `merge_objective_results` adds component numerators and
counts, merges **each auxiliary field** before its mean, then reconstructs the
weighted objective. Averaging microbatch means is incorrect when masks/support
differ. This helper retains the graphs; it does not implement a memory-saving
training loop. P2.4 must use global component/per-field denominators for each
microbatch's backward contribution when implementing gradient accumulation.

## State/transition alignment and masks

The [batch adapter](../src/world_models/training/objective_batch.py) accepts
already computed P2.2 outputs. It does not choose training windows or run an
optimizer. For T transitions, supply T+1 state records and T transition records:

| Axis | Predictions and targets |
|---|---|
| State t | RGB at t and auxiliary player/task/enemy state at t |
| Transition t | Action t causes t+1; reward bonus, outcome, and event deltas describe t→t+1 |
| KL t | Prior/posterior for destination t+1 |

`objective_inputs(initial_prediction, transitions, state_arrays,
transition_arrays, task, ...)` stacks the initial state's predictions followed
by each destination state exactly once. State arrays have leading axes
`[T+1,B]`; transition arrays have `[T,B]`. The adapter checks the predicted
source/destination clocks against replay source indices. When `next_obs_valid`
is supplied it must match the destination state's image mask.

The joint posterior requires both synchronized local views. If either required
destination view is missing, the model uses the prior and P2.3 skips **both-view
next-RGB loss and posterior KL** for that row. The adapter rejects a claimed
posterior whose validity disagrees with replay image availability. Valid
reward/outcome, event, and auxiliary state targets remain independent and can
train the prior-based destination. Initial RGB and state supervision are also
supported with zero transitions and no preceding reward or KL target.

The low-level objective accepts explicit state/transition inputs for fixtures
or other loaders. Production loaders should use the alignment adapter and
validate normalization provenance against their `ReplayDataset` first.

`state_loss_mask` and `transition_loss_mask` are independent loss-bearing masks
for padding/burn-in. They are applied before decoding labels, so invalid padded
reason codes, NaN rewards, and unavailable auxiliary values are ignored safely.
They do not change the episode's clock or supplied model predictions. P2.4
still owns window construction, reset handling, and its burn-in approximation.

## Fitting-only continuous normalization

Statistics are frozen independently for each task/seed's initial real replay
and physically shared within B/D. The four
[bindings](../artifacts/phase23/normalization_bindings.json) point B and D to the
same statistics file/hash for that pair; no new data-store copies are created.
Continuous inputs to statistics are the replay's existing geometry-normalized
player/task/enemy labels, then P2.3 standardizes them as `(value-mean)/scale`.
Model inputs remain RGB, actions, memory, and clock. Privileged state stays in
the supervision path.

`fit_normalization` streams only `iter_episodes(purpose='world_model_fit')`.
It includes each unique valid state once, including the initial observation
and valid terminal state targets, independently of whether terminal RGB exists.
Diagnostic episodes and non-training geometry never supply these statistics.
Population mean/variance use chunked float64 Welford accumulation. Each field
has its own validity mask and count. Memory stays bounded by chunk/cache use.

For a field with population std ≥1e-6, use that std as scale. Otherwise use
scale 1.0 and record it as constant. Unobserved fields retain null moments and
unit scale, and reject any later nonempty supervision instead of inventing
statistics. All pilot continuous fields had fitting support. Masks select
values before normalization, so count-zero fields with no valid labels are safe.

JSON files include field counts/moments/extrema, exact fitting episode IDs and
their hash, dataset manifest/content identity, seed/task, excluded diagnostic
episode count, task constants, and label-imbalance profiles. Saving never
overwrites an existing JSON/receipt. Loading checks SHA and canonical identity;
`validate_for_dataset` checks the precise source store and fit membership.

| Task / pilot seed | Fitting episodes | Fitting transitions | Unique fitting states |
|---|---:|---:|---:|
| Relay / 0 | 314 | 43,777 | 44,091 |
| Relay / 1 | 324 | 46,729 | 47,053 |
| Cover / 0 | 228 | 44,725 | 44,953 |
| Cover / 1 | 222 | 47,604 | 47,826 |
| Total | 1,088 | 182,835 | 183,923 |

The remaining 104 diagnostic episodes are excluded. These are **initial**
normalizers. P2.4 must bind their identity to checkpoints and version any later
online normalization change; never overwrite them or mix B/D branch replay.

## Diagnostics and interpretation

`ObjectiveResult.metrics()` exports component and weighted losses, sufficient
statistics, per-field losses/support, raw KL, prior/posterior entropy, event
prevalence/probability errors, stage/outcome confusion, and total-reward error.
Continuous auxiliary MAE is also mapped back to the replay's geometry-normalized
units for interpretation. Binary event labels mean `event_delta>0`; raw counts
remain in replay. Undefined precision/recall and empty entropy are null with
zero support, not evidence of good prediction or latent collapse.

Total-reward diagnostics use the deterministic expected reward, without an
extra outcome draw. Before the deadline, the terminal expectation is
`p_success*R_success + p_failure*R_failure`. At the deadline add
`p_continue*R_failure`, because continue is then resolved as deadline failure.
Add the bonus prediction and known step cost, then measure MAE/MSE/bias against
real total team reward. A separate bonus MAE distinguishes reward-head error.
Real deadline-only and external collection-cutoff labels still train the
learned outcome as continue. Label decomposition remains strict and bounded.

`gradient_diagnostics(model)` reports global and module L2 norms after backward
and **before clipping**, parameter tensors with/without gradients, and nonfinite
entries. It neither clips nor changes gradients, parameters, or train/eval mode.
Norm reductions use scaled float64 calculations. Nonfinite valid gradients
raise by default. P2.4 will choose logging cadence and profile synchronization
cost on the training host.

The full fitting profile is strongly imbalanced: only five non-deadline failure
transitions occur across these fitting stores. Console-completion presence is
about 0.19–0.29% of cover transitions; relay latch activation is about 0.43–0.45%.
P2.4/P2.5 must inspect rare-event learning rather than relying on average
classification accuracy. The current untrained checks cannot establish that
events are ignored. No reweighting or additional collection was performed.
If pilots later reveal missed events, inspect coverage and alignment first;
any change to weighting/scales must be recorded and applied to both variants
before protocol freeze, as required by the plan.

## Reproduce and use

Use the P2.2 [CPU dependency lock](../requirements-phase2-cpu.lock) and run from
the repository root. The measured implementation checks use Windows/Python3.12,
PyTorch2.15.0+cpu, four CPU affinity slots, and one torch thread. No CUDA/H200
execution or optimizer performance is claimed.

```powershell
# Recompute statistics and objective checks into fresh paths.
.\.venv\Scripts\python.exe -m scripts.phase23 complete --output-dir artifacts/phase23_recheck --data-dir data/initial_replay/pilot --workers 4

# Recheck using the frozen initial normalizers; no second full statistics pass.
.\.venv\Scripts\python.exe -m scripts.phase23 verify --output-dir artifacts/phase23_verify_only --data-dir data/initial_replay/pilot --statistics-dir artifacts/phase23/normalization

.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`stats` can separately generate statistics with an exact generation receipt.
`complete` keeps a sibling `<output-dir>_statistics` generation folder plus
verified canonical copies and B/D bindings in its report bundle. Use fresh
paths; existing evidence is never overwritten by these commands. The canonical
`normalization/` directory includes its generation receipt so it can directly
supply the second command. These are small metadata files containing no RGB
or model checkpoint.

Cached statistics reuse checks exact generation-source byte hashes. If a
checkout changes source bytes, including line endings, run `complete` into
fresh paths to regenerate its statistics receipt; do not rewrite the old
receipt or its frozen statistics in place.

```python
from world_models.training.normalization import NormalizationStats
from world_models.training.objective import ObjectiveConfig, WorldModelObjective
from world_models.training.objective_batch import objective_inputs
from world_models.training.diagnostics import gradient_diagnostics

stats = NormalizationStats.load("artifacts/phase23/normalization/relay/seed_0.json")
stats.validate_for_dataset(replay_dataset)
criterion = WorldModelObjective(model.task, stats, ObjectiveConfig.from_experiment(cfg))
state_inputs, transition_inputs = objective_inputs(
    initial_prediction, observed_steps, state_arrays, transition_arrays, model.task,
    event_fields=replay_dataset.manifest["schema"]["event_fields"],
    state_loss_mask=state_mask, transition_loss_mask=transition_mask)
model.zero_grad(set_to_none=True)
losses = criterion(state_inputs, transition_inputs)
losses.total.backward()
metrics = losses.metrics()
gradient_metrics = gradient_diagnostics(model)
# Optimizer updates are introduced in P2.4.
```

The verifier reconstructs complete real fitting prefixes without keeping a
graph, then checks short windows for representative end reasons and positive
latch/console milestone events in each store. It records all losses and gradients and confirms
weights and replay manifests remain unchanged. This is an interface test on
an untrained model, not held-out predictive-quality evaluation. Existing
P2.1/P2.2 acceptance artifacts remain historical, unchanged evidence.
