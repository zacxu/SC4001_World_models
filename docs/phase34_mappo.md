# P3.4: recurrent MAPPO

P3.4 implements one complete PPO update cycle using the P3.3 imagined trajectories.
Local CPU checks work without server checkpoints. Research policy training and
the real-data/model/policy schedule are connected by [P3.5](phase35_training.md); trained-policy
acceptance remains P3.6.

## Verify locally

From the project root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m scripts.phase34 verify --output-dir artifacts/phase34_checks/local_v1
```

Or in the existing Linux environment:

```bash
python -m scripts.phase34 verify \
  --output-dir "artifacts/phase34_checks/$(date +%Y%m%d_%H%M%S)"
```

Use a new output directory each time. Checks include small optimizer updates on
synthetic CPU fixtures, with no research replay or environment collection.

## Use the update API

With the loaded `model`, `actor`, `critic`, resolved `settings`, and fitting-replay
`sampler` from P3.1–P3.3:

```python
from world_models.imagination import AnchorBuilder
from world_models.mappo import RecurrentMAPPO
from world_models.rollouts import RolloutGenerator, RolloutStreams

precision = "float32"  # Use the same precision for anchors, collection and PPO.
learner = RecurrentMAPPO.from_experiment(
    model, actor, critic, settings, master_seed=seed, precision=precision)
builder = AnchorBuilder(model, actor, seed=prefix_seed, precision=precision)
collector = RolloutGenerator.from_experiment(
    model, actor, critic, settings, precision=precision)
streams = RolloutStreams.from_experiment(
    settings, master_seed=seed, device=next(model.parameters()).device)

anchors = builder.build(sampler)
rollout = collector.collect(anchors, streams=streams)
report = learner.update(rollout, sampler)
print(report["actor_optimizer_steps"], report["critic_optimizer_steps"])
print(report["metrics"])
```

Keep the learner and streams across cycles. Rebuild anchors and collect fresh
trajectories after each update. With defaults, each cycle runs **4 epochs × 4
whole-sequence minibatches = 16 actor and 16 critic optimizer steps**. Updates
reject stale model/actor/critic versions and require at least four trajectories
for four minibatches. Each trajectory appears once per epoch, including shorter
terminal trajectories; padding contributes to neither loss.

Save `learner.state_dict()` alongside the actor, critic, world-model identities,
rollout stream states and sampler state for a completed-cycle checkpoint. Restore
matching network weights first, then `learner.load_state_dict(...)`. Optimizer
state includes both Adam moments, counters and the independent shuffle RNG.
File publication and the complete training/resume controller are implemented in P3.5.

## Correctness and diagnostics

- Real actor history is rebuilt from reset under current parameters before every
  minibatch. Prefix work ends before the anchor image and runs without gradients;
  the anchor and imagined source images are then replayed once with recurrent
  gradients. Stored behavior memory never initializes current actor recurrence.
- One raw team GAE is computed from frozen old values. True terminals bootstrap
  zero; an alive horizon cutoff includes the final value but stops recursion.
  Raw GAE plus old value forms the critic return. A separate actor copy is
  normalized across valid joint samples.
- PPO uses each player's individual likelihood ratio and the same team advantage,
  averaged across valid player decisions. The critic uses fixed stored tokens,
  actor memories and clock, with one raw Huber loss per joint sample.
- Before any update, replay must reproduce behavior log probabilities and values.
  The local actor uses IEEE FP32 throughout its encoder, recurrent state and
  action head, even inside a BF16 world-model/critic context. The maximum actor
  log-probability error limit is `2e-5` for both settings. Critic replay retains
  its BF16 `0.02` relative/absolute tolerance (FP32 `2e-5`). Errors include the
  measured mismatch and offending step/trajectory/player; stored behavior
  probabilities are never replaced to make the check pass. The numerical repair
  is validated on CPU; the full H200 continuation remains a user-run check.
- Separate Adam optimizers use learning rate `3e-4`, epsilon `1e-5`, no weight
  decay, and separate gradient norm clipping at `0.5`. Epoch count stays fixed.
  An exception restores the cycle's actor/critic weights, gradients, optimizer
  state and shuffle RNG; numerical divergence must be diagnosed before retrying.
- Reports contain per-minibatch and aggregate approximate KL, clip fraction,
  entropy, value loss, final raw return MSE, explained variance, pre-clipping
  gradient norms, mean active length and terminal causes. Undefined explained
  variance for constant returns is JSON `null`. Termination frequency is the
  fraction of trajectories ending terminal; a separate per-step rate is recorded.

The world model is never executed during PPO or included in either optimizer;
its content hash must remain unchanged. See the [implementation evidence](../artifacts/phase34/implementation_v1/implementation_checks.json).
