# P3.3: imagined on-policy trajectories

P3.3 collects synthetic trajectories with frozen world-model, actor and critic
weights. It performs no optimizer updates or new environment collection. PPO
optimization is implemented in [P3.4](phase34_mappo.md); the research training loop belongs to P3.5.

## Verify locally

On Windows, from the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m scripts.phase33 verify --output-dir artifacts/phase33_checks/local_v1
```

Use a new output directory on each run. On Linux, in the existing environment:

```bash
python -m scripts.phase33 verify \
  --output-dir "artifacts/phase33_checks/$(date +%Y%m%d_%H%M%S)"
```

This runs small CPU fixtures with untrained networks. No GPU allocation, server
checkpoint or research replay is needed. A successful result is
`complete_p33_implementation`; the fresh directory contains
`implementation_checks.json`, its SHA256 receipt and `tests.txt`.

## Integration API

With the selected world model, current local actor/critic and fitting-replay
sampler already loaded:

```python
from world_models.imagination import AnchorBuilder
from world_models.rollouts import RolloutGenerator, RolloutStreams, save_rollout

# Keep the same world-model/critic precision for anchors, collection and replay.
# The local actor always computes in IEEE FP32.
precision = "float32"  # CPU-friendly; bfloat16 is also supported
anchors = AnchorBuilder(model, actor, seed=prefix_sampling_seed,
                        precision=precision).build(sampler)
streams = RolloutStreams.from_experiment(settings, master_seed=training_seed,
                                        device=next(model.parameters()).device)
collector = RolloutGenerator.from_experiment(model, actor, critic, settings,
                                             precision=precision)
rollout = collector.collect(anchors, streams=streams)
save_rollout(rollout, "artifacts/imagined_rollouts/cycle_000000")
```

Create the streams once and retain them across collection cycles. Save their
`state_dict()` with the later run checkpoint; recreating them with the same seed
each cycle would repeat random draws. The buffer also records states before and
after collection. `load_rollout(path, device="cpu")` verifies the stored receipts,
provenance and trajectory semantics. Every save requires a fresh directory.

- **On-policy actions:** the first actions use the P3.2 anchor logits directly;
  the anchor image is already consumed. Each player samples from its own explicit
  RNG stream. Later memory updates use that player's decoded image and the action
  actually executed. Generated images are rounded to uint8 and normalized by the
  same actor interface as real images.
- **Shared dynamics:** each joint decision calls the world transition once with
  both actions, sampling one shared next latent and one joint outcome. No future
  real observations, privileged targets or posterior corrections enter the rollout.
- **Stopping:** each row ends at sampled success/failure, exact task deadline or
  the configured horizon (default 15). Success at the deadline remains success.
  A terminal at the horizon takes precedence over a cutoff. Inactive rows neither
  draw actions nor update their memories; padding is explicitly masked.
- **State alignment:** states have shape `[T+1,B,...]`, transitions `[T,B,...]`.
  `local_rgb`, `previous_actions`, `actor_hidden`, `actor_logits`, `shared_tokens`,
  raw critic `values` and the full shared `world` trace include final destinations.
  Each memory has already consumed the image at its state index. Transition data
  include actions, per-player behavior `log_probs`, shared rewards/bonus, validity,
  terminal/cutoff masks, outcome probabilities/sample/resolution, deadline flags
  and `end_reason`. Actual `T` can be shorter when all rows terminate early.
- **Bootstrap:** `old_values` and `next_values` retain raw critic predictions.
  `bootstrap_values` zeros true terminals and keeps the final value for an alive
  horizon cutoff. `recursion_mask` stops at both kinds of boundary. P3.4 uses
  these to compute GAE and separate raw returns from normalized actor advantages.
- **Provenance:** each buffer records anchor/prefix references, replay cohort,
  exact model/actor/critic versions, precision, task reward rule and RNG states.
  Stale anchors are rejected. Collection detects concurrent weight changes,
  restores module modes and leaves weights, existing gradients and global RNG
  unchanged. On failure, explicit RNG streams roll back to their starting state.
- **Separate synthetic storage:** the receipt-bound store is marked
  `world_model_imagination` and excluded from world-model fitting. Saves inside
  an existing real-replay store are rejected. Do not add imagined trajectories to
  the real-interaction ledger or anchor sampler.
- **Failure diagnostics:** the model predicts a combined learned failure class.
  Causes distinguish sampled success, sampled failure, exact deadline and horizon;
  they do not claim to distinguish death from other learned failures.

P3.4 reconstructs current actor memories from the saved local prefix/history
references when parameters change. Stored post-observation memory is not a
pre-image memory: feeding its source image again would consume that image twice.
Behavior likelihoods and critic features remain fixed rollout data during PPO.
See [the implemented recurrent MAPPO API](phase34_mappo.md).

The [saved implementation evidence](../artifacts/phase33/implementation_v1/implementation_checks.json)
covers correctness with synthetic fixtures. Trained model/policy quality and
Phase 3 acceptance remain pending; this command does not run a research campaign.
