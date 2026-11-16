# P3.1: local actors and centralized critic

P3.1 implements the networks used by B and D students. It does not run policy
training, replace the P2.5 model-quality gate, or complete the rest of Phase 3.

## Verify

In the existing Linux project environment, from the project root:

```bash
python -m scripts.phase31 verify \
  --output-dir "artifacts/phase31_checks/$(date +%Y%m%d_%H%M%S)"
```

This uses CPU synthetic fixtures and needs no GPU allocation, replay or trained
checkpoint. A successful result is `complete_p31_implementation`; the new output
directory contains `implementation_checks.json`, its SHA256 receipt and
`tests.txt`. Existing output directories are never overwritten.

## Network interfaces

```python
import torch
from world_models.infrastructure.config import resolve_config
from world_models.policies import build_actor_critic, imagined_rgb_to_uint8, raw_value_loss

settings = resolve_config("relay", "B", "pilot").settings
policy = build_actor_critic(settings, master_seed=0, variant="B")
actor, critic = policy.actor, policy.critic

# One actor network is shared by both players. Each row holds its own memory.
memory = actor.initial_hidden(2)                      # [2, 256], zeros
previous = torch.full((2,), 12, dtype=torch.long)     # START at episode reset
rgb = torch.zeros(2, 3, 96, 128, dtype=torch.uint8)    # synthetic example only
out = actor(rgb, previous, memory)
memory = out.next_local_hidden                       # [2, 256], after current RGB
probabilities = out.distribution.probs               # [2, 12], FP32
```

- **Actor:** own uint8 RGB → CNN `32/64/128/256` → flattened spatial projection
  to 256 → concatenate the 13-way previous-action one-hot → GRU256 → two-layer
  categorical head. GroupNorm replaces batch-dependent normalization; no dropout.
  Actor CNN parameters are separate from the world-model encoder.
- **Reset:** START=12 zeros the incoming memory before consuming the first image.
  Later calls supply the action actually executed, in `0..11`. START cannot be
  sampled as an action. A cutoff inside an episode is not a reset.
- **Deployment:** `actor.step(LocalActorInput(...))` implements the existing
  simulator contract. Only local RGB, own previous action and own memory enter.
- **Sequences:** `actor.forward_sequence(rgb, previous, initial_memory, valid=mask)`
  takes time-major `[T,B,...]` inputs and a boolean `[T,B]` mask. Padding is skipped
  and preserves memory; mask its placeholder logits out of all future losses.
  Gradients flow through active recurrence and stop at START. P3.2 constructs
  full real prefixes; process an anchor's current image only once.
- **Sampling:** call `out.sample(generator=...)` with a separate generator for each
  player's batch, derived from the existing `local_action_sampling` namespace.
  `out.log_prob(actions)` and `out.distribution.entropy()` use FP32. Forward calls
  consume no random draws; the module stores no recurrent state.
- **Imagined RGB:** `imagined_rgb_to_uint8(decoded_rgb)` detaches, rounds to the
  nearest 8-bit pixel and retains leading batch/player axes. The actor divides
  both real and imagined uint8 images by 255 identically.
- **Critic:** `critic(Z, memories, remaining)` takes `[B,48,256]` shared tokens,
  ordered `[B,2,256]` actor memories **after the current observations**, and `[B]`
  remaining time `(deadline - decision_index) / deadline`. It mean-pools tokens,
  detaches all input features, and applies a `769→512→512→1` MLP to return `[B]`
  raw team values. The actor never receives these centralized inputs.
- **Value loss:** `raw_value_loss(values, raw_returns, valid=mask, delta=1.0)`
  detaches targets and averages Huber loss once per valid joint sample. It applies
  no value/reward normalization or value coefficient. P3.4 forms raw returns,
  normalize a separate copy of actor advantages and apply the configured loss
  coefficient in the MAPPO update.

The builder uses separate actor/critic initialization seeds. Paired B/D settings
and master seeds produce identical initial values with independent parameter
storage, without changing global RNG state. Move the resulting module and its
inputs to the desired device explicitly; each agent's generator must use that
same device. Keep parameters in FP32. The local actor always computes in IEEE
FP32, including inside an outer BF16 context, so collecting and reconstructing
its history with different batch sizes does not introduce BF16 rounding drift.
The world model and critic can continue using their configured autocast precision.

The saved [implementation evidence](../artifacts/phase31/implementation_v1/implementation_checks.json)
covers locality, resets, recurrent likelihood replay, padding, explicit sampling
streams, paired initialization, raw value loss and gradient isolation from an
actual untrained world model. These are implementation checks; trained policy
behavior and the Phase 3 acceptance gate remain pending. Full-prefix anchors are
implemented in [P3.2](phase32_anchors.md), and imagined trajectories in
[P3.3](phase33_rollouts.md), with recurrent MAPPO in [P3.4](phase34_mappo.md).
The training controller is implemented in [P3.5](phase35_training.md).
Next is **P3.6's real-policy acceptance gate**.
