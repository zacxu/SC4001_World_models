# P3.2: valid imagination anchors

P3.2 reconstructs starting states for policy imagination. It uses existing real
replay and current model/actor weights; it performs no policy training or new
environment collection. [P3.3](phase33_rollouts.md) generates the imagined trajectories.

## Verify

From the project root in the existing Linux environment:

```bash
python -m scripts.phase32 verify \
  --output-dir "artifacts/phase32_checks/$(date +%Y%m%d_%H%M%S)"
```

The check uses CPU synthetic replay and untrained networks. No GPU allocation,
server checkpoint or research replay is needed. Success is
`complete_p32_implementation`; the fresh directory contains
`implementation_checks.json`, its SHA256 receipt and `tests.txt`.

## Integration API

With `settings` resolved for the current task/variant, `model` holding the frozen
world-model snapshot and `actor` holding the current local-policy weights:

```python
from world_models.data.replay import ReplayDataset
from world_models.imagination import AnchorBuilder, AnchorSampler

# List the shared initial store, then this branch's later stores chronologically.
stores = [ReplayDataset(replay_path, verify=False)]
sampler = AnchorSampler.from_experiment(stores, settings, seed=anchor_sampling_seed)
builder = AnchorBuilder(model, actor, seed=prefix_sampling_seed,
                        precision="float32", cache_capacity=512)

anchors = builder.build(sampler)              # configured 256 starts
first_player = anchors.actor_output(0)        # anchor image already consumed
second_player = anchors.actor_output(1)
value = critic(anchors.shared_tokens, anchors.actor_hidden, anchors.remaining_time)
```

Use dedicated seeds derived from the existing RNG namespace for sampling and
prefix reconstruction. Save `sampler.state_dict()` to resume its sampling stream;
restore only into the same ordered cohort and settings. Recreate the sampler
when replay grows. Prefix caches are disposable and need not be checkpointed.

- **Balanced sampling:** the existing `world_model.recent_fraction=0.5` gives
  128 uniform draws from all eligible fitting states and 128 from the last
  `recent_transitions=100000` physical transitions, followed by a shuffle.
  Draws use replacement; the pools overlap. Diagnostic transitions count toward
  recency but are never anchors. No reward, success or event weighting is used.
  B and D use the same rule. At initialization the recent pool can equal the
  complete pool, making this uniform sampling over fitting states.
- **Eligible states:** fitting episodes on training geometry, real data only,
  both images valid throughout the prefix, and a source decision before the task
  deadline. A state whose next real transition terminates remains eligible.
  Final destination images, including collection-cutoff endpoints, are excluded
  under the existing replay source-state convention. No reset is fabricated at
  a cutoff. Semantic checks still run when opening replay with `verify=False`.
- **Full prefixes:** at decision `t`, only `RGB[0:t+1]` and executed `actions[0:t]`
  are passed to reconstruction. There is no burn-in cap. The world posterior
  uses both views; each actor receives only its own view, own executed action and
  own recurrent memory, beginning with zero memory and START at reset.
- **Returned batch:** `world` is the shared posterior; `actor_hidden` is
  `[B,2,256]` after observing the anchor; `actor_logits` is `[B,2,12]`;
  `local_rgb` is real uint8 `[B,2,3,96,128]`; `previous_actions` is `[B,2]`;
  `shared_tokens` is `[B,48,256]`; `remaining_time` is `[B]`. The recorded decision
  clock is preserved. Sample the first actions from `actor_output(player)` with
  separate actor generators; do not feed `local_rgb` through the actor again.
  Subsequent imagined RGB uses the P3.1 `imagined_rgb_to_uint8` adapter.
- **Cache correctness:** identity includes episode/prefix, replay cohort, exact
  model and actor state hashes, seed, device and precision. Each build checks the
  current network versions; weight updates or checkpoint loads invalidate cached
  memories automatically. Posterior RNG is episode-specific and independent of
  actor updates, cache hits, request order and eviction. Returned tensors cannot
  mutate cached entries. Set `cache_capacity=0` for an uncached reference.
- **Efficiency and isolation:** requested anchors from one episode share a
  forward prefix pass. Only one episode's RGB prefix is loaded at a time and
  cached snapshots are bounded. Reconstruction skips image decoding and reward
  heads, runs without gradients, and restores network modes and RNG state.
  Parameters remain FP32; `precision="bfloat16"` enables autocast. Keep the
  chosen precision consistent with later policy rollout/replay computations.

`sampler.reference(dataset_index, episode_id, decision_index)` selects an explicit
eligible anchor; `builder.build(sampler, references)` reconstructs it without
advancing the sampler RNG. `anchors.metadata()` records prefix IDs, versions,
precision and cache work. Output tensors are detached ordinary tensors suitable
for subsequent gradient-bearing policy operations.

See the [saved implementation evidence](../artifacts/phase32/implementation_v1/implementation_checks.json).
These checks do not establish model quality, trained-policy competence or Phase
3 acceptance. The P2.5 quality review remains separate.
