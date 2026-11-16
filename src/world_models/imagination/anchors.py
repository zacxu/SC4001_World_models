"""Full real-prefix posterior and local-memory reconstruction, with no gradients.

Only source images and actions executed BEFORE each anchor enter reconstruction.
Cached stochastic posteriors have named episode streams, so ordering, duplicates
and cache eviction cannot change an anchor. Both agents consume their current
image exactly once; the returned logits are ready for the first imagined action.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from world_models.data.replay import canonical_hash
from world_models.models.core import WorldState
from world_models.models.world_model import SamplingStreams, SharedWorldModel
from world_models.policies.actor import ActorOutput, LocalActor
from .replay import AnchorPrefix, AnchorRef, AnchorSampler
from .versions import module_version


@dataclass(frozen=True)
class AnchorBatch:
    references: tuple[AnchorRef, ...]
    world: WorldState
    actor_hidden: Tensor
    actor_logits: Tensor
    local_rgb: Tensor
    previous_actions: Tensor
    shared_tokens: Tensor
    remaining_time: Tensor
    world_model_version: str
    actor_version: str
    cohort_identity: str
    seed: int
    precision: str
    stats: dict

    def actor_output(self, player: int) -> ActorOutput:
        """Already-conditioned local distributions; do not encode RGB again."""
        if type(player) is not int or player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        return ActorOutput(self.actor_logits[:, player], self.actor_hidden[:, player])

    def metadata(self) -> dict:
        return {"schema_version": "p32-anchor-batch-v1", "references": [asdict(ref) for ref in self.references],
                "world_model_version": self.world_model_version, "actor_version": self.actor_version,
                "cohort_identity": self.cohort_identity, "seed": self.seed, "precision": self.precision,
                "device": str(self.world.memory.device), "prefix_policy": "full_episode_from_reset",
                "anchor_observation_consumed": True, "stats": dict(self.stats)}


def _clone_world(state: WorldState) -> WorldState:
    return WorldState(*(getattr(state, name).detach().clone() for name in
                        ("memory", "stochastic", "logits", "decision_index")))


@dataclass(frozen=True)
class _PrefixKey:
    dataset_identity: str
    episode_id: str
    prefix_length: int
    world_model_version: str
    actor_version: str
    cohort_identity: str
    seed: int
    precision: str
    device: str


@dataclass(frozen=True)
class _Snapshot:
    world: WorldState
    hidden: Tensor
    logits: Tensor
    rgb: Tensor
    previous: Tensor
    streams: dict[str, Tensor]

    def clone(self) -> _Snapshot:
        return _Snapshot(_clone_world(self.world), self.hidden.detach().clone(), self.logits.detach().clone(),
                         self.rgb.detach().clone(), self.previous.detach().clone(),
                         {name: value.clone() for name, value in self.streams.items()})


@contextmanager
def _reconstruction(model, actor, precision, device):
    # Preserve mixed submodule train/eval flags, not only the root module flag.
    modes = [(module, module.training) for network in (model, actor) for module in network.modules()]
    try:
        model.eval()
        actor.eval()
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                              enabled=precision == "bfloat16"):
            yield
    finally:
        for module, training in modes:
            module.training = training


class AnchorBuilder:
    """Reconstruct current-model/current-policy anchors from guarded real replay.

    A builder references the live networks, never an optimizer. Each build hashes
    actual network state (including buffers and configuration), automatically
    invalidating old caches after parameter updates, checkpoint loads or device
    moves. Callers must not mutate either network concurrently with a build.

    The bounded cache stores requested prefixes only. Within a build, sorted
    anchors from one episode reuse the same recurrence even with capacity zero.
    Capacity zero disables retention between builds and serves as a reference.
    """
    actor_type = LocalActor
    batch_type = AnchorBatch

    def __init__(self, model: SharedWorldModel, actor: LocalActor, *, seed: int,
                 precision: str = "float32", cache_capacity: int = 512):
        if not isinstance(model, SharedWorldModel) or not isinstance(actor, self.actor_type):
            raise ValueError(f"provide SharedWorldModel and {self.actor_type.__name__} instances")
        self.model, self.actor = model, actor
        self.seed, self.precision, self.cache_capacity = seed, precision, cache_capacity
        self._validate_options()
        self._cache: OrderedDict[_PrefixKey, _Snapshot] = OrderedDict()
        self._context = None
        self._invalidations = 0
        self._device()

    def _validate_options(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be a nonnegative signed 64-bit integer")
        if self.precision not in ("float32", "bfloat16"):
            raise ValueError("precision must be float32 or bfloat16")
        if type(self.cache_capacity) is not int or self.cache_capacity < 0:
            raise ValueError("cache_capacity must be a nonnegative integer")

    def _device(self) -> torch.device:
        parameters = tuple(self.model.parameters()) + tuple(self.actor.parameters())
        devices = {parameter.device for parameter in parameters}
        if len(devices) != 1 or next(iter(devices)).type not in ("cpu", "cuda"):
            raise ValueError("world model and actor must be on the same CPU or CUDA device")
        if any(parameter.dtype != torch.float32 for parameter in parameters):
            raise ValueError("keep model and actor parameters in float32; use precision for autocast")
        model, actor = self.model.config, self.actor.config
        if (self.model.core.config != model or self.model.core.encoder.config != model
                or self.model.decoder.config != model or self.model.task.deadline != model.deadline):
            raise ValueError("world-model submodule configs and task deadline must agree")
        if (model.image_shape != actor.image_shape or model.action_count != actor.action_count
                or model.encoder_channels != actor.encoder_channels
                or model.width != actor.world_width or model.memory_tokens + model.latent_variables != actor.world_tokens):
            raise ValueError("actor and world-model architecture contracts differ")
        return next(iter(devices))

    @property
    def cache_entries(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()

    def posterior_seed(self, ref: AnchorRef, world_model_version: str) -> int:
        """Episode draw namespace, independent of actor version and anchor order."""
        value = {"schema": "p32-prefix-rng-v1", "seed": self.seed, "dataset": ref.dataset_identity,
                 "episode": ref.episode_id, "world_model_version": world_model_version,
                 "precision": self.precision}
        return int(canonical_hash(value)[:15], 16)

    def _key(self, ref, world_version, actor_version, device, *, prefix_length=None):
        return _PrefixKey(ref.dataset_identity, ref.episode_id,
                          ref.prefix_length if prefix_length is None else prefix_length,
                          world_version, actor_version, ref.cohort_identity, self.seed, self.precision, str(device))

    def _remember(self, key, snapshot):
        if not self.cache_capacity:
            return
        self._cache[key] = snapshot.clone()
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_capacity:
            self._cache.popitem(last=False)

    def _nearest(self, key):
        candidates = [(old_key, value) for old_key, value in self._cache.items()
                      if old_key.dataset_identity == key.dataset_identity and old_key.episode_id == key.episode_id
                      and old_key.prefix_length < key.prefix_length]
        if not candidates:
            return None
        selected, value = max(candidates, key=lambda pair: pair[0].prefix_length)
        self._cache.move_to_end(selected)
        return value.clone()

    def _actor_step(self, rgb, previous, hidden, world):
        return self.actor(rgb, previous, hidden)

    def _extend(self, prefix: AnchorPrefix, target: int, current: _Snapshot | None,
                world_version: str, device: torch.device, stats: dict) -> _Snapshot:
        streams = SamplingStreams.from_seed(self.posterior_seed(prefix.ref, world_version), device)
        if current is None:
            rgb = torch.tensor(prefix.rgb[0], device=device)
            previous = torch.full((2,), self.actor.config.action_count, dtype=torch.long, device=device)
            world = self.model.initialize(rgb.unsqueeze(0), streams=streams)
            output = self._actor_step(rgb, previous, self.actor.initial_hidden(2), world)
            current = _Snapshot(world, output.next_local_hidden, output.logits, rgb, previous, streams.state_dict())
            stats["world_observation_steps"] += 1
            stats["actor_observation_steps"] += 2
        else:
            streams.load_state_dict(current.streams)
        start = int(current.world.decision_index.item()) + 1
        for index in range(start, target + 1):
            rgb = torch.tensor(prefix.rgb[index], device=device)
            previous = torch.tensor(prefix.actions[index - 1], dtype=torch.long, device=device)
            # No decoder, reward, auxiliary targets or future observations are
            # needed to reconstruct a posterior over the recorded real prefix.
            prior = self.model.core.prior_step(current.world, previous.unsqueeze(0), generator=streams.prior)
            world = self.model.core.observe(prior, rgb.unsqueeze(0).float().div(255), generator=streams.posterior)
            output = self._actor_step(rgb, previous, current.hidden, world)
            current = _Snapshot(world, output.next_local_hidden, output.logits, rgb, previous, streams.state_dict())
            stats["world_observation_steps"] += 1
            stats["actor_observation_steps"] += 2
        if int(current.world.decision_index.item()) != int(prefix.decision_indices[target]):
            raise ValueError("reconstructed clock differs from the recorded anchor decision index")
        return current

    @torch.inference_mode(False)
    def build(self, sampler: AnchorSampler, references=None) -> AnchorBatch:
        """Materialize requested refs, or sample the configured 256 starts.

        Returned observations/memories already include o_t; sample the first
        action from actor_output(player). A later step uses predicted RGB and
        the actions actually executed. No actor/global sampling RNG is consumed.
        """
        if not isinstance(sampler, AnchorSampler):
            raise ValueError("provide an AnchorSampler of fitting replay")
        self._validate_options()
        while len(self._cache) > self.cache_capacity:
            self._cache.popitem(last=False)
        if sampler.task_family != self.model.task.task_family or sampler.deadline != self.model.task.deadline:
            raise ValueError("replay task/deadline must match the current world model")
        device = self._device()
        refs = tuple(sampler.sample() if references is None else references)
        if not refs:
            raise ValueError("provide at least one anchor reference")
        for ref in refs:
            sampler.validate_reference(ref)
        world_version, actor_version = module_version(self.model), module_version(self.actor)
        context = (world_version, actor_version, sampler.cohort_identity, self.seed, self.precision, str(device))
        if self._context != context:
            if self._context is not None:
                self._invalidations += 1
            self.clear_cache()
            self._context = context
        stats = {"cache_hits": 0, "cache_misses": 0, "world_observation_steps": 0,
                 "actor_observation_steps": 0, "requested_anchors": len(refs), "distinct_prefixes": len(set(refs))}
        grouped = {}
        for ref in refs:
            grouped.setdefault((ref.dataset_index, ref.episode_id), set()).add(ref)
        snapshots = {}
        try:
            with _reconstruction(self.model, self.actor, self.precision, device):
                for group in grouped.values():
                    ordered = sorted(group, key=lambda ref: ref.decision_index)
                    # Only one episode prefix is materialized at a time. It
                    # contains no recorded actions or RGB beyond its last anchor.
                    prefix = None
                    current = None
                    for ref in ordered:
                        key = self._key(ref, world_version, actor_version, device)
                        if key in self._cache:
                            current = self._cache[key].clone()
                            self._cache.move_to_end(key)
                            stats["cache_hits"] += 1
                        else:
                            stats["cache_misses"] += 1
                            if prefix is None:
                                prefix = sampler.read_prefix(ordered[-1])
                            earlier = self._nearest(key)
                            if earlier is not None and (current is None or
                                    earlier.world.decision_index.item() > current.world.decision_index.item()):
                                current = earlier
                            current = self._extend(prefix, ref.decision_index, current, world_version, device, stats)
                            self._remember(key, current)
                        snapshots[ref] = current.clone()
                ordered_snapshots = [snapshots[ref] for ref in refs]
                world = WorldState(*(torch.cat([getattr(snapshot.world, field) for snapshot in ordered_snapshots], dim=0)
                                     for field in ("memory", "stochastic", "logits", "decision_index")))
                tokens = self.model.core.representation(world).detach()
                remaining = (self.model.task.deadline - world.decision_index).float() / self.model.task.deadline
            # Detect unsupported concurrent mutations, rather than returning a
            # mixture of old and new versions stamped with one checkpoint ID.
            if module_version(self.model) != world_version or module_version(self.actor) != actor_version:
                raise RuntimeError("model or actor changed during anchor reconstruction; retry with frozen snapshots")
        except BaseException:
            self.clear_cache()
            raise
        stats.update(cache_entries=len(self._cache), cache_invalidations=self._invalidations)
        return self.batch_type(refs, world,
                           torch.stack([snapshot.hidden for snapshot in ordered_snapshots]),
                           torch.stack([snapshot.logits for snapshot in ordered_snapshots]),
                           torch.stack([snapshot.rgb for snapshot in ordered_snapshots]),
                           torch.stack([snapshot.previous for snapshot in ordered_snapshots]),
                           tokens, remaining, world_version, actor_version, sampler.cohort_identity,
                           self.seed, self.precision, stats)
