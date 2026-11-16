"""Independent, auditable sampling streams for on-policy imagined rollouts."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Mapping

import torch

from world_models.infrastructure.config import validate_config
from world_models.infrastructure.reproducibility import canonical_json, derive_seed
from world_models.models.world_model import SamplingStreams


_VERSION = "p33-rollout-streams-v1"
_WORLD_NAMES = ("prior", "posterior", "outcome")
_SEED_NAMES = {"world", *_WORLD_NAMES, "actor_0", "actor_1"}


def _seed(value: int) -> int:
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError("seed must be a nonnegative signed 64-bit integer")
    return value


def _named_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{_VERSION}|{seed}|{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 2**63


def _world_seed(seed: int, name: str) -> int:
    # SamplingStreams owns this existing namespace; reproduce it for the audit.
    digest = hashlib.sha256(f"p22-sampling-v1|{seed}|{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 2**63


def _device(value: str | torch.device) -> torch.device:
    device = torch.device(value)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("rollout sampling supports CPU or CUDA generators")
    if device.type == "cpu":
        return torch.device("cpu")
    if device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


class RolloutStreams:
    """Three world streams and two agent streams, without global RNG draws.

    The posterior stream is retained for exact audit/resume, although imagined
    trajectory generation must never consume it. Serializing owns every tensor;
    loading validates all five candidate states before changing any live stream.
    """

    def __init__(self, world: SamplingStreams,
                 actors: tuple[torch.Generator, torch.Generator], identity: dict):
        self.world = world
        self.actors = actors
        try:
            canonical_json(identity)
        except (TypeError, ValueError) as exc:
            raise ValueError("stream identity must be finite JSON data") from exc
        self.identity = deepcopy(identity)
        self._construction_identity = deepcopy(identity)
        self.validate()

    @classmethod
    def _create(cls, world_seed: int, actor_seeds: tuple[int, int],
                device: str | torch.device, **metadata) -> "RolloutStreams":
        target = _device(device)
        seeds = {"world": _seed(world_seed),
                 **{name: _world_seed(world_seed, name) for name in _WORLD_NAMES},
                 **{f"actor_{agent}": _seed(value) for agent, value in enumerate(actor_seeds)}}
        identity = {"schema_version": _VERSION, "device": str(target), "seeds": seeds, **metadata}
        return cls(SamplingStreams.from_seed(world_seed, device=target),
                   tuple(torch.Generator(device=target).manual_seed(seed) for seed in actor_seeds),
                   identity)

    @classmethod
    def from_seed(cls, seed: int, device: str | torch.device = "cpu") -> "RolloutStreams":
        """Create deterministic named components for fixtures or explicit runs."""
        _seed(seed)
        return cls._create(_named_seed(seed, "world"),
                           tuple(_named_seed(seed, f"actor_{agent}") for agent in (0, 1)),
                           device)

    @classmethod
    def from_experiment(cls, experiment: Mapping, *, master_seed: int,
                        device: str | torch.device = "cpu") -> "RolloutStreams":
        """Keep B/D initialization paired while their training draws differ."""
        validate_config(experiment)
        task, variant = experiment["task"]["name"], experiment["variant"]["name"]
        _seed(master_seed)
        return cls._create(derive_seed(task, master_seed, "imagination", variant),
                           tuple(derive_seed(task, master_seed, "local_action_sampling", variant, agent)
                                 for agent in (0, 1)), device,
                           task_family=task, variant=variant, master_seed=master_seed)

    def _generators(self) -> dict[str, torch.Generator]:
        if type(self.world) is not SamplingStreams:
            raise ValueError("world must be SamplingStreams")
        if type(self.actors) is not tuple or len(self.actors) != 2:
            raise ValueError("actors must contain exactly two independent generators")
        return {**{name: getattr(self.world, name) for name in _WORLD_NAMES},
                **{f"actor_{agent}": value for agent, value in enumerate(self.actors)}}

    def validate(self, device: str | torch.device | None = None) -> None:
        """Reject aliased generators, changed identity, and device/seed mismatch."""
        identity = self.identity
        required = {"schema_version", "device", "seeds"}
        optional = {"task_family", "variant", "master_seed"}
        if type(identity) is not dict or set(identity) not in (required, required | optional):
            raise ValueError("stream identity has missing or unknown fields")
        try:
            if canonical_json(identity) != canonical_json(self._construction_identity):
                raise ValueError("stream identity changed after construction")
        except (TypeError, ValueError) as exc:
            raise ValueError("stream identity is invalid or changed after construction") from exc
        if identity["schema_version"] != _VERSION or type(identity["device"]) is not str:
            raise ValueError("unsupported stream identity schema/device")
        seeds = identity["seeds"]
        if type(seeds) is not dict or set(seeds) != _SEED_NAMES:
            raise ValueError("stream identity must record each component seed")
        for value in seeds.values():
            _seed(value)
        if len(set(seeds.values())) != len(seeds):
            raise ValueError("stream component seeds must be distinct")
        if any(seeds[name] != _world_seed(seeds["world"], name) for name in _WORLD_NAMES):
            raise ValueError("world stream seeds disagree with the named namespace")
        if optional <= identity.keys():
            task, variant, master = (identity[name] for name in ("task_family", "variant", "master_seed"))
            if (seeds["world"] != derive_seed(task, master, "imagination", variant)
                    or any(seeds[f"actor_{agent}"] != derive_seed(
                        task, master, "local_action_sampling", variant, agent) for agent in (0, 1))):
                raise ValueError("experiment stream namespaces disagree with identity")
        target = _device(identity["device"])
        if str(target) != identity["device"]:
            raise ValueError("identity device must be explicit and canonical")
        if device is not None and _device(device) != target:
            raise ValueError("sampling streams do not match requested device")
        generators = self._generators()
        if any(not isinstance(generator, torch.Generator) for generator in generators.values()):
            raise ValueError("all sampling streams must be torch generators")
        if len({id(generator) for generator in generators.values()}) != len(generators):
            raise ValueError("sampling streams must not alias generator objects")
        for name, generator in generators.items():
            if _device(generator.device) != target:
                raise ValueError("sampling generator device disagrees with identity")
            if generator.initial_seed() != seeds[name]:
                raise ValueError("sampling generator seed disagrees with identity")

    def state_dict(self) -> dict:
        """Return owned, serializable generator states and a JSON identity."""
        self.validate()
        return {"schema_version": _VERSION, "identity": deepcopy(self.identity),
                "world": self.world.state_dict(),
                "actors": {str(agent): generator.get_state().clone()
                           for agent, generator in enumerate(self.actors)}}

    def load_state_dict(self, state: dict) -> None:
        """Restore exactly this experiment/device, with transactional validation."""
        self.validate()
        if type(state) is not dict or set(state) != {"schema_version", "identity", "world", "actors"}:
            raise ValueError("rollout RNG state has missing or unknown fields")
        try:
            matches = canonical_json(state["identity"]) == canonical_json(self.identity)
        except (TypeError, ValueError) as exc:
            raise ValueError("rollout RNG identity must be finite JSON data") from exc
        if state["schema_version"] != _VERSION or not matches:
            raise ValueError("rollout RNG state identity/schema does not match")
        if type(state["world"]) is not dict or set(state["world"]) != set(_WORLD_NAMES):
            raise ValueError("world RNG states must contain prior/posterior/outcome")
        if type(state["actors"]) is not dict or set(state["actors"]) != {"0", "1"}:
            raise ValueError("actor RNG states must contain exactly 0/1")
        values = {**state["world"],
                  **{f"actor_{agent}": state["actors"][str(agent)] for agent in (0, 1)}}
        candidate_states = {}
        for name, value in values.items():
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.uint8
                    or value.device.type != "cpu" or value.ndim != 1 or value.layout != torch.strided):
                raise ValueError("generator state must be a one-dimensional CPU byte tensor")
            candidate = torch.Generator(device=self.identity["device"])
            try:
                candidate.set_state(value.clone())
            except (RuntimeError, TypeError) as exc:
                raise ValueError(f"malformed {name} generator state") from exc
            if candidate.initial_seed() != self.identity["seeds"][name]:
                raise ValueError(f"{name} state belongs to a different component seed")
            candidate_states[name] = candidate.get_state().clone()
        # Candidate creation/set_state above cannot advance any live stream.
        for name, generator in self._generators().items():
            generator.set_state(candidate_states[name])
