"""Stable, explicit RNG namespaces for the paired B/D research protocol.

Derivation is independent of Python's randomized ``hash()`` and of the order in
which components request seeds.  Callers must still initialize their own RNGs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


TASK_FAMILIES = frozenset({"relay", "cover"})
VARIANTS = frozenset({"B", "D"})
SEED_DERIVATION_VERSION = "world-models-seeds-v1"
PAIRED_INITIAL_COMPONENTS = frozenset(
    {
        "initial_collection",
        "world_model_initialization",
        "world_model_fitting_initial",
        "local_actor_initialization",
        "local_critic_initialization",
    }
)


def canonical_json(value: Any) -> str:
    """Serialize an explicitly JSON-compatible value reproducibly.

    NaN and infinity are rejected rather than creating nonstandard manifests.
    """
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def canonical_hash(value: Any) -> str:
    """Return a SHA-256 identifier for a canonical JSON value."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def derive_seed(
    task_family: str,
    master_seed: int,
    component: str,
    variant: str,
    agent_id: int | None = None,
) -> int:
    """Derive a stable uint32 seed for one component's random stream.

    Only the declared paired-initialization components ignore the B/D branch.
    Different task families and agent IDs always have distinct namespaces.
    ``agent_id`` is zero-based and is not an actor input or communication channel.
    """
    if not isinstance(task_family, str) or task_family not in TASK_FAMILIES:
        raise ValueError("task_family must be 'relay' or 'cover'")
    if not isinstance(variant, str) or variant not in VARIANTS:
        raise ValueError("variant must be 'B' or 'D'")
    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise ValueError("master_seed must be an integer")
    if not 0 <= master_seed <= 2**63 - 1:
        raise ValueError("master_seed must be between 0 and 2**63 - 1")
    if not isinstance(component, str) or not component or component.strip() != component:
        raise ValueError("component must be a nonempty name without surrounding whitespace")
    if len(component) > 128 or not component.isprintable():
        raise ValueError("component name must be printable and at most 128 characters")
    if agent_id is not None and (
        isinstance(agent_id, bool) or not isinstance(agent_id, int) or agent_id not in (0, 1)
    ):
        raise ValueError("agent_id must be 0, 1, or None")

    namespace = {
        "version": SEED_DERIVATION_VERSION,
        "task_family": task_family,
        "master_seed": master_seed,
        "component": component,
        "branch": "paired-initialization" if component in PAIRED_INITIAL_COMPONENTS else variant,
        "agent_id": agent_id,
    }
    digest = hashlib.sha256(canonical_json(namespace).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def seed_manifest(task_family: str, master_seed: int, variant: str) -> dict[str, Any]:
    """Describe the initial and independent streams required by the protocol."""
    components = sorted(PAIRED_INITIAL_COMPONENTS) + [
        "real_collection",
        "world_model_fitting_online",
        "imagination",
        "local_policy_updates",
        "teacher_initialization",
        "teacher_rollouts",
        "teacher_updates",
        "distillation",
    ]
    return {
        "derivation_version": SEED_DERIVATION_VERSION,
        "task_family": task_family,
        "master_seed": master_seed,
        "variant": variant,
        "components": {
            component: derive_seed(task_family, master_seed, component, variant)
            for component in components
        },
        "local_action_sampling": {
            str(agent): derive_seed(task_family, master_seed, "local_action_sampling", variant, agent)
            for agent in (0, 1)
        },
        "paired_initial_components": sorted(PAIRED_INITIAL_COMPONENTS),
        "note": (
            "Evaluation uses a separately frozen paired bank; training branch streams "
            "must not select test scenarios. Distinct uint32 hashes are not a proof "
            "of statistical independence, and manifests should be checked for collisions."
        ),
    }
