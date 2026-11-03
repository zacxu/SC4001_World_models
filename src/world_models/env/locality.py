"""A local-call adapter and Phase 1 information-flow contract probe.

The probe is deliberately not a learned research actor. The same audit must be
repeated with actual B/D deployment policies when Phase 3 supplies them.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import math
import struct

from world_models.contracts import LocalActorInput, LocalActorOutput, RGBFrame


def actor_step(actor, local_rgb: RGBFrame, previous_action: int, local_hidden: tuple[float, ...]) -> LocalActorOutput:
    output = actor.step(LocalActorInput(local_rgb, previous_action, local_hidden))
    if not isinstance(output, LocalActorOutput):
        raise TypeError("a local actor must return LocalActorOutput")
    if len(output.next_local_hidden) != len(local_hidden):
        raise ValueError("actor changed its configured recurrent-memory size")
    return output


class LocalContractProbe:
    """Deterministic pixel/action/memory-dependent actor for adapter testing."""

    def step(self, inputs: LocalActorInput) -> LocalActorOutput:
        digest = hashlib.sha256(inputs.local_rgb.data + bytes([inputs.previous_action])
                                + struct.pack(f"<{len(inputs.local_hidden)}d", *inputs.local_hidden)).digest()
        weights = [float(value + 1) for value in digest[:12]]
        total = sum(weights)
        hidden = tuple(math.tanh(value + (digest[index % len(digest)] - 127) / 255)
                       for index, value in enumerate(inputs.local_hidden))
        return LocalActorOutput(tuple(value / total for value in weights), hidden)


def audit_locality(history: list, *, hidden_size: int = 256, actor=None) -> dict:
    """Hold local history fixed while replacing all metadata.

    Supplied actors must return deterministic action distributions in evaluation
    mode and keep recurrent state in the explicit memory argument. This audit
    deliberately does not conflate sampled-action randomness with information
    leakage.
    """
    if not history or type(hidden_size) is not int or hidden_size < 1:
        raise ValueError("provide a nonempty native history and a valid hidden size")
    supplied_actor = actor is not None
    probe = actor if supplied_actor else LocalContractProbe()
    results = []
    for player in (0, 1):
        ordinary_hidden = changed_hidden = (0.0,) * hidden_size
        ordinary, changed, image_hashes, action_history = [], [], [], []
        for index, result in enumerate(history):
            previous = result.metadata.actions[player] if hasattr(result,"metadata") else 12
            frame = result.observations[player]
            if frame is None:
                raise ValueError("locality audit requires valid local images")
            # The altered result has different global metadata AND teammate RGB.
            other = RGBFrame(bytes([index % 256]) * len(frame.data))
            observations = list(result.observations)
            observations[1 - player] = other
            altered = replace(result, observations=tuple(observations), training_only_info={
                "positions": [999999, -999999], "teacher_tokens": [float(index + 1)],
                "reward": 12345, "role": 1-player, "arbitrary_global_metadata": object()})
            first = actor_step(probe, frame, previous, ordinary_hidden)
            second = actor_step(probe, altered.observations[player], previous, changed_hidden)
            if first != second:
                raise AssertionError("global metadata/teammate RGB changed a fixed local call")
            ordinary.append(first.action_probabilities)
            changed.append(second.action_probabilities)
            ordinary_hidden, changed_hidden = first.next_local_hidden, second.next_local_hidden
            action_history.append(previous)
            image_hashes.append(hashlib.sha256(frame.data).hexdigest())
        first_frame = history[0].observations[player]
        changed_bytes = bytes([first_frame.data[0] ^ 255]) + first_frame.data[1:]
        control = actor_step(probe, first_frame, 12, (0.0,) * hidden_size)
        local_change = actor_step(probe, RGBFrame(changed_bytes), 12, (0.0,) * hidden_size)
        if not supplied_actor and control == local_change:
            raise AssertionError("locality probe failed its own-image positive control")
        results.append({"player": player, "history_length": len(history), "image_sha256": image_hashes,
                        "executed_previous_action_history":action_history,
                        "outputs_equal_after_global_and_teammate_changes": ordinary == changed,
                        "own_image_change_changes_output": control != local_change,
                        "final_memories_equal": ordinary_hidden == changed_hidden})
    return {"status": "passed", "players": results,
            "actor": type(probe).__name__,
            "scope": "Supplied actor isolation check." if supplied_actor else "Phase 1 adapter/contract isolation using a pixel/action/memory-dependent probe. No trained B/D actor exists yet; rerun with deployment actors in Phase 3."}
