"""Detached teacher-action targets and local histories, never world fitting data.

The compact payload deliberately excludes teacher memories, rewards, values and
global tokens. Global states are retained as content identities for provenance;
the student's only inputs are its own RGB and actually executed past actions.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import re

import numpy as np
import torch

from world_models.imagination.replay import AnchorPrefix, AnchorRef
from world_models.imagination.versions import module_version
from world_models.mappo.recurrent import PrefixBank
from world_models.rollouts.schema import _stream_tree, _tree_copy
from world_models.teacher.learning import TeacherImaginedRollout, TeacherPrefixBank


TENSOR_FIELDS = ("local_rgb", "previous_actions", "actions", "valid", "terminated", "cutoff", "targets")


def _tensor_hash(value):
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(tensor.dtype), tuple(tensor.shape))).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class DistillationBuffer:
    local_rgb: torch.Tensor
    previous_actions: torch.Tensor
    actions: torch.Tensor
    valid: torch.Tensor
    terminated: torch.Tensor
    cutoff: torch.Tensor
    targets: torch.Tensor
    prefixes: PrefixBank
    metadata: dict
    streams: tuple

    def __post_init__(self):
        self.validate()

    @property
    def batch_size(self):
        return self.valid.shape[1]

    @property
    def steps(self):
        return self.valid.shape[0]

    @property
    def imagined_joint_steps(self):
        return int(self.valid.sum().item())

    def validate(self):
        if (not isinstance(self.valid, torch.Tensor) or self.valid.dtype != torch.bool
                or self.valid.ndim != 2 or min(self.valid.shape) < 1):
            raise ValueError("distillation valid must be nonempty boolean [T,B]")
        t, b = self.valid.shape
        if t > 15 or b > 128:
            raise ValueError("distillation permits at most 128 trajectories of at most 15 steps")
        specs = {"local_rgb": ((t, b, 2, 3, 96, 128), torch.uint8),
                 "previous_actions": ((t, b, 2), torch.long), "actions": ((t, b, 2), torch.long),
                 "valid": ((t, b), torch.bool), "terminated": ((t, b), torch.bool),
                 "cutoff": ((t, b), torch.bool), "targets": ((t, b, 2, 12), torch.float32)}
        for name, (shape, dtype) in specs.items():
            value = getattr(self, name)
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype
                    or value.device.type != "cpu" or value.layout != torch.strided
                    or value.requires_grad or value.grad_fn is not None or value.is_inference()):
                raise ValueError(f"{name} must be detached ordinary CPU {dtype} with shape {shape}")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite, including padding")
            if bool((value[~self.valid] != 0).any()):
                raise ValueError(f"{name} padding must be zero")
        if not bool(self.valid[0].all()) or not bool(self.valid[-1].any()):
            raise ValueError("every trajectory must begin with a valid step and have no all-padding final row")
        if bool((self.valid[1:] & ~self.valid[:-1]).any()):
            raise ValueError("valid masks must form uninterrupted prefixes")
        last = self.valid & ~torch.cat((self.valid[1:], torch.zeros_like(self.valid[:1])))
        if bool((self.terminated & self.cutoff).any()) or not torch.equal(self.terminated | self.cutoff, last):
            raise ValueError("each trajectory must end exactly once by termination or horizon cutoff")
        if bool(((self.actions[self.valid] < 0) | (self.actions[self.valid] >= 12)).any()):
            raise ValueError("actions must be executed teacher indices 0..11")
        if bool(((self.previous_actions[self.valid] < 0) | (self.previous_actions[self.valid] > 12)).any()):
            raise ValueError("previous actions must be executed indices or START=12")
        if not torch.equal(self.previous_actions[1:][self.valid[1:]], self.actions[:-1][self.valid[1:]]):
            raise ValueError("imagined previous actions must be the actually executed teacher actions")
        probabilities = self.targets[self.valid]
        if (bool(((probabilities < 0) | (probabilities > 1)).any())
                or not torch.allclose(probabilities.sum(-1), torch.ones_like(probabilities[..., 0]),
                                      rtol=0, atol=2e-6)):
            raise ValueError("valid soft targets must be normalized probabilities")
        if type(self.metadata) is not dict:
            raise ValueError("buffer metadata must be a dictionary")
        json.dumps(self.metadata, allow_nan=False)
        for key, expected in (("schema_version", "p42-synthetic-buffer-v1"),
                              ("data_source", "teacher_world_model_imagination"),
                              ("temperature", 1.0), ("real_interactions", 0),
                              ("world_model_fitting_eligible", False)):
            if type(self.metadata.get(key)) is not type(expected) or self.metadata.get(key) != expected:
                raise ValueError(f"buffer {key} must be {expected}")
        for key in ("world_model_version", "teacher_version", "cohort_identity"):
            if re.fullmatch(r"[a-f0-9]{64}", str(self.metadata.get(key))) is None:
                raise ValueError(f"buffer {key} must be a SHA256 identity")
        if type(self.prefixes) is not PrefixBank or len(self.prefixes.prefixes) != b:
            raise ValueError("buffer must own one safe full real prefix per trajectory")
        if (self.prefixes.cohort_identity != self.metadata["cohort_identity"]
                or self.prefixes.task_family != self.metadata.get("task_family")
                or self.prefixes.deadline != self.metadata.get("deadline")
                or [asdict(ref) for ref in self.prefixes.references] != self.metadata.get("anchor_references")):
            raise ValueError("prefix identities differ from the buffer metadata")
        for row, prefix in enumerate(self.prefixes.prefixes):
            previous = prefix.actions[-1] if len(prefix.actions) else np.full(2, 12, dtype=np.int64)
            if (not np.array_equal(prefix.rgb[-1], self.local_rgb[0, row].numpy())
                    or not np.array_equal(previous, self.previous_actions[0, row].numpy())):
                raise ValueError("anchor observation/actions differ from the complete real prefix")
        hashes = self.metadata.get("shared_state_sha256")
        if (type(hashes) is not list or len(hashes) != b
                or any(re.fullmatch(r"[a-f0-9]{64}", str(value)) is None for value in hashes)):
            raise ValueError("each trajectory must preserve a shared-state content identity")
        if type(self.streams) is not tuple or len(self.streams) < 1:
            raise ValueError("buffer must preserve collection sampling streams")
        _stream_tree(self.streams, "buffer_streams")

    @classmethod
    @torch.inference_mode(False)
    def from_teacher_rollout(cls, rollout, sampler, teacher, *, verification_batch_size=32):
        if type(rollout) is not TeacherImaginedRollout:
            raise ValueError("distillation requires a teacher-policy rollout")
        if type(verification_batch_size) is not int or verification_batch_size < 1:
            raise ValueError("verification batch size must be positive")
        rollout.validate()
        version = module_version(teacher)
        if version != rollout.metadata["actor_version"]:
            raise ValueError("teacher changed since the recorded trajectories")
        teacher_prefixes = TeacherPrefixBank.from_rollout(rollout, sampler)
        with torch.no_grad():
            for start in range(0, rollout.batch_size, verification_batch_size):
                rows = list(range(start, min(start + verification_batch_size, rollout.batch_size)))
                replayed = teacher_prefixes.replay(teacher, rollout, rows, precision=rollout.metadata["precision"])
                mask = rollout.valid[:, rows]
                if not torch.allclose(replayed[mask], rollout.actor_logits[:-1, rows][mask], rtol=1e-5, atol=2e-5):
                    raise ValueError("frozen teacher replay disagrees with recorded soft-target logits")
        if module_version(teacher) != version:
            raise RuntimeError("teacher changed during soft-target verification")
        targets = rollout.actor_logits[:-1].detach().float().softmax(-1)
        targets = torch.where(rollout.valid[:, :, None, None], targets, 0).cpu().clone()
        # Terminal destination images/logits are excluded: only action-source
        # observations are supervised, and every executed terminal action counts.
        tensors = {key: getattr(rollout, key)[:-1].detach().cpu().clone()
                   for key in ("local_rgb", "previous_actions")}
        tensors.update({key: getattr(rollout, key).detach().cpu().clone()
                        for key in ("actions", "valid", "terminated", "cutoff")})
        for key in ("local_rgb", "previous_actions"):
            tensors[key][~rollout.valid.cpu()] = 0
        metadata = {"schema_version": "p42-synthetic-buffer-v1", "temperature": 1.0,
                    "data_source": "teacher_world_model_imagination", "real_interactions": 0,
                    "world_model_fitting_eligible": False,
                    "teacher_version": version,
                    **{key: deepcopy(rollout.metadata[key]) for key in
                       ("world_model_version", "cohort_identity", "task_family", "deadline", "anchor_references")},
                    "shared_state_sha256": [_tensor_hash(rollout.shared_tokens[:, row])
                                            for row in range(rollout.batch_size)]}
        prefixes = PrefixBank(teacher_prefixes.prefixes, teacher_prefixes.cohort_identity,
                              teacher_prefixes.task_family, teacher_prefixes.deadline)
        return cls(**tensors, targets=targets, prefixes=prefixes, metadata=metadata,
                   streams=(_tree_copy({"before": rollout.streams_before, "after": rollout.streams_after,
                                        "anchor_metadata": rollout.metadata.get("anchor_metadata", {})}),))

    @classmethod
    @torch.inference_mode(False)
    def concat(cls, chunks):
        chunks = tuple(chunks)
        if not chunks or any(type(chunk) is not cls for chunk in chunks):
            raise ValueError("concat requires nonempty distillation buffers")
        first, horizon = chunks[0], max(chunk.steps for chunk in chunks)
        for chunk in chunks:
            chunk.validate()
            for key in ("teacher_version", "world_model_version", "cohort_identity", "task_family", "deadline"):
                if chunk.metadata[key] != first.metadata[key]:
                    raise ValueError("all buffer shards must use the same frozen teacher/model/replay cohort")
        tensors = {}
        for name in TENSOR_FIELDS:
            padded = []
            for chunk in chunks:
                value = getattr(chunk, name)
                if chunk.steps != horizon:
                    value = torch.cat((value, value.new_zeros((horizon - chunk.steps, *value.shape[1:]))))
                padded.append(value)
            tensors[name] = torch.cat(padded, dim=1)
        metadata = deepcopy(first.metadata)
        for key in ("anchor_references", "shared_state_sha256"):
            metadata[key] = [deepcopy(value) for chunk in chunks for value in chunk.metadata[key]]
        prefixes = PrefixBank(tuple(prefix for chunk in chunks for prefix in chunk.prefixes.prefixes),
                              first.prefixes.cohort_identity, first.prefixes.task_family, first.prefixes.deadline)
        return cls(**tensors, prefixes=prefixes, metadata=metadata,
                   streams=tuple(_tree_copy(stream) for chunk in chunks for stream in chunk.streams))

    @torch.inference_mode(False)
    def state_dict(self):
        self.validate()
        return {"schema_version": "p42-buffer-payload-v1",
                "tensors": {key: getattr(self, key).detach().clone() for key in TENSOR_FIELDS},
                "prefixes": [{"reference": asdict(prefix.ref),
                              "rgb": torch.from_numpy(prefix.rgb.copy()),
                              "actions": torch.from_numpy(prefix.actions.copy()),
                              "decision_indices": torch.from_numpy(prefix.decision_indices.copy())}
                             for prefix in self.prefixes.prefixes],
                "metadata": deepcopy(self.metadata), "streams": _tree_copy(self.streams)}

    @classmethod
    @torch.inference_mode(False)
    def from_state_dict(cls, state):
        if (type(state) is not dict or set(state) != {"schema_version", "tensors", "prefixes", "metadata", "streams"}
                or state["schema_version"] != "p42-buffer-payload-v1"
                or type(state["tensors"]) is not dict or set(state["tensors"]) != set(TENSOR_FIELDS)):
            raise ValueError("invalid distillation buffer payload")
        metadata = deepcopy(state["metadata"])
        prefixes = []
        for value in state["prefixes"]:
            if type(value) is not dict or set(value) != {"reference", "rgb", "actions", "decision_indices"}:
                raise ValueError("invalid safe prefix payload")
            for key in ("rgb", "actions", "decision_indices"):
                if (not isinstance(value[key], torch.Tensor) or value[key].device.type != "cpu"
                        or value[key].requires_grad or value[key].grad_fn is not None or value[key].is_inference()):
                    raise ValueError("safe prefix payload tensors must be detached ordinary CPU tensors")
            prefixes.append(AnchorPrefix(AnchorRef(**value["reference"]),
                                         **{key: value[key].numpy() for key in ("rgb", "actions", "decision_indices")}))
        bank = PrefixBank(tuple(prefixes), metadata["cohort_identity"], metadata["task_family"], metadata["deadline"])
        for value in state["tensors"].values():
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.requires_grad
                    or value.grad_fn is not None or value.is_inference()):
                raise ValueError("buffer payload requires detached ordinary CPU tensors")
        return cls(**{key: value.detach().clone() for key, value in state["tensors"].items()},
                   prefixes=bank, metadata=metadata, streams=_tree_copy(state["streams"]))
