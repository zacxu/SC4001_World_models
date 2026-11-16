"""Student-only forward KL over complete recorded teacher trajectories."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from world_models.imagination.versions import module_version
from world_models.mappo.recurrent import _indices, _ordinary_replay, _replay_mode
from world_models.policies.actor import LocalActor
from world_models.policies.critic import CentralizedCritic
from .buffer import DistillationBuffer


@dataclass(frozen=True)
class DistillationConfig:
    updates: int = 100
    sequences_per_update: int = 32
    learning_rate: float = 3e-4
    epsilon: float = 1e-5
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 0.0
    gradient_clip: float = 0.5

    def __post_init__(self):
        for name in ("updates", "sequences_per_update"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "epsilon", "gradient_clip"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (type(self.betas) is not tuple or len(self.betas) != 2
                or any(type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value < 1
                       for value in self.betas) or self.weight_decay != 0):
            raise ValueError("Adam requires two finite betas in [0,1) and zero weight decay")


@_ordinary_replay
def replay_student(actor, buffer, indices):
    """Current student's own real history followed by full synthetic sequences.

    No hidden states are stored in the buffer. Prefix work is recomputed with
    current parameters under no_grad, excluding the anchor image. The anchor
    and every subsequent source image are consumed exactly once with BPTT.
    """
    if type(actor) is not LocalActor or type(buffer) is not DistillationBuffer:
        raise ValueError("student replay requires a local actor and a synthetic distillation buffer")
    selected = _indices(indices, buffer.batch_size)
    device = actor.gru.weight_hh.device
    if device.type not in ("cpu", "cuda") or any(
            parameter.device != device or parameter.dtype != torch.float32 for parameter in actor.parameters()):
        raise ValueError("student parameters must be FP32 on one CPU or CUDA device")
    prefixes = tuple(buffer.prefixes.prefixes[index] for index in selected)
    with _replay_mode(actor, precision="float32", device=device):
        with torch.no_grad():
            hidden = actor.initial_hidden(len(selected) * 2).reshape(len(selected), 2, -1)
            for step in range(max(prefix.ref.decision_index for prefix in prefixes)):
                active = [row for row, prefix in enumerate(prefixes) if step < prefix.ref.decision_index]
                rgb = torch.from_numpy(np.stack([prefixes[row].rgb[step] for row in active])).to(device)
                previous = torch.from_numpy(np.stack([
                    prefixes[row].actions[step - 1] if step else np.full(2, 12, dtype=np.int64)
                    for row in active])).to(device)
                rows = torch.tensor(active, dtype=torch.long, device=device)
                output = actor(rgb.flatten(0, 1), previous.flatten(), hidden[rows].flatten(0, 1))
                hidden[rows] = output.next_local_hidden.reshape(len(active), 2, -1)
            initial = hidden.detach().reshape(len(selected) * 2, -1)
        # The CPU buffer remains fixed. Only this full-sequence minibatch moves
        # to the student device; no teacher/global state is on this path.
        rows = torch.tensor(selected, dtype=torch.long)
        images = buffer.local_rgb[:, rows].to(device).flatten(1, 2)
        previous = buffer.previous_actions[:, rows].to(device).flatten(1, 2)
        valid = buffer.valid[:, rows, None].expand(-1, -1, 2).to(device).flatten(1, 2)
        result = actor.forward_sequence(images, previous, initial, valid=valid)
        return result.logits.reshape(buffer.steps, len(selected), 2, actor.config.action_count)


def forward_kl(logits, targets, valid):
    """Mean KL(teacher || student) over valid player decisions, in FP32."""
    if (not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor)
            or not isinstance(valid, torch.Tensor) or logits.dtype != torch.float32 or logits.ndim != 4
            or logits.shape[-2:] != (2, 12) or targets.shape != logits.shape or targets.dtype != torch.float32
            or valid.shape != logits.shape[:2] or valid.dtype != torch.bool
            or targets.device != logits.device or valid.device != logits.device):
        raise ValueError("forward KL expects FP32 [T,B,2,12] logits/targets and bool [T,B] masks")
    if targets.requires_grad or targets.grad_fn is not None:
        raise ValueError("teacher targets must be detached")
    if not bool(valid.any()) or not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(targets).all()):
        raise ValueError("forward KL requires finite tensors and at least one valid step")
    probabilities = targets[valid]
    if (bool(((probabilities < 0) | (probabilities > 1)).any())
            or not torch.allclose(probabilities.sum(-1), torch.ones_like(probabilities[..., 0]), rtol=0, atol=2e-6)):
        raise ValueError("valid teacher soft targets must be normalized probabilities")
    log_student = logits[valid].log_softmax(-1)
    terms = torch.xlogy(probabilities, probabilities) - probabilities * log_student
    loss = terms.sum(-1).mean()
    entropy = -(log_student.exp() * log_student).sum(-1).mean()
    return loss, entropy


class StudentDistiller:
    """Dedicated supervised optimizer; never a teacher or MAPPO optimizer.

    Construction checks the saved paired B initialization. For resume, construct
    from that initialization, load the current saved actor, then load this state.
    Downstream local PPO must construct fresh optimizers after warm-start.
    """
    def __init__(self, actor, critic, *, paired_actor_version, paired_critic_version,
                 seed, config=DistillationConfig()):
        if (type(actor) is not LocalActor or type(critic) is not CentralizedCritic
                or not isinstance(config, DistillationConfig)):
            raise ValueError("distillation requires a local actor, centralized critic and DistillationConfig")
        if type(seed) is not int or not 0 <= seed < 2 ** 63:
            raise ValueError("distillation sampling seed must be an integer in [0,2**63)")
        if module_version(actor) != paired_actor_version or module_version(critic) != paired_critic_version:
            raise ValueError("student actor and critic must exactly match B's paired initial hashes")
        if any(parameter.requires_grad or parameter.grad is not None for parameter in critic.parameters()):
            raise ValueError("student critic must be frozen with no accumulated gradients")
        actor_storage = {parameter.untyped_storage().data_ptr() for parameter in actor.parameters()}
        if any(parameter.untyped_storage().data_ptr() in actor_storage for parameter in critic.parameters()):
            raise ValueError("student actor and critic must not share parameter storage")
        self.actor, self.critic, self.config = actor, critic, config
        self.paired_actor_version, self.paired_critic_version = paired_actor_version, paired_critic_version
        self.seed = seed
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate,
                                          eps=config.epsilon, betas=config.betas,
                                          weight_decay=config.weight_decay)
        self.updates_completed = 0
        self._failed = False

    def _guard(self, buffer):
        if self._failed:
            raise RuntimeError("distillation learner failed; reconstruct from the last committed checkpoint")
        if type(buffer) is not DistillationBuffer:
            raise ValueError("student distillation requires a synthetic fixed buffer")
        buffer.validate()
        if module_version(self.critic) != self.paired_critic_version:
            raise ValueError("student critic changed during the supervised-only stage")
        if any(parameter.requires_grad or parameter.grad is not None for parameter in self.critic.parameters()):
            raise ValueError("student critic must remain frozen with no accumulated gradients")
        if any(not parameter.requires_grad for parameter in self.actor.parameters()):
            raise ValueError("all local student parameters must be trainable")

    @_ordinary_replay
    def step(self, buffer):
        self._guard(buffer)
        if self.updates_completed >= self.config.updates:
            raise ValueError("registered supervised update budget is already complete")
        if buffer.batch_size < self.config.sequences_per_update:
            raise ValueError("fixed buffer is smaller than the full-sequence minibatch")
        rng_before = self.generator.get_state().clone()
        try:
            indices = torch.randperm(buffer.batch_size, generator=self.generator)[:self.config.sequences_per_update].tolist()
            logits = replay_student(self.actor, buffer, indices)
            device = logits.device
            valid = buffer.valid[:, indices].to(device)
            targets = buffer.targets[:, indices].to(device)
            loss, entropy = forward_kl(logits, targets, valid)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip,
                                                 error_if_nonfinite=True)
            self.optimizer.step()
            if any(not bool(torch.isfinite(parameter).all()) for parameter in self.actor.parameters()):
                raise RuntimeError("nonfinite student parameters after distillation update")
            self.updates_completed += 1
            return {"schema_version": "p42-student-update-v1", "optimizer_update": self.updates_completed,
                    "sequence_indices": indices, "valid_player_decisions": 2 * int(valid.sum().item()),
                    "forward_kl": float(loss.detach()), "student_entropy": float(entropy.detach()),
                    "gradient_norm": float(norm), "teacher_optimizer_steps": 0,
                    "critic_optimizer_steps": 0, "model_optimizer_steps": 0,
                    "new_environment_interactions": 0}
        except BaseException:
            self._failed = True
            self.generator.set_state(rng_before)
            self.optimizer.zero_grad(set_to_none=True)
            raise

    @torch.inference_mode(False)
    @torch.no_grad()
    def diagnostics(self, buffer, *, batch_size=32):
        self._guard(buffer)
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("diagnostic minibatch size must be positive")
        count, kl_total, entropy_total = 0, 0.0, 0.0
        for start in range(0, buffer.batch_size, batch_size):
            indices = list(range(start, min(start + batch_size, buffer.batch_size)))
            logits = replay_student(self.actor, buffer, indices)
            valid = buffer.valid[:, indices].to(logits.device)
            loss, entropy = forward_kl(logits, buffer.targets[:, indices].to(logits.device), valid)
            decisions = 2 * int(valid.sum().item())
            count += decisions
            kl_total += float(loss) * decisions
            entropy_total += float(entropy) * decisions
        return {"schema_version": "p42-training-diagnostics-v1", "forward_kl": kl_total / count,
                "student_entropy": entropy_total / count, "valid_player_decisions": count,
                "trajectories": buffer.batch_size, "optimizer_updates": self.updates_completed,
                "scope": "fixed training buffer; not validation or real performance"}

    def state_dict(self):
        if self._failed:
            raise RuntimeError("cannot save a failed distillation learner; resume the last committed checkpoint")
        return {"schema_version": "p42-student-optimizer-v1", "config": asdict(self.config),
                "updates_completed": self.updates_completed,
                "paired_actor_version": self.paired_actor_version,
                "paired_critic_version": self.paired_critic_version,
                "current_actor_version": module_version(self.actor),
                "sampling_seed": self.seed,
                "sampling_rng": self.generator.get_state().clone(),
                "optimizer": deepcopy(self.optimizer.state_dict()), "next_local_ppo_optimizer": "fresh"}

    def load_state_dict(self, state):
        if self._failed:
            raise RuntimeError("failed learner must be reconstructed before loading a committed checkpoint")
        expected = {"schema_version", "config", "updates_completed", "paired_actor_version", "paired_critic_version",
                    "current_actor_version", "sampling_seed", "sampling_rng", "optimizer", "next_local_ppo_optimizer"}
        if (type(state) is not dict or set(state) != expected or state["schema_version"] != "p42-student-optimizer-v1"
                or state["config"] != asdict(self.config) or state["next_local_ppo_optimizer"] != "fresh"
                or state["paired_actor_version"] != self.paired_actor_version
                or state["paired_critic_version"] != self.paired_critic_version
                or type(state["sampling_seed"]) is not int or state["sampling_seed"] != self.seed
                or state["current_actor_version"] != module_version(self.actor)
                or module_version(self.critic) != self.paired_critic_version
                or type(state["updates_completed"]) is not int
                or not 0 <= state["updates_completed"] <= self.config.updates):
            raise ValueError("distillation optimizer state differs from the actor, critic or registered schedule")
        probe = torch.Generator(device="cpu")
        probe.set_state(state["sampling_rng"])
        if probe.initial_seed() != self.seed:
            raise ValueError("saved supervised sampling RNG belongs to a different stream seed")
        raw = state["optimizer"]
        parameters = tuple(self.actor.parameters())
        if (type(raw) is not dict or set(raw) != {"state", "param_groups"}
                or type(raw["state"]) is not dict or type(raw["param_groups"]) is not list
                or len(raw["param_groups"]) != 1 or type(raw["param_groups"][0]) is not dict
                or raw["param_groups"][0].get("params") != list(range(len(parameters)))):
            raise ValueError("saved supervised optimizer parameter order differs from the complete student")
        expected_slots = set(range(len(parameters))) if state["updates_completed"] else set()
        if set(raw["state"]) != expected_slots:
            raise ValueError("saved supervised optimizer must contain moments for every student parameter")
        for index, values in raw["state"].items():
            for key in ("exp_avg", "exp_avg_sq"):
                value = values.get(key) if isinstance(values, dict) else None
                if (not isinstance(value, torch.Tensor) or value.shape != parameters[index].shape
                        or value.dtype != parameters[index].dtype):
                    raise ValueError("saved supervised optimizer moment shape/dtype mismatch")
        optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.learning_rate,
                                     eps=self.config.epsilon, betas=self.config.betas, weight_decay=0.)
        optimizer.load_state_dict(deepcopy(state["optimizer"]))
        groups = optimizer.param_groups
        if len(groups) != 1 or any(groups[0][key] != expected for key, expected in
                                  (("lr", self.config.learning_rate), ("eps", self.config.epsilon),
                                   ("betas", self.config.betas), ("weight_decay", 0.),
                                   ("amsgrad", False), ("maximize", False), ("capturable", False),
                                   ("differentiable", False), ("decoupled_weight_decay", False))):
            raise ValueError("saved supervised Adam hyperparameters differ from the registration")
        if state["updates_completed"] and set(optimizer.state) != set(parameters):
            raise ValueError("saved supervised optimizer must contain moments for every student parameter")
        for parameter, values in optimizer.state.items():
            if set(values) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("saved supervised optimizer has invalid moment fields")
            for key, value in values.items():
                if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
                    raise ValueError("nonfinite saved supervised optimizer state")
            if (values["step"].shape != torch.Size([]) or values["step"].item() != state["updates_completed"]):
                raise ValueError("saved supervised optimizer steps disagree with the update count")
            for key in ("exp_avg", "exp_avg_sq"):
                value = values[key]
                if value.shape != parameter.shape or value.dtype != parameter.dtype or value.device != parameter.device:
                    raise ValueError("saved supervised optimizer moment shape/dtype/device mismatch")
            if bool((values["exp_avg_sq"] < 0).any()):
                raise ValueError("saved supervised optimizer second moments must be nonnegative")
        if bool(optimizer.state) != bool(state["updates_completed"]):
            raise ValueError("saved supervised optimizer moments disagree with the update count")
        self.optimizer, self.generator = optimizer, probe
        self.updates_completed = state["updates_completed"]
