"""Explicit global-teacher adapters for guarded prefixes and recurrent MAPPO.

Teacher buffers have their own schema and role. The local training path cannot
consume them. Only the current action head sees shared model tokens; recurrent
history always reconstructs each player's own images and executed actions.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json

import numpy as np
import torch

from world_models.contracts import PREVIOUS_ACTION_START
from world_models.imagination.anchors import AnchorBatch, AnchorBuilder, _reconstruction
from world_models.imagination.versions import module_version
from world_models.infrastructure.reproducibility import derive_seed
from world_models.mappo.config import MAPPOConfig
from world_models.mappo.recurrent import PrefixBank, _indices, _ordinary_replay, _replay_mode
from world_models.mappo.updater import RecurrentMAPPO
from world_models.policies.teacher import GlobalTeacher
from world_models.policies.actor import ActorOutput
from world_models.rollouts.generator import RolloutGenerator
from world_models.rollouts.schema import ImaginedRollout


ROLE = "global_teacher"


def _paired_context(tokens):
    """Expand current joint Z to its two player rows, preserving player order."""
    batch, count, width = tokens.shape
    repeated = tokens[:, None].expand(batch, 2, count, width).reshape(batch * 2, count, width)
    players = torch.arange(2, dtype=torch.long, device=tokens.device).repeat(batch)
    return repeated, players


class TeacherAnchorBatch(AnchorBatch):
    def metadata(self):
        return {**super().metadata(), "schema_version": "p41-teacher-anchor-v1", "policy_role": ROLE}


class TeacherAnchorBuilder(AnchorBuilder):
    """Full real-prefix local teacher memory and current global action logits.

    The inherited content-addressed cache hashes the entire teacher, including
    attention and action parameters. Head changes therefore invalidate logits
    even when the underlying local CNN/GRU weights have not changed.
    """
    actor_type = GlobalTeacher
    batch_type = TeacherAnchorBatch

    def _actor_step(self, rgb, previous, hidden, world):
        # Prefix snapshots cache only recurrence. The global head is evaluated
        # once per requested anchor below, rather than at every real-history step.
        memory = self.actor.memory_step(rgb, previous, hidden)
        return ActorOutput(memory.new_zeros((len(memory), self.actor.config.action_count)), memory)

    @torch.inference_mode(False)
    def build(self, sampler, references=None):
        anchors = super().build(sampler, references)
        try:
            with _reconstruction(self.model, self.actor, self.precision, anchors.actor_hidden.device):
                tokens, players = _paired_context(anchors.shared_tokens)
                output = self.actor.action_from_memory(anchors.actor_hidden.flatten(0, 1), tokens, players)
                result = replace(anchors, actor_logits=output.logits.reshape(len(anchors.references), 2, -1).detach().clone())
            if (module_version(self.model), module_version(self.actor)) != (
                    anchors.world_model_version, anchors.actor_version):
                raise RuntimeError("model or teacher changed during anchor action-head evaluation")
            return result
        except BaseException:
            self.clear_cache()
            raise


class TeacherImaginedRollout(ImaginedRollout):
    schema_version = "p41-teacher-rollout-v1"

    def _metadata(self):
        result = super()._metadata()
        if self.metadata.get("policy_role") != ROLE:
            raise ValueError("teacher rollout must identify the global_teacher policy role")
        return result


class TeacherRolloutGenerator(RolloutGenerator):
    """Teacher actions drive every shared transition; critic sees teacher memory."""
    actor_type = GlobalTeacher
    anchor_type = TeacherAnchorBatch
    builder_type = TeacherAnchorBuilder
    rollout_type = TeacherImaginedRollout

    @classmethod
    def from_experiment(cls, model, actor, critic, experiment, *, precision=None):
        if experiment.get("variant", {}).get("name") != "D":
            raise ValueError("global teacher collection requires variant D")
        return super().from_experiment(model, actor, critic, experiment, precision=precision)

    def _actor_step(self, rgb, previous, hidden, tokens):
        paired, players = _paired_context(tokens)
        return self.actor(rgb, previous, hidden, paired, players)

    def _role_metadata(self, joint_steps):
        return {"policy_role": ROLE, "teacher_attention_heads": self.actor.attention_heads,
                "teacher_player_decisions": 2 * joint_steps, "local_player_decisions": 0,
                "teacher_recurrence": "own_rgb_and_executed_previous_action_only",
                "teacher_global_context": "current_shared_tokens_at_action_head_only"}


class TeacherPrefixBank(PrefixBank):
    """Current teacher recurrence from complete real histories, then imagined RGB."""

    @_ordinary_replay
    def replay(self, actor, rollout, indices, *, precision="float32"):
        self.validate_rollout(rollout)
        if type(rollout) is not TeacherImaginedRollout or rollout.metadata.get("policy_role") != ROLE:
            raise ValueError("teacher replay requires a teacher-policy rollout")
        if not isinstance(actor, GlobalTeacher):
            raise ValueError("teacher replay requires a GlobalTeacher")
        selected = _indices(indices, len(self.prefixes))
        if precision not in ("float32", "bfloat16") or precision != rollout.metadata["precision"]:
            raise ValueError("teacher replay precision must match behavior collection")
        device = actor.gru.weight_hh.device
        if device.type not in ("cpu", "cuda") or any(
                parameter.device != device or parameter.dtype != torch.float32 for parameter in actor.parameters()):
            raise ValueError("teacher parameters must be FP32 on one CPU or CUDA device")
        if rollout.local_rgb.device != device:
            raise ValueError("teacher and rollout must use the same device")
        if (rollout.metadata.get("actor_config") != json.loads(json.dumps(asdict(actor.config)))
                or rollout.metadata.get("teacher_attention_heads") != actor.attention_heads):
            raise ValueError("teacher architecture differs from behavior collection")
        prefixes = tuple(self.prefixes[index] for index in selected)
        with _replay_mode(actor, precision=precision, device=device):
            with torch.no_grad():
                hidden = actor.initial_hidden(len(selected) * 2).reshape(len(selected), 2, -1)
                # Exclude the anchor observation here: it is consumed exactly
                # once with gradients below. No global tokens enter this loop.
                for step in range(max(prefix.ref.decision_index for prefix in prefixes)):
                    active = [row for row, prefix in enumerate(prefixes) if step < prefix.ref.decision_index]
                    rgb = torch.from_numpy(np.stack([prefixes[row].rgb[step] for row in active])).to(device)
                    previous = torch.from_numpy(np.stack([
                        prefixes[row].actions[step - 1] if step else
                        np.full(2, PREVIOUS_ACTION_START, dtype=np.int64) for row in active])).to(device)
                    rows = torch.tensor(active, dtype=torch.long, device=device)
                    memory = actor.memory_step(rgb.flatten(0, 1), previous.flatten(), hidden[rows].flatten(0, 1))
                    hidden[rows] = memory.reshape(len(active), 2, -1)
                hidden = hidden.detach()
            rows = torch.tensor(selected, dtype=torch.long, device=device)
            outputs = []
            for step in range(rollout.steps):
                active = rollout.valid[step, rows].nonzero().flatten()
                logits = torch.zeros((len(selected), 2, actor.config.action_count), device=device)
                if len(active):
                    source_rows = rows[active]
                    tokens, players = _paired_context(rollout.shared_tokens[step, source_rows])
                    output = actor(rollout.local_rgb[step, source_rows].flatten(0, 1),
                                   rollout.previous_actions[step, source_rows].flatten(),
                                   hidden[active].flatten(0, 1), tokens, players)
                    hidden = hidden.index_copy(0, active, output.next_local_hidden.reshape(len(active), 2, -1))
                    logits = logits.index_copy(0, active, output.logits.reshape(len(active), 2, -1))
                outputs.append(logits)
            return torch.stack(outputs)


class TeacherMAPPO(RecurrentMAPPO):
    """Independent teacher Adam state; frozen-world PPO math shared with B."""
    generator_type = TeacherRolloutGenerator
    rollout_type = TeacherImaginedRollout
    prefix_bank_type = TeacherPrefixBank
    optimizer_schema = "p41-teacher-optimizer-v1"
    update_schema = "p41-teacher-update-v1"

    @classmethod
    def from_experiment(cls, model, actor, critic, experiment, *, master_seed, precision=None):
        generator = cls.generator_type.from_experiment(model, actor, critic, experiment, precision=precision)
        seed = derive_seed(experiment["task"]["name"], master_seed, "teacher_updates", "D")
        return cls(model, actor, critic, config=MAPPOConfig.from_experiment(experiment),
                   seed=seed, precision=generator.precision)

    def update(self, rollout, sampler):
        result = super().update(rollout, sampler)
        result["policy_role"] = ROLE
        result["teacher_player_decisions"] = result["imagined_joint_steps"] * 2
        rewards = rollout.rewards.detach()
        result["metrics"]["imagined_reward_mean"] = float(rewards[rollout.valid].mean().item())
        result["metrics"]["mean_imagined_return"] = float((rewards * rollout.valid).sum(0).mean().item())
        return result
