"""CPU policy-trainability evidence on an explicitly analytic engineering task.

The production actor, recurrent PPO updater, GAE, rollout collector and prefix
replay are used unchanged. The world is an exact visual-cue bandit, not a fitted
ViZDoom world model. Its prefixes live only in an explicitly marked fixture;
they are never written into the real replay format or a research budget.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from world_models.cycle.checkpoints import save_cycle_checkpoint
from world_models.imagination.anchors import AnchorBuilder
from world_models.imagination.replay import AnchorPrefix, AnchorRef, AnchorSampler
from world_models.imagination.versions import module_version
from world_models.mappo.config import MAPPOConfig
from world_models.mappo.updater import RecurrentMAPPO
from world_models.models.core import WorldState
from world_models.models.heads import TaskSpec, TransitionPredictions, resolve_outcomes, total_reward
from world_models.models.world_model import SharedWorldModel, ImaginationState, ImaginedTransition, ModelTransition
from world_models.policies import LocalActor, CentralizedCritic
from world_models.policies.config import PolicyConfig
from world_models.rollouts.generator import RolloutGenerator
from world_models.rollouts.streams import RolloutStreams


def cue_images(seed: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Balanced independent local red/green cues; nuisance brightness is seeded."""
    if type(count) is not int or count < 4 or count % 4:
        raise ValueError("cue count must be a positive multiple of four")
    rng = np.random.default_rng(seed)
    targets = np.tile(np.array(((0, 0), (0, 1), (1, 0), (1, 1)), dtype=np.int64), (count // 4, 1))
    rng.shuffle(targets)
    images = np.zeros((count, 2, 3, 96, 128), dtype=np.uint8)
    for index in range(count):
        for player in (0, 1):
            images[index, player] = rng.integers(5, 30, size=(3, 1, 1), dtype=np.uint8)
            images[index, player, targets[index, player]] = int(rng.integers(170, 256))
    return images, targets


class EngineeringCuePrefixes(AnchorSampler):
    """In-memory test double with no real-data manifest or learning eligibility.

    Subclassing makes the fixture usable by the production guarded prefix API;
    this object is deliberately not constructible from a research dataset.
    """

    data_source = "analytic_engineering_fixture"
    real_environment_interactions = 0

    def __init__(self, images: np.ndarray):
        if images.dtype != np.uint8 or images.shape[1:] != (2, 3, 96, 128):
            raise ValueError("expected uint8 joint cue images")
        self.task_family, self.deadline, self.run_seed = "relay", 400, 0
        self.cohort_identity = hashlib.sha256(b"p36-cue-fixture-v1" + images.tobytes()).hexdigest()
        identity = hashlib.sha256(b"engineering-only-no-real-replay" + images.tobytes()).hexdigest()
        self.references = tuple(AnchorRef(self.cohort_identity, 0, identity, f"cue-{index}", 0)
                                for index in range(len(images)))
        self.prefixes = {ref: AnchorPrefix(ref, images[index:index+1], np.empty((0, 2), dtype=np.int64),
                                          np.array([0], dtype=np.int64))
                         for index, ref in enumerate(self.references)}

    def sample(self):
        return self.references

    def validate_reference(self, reference):
        if reference not in self.prefixes:
            raise ValueError("reference is outside the engineering cue fixture")

    def read_prefix(self, reference):
        self.validate_reference(reference)
        return self.prefixes[reference]


@dataclass(frozen=True)
class CueConfig:
    image_shape: tuple = (3, 96, 128)
    encoder_channels: tuple = (8, 8, 8, 8)
    action_count: int = 12
    width: int = 256
    memory_tokens: int = 16
    latent_variables: int = 32
    deadline: int = 400


class CueCore(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = torch.nn.Identity()
        self.encoder.config = config

    def representation(self, state):
        latent = torch.nn.functional.pad(state.stochastic, (0, 224))
        return torch.cat((state.memory, latent), dim=1)


class AnalyticCueWorld(SharedWorldModel):
    """Exact, frozen one-step cue dynamics, distinct from a research model.

    Fixed image/action/token interfaces are retained for production API guards.
    A small configurable actor CNN and analytic world keep this a CPU fixture.
    Each player chooses its own visible color (red=0, green=1), with the full
    12-action space. Both correct terminates in success; otherwise in failure.
    A bounded shared bonus of 0.5 per correct player gives local credit.
    """

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.config = CueConfig()
        self.task = TaskSpec(step_cost=0., milestone_bonus=1.)
        self.core = CueCore(self.config)
        self.decoder = torch.nn.Identity()
        self.decoder.config = self.config
        self.register_buffer("frozen_signature", torch.tensor([36.]))

    def initialize(self, observations, *, streams):
        count, device = observations.shape[0], observations.device
        memory = torch.zeros(count, 16, 256, device=device)
        memory[:, 0, :6] = observations.float().mean((-1, -2)).reshape(count, 6) / 255
        stochastic = torch.zeros(count, 32, 32, device=device)
        stochastic[:, :, 0] = 1
        return WorldState(memory, stochastic, torch.zeros_like(stochastic),
                          torch.zeros(count, dtype=torch.long, device=device))

    def imagine_step(self, current, actions, *, streams):
        if current.ended.any().item() or (current.world.decision_index != 0).any().item():
            raise ValueError("analytic cue mechanics has exactly one decision per episode")
        source = current.world
        colors = source.memory[:, 0, :6].reshape(-1, 2, 3)
        targets = colors[:, :, :2].argmax(-1)
        correct = actions == targets
        outcomes = torch.where(correct.all(-1), 1, 2)
        logits = torch.full((len(actions), 3), -50., device=actions.device)
        logits.scatter_(1, outcomes[:, None], 50.)
        sampled = torch.multinomial(logits.softmax(-1), 1, generator=streams.outcome).squeeze(-1)
        world = WorldState(source.memory.clone(), source.stochastic.clone(), source.logits.clone(),
                           source.decision_index + 1)
        resolved = resolve_outcomes(sampled, world.decision_index, self.task)
        predictions = TransitionPredictions(correct.float().sum(-1) * .5, logits)
        rgb = colors[..., None, None].expand(-1, -1, -1, 96, 128).clone()
        valid = torch.ones(len(actions), dtype=torch.bool, device=actions.device)
        prediction = ModelTransition(world, world, self.core.representation(world), rgb, predictions,
                                     {}, torch.zeros_like(valid))
        return ImaginedTransition(prediction, ImaginationState(world, resolved.terminated), valid,
                                  sampled, resolved.outcome, total_reward(predictions.bonus, resolved, self.task),
                                  resolved.terminated, resolved.deadline_failure)


@torch.no_grad()
def evaluate_cues(actor: LocalActor, images: np.ndarray, targets: np.ndarray) -> dict:
    """Exact expected metrics over all actions, without Monte Carlo reward noise."""
    rgb = torch.tensor(images).flatten(0, 1)
    previous = torch.full((len(rgb),), 12, dtype=torch.long)
    output = actor(rgb, previous, actor.initial_hidden(len(rgb)))
    probabilities = output.distribution.probs.reshape(-1, 2, 12)
    target = torch.tensor(targets)
    correct = probabilities.gather(-1, target[..., None]).squeeze(-1)
    greedy = probabilities.argmax(-1) == target
    joint = correct.prod(-1)
    # +1 success / -1 failure, plus .5 for each correct local action.
    expected_return = 2 * joint - 1 + .5 * correct.sum(-1)
    return {"episodes": len(images), "expected_individual_accuracy": float(correct.mean()),
            "expected_joint_success": float(joint.mean()),
            "greedy_individual_accuracy": float(greedy.float().mean()),
            "greedy_joint_success": float(greedy.all(-1).float().mean()),
            "mean_expected_team_return": float(expected_return.mean())}


def _export(directory: Path, actor, world_version, label):
    metadata = {"schema_version": "p36-mechanics-actor-v1", "engineering_only": True,
                "task_family": "analytic_local_color_cue", "label": label,
                "actor_config": json.loads(json.dumps(asdict(actor.config))),
                "actor_version": module_version(actor), "world_model_version": world_version,
                "real_joint_transitions": 0, "evaluation_executed": True, "phase3_acceptance": False}
    manifest = save_cycle_checkpoint(directory, state={"actor": actor.state_dict()}, metadata=metadata)
    from scripts.phase36_support.evidence import file_sha256
    return {"path": directory.name, "actor_version": metadata["actor_version"],
            "manifest_sha256": file_sha256(manifest), "payload_sha256": file_sha256(directory / "state.pt")}


def run_mechanics(output_dir, *, seed=3606, cycles=40, batch_size=32) -> dict:
    """Fit a small cue policy using genuine sampled, clipped recurrent MAPPO.

    This optional command performs tiny engineering optimizer updates on CPU.
    It does not use a research checkpoint, native game, or research replay.
    """
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed must be a nonnegative signed 32-bit integer")
    if type(cycles) is not int or cycles < 1:
        raise ValueError("cycles must be a positive integer")
    if type(batch_size) is not int or batch_size < 4 or batch_size % 4:
        raise ValueError("batch_size must be a multiple of four")
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    from scripts.phase36_support.evidence import source_identity, file_sha256
    sources_before = source_identity()
    started, previous_threads = time.perf_counter(), torch.get_num_threads()
    global_rng = torch.random.get_rng_state().clone()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = AnalyticCueWorld()
            policy = PolicyConfig(encoder_channels=(8, 8, 8, 8), embedding_size=32, hidden_size=32,
                                  critic_hidden_sizes=(32, 32))
            actor, critic = LocalActor(policy), CentralizedCritic(policy)
        model.eval()
        actor.eval()
        critic.eval()
        model_before, actor_before, critic_before = (module_version(net) for net in (model, actor, critic))
        images, _ = cue_images(seed + 1, batch_size)
        heldout, targets = cue_images(seed + 2, 64)
        sampler = EngineeringCuePrefixes(images)
        builder = AnchorBuilder(model, actor, seed=seed + 3, cache_capacity=0)
        generator = RolloutGenerator(model, actor, critic, horizon=1)
        streams = RolloutStreams.from_seed(seed + 4)
        config = MAPPOConfig(epochs=2, minibatches=1, actor_lr=.003, critic_lr=.003,
                            entropy_coefficient=.01)
        learner = RecurrentMAPPO(model, actor, critic, config=config, seed=seed + 5)
        before = evaluate_cues(actor, heldout, targets)
        exports = {"before": _export(directory / "actor_before", actor, model_before, "before")}
        reports = []
        with (directory / "ppo_updates.jsonl").open("w", encoding="utf-8") as output:
            for _ in range(cycles):
                rollout = generator.collect(builder.build(sampler), streams=streams)
                update = learner.update(rollout, sampler)
                output.write(json.dumps(update, allow_nan=False) + "\n")
                output.flush()
                reports.append(update)
        after = evaluate_cues(actor, heldout, targets)
        exports["after"] = _export(directory / "actor_after", actor, model_before, "after")
        finite = all(all(value is None or math.isfinite(value) for value in row["metrics"].values()) for row in reports)
        sources_after = source_identity()
        checks = {"world_unchanged": module_version(model) == model_before,
                  "actor_changed": module_version(actor) != actor_before,
                  "critic_changed": module_version(critic) != critic_before,
                  "improvement_over_before": after["expected_individual_accuracy"] >= before["expected_individual_accuracy"] + .20,
                  "above_chance": after["expected_individual_accuracy"] >= .70 and after["expected_joint_success"] >= .45,
                  "finite_ppo_diagnostics": finite,
                  "initial_ratios_replayed": max(row["metrics"]["initial_ratio_max_error"] for row in reports) < 2e-5,
                  "independent_actor_streams": streams.actors[0] is not streams.actors[1],
                  "global_rng_unchanged": torch.equal(global_rng, torch.random.get_rng_state()),
                  "source_unchanged": sources_before == sources_after}
        report = {"schema_version": "p36-mechanics-v1", "status": "passed" if all(checks.values()) else "failed",
                  "engineering_only": True, "native_environment_evaluation": False,
                  "research_training_executed": False, "phase3_acceptance": False,
                  "new_environment_interactions": 0, "learning_algorithm": "production RecurrentMAPPO",
                  "task": "one-step two-player independent visual color cues with shared team reward",
                  "task_limitations": ["analytic dynamics", "one-step task does not establish memory-dependent skill",
                                       "does not establish ViZDoom mechanics or B-versus-D efficacy"],
                  "seed": seed, "training_cue_seed": seed + 1, "heldout_cue_seed": seed + 2,
                  "training_episodes_per_cycle": batch_size, "heldout_episodes": len(heldout),
                  "cycles": cycles, "actor_optimizer_steps": learner.actor_optimizer_steps,
                  "critic_optimizer_steps": learner.critic_optimizer_steps, "mappo_config": asdict(config),
                  "policy_config": asdict(policy), "before": before, "after": after,
                  "chance": {"expected_individual_accuracy": 1 / 12, "expected_joint_success": 1 / 144},
                  "checks": checks, "world_model_version": model_before, "actor_exports": exports,
                  "source_sha256": sources_before, "source_sha256_after": sources_after,
                  "ppo_updates_path": "ppo_updates.jsonl", "ppo_updates_sha256": file_sha256(directory / "ppo_updates.jsonl"),
                  "elapsed_seconds": time.perf_counter() - started}
        from scripts.phase36_support.evidence import write_report
        write_report(directory / "report.json", report)
        return report
    finally:
        torch.set_num_threads(previous_threads)
