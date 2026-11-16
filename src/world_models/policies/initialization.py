"""Paired B/D local network initialization, independent of training RNG streams."""
from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from world_models.infrastructure.reproducibility import derive_seed
from .actor import LocalActor
from .config import PolicyConfig
from .critic import CentralizedCritic


class LocalActorCritic(nn.Module):
    """One shared actor, two caller-owned memories, and one centralized value.

    B and D students use this same module. Deliberately no combined forward:
    centralized inputs cannot accidentally become local actor arguments.
    """
    def __init__(self, actor: LocalActor, critic: CentralizedCritic):
        super().__init__()
        self.actor = actor
        self.critic = critic


def build_actor_critic(experiment: Mapping, *, master_seed: int, variant: str) -> LocalActorCritic:
    """Construct paired CPU weights; moving to a GPU is an explicit caller step."""
    config = PolicyConfig.from_experiment(experiment)
    if variant != experiment["variant"]["name"]:
        raise ValueError("variant must match the resolved experiment")
    family = experiment["task"]["name"]
    actor_seed = derive_seed(family, master_seed, "local_actor_initialization", variant)
    critic_seed = derive_seed(family, master_seed, "local_critic_initialization", variant)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(actor_seed)
        actor = LocalActor(config)
        torch.random.default_generator.manual_seed(critic_seed)
        critic = CentralizedCritic(config)
    return LocalActorCritic(actor, critic)
