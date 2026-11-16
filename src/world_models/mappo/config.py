"""Recurrent MAPPO optimization settings for both local policy variants."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class MAPPOConfig:
    epochs: int = 4
    minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    adam_epsilon: float = 1e-5
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    value_huber_delta: float = 1.0
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True

    def __post_init__(self) -> None:
        for name in ("epochs", "minibatches"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("gamma", "gae_lambda", "clip_epsilon", "actor_lr", "critic_lr",
                     "adam_epsilon", "entropy_coefficient", "value_coefficient",
                     "value_huber_delta", "max_grad_norm"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if name in ("gamma", "gae_lambda"):
                if not 0 <= value <= 1:
                    raise ValueError(f"{name} must be between zero and one")
            elif name == "clip_epsilon":
                if not 0 < value < 1:
                    raise ValueError("clip_epsilon must be strictly between zero and one")
            elif name == "entropy_coefficient":
                if value < 0:
                    raise ValueError("entropy_coefficient must be nonnegative")
            elif value <= 0:
                raise ValueError(f"{name} must be positive")
        if type(self.normalize_advantages) is not bool:
            raise ValueError("normalize_advantages must be boolean")

    @classmethod
    def from_experiment(cls, experiment: Mapping) -> MAPPOConfig:
        from world_models.infrastructure.config import validate_config

        validate_config(experiment)
        settings = experiment["mappo"]
        if experiment["actor"]["dropout"] != 0:
            raise ValueError("recurrent likelihood replay requires zero actor dropout")
        if settings["normalize_values"]:
            raise ValueError("P3.4 uses raw team returns and values")
        return cls(
            epochs=settings["epochs"], minibatches=settings["recurrent_minibatches"],
            gamma=settings["gamma"], gae_lambda=settings["gae_lambda"],
            clip_epsilon=settings["clip_epsilon"], actor_lr=settings["learning_rate"],
            critic_lr=settings["learning_rate"], adam_epsilon=settings["optimizer_epsilon"],
            entropy_coefficient=settings["entropy_coefficient"],
            value_coefficient=settings["value_coefficient"],
            value_huber_delta=settings["value_huber_delta"],
            max_grad_norm=settings["gradient_clip"],
            normalize_advantages=settings["normalize_actor_advantages"],
        )
