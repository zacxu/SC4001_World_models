"""Local B/D policy dimensions, without changing frozen world-model configs."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from world_models.contracts import ACTION_COUNT, LOCAL_RGB_SHAPE


@dataclass(frozen=True)
class PolicyConfig:
    image_shape: tuple[int, int, int] = LOCAL_RGB_SHAPE
    encoder_channels: tuple[int, ...] = (32, 64, 128, 256)
    embedding_size: int = 256
    hidden_size: int = 256
    action_count: int = ACTION_COUNT
    world_tokens: int = 48
    world_width: int = 256
    critic_hidden_sizes: tuple[int, int] = (512, 512)
    value_huber_delta: float = 1.0

    def __post_init__(self) -> None:
        if self.image_shape != LOCAL_RGB_SHAPE or type(self.image_shape) is not tuple:
            raise ValueError("local actors require CHW images of shape (3,96,128)")
        if type(self.action_count) is not int or self.action_count != ACTION_COUNT:
            raise ValueError("local actors require exactly 12 executable actions")
        if (type(self.encoder_channels) is not tuple or len(self.encoder_channels) != 4
                or any(type(x) is not int or x <= 0 or x % 8 for x in self.encoder_channels)):
            raise ValueError("actor CNN needs four positive channel widths divisible by 8")
        for name in ("embedding_size", "hidden_size", "world_tokens", "world_width"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (type(self.critic_hidden_sizes) is not tuple or len(self.critic_hidden_sizes) != 2
                or any(type(x) is not int or x <= 0 for x in self.critic_hidden_sizes)):
            raise ValueError("the critic requires two positive hidden widths")
        if (isinstance(self.value_huber_delta, bool)
                or not isinstance(self.value_huber_delta, (int, float))
                or not math.isfinite(self.value_huber_delta) or self.value_huber_delta <= 0):
            raise ValueError("value_huber_delta must be finite and positive")

    @classmethod
    def from_experiment(cls, experiment: Mapping) -> PolicyConfig:
        from world_models.infrastructure.config import validate_config
        from world_models.models.core import ModelConfig

        validate_config(experiment)
        model = ModelConfig.from_experiment(experiment)
        if experiment["actor"]["dropout"] != 0:
            raise ValueError("replayed actor probabilities require zero dropout")
        if experiment["mappo"]["normalize_values"]:
            raise ValueError("P3.1 implements raw values; value normalization is not enabled")
        return cls(image_shape=model.image_shape, encoder_channels=model.encoder_channels,
                   embedding_size=experiment["actor"]["embedding_size"],
                   hidden_size=experiment["actor"]["hidden_size"], action_count=model.action_count,
                   world_tokens=model.memory_tokens + model.latent_variables, world_width=model.width,
                   critic_hidden_sizes=tuple(experiment["critic"]["hidden_sizes"]),
                   value_huber_delta=experiment["mappo"]["value_huber_delta"])
