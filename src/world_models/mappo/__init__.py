"""Recurrent local MAPPO with full-prefix policy replay and raw team returns."""
from .config import MAPPOConfig
from .objectives import GAETargets, actor_objective, compute_gae, explained_variance
from .recurrent import PrefixBank, replay_actor
from .updater import RecurrentMAPPO

__all__ = ["MAPPOConfig", "GAETargets", "compute_gae", "actor_objective", "explained_variance",
           "PrefixBank", "replay_actor", "RecurrentMAPPO"]
