"""P4.2 one-time synthetic action distillation for local recurrent students."""
from .buffer import DistillationBuffer
from .learning import DistillationConfig, StudentDistiller, forward_kl, replay_student
from .protocol import DISTILLATION_PROTOCOL, load_protocol, plan, protocol_identity

__all__ = ["DistillationBuffer", "DistillationConfig", "StudentDistiller", "forward_kl",
           "replay_student", "DISTILLATION_PROTOCOL", "load_protocol", "plan", "protocol_identity"]
