import torch
from torch.distributions import Independent, Normal, kl_divergence

from s2p.models.agent_model import AgentModel

from s2p.losses.loss_function_base import LossFunctionBase

def parse_index(key: str):
    """Convert a YAML string into an int, slice, or tuple of int/slice, for indexing.

    Examples:
        "3"          -> 3
        ":"          -> slice(None)
        "0:3"        -> slice(0, 3)
        ":,:,0:3"    -> (slice(None), slice(None), slice(0, 3))
        "cube_pose"  -> "cube_pose"  (plain dict key, passed through)
    """
    def parse_single(part: str):
        if ":" in part:
            pieces = part.split(":")
            pieces = [int(p) if p else None for p in pieces]
            return slice(*pieces)
        try:
            return int(part)
        except ValueError:
            return part  # dict key like "state", "cube_pose"

    parts = [parse_single(p) for p in key.split(",")]
    return parts[0] if len(parts) == 1 else tuple(parts)
class ProbeLoss(LossFunctionBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
    
    def __call__(self, batch, model:AgentModel):
        T, B = batch.shape

        y_hat = model.encoder_model.encode(batch["obs"])
        y = batch
        for key in self.cfg.target_path:
            y = y[parse_index(key)]
        
        # Value Loss
        loss = (0.5*(y - y_hat)**2).mean()
        
        # Sum them up
        return loss, {
            # Loss but as a logged metric
            "loss": loss.detach(),
        }