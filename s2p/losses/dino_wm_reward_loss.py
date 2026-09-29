import torch
from torch.distributions import Independent, Normal, kl_divergence

from s2p.models.agent_model import AgentModel

from s2p.losses.loss_function_base import LossFunctionBase
from s2p.models.dino_wm_reward_model import DINOWMRewardModel

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

class DINOWMRewardLoss(LossFunctionBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
    
    def __call__(self, batch, model:AgentModel):
        T, B = batch.shape

        # Checked by interface, not by class: this loss serves every DINO-WM-family wrapper
        # (DINO-WM, TC-WM, DINO-Bisim, Sparse Imagination), and importing one of them to name
        # it here would make the other three unimportable in the same process, since all four
        # repos claim the same rootless top-level module names (see `s2p.lib.tc_wm_path`).
        # `write_action_into_newest_frame` is what the reward head needs of the encoder.
        assert isinstance(model.reward_model, DINOWMRewardModel), (
            f"{type(self).__name__} trains a DINOWMRewardModel, got "
            f"{type(model.reward_model).__name__}."
        )
        assert hasattr(model.encoder_model, "write_action_into_newest_frame"), (
            f"{type(model.encoder_model).__name__} is not a DINO-WM-family world model: it has "
            "no `write_action_into_newest_frame`, so the reward head cannot condition on an "
            "action the way its predictor does."
        )

        z_all = model.encoder_model.encode(batch["obs"])
        z = z_all[:-1]
        z_prime = z_all[1:]
        a = batch["action"][1:]
        r = batch["reward"][1:]
        r_prediction = model.reward_model.reward(z, a)
        r_error = torch.mean(0.5*(r_prediction - r)**2)
        
        # Sum them up
        return r_error, {
            # Loss but as a logged metric
            "loss": r_error.detach(),
        }