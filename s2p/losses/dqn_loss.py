import torch
from torch.distributions import Independent, Normal, kl_divergence

from s2p.models.agent_model import AgentModel
from s2p.models.base.state_value_model_base import StateValueModelBase
from s2p.models.base.state_action_value_model_base import StateActionValueModelBase
from s2p.lib.ensembles import Ensemble
from s2p.losses.loss_function_base import LossFunctionBase

class DQNLoss(LossFunctionBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
    
    def __call__(self, batch, model:AgentModel):
        T, B = batch.shape

        # Check that the model is a valid choice for this loss
        assert (
            model.value_model is not None
            and isinstance(model.value_model, StateActionValueModelBase)
            and isinstance(model.value_model, Ensemble)
            and model.policy_model is not None
        )

        # Assert that trajectories are long enough for the desired latent imagination horizons
        assert self.cfg.latent_policy_imagination_horizon < batch.shape[0]

        # Lambda value for weighting more distant losses less
        t = torch.arange(T-1, dtype=torch.float32, device=batch.device).view(T-1, 1)  # 0..T-1
        lam = (self.cfg.lam ** t).expand(T-1, B)  # shape [T,B,N]

        # Encode observations to Teacher latent space 
        full_state = model.encoder_model.encode(batch["obs"]) # [T, B, Z]
        state = full_state[:-1]
        action = batch["action"][1:]

        # Policy Loss
        state_ = state.detach() # Don't let policy error flow to the encoder
        a_ = action
        policy_loss = torch.tensor(0.0, device=state.device)
        lam_ = lam
        rho_ = 1.0
        for _ in range(self.cfg.latent_policy_imagination_horizon):
            policy_action = model.policy_model.policy(state_)

            # No gradients to value function
            model.value_model.requires_grad_(False)
            l = - model.value_model.state_action_value(state_, policy_action)
            model.value_model.requires_grad_(True)

            policy_loss += torch.mean(l)
            # Increment the imagined state
            state_ = model.dynamics_model.dynamics(state_, a_)[:-1]
            a_ = a_[1:]
            lam_ = lam_[1:]
            rho_ = rho_ * self.cfg.rho
            
        if self.cfg.latent_policy_imagination_horizon != 0: policy_loss /= self.cfg.latent_policy_imagination_horizon
        
        loss = policy_loss
        
        return loss, {
            # Loss but as a logged metric
            "loss": loss.detach(),
        }