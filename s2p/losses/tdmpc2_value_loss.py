import torch
from tensordict import TensorDict
from tdmpc2.common import math
import torch.nn.functional as F

from s2p.models.agent_model import AgentModel
from s2p.models.tdmpc2_world_model import TDMPC2WorldModel
from s2p.losses.loss_function_base import LossFunctionBase

class TDMPC2ValueLoss(LossFunctionBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
    

    def update_pi(self, model, zs, task):
        """
        Update policy using a sequence of latent states.

        Args:
            zs (torch.Tensor): Sequence of latent states.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            float: Loss of the policy update.
        """
        action, info = model.value_model.tdmpc2.model.pi(zs, task)
        qs = model.value_model.tdmpc2.model.Q(zs, action, task, return_type='avg', detach=True)
        model.value_model.tdmpc2.scale.update(qs[0])
        qs = model.value_model.tdmpc2.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(model.value_model.parsed_tdmpc2_cfg.rho, torch.arange(len(qs), device=self.cfg.device))
        pi_loss = (-(model.value_model.parsed_tdmpc2_cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
        # pi_loss.backward()
        # pi_grad_norm = torch.nn.utils.clip_grad_norm_(model.student.tdmpc2.model._pi.parameters(), self.cfg.grad_clip_norm)
        # self.pi_optim.step()
        # self.pi_optim.zero_grad(set_to_none=True)

        info = TensorDict({
            "pi_loss": pi_loss,
            # "pi_grad_norm": pi_grad_norm,
            "pi_entropy": info["entropy"],
            "pi_scaled_entropy": info["scaled_entropy"],
            # "pi_scale": self.scale.value,
        })
        return pi_loss, info

    @torch.no_grad()
    def _td_target(self, model:AgentModel, next_z:torch.Tensor, reward:torch.Tensor, terminated:torch.Tensor, task:torch.Tensor):
        """
        Compute the TD-target from a reward and the observation at the following time step.

        Args:
            next_z (torch.Tensor): Latent state at the following time step.
            reward (torch.Tensor): Reward at the current time step.
            terminated (torch.Tensor): Termination signal at the current time step.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            torch.Tensor: TD-target.
        """
        action, _ = model.value_model.tdmpc2.model.pi(next_z, task)
        discount = model.value_model.tdmpc2.discount[task].unsqueeze(-1) if model.value_model.parsed_tdmpc2_cfg.multitask else model.value_model.tdmpc2.discount
        return reward + discount * (1-terminated) * model.value_model.tdmpc2.model.Q(next_z, action, task, return_type='min', target=True)

    def _update(self, model:AgentModel, obs, action, reward, terminated, task=None):
        T, B = reward.shape[:2]
        
        # Compute targets
        with torch.no_grad():
            next_z = model.encoder_model.encode(obs[1:])
            td_targets = self._td_target(model, next_z, reward, terminated, task)

        # Latent rollout
        zs = torch.empty(T+1, B, model.value_model.parsed_tdmpc2_cfg.latent_dim, device=self.cfg.device)
        z = model.encoder_model.encode(obs[0].unsqueeze(0)).squeeze(0)
        zs[0] = z
        consistency_loss = 0
        for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
            z = model.dynamics_model.dynamics(z.unsqueeze(0), _action.unsqueeze(0)).squeeze(0)
            consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * model.value_model.parsed_tdmpc2_cfg.rho**t
            zs[t+1] = z

        # Predictions
        _zs = zs[:-1]
        qs = model.value_model.tdmpc2.model.Q(_zs, action, task, return_type='all')

        # Compute losses
        value_loss = 0
        for t, (td_targets_unbind, qs_unbind) in enumerate(zip(td_targets.unbind(0), qs.unbind(1))):
            for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
                value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, model.value_model.parsed_tdmpc2_cfg).mean() * model.value_model.parsed_tdmpc2_cfg.rho**t

        value_loss = value_loss / (model.value_model.parsed_tdmpc2_cfg.horizon * model.value_model.parsed_tdmpc2_cfg.num_q)
        
        # Update model
        # total_loss.backward()
        # grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        # self.optim.step()
        # self.optim.zero_grad(set_to_none=True)

        # Update policy
        pi_loss, pi_info = self.update_pi(model, zs.detach(), task)

        # Update target Q-functions
        model.value_model.tdmpc2.model.soft_update_target_Q()

        # Return training statistics
        info = TensorDict({
            # "consistency_loss": consistency_loss,
            # "reward_loss": reward_loss,
            "value_loss": value_loss,
            # "termination_loss": termination_loss,
            # "total_loss": total_loss,
            # "grad_norm": grad_norm,
        })
        info.update(pi_info)
        return pi_loss, value_loss, info

    
    def __call__(self, batch, model:AgentModel):

        assert (
            model.value_model is not None
            and isinstance(model.value_model, TDMPC2WorldModel)
        )

        pi_loss, value_loss, info = self._update(
            model, 
            obs=batch["obs"], 
            action=batch["action"][1:], 
            reward=batch["reward"][1:], 
            terminated=torch.zeros_like(batch["reward"][1:])
        )

        # Sum them up
        total_loss = pi_loss + value_loss
        out_info = {
            "loss": total_loss.detach(),
            "pi_loss": pi_loss.detach(),
            "value_loss": value_loss.detach()
        }
        return total_loss, out_info