import abc
import typing
from enum import Enum
import os

from omegaconf import OmegaConf
import torch
import numpy as np


from s2p.lib.base import BaseClass
from s2p.models.base.encoder_model_base import EncoderModelBase
from s2p.models.base.dynamics_model_base import DynamicsModelBase
from s2p.models.base.reward_model_base import RewardModelBase
from s2p.models.base.state_value_model_base import StateValueModelBase
from s2p.models.base.state_action_value_model_base import StateActionValueModelBase
from s2p.models.base.policy_model_base import PolicyModelBase
from s2p.planners.planner_base import PlannerBase

class ActionMode(Enum):
    PLANNING = 0
    POLICY = 1

class AgentModel(BaseClass, torch.nn.Module):
    def __init__(
            self, 
            cfg:OmegaConf,
            encoder_model:EncoderModelBase = None,
            dynamics_model:DynamicsModelBase = None,
            reward_model:RewardModelBase = None,
            planner:PlannerBase = None,
            value_model:typing.Union[StateValueModelBase, StateActionValueModelBase] = None,
            policy_model:PolicyModelBase = None
    ):

        super().__init__(cfg)
        self.cfg = cfg

        self.encoder_model = encoder_model
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.planner = planner
        self.value_model = value_model
        self.policy_model = policy_model

        self.models = {
            "encoder": self.encoder_model,
            "dynamics": self.dynamics_model,
            "reward": self.reward_model,
            "value": self.value_model,
            "policy": self.policy_model
        }
        # Verify that these define a proper agent
        agent_has_encoder = self.encoder_model is not None
        if not agent_has_encoder:
            raise ValueError("Agent has no encoder. If identity encoder is desired, use IdentityEncoder class.")
        
        self.action_modes = []
        agent_can_plan = None not in (self.dynamics_model, self.reward_model)
        if agent_can_plan:
            self.action_modes.append(ActionMode.PLANNING)
            
        agent_can_use_policy = self.policy_model is not None
        if agent_can_use_policy:
            self.action_modes.append(ActionMode.POLICY)
        
        if not (agent_can_use_policy or agent_can_plan):
            raise ValueError("Agent cannot plan, nor does it contain a policy. Cannot compute actions.")
            
        # Load checkpoints from a folder if provided
        if self.cfg.checkpoint_folder is not None:
            self.load_from_folder(self.cfg.checkpoint_folder)
        return

    def requires_grad_(self, requires_grad):
        for name, model in self.models.items():
            if model is not None:
                model.requires_grad_(requires_grad and not self.cfg.freeze)

    @torch.no_grad
    def plan(self, state:torch.Tensor, action_prior=None) -> typing.Tuple[torch.Tensor, typing.Dict]:
        """
        Plan an action sequence from `state`, returning it with the planner's info dict.

        The info dict describes how the planner converged to that plan (see the planner's
        own `plan` for what it contains); callers which only want the actions bind it to `_`.
        """
        if not ActionMode.PLANNING in self.action_modes:
            raise ValueError("Planning is not available for this Agent.")
        
        # If no action prior is provided, use all zeros
        if action_prior is None:
            action_prior = torch.zeros(self.planner.task.action_dimension, device=state.device)
       
        # Generate the plan
        plan, info = self.planner.plan(
            dynamics_model=self.dynamics_model, 
            reward_model=self.reward_model,
            policy_model=self.policy_model,
            value_model=self.value_model,
            current_state=state, 
            eval_mode=not self.training,
            action_prior=action_prior)
        
        return plan, info
    
    def act(self, state:torch.Tensor):
        if not ActionMode.POLICY in self.action_modes:
            raise ValueError("Policy actions are not available for this Agent.")
        
        return self.policy_model.policy(state)
    
    def save_to_folder(self, folder):
        for name, module in self.models.items():
            if module is not None:
                module.save_to_file(os.path.join(folder, f"{name}.pt"))

    def load_from_folder(self, checkpoints_folder):
        for name in self.models.keys():
            if self.models[name] is not None:
                module_checkpoint_file = os.path.join(checkpoints_folder, f"{name}.pt")
                self.models[name].load_from_file(module_checkpoint_file)
        return


        