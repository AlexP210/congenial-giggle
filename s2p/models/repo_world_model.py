import typing

import torch
import torch.nn.functional as F

from s2p.models.base.encoder_model_base import EncoderModelBase
from s2p.models.base.dynamics_model_base import DynamicsModelBase
from s2p.models.base.reward_model_base import RewardModelBase
from s2p.models.base.state_value_model_base import StateValueModelBase
from s2p.models.base.policy_model_base import PolicyModelBase
from s2p.tasks.base.online_task_base import OnlineTaskBase
from s2p.tasks.base.task_base import TaskBase
from s2p.lib.checkpointing import resolve_checkpoint_path, save_checkpoint_reference

from repo.algorithms.repo import RePo
from repo.algorithms.repo.models.utils import bottle
from repo.common.logger import Logger
from repo.experiments.setup import set_device, set_seed
import hashlib

class RePoWorldModel(
	EncoderModelBase,
	DynamicsModelBase,
	RewardModelBase,
	StateValueModelBase,
	PolicyModelBase,
	torch.nn.Module
):
	"""
	RePo World Model wrapper class to enable us to evaluate trained RePo world models in our planning environment.
	For training, use the original RePo training scripts.
	"""

	class RePoEncoderModelForFLOPSCalculation(torch.nn.Module):
		"""
		Module whose forward pass repeats the computation of the posterior encoding from
		RSSM. Used for computing FLOPs of the encoding path only.
		"""
		def __init__(
				self, belief_size, state_size, action_size, observation_size,
				act_fn:torch.nn.Module, fc_embed_state_action:torch.nn.Module, rnn:torch.nn.Module,
				fc_embed_belief_posterior:torch.nn.Module, fc_state_posterior:torch.nn.Module
			   ):
			self.belief_size = belief_size
			self.state_size = state_size
			self.action_size = action_size
			self.observation_size = observation_size
			self.input_size = (self.belief_size + self.state_size + self.action_size + self.observation_size,)
			self.act_fn = act_fn
			self.fc_embed_state_action = fc_embed_state_action
			self.rnn = rnn
			self.fc_embed_belief_posterior = fc_embed_belief_posterior
			self.fc_state_posterior = fc_state_posterior
		
		def forward(self, belief_state_action_observation:torch.Tensor):
			belief, state, action, observation = belief_state_action_observation.split(
				(self.belief_size, self.state_size, self.action_size, self.observation_size),
				dim=-1
			)
			# repo.transition_model.compute_belief()
			hidden = self.act_fn(
				self.fc_embed_state_action(torch.cat([state, action], dim=1))
			)
			belief = self.rnn(hidden, belief)
			# repo.transition_model.compute_posterior_state()
			hidden = self.act_fn(
				self.fc_embed_belief_posterior(torch.cat([belief, observation], dim=1))
			)
			posterior_mean, posterior_std_dev = torch.chunk(
				self.fc_state_posterior(hidden), chunks=2, dim=1
			)
			posterior_std_dev = F.softplus(posterior_std_dev) + self.min_std_dev
			posterior_state = posterior_mean + (
				posterior_std_dev * torch.randn_like(posterior_mean)
			)
			return torch.cat([belief, posterior_state], dim=-1)
		
	class RePoDynamicsModelForFLOPSCalculation(torch.nn.Module):
		"""
		Module whose forward pass repeats the computation of the dynamics prediction from
		RSSM. Used for computing FLOPs of the dynamics prediction only.
		"""
		def __init__(
				self, belief_size, state_size, action_size, observation_size,
				act_fn:torch.nn.Module, fc_embed_state_action:torch.nn.Module, rnn:torch.nn.Module,
				fc_embed_belief_posterior:torch.nn.Module, fc_state_posterior:torch.nn.Module
			   ):
			self.belief_size = belief_size
			self.state_size = state_size
			self.action_size = action_size
			self.input_size = (self.belief_size + self.state_size + self.action_size,)
			self.act_fn = act_fn
			self.fc_embed_state_action = fc_embed_state_action
			self.rnn = rnn
			self.fc_embed_belief_posterior = fc_embed_belief_posterior
			self.fc_state_posterior = fc_state_posterior
		
		def forward(self, belief_state_action:torch.Tensor):
			belief, state, action = belief_state_action.split(
				(self.belief_size, self.state_size, self.action_size),
				dim=-1
			)
			# repo.transition_model.compute_belief()
			hidden = self.act_fn(
				self.fc_embed_state_action(torch.cat([state, action], dim=1))
			)
			belief = self.rnn(hidden, belief)
			# repo.transition_model.compute_prior_state()
			hidden = self.act_fn(self.fc_embed_belief_prior(belief))
			prior_mean, prior_std_dev = torch.chunk(
				self.fc_state_prior(hidden), chunks=2, dim=1
			)
			prior_std_dev = F.softplus(prior_std_dev) + self.min_std_dev
			prior_state = prior_mean + prior_std_dev * torch.randn_like(prior_mean)
			return torch.cat([belief, prior_state], dim=-1)

	class RePoRewardModelForFLOPSCalculation(torch.nn.Module):
		"""
		Module whose forward pass repeats the computation of the reward prediction from
		RSSM. Used for computing FLOPs of the reward model only.
		"""
		def __init__(
				self, belief_size, state_size,
				reward_model:torch.nn.Module
			   ):
			self.belief_size = belief_size
			self.state_size = state_size
			self.reward_model = reward_model
			self.input_size = (self.belief_size+self.state_size,)
		
		def forward(self, belief_state_action:torch.Tensor):
			belief, state = belief_state_action.split(
				(self.belief_size, self.state_size),
				dim=-1
			)
			return self.reward_model(belief, state)

	class RePoStateValueModelForFLOPSCalculation(torch.nn.Module):
		"""
		Module whose forward pass repeats the computation of the reward prediction from
		RSSM. Used for computing FLOPs of the reward model only.
		"""
		def __init__(
				self, belief_size, state_size,
				value_model:torch.nn.Module
			   ):
			self.belief_size = belief_size
			self.state_size = state_size
			self.value_model = value_model
			self.input_size = (self.belief_size+self.state_size,)
		
		def forward(self, belief_state:torch.Tensor):
			belief, state = belief_state.split(
				(self.belief_size, self.state_size),
				dim=-1
			)
			return self.value_model(belief, state)

	class RePoPolicyModelForFLOPSCalculation(torch.nn.Module):
		"""
		Module whose forward pass repeats the computation of the reward prediction from
		RSSM. Used for computing FLOPs of the reward model only.
		"""
		def __init__(
				self, belief_size, state_size,
				policy_model:torch.nn.Module
			   ):
			self.belief_size = belief_size
			self.state_size = state_size
			self.policy_model = policy_model
			self.input_size = (self.belief_size+self.state_size,)
		
		def forward(self, belief_state:torch.Tensor):
			belief, state = belief_state.split(
				(self.belief_size, self.state_size),
				dim=-1
			)
			return self.policy_model(belief, state)

	def __init__(self, cfg, task:OnlineTaskBase):
		super().__init__(cfg=cfg)
		self.cfg = cfg
		self.task = task
		self.repo_config = self.cfg.repo_cfg
		set_seed(self.repo_config.seed)
		set_device(self.repo_config.use_gpu, self.repo_config.gpu_id)
		# `.env` unwraps `SingleEnvBatch`: every task now hands out a batched env, and RePo is
		# vendored code that steps one dm_control sim with unbatched observations and a bare
		# `(A,)` action. It gets the same env it always did, from underneath the batch axis.
		self.repo = RePo(
			config=self.repo_config,
			env=task.make_env().env,
			eval_env=task.make_env().env,
			logger=Logger(folder="/dev/null", output_formats=[])
		)
		if self.cfg.checkpoint is not None:
			self.load_from_file(self.cfg.checkpoint)

	def encode(self, observation:torch.Tensor, previous_state:torch.Tensor=None, a:torch.Tensor=None):

		embedding = bottle(self.repo.encoder, [observation,])

		if previous_state is None or a is None:
			prev_belief, prev_state, a = self.repo.init_latent_and_action()
			prev_belief = prev_belief.unsqueeze(0).expand(embedding.shape[0], embedding.shape[1], -1) # Add time dimension
			prev_state = prev_state.unsqueeze(0).expand(embedding.shape[0], embedding.shape[1], -1) # Add time dimension
			a = a.unsqueeze(0).expand(embedding.shape[0], embedding.shape[1], -1) # Add time dimension
		else:
			prev_belief, prev_state = previous_state.split(
				(
					self.repo_config.belief_size, 
					self.repo_config.state_size
				), 
				dim=-1
			)

		next_belief = torch.empty_like(prev_belief)
		next_state = torch.empty_like(prev_state)
		for t in range(embedding.shape[0]):
			next_belief[t] = self.repo.transition_model.compute_belief(prev_belief[t], prev_state[t], a[t])
			next_state[t], _, _ = self.repo.transition_model.compute_posterior_state(
					belief=next_belief[t],
					observation=embedding[t]
				)
		s = torch.cat([next_belief, next_state], dim=-1)
		return s
	
	def get_encoding_function(self) -> typing.Tuple[typing.Tuple[int], torch.nn.Module]:
		encoder_module = self.RePoEncoderModelForFLOPSCalculation(
			self.repo_config.belief_size, self.repo_config.state_size, 
			self.task.action_dimension[-1], self.task.observation_dimension[-1],
			act_fn=self.repo.transition_model.act_fn,
			fc_embed_state_action=self.repo.transition_model.fc_embed_state_action,
			fc_embed_belief_posterior=self.repo.transition_model.fc_embed_belief_posterior,
			fc_state_posterior=self.repo.transition_model.fc_state_posterior
		)
		return (
			encoder_module.input_size,
			encoder_module
		)

	def dynamics(self, s:torch.Tensor, a:torch.Tensor):
		belief, state = s.split(
			(
				self.repo_config.belief_size, 
				self.repo_config.state_size
			), 
			dim=-1
		)
		next_belief = torch.empty_like(belief)
		next_state = torch.empty_like(state)
		for t in range(a.shape[0]):
			next_belief[t] = self.repo.transition_model.compute_belief(belief[t], state[t], a[t])
			next_state[t], _, _ = self.repo.transition_model.compute_prior_state(next_belief[t])
		s = torch.cat([next_belief, next_state], dim=-1)
		return s
	
	def get_dynamics_function(self):
		dynamics_module = self.RePoDynamicsModelForFLOPSCalculation(
			self.repo_config.belief_size, self.repo_config.state_size, 
			self.task.action_dimension[-1], self.task.observation_dimension[-1],
			act_fn=self.repo.transition_model.act_fn,
			fc_embed_state_action=self.repo.transition_model.fc_embed_state_action,
			fc_embed_belief_posterior=self.repo.transition_model.fc_embed_belief_posterior,
			fc_state_posterior=self.repo.transition_model.fc_state_posterior
		)
		return (
			dynamics_module.input_size,
			dynamics_module
		)

	def reward(self, s, a):
		belief, state = s.split(
			(
				self.repo_config.belief_size, 
				self.repo_config.state_size
			), 
			dim=-1
		)
		return self.repo.reward_model(belief, state)
	
	def get_reward_function(self):
		reward_module = self.RePoRewardModelForFLOPSCalculation(
			self.repo_config.belief_size, self.repo_config.state_size,
			self.repo.reward_model
		)
		return (
			reward_module.input_size,
			reward_module
		)

	def state_value(self, s):
		belief, state = s.split(
			(
				self.repo_config.belief_size, 
				self.repo_config.state_size
			), 
			dim=-1
		)
		value = torch.empty(size=(*s.shape[:-1], 1), device=s.device)
		for t in range(s.shape[0]):
			value[t] = self.repo.value_model(belief[t], state[t]).unsqueeze(-1)
		return value
	
	def get_state_value_function(self):
		value_module = self.RePoStateValueModelForFLOPSCalculation(
			self.repo_config.belief_size, self.repo_config.state_size,
			self.repo.value_model
		)
		return (
			value_module.input_size,
			value_module
		)
	
	def policy(self, s):
		belief, state = s.split(
			(
				self.repo_config.belief_size, 
				self.repo_config.state_size
			), 
			dim=-1
		)
		return self.repo.actor_model.get_action(belief, state)
	
	def get_policy_function(self):
		policy_module = self.RePoPolicyModelForFLOPSCalculation(
			self.repo_config.belief_size, self.repo_config.state_size,
			self.repo.actor_model
		)
		return (
			policy_module.input_size,
			policy_module
		)
	
	def save_to_file(self, filepath):
		# Frozen: these weights are still exactly the checkpoint this was built from, so
		# point at it instead of copying it. See s2p.lib.checkpointing.
		if self.cfg.freeze and self.cfg.checkpoint is not None:
			save_checkpoint_reference(filepath, self.cfg.checkpoint, type(self).__name__)
			return

		self.repo.save_checkpoint(filepath)

	def load_from_file(self, path):
		# May be a reference written by `save_to_file` above rather than weights.
		path = resolve_checkpoint_path(path)
		self.repo.load_checkpoint(model_path=path)

	def requires_grad_(self, requires_grad):
		return super().requires_grad_(requires_grad and not self.cfg.freeze)


_registry: dict = {}

def get_or_create(cfg, task) -> "RePoWorldModel":
	key = hashlib.md5(OmegaConf.to_yaml(cfg).encode()).hexdigest()
	if key not in _registry:
		_registry[key] = RePoWorldModel(cfg=cfg, task=task)
	return _registry[key]
