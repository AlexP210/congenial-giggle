import typing

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset, DataLoader, RandomSampler, random_split
from s2p.evaluators.evaluator_base import EvaluatorBase
import tqdm
import time

from calflops import calculate_flops

from s2p.models.agent_model import AgentModel
from s2p.models.base.encoder_model_base import EncoderModelBase
from s2p.models.base.dynamics_model_base import DynamicsModelBase
from s2p.models.base.reward_model_base import RewardModelBase
from s2p.models.base.state_value_model_base import StateValueModelBase
from s2p.models.base.state_action_value_model_base import StateActionValueModelBase

class FLOPSEvaluator(EvaluatorBase):

	def __init__(self, cfg):

		super().__init__(cfg)
		self.cfg = cfg

	def __call__(self, model:AgentModel) -> typing.Dict[str, typing.Any]:
		info = {}
		if model.encoder_model is not None:
			encoder_input_dims, encoder = model.encoder_model.get_encoding_function()
			flops, macs, params, t = self.benchmark(encoder, encoder_input_dims, self.cfg.device)
			info["encoder/flops"] = flops
			info["encoder/macs"] = macs
			info["encoder/params"] = params
			info["encoder/inference_time"] = t
		if model.dynamics_model is not None:
			dynamics_input_dims, dynamics_function = model.dynamics_model.get_dynamics_function()
			flops, macs, params, t = self.benchmark(dynamics_function, dynamics_input_dims, self.cfg.device)
			info["dynamics/flops"] = flops
			info["dynamics/macs"] = macs
			info["dynamics/params"] = params
			info["dynamics/inference_time"] = t
		if model.reward_model is not None:
			reward_input_dims, reward_function = model.reward_model.get_reward_function()
			flops, macs, params, t = self.benchmark(reward_function, reward_input_dims, self.cfg.device)
			info["reward/flops"] = flops
			info["reward/macs"] = macs
			info["reward/params"] = params
			info["reward/inference_time"] = t
		if model.value_model is not None:
			if isinstance(model.value_model, StateValueModelBase):
				value_input_dims, value_function = model.value_model.get_state_value_function()
			elif isinstance(model.value_model, StateActionValueModelBase):
				value_input_dims, value_function = model.value_model.get_state_action_value_function()
			flops, macs, params, t = self.benchmark(value_function, value_input_dims, self.cfg.device)
			info["value/flops"] = flops
			info["value/macs"] = macs
			info["value/params"] = params
			info["value/inference_time"] = t

		return info
	
	def benchmark(self, model, input_shape, device):
		input_shape = (1, *tuple(input_shape))
		flops, macs, params = calculate_flops(model=model, 
											input_shape=input_shape,
											output_as_string=False,
											print_results=False,
											print_detailed=False,
											output_precision=4)
		t_start = time.time()
		model(torch.zeros(size=input_shape).to(device=device))
		t_stop = time.time()
		t = t_stop - t_start
		return flops, macs, params, t

