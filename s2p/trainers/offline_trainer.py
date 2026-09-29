import typing
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from omegaconf import OmegaConf

from s2p.trainers.trainer_base import TrainerBase
from s2p.tasks.base.offline_task_base import OfflineTaskBase
from s2p.models.agent_model import AgentModel
from s2p.losses.loss_function_base import LossFunctionBase
from s2p.lib.utils import collate_batch, infinite_loader

class OfflineTrainer(TrainerBase):

    def __init__(self, cfg:OmegaConf, loss_functions:typing.Dict[str, LossFunctionBase], loss_function_weights:typing.Dict[str, float], task:OfflineTaskBase):

        super().__init__(cfg, task, loss_functions, loss_function_weights)
        self.cfg = cfg
        self.task = task

        self.training_dataset, self.validation_dataset = task.make_dataset()
        # With `load=False` the dataset is a set of memmap views, so __getitem__ is a page
        # fault rather than a computation and a single-process loader spends the whole step
        # blocked in D state with the GPU idle. Workers turn those faults into concurrent
        # ones, which is the entire point -- the dataset must therefore hand back CPU
        # tensors (see the task's `batch_device`), because a forked worker cannot touch the
        # parent's CUDA context. `fetch_batch` does the one transfer to the GPU instead.
        num_workers = self.cfg.get("num_workers", 0)
        self.data_loader = DataLoader(
            dataset=self.training_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=collate_batch,
            num_workers=num_workers,
            # `infinite_loader` restarts the loader on every pass; without this each pass
            # would tear down and respawn every worker.
            persistent_workers=num_workers > 0,
            prefetch_factor=self.cfg.get("prefetch_factor", 2) if num_workers > 0 else None,
            pin_memory=num_workers > 0,
        )
        self.data_iterator = infinite_loader(self.data_loader)

    def fetch_batch(self):
        batch = next(self.data_iterator)
        # A no-op when the dataset already placed the sample (batch_device set to a GPU and
        # no workers); the real transfer when it handed back pinned CPU tensors.
        batch = batch.to(self.cfg.device, non_blocking=True)
        # Permute [B, T, ...] → [T, B, ...] to match TD-MPC2 convention
        return torch.permute(batch, dims=(1, 0)).contiguous()

    def compute_step(self, batch, model:AgentModel, optimizer:Optimizer) -> typing.Dict[str, typing.Any]:
        optimizer.zero_grad()
        total_info = {}
        total_loss = torch.tensor(0.0, device=self.cfg.device)
        for loss_function_name, loss_function in self.loss_functions.items():
            loss, info = loss_function(
                batch=batch,
                model=model,
            )
            total_loss += self.loss_function_weights[loss_function_name] * loss
            prefix = self._loss_prefixes[loss_function_name]
            total_info.update({prefix + key: value for key, value in info.items()})
        total_info["total_loss"] = total_loss.detach()
        total_loss.backward()
        optimizer.step()
        for name, sub_model in model.models.items():
            if sub_model is not None: sub_model.on_parameter_update_callback()
        return total_info