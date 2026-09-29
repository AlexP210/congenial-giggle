import numpy as np
import torch
from torch import nn
from tensordict import stack
from torch.utils.data import Subset, DataLoader, RandomSampler, random_split
from s2p.evaluators.offline_evaluator_base import OfflineEvaluatorBase
from latentmi import lmi
from tqdm import tqdm

from s2p.models.encoder_base import EncoderBase
from s2p.lib.transition_data import OfflineTransitionDataset


class LatentMIEvaluator(OfflineEvaluatorBase):

    def __init__(self, cfg, dataset:OfflineTransitionDataset):
        
        super().__init__(cfg, dataset)
        self.cfg = cfg

    def __call__(self, model:EncoderBase):
        mi_estimates = np.empty(self.cfg.num_estimates)
        for estimate_idx in range(self.cfg.num_estimates):
            # Split the validation dataset further into an MI-train and MI-val dataset
            train_size = int(self.cfg.train_fraction * len(self.data_set))
            val_size = len(self.data_set) - train_size

            train_dataset, val_dataset = random_split(self.data_set, [train_size, val_size])
            mi_train_dataloader = DataLoader(train_dataset, batch_size=self.cfg.lmi_batch_size, shuffle=True, collate_fn=stack)
            mi_val_dataloader = DataLoader(val_dataset, batch_size=self.cfg.lmi_batch_size, shuffle=False, collate_fn=stack)     
            
            info = {}

            # Prepare the LMI embedding model
            example_datum = next(iter(mi_train_dataloader))
            latent = model.encode(example_datum["obs"])

            latent_dim = latent.shape[-1]
            lmi_encoder = lmi.models.AECross(
                x_dim=latent_dim,
                y_dim=example_datum[self.cfg.target_variable_name].shape[-1],
                latent_size=self.cfg.lmi_latent_size,
            ).to(self.cfg.device)

            # Prepare the stuff to train LMI embedding model
            optimizer = torch.optim.Adam(lmi_encoder.parameters(), lr=self.cfg.lmi_lr, eps=1e-07) 
            val_losses = []
            early_stopper = lmi.EarlyStopper(patience=self.cfg.lmi_patience)

            # Train the LMI embedding model
            lmi_training_bar = tqdm(
                iterable=range(self.cfg.lmi_epochs*len(mi_train_dataloader)), 
                desc=f"MI: {self.cfg.target_variable_name}", leave=False
            )
            val_loss = None
            for epoch in range(self.cfg.lmi_epochs):

                # Train 1 epoch
                for batch in mi_train_dataloader:
                    latent = model.encode(batch["observation"])
                    train_loss = lmi_encoder.learning_loss(latent, batch[self.cfg.target_variable_name])
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()
                    lmi_training_bar.set_postfix(train_loss=train_loss.item(), val_loss=val_loss)
                    lmi_training_bar.update(1)

                # Validate 
                with torch.no_grad():
                    epoch_validate_loss = []
                    for batch in mi_val_dataloader:
                        latent = model.encode(batch["observation"])
                        epoch_validate_loss.append(lmi_encoder.learning_loss(
                            x_samples=latent, 
                            y_samples=batch[self.cfg.target_variable_name]).item()
                        )
                    val_loss = np.mean(epoch_validate_loss).item()
                    val_losses.append(val_loss)
                    lmi_training_bar.set_postfix(train_loss=train_loss.item(), val_loss=val_loss)

                # Whether to stop early
                es = early_stopper.early_stop(val_losses[-1], lmi_encoder)
                if es:
                    lmi_encoder.load_state_dict(es)
                    break

            # Encode the data
            Z_Xs = []
            Z_Ys = []
            for batch in self.data_loader:
                latent = model.encode(batch["observation"])
                Z_X, Z_Y = lmi_encoder.encode(latent, batch[self.cfg.target_variable_name])
                Z_Xs.extend(Z_X.tolist())
                Z_Ys.extend(Z_Y.tolist())
            
            # Estimate MI with KSG
            mi_estimate = np.mean(lmi.ksg.mi(Z_Xs, Z_Ys))
            mi_estimates[estimate_idx] = mi_estimate
        info[f"mutual_information_mean"] = np.mean(mi_estimates)
        info[f"mutual_information_sem"] = np.std(mi_estimates)/np.sqrt(len(mi_estimates))
        info[f"mutual_information_distribution"] = mi_estimates

        return info

