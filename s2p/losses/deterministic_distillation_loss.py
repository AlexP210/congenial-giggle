import torch

from s2p.models.agent_model import AgentModel
from s2p.models.teacher_student_encoder_model import TeacherStudentEncoderModel
from s2p.models.base.ema_target_encoder_base import EMATargetEncoderBase
from s2p.models.deterministic_mlp_dynamics_model import DeterministicMLPDynamicsModel
from s2p.models.deterministic_mlp_reward_model import DeterministicMLPRewardModel
from s2p.losses.loss_function_base import LossFunctionBase

class DeterministicDistillationLoss(LossFunctionBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
    
    def __call__(self, batch, model:AgentModel):

        T, B = batch.shape

        # Check that the model is a valid choice for this loss
        assert isinstance(model.encoder_model, TeacherStudentEncoderModel)
        assert (
            isinstance(model.encoder_model.student_encoder, torch.nn.Module)
            and isinstance(model.dynamics_model, DeterministicMLPDynamicsModel)
            and isinstance(model.reward_model, DeterministicMLPRewardModel)
        )
        if self.cfg.use_target_encoder:
            assert isinstance(model.encoder_model.student_encoder, EMATargetEncoderBase)

        # Assert that trajectories are long enough for the desired latent imagination horizons
        assert self.cfg.latent_rewards_imagination_horizon < batch.shape[0]
        assert self.cfg.latent_dynamics_imagination_horizon < batch.shape[0]
        
        # Lambda value for weighting more distant losses less
        t = torch.arange(T-1, dtype=torch.float32, device=batch.device).view(T-1, 1)  # 0..T-1
        lam = (self.cfg.lam ** t).expand(T-1, B)  # shape [T,B,N]

        # Encode observations to Teacher latent space 
        teacher_latent = model.encoder_model.teacher_encoder.encode(batch["obs"]) # [T, B, Z]
        # Encode observations to the Student latent space
        all_zbar = model.encoder_model.student_encoder.encode(teacher_latent) # [T, B, Zbar]
        # Get the target encodings, if requested
        if self.cfg.use_target_encoder:
            target_zbar = model.encoder_model.student_encoder.encode_target(teacher_latent)

        # Create useful quantities
        a = batch["action"][1:]
        r = batch["reward"][1:]
        zbar = all_zbar[:-1]
        zbar_prime = all_zbar[1:] if not self.cfg.use_target_encoder else target_zbar[1:]
        zbar_prime_prediction = model.dynamics_model.dynamics(zbar, a) # [T-1, B, Zbar]
        r_prediction = model.reward_model.reward(zbar, a) # [T-1, B, 1]

        # KL Loss: Get the Student's encoder distribution & the prior we are regularizing towards
        #
        # `stop_kl_loss` ablates the KL term's *gradient into the encoder* rather than the term
        # itself, as in the stochastic loss: every encoder-produced quantity it touches is a
        # constant, including the `zbar` the `dynamics`-mode prior is conditioned on. In
        # `dynamics` mode the term still trains the dynamics model; in `centered` and `standard`
        # modes there is no head behind it, so it trains nothing.
        kl_zbar = all_zbar.detach() if self.cfg.stop_kl_loss else all_zbar
        if self.cfg.kl_mode == "centered":
            mean_diff = kl_zbar - torch.mean(kl_zbar, axis=-1, keepdim=True)
        elif self.cfg.kl_mode == "dynamics":
            # dynamics prediction over the next latent, [T-1, B, Zbar]
            prior = (
                model.dynamics_model.dynamics(zbar.detach(), a)
                if self.cfg.stop_kl_loss else zbar_prime_prediction
            )
            posterior = kl_zbar[1:] # the encoder's own next latent, [T-1, B, Zbar]
            mean_diff = prior-posterior
        elif self.cfg.kl_mode == "standard":
            mean_diff = kl_zbar
        else:
            raise ValueError(
                f"{self.cfg.kl_mode} is not a valid choice for `kl_mode`."
                "Please choose one of 'centered', 'dynamics', or 'standard'."
            )
        zbar_kl = 0.5*torch.mean(mean_diff**2) # mean over [T, B] of distributions

        # Dynamics Loss: Get the true next states, and the Student's distribution over next states
        #
        # `stop_dynamics_loss` detaches both the latent the rollout starts from and the target it
        # is scored against, so the term trains the dynamics model alone. `alpha` already controls
        # how much of the *target's* gradient reaches the encoder; the flag overrides it and also
        # cuts the input side, which `alpha` never touched.
        zbar_ = zbar.detach() if self.cfg.stop_dynamics_loss else zbar
        zbar_prime_ = (1-self.cfg.alpha)*zbar_prime.detach() + self.cfg.alpha*zbar_prime # Don't let dynamics error flow to the encoder
        if self.cfg.stop_dynamics_loss: zbar_prime_ = zbar_prime_.detach()
        a_ = a
        zbar_prime_mse = torch.tensor(0.0, device=a_.device)
        lam_ = lam
        rho_ = 1.0
        for _ in range(self.cfg.latent_dynamics_imagination_horizon):
            # Compute prediction loss
            zbar_prime_prediction_ = model.dynamics_model.dynamics(zbar_, a_)
            squared_error = ((zbar_prime_prediction_ - zbar_prime_)**2).sum(dim=-1)
            l = rho_ * lam_ * 0.5*squared_error / zbar_prime_.shape[-1]
            zbar_prime_mse += torch.mean(l)
            # Step the variables
            zbar_ = zbar_prime_prediction_[:-1]
            a_ = a_[1:]
            zbar_prime_ = zbar_prime_[1:]
            lam_ = lam_[1:]
            rho_ = rho_*self.cfg.rho
        if self.cfg.latent_dynamics_imagination_horizon != 0: zbar_prime_mse /= self.cfg.latent_dynamics_imagination_horizon

        # Reward Loss: Get the Student's distribution over rewards
        # For multi-step loss, r_ll is the ll
        # `stop_reward_loss` detaches the starting latent, so reward error trains the reward head
        # (and, past the first imagined step, the dynamics model) but not the encoder.
        zbar_ = zbar.detach() if self.cfg.stop_reward_loss else zbar
        a_ = a
        r_ = r
        r_mse = torch.tensor(0.0, device=a_.device)
        lam_ = lam
        rho_ = 1.0
        for _ in range(self.cfg.latent_rewards_imagination_horizon):
            r_prediction_ = model.reward_model.reward(zbar_, a_)
            squared_error = ((r_prediction_ - r_)**2).sum(dim=-1)
            l = rho_*lam_* 0.5*squared_error
            r_mse += torch.mean(l)
            zbar_ = model.dynamics_model.dynamics(zbar_, a_)[:-1]
            a_ = a_[1:]
            r_ = r_[1:]
            lam_ = lam_[1:]
            rho_ = rho_ * self.cfg.rho
        if self.cfg.latent_rewards_imagination_horizon != 0: r_mse /= self.cfg.latent_rewards_imagination_horizon
        
        loss = torch.mean(
            self.cfg.dynamics_mse_coefficient*zbar_prime_mse
            +self.cfg.reward_mse_coefficient*r_mse
            +self.cfg.kl_coefficient*zbar_kl
        )
        
        # Sum them up
        return loss, {
            # Loss but as a logged metric
            "loss": loss.detach(),
            # Weighted, horizon-averaged components of the loss
            "dynamics_loss": zbar_prime_mse.detach(),
            "reward_loss": r_mse.detach(),
            "encoder_kl_divergence": zbar_kl.detach(),
            # Variance of the outputs across time (should never go to zero)
            "dynamics_output_variance": torch.mean(zbar_prime_prediction.var(dim=0)).detach(),
            "reward_output_variance": torch.mean(r_prediction.var(dim=0)).detach(),
            "encoder_output_variance": torch.mean(all_zbar.var(dim=0)).detach(),
            # Single-step MSE, defined as in the stochastic loss so the two are comparable
            "reward_mse": torch.mean((r_prediction - r) ** 2).detach(),
            "dynamics_mse": torch.mean((zbar_prime_prediction - zbar_prime) ** 2).detach(),
        }