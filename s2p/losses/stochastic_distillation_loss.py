import torch
from torch.distributions import Independent, Normal, kl_divergence

from s2p.models.agent_model import AgentModel
from s2p.models.teacher_student_encoder_model import TeacherStudentEncoderModel
from s2p.models.base.ema_target_encoder_base import EMATargetEncoderBase
from s2p.models.stochastic_mlp_dynamics_model import StochasticMLPDynamicsModel
from s2p.models.stochastic_mlp_reward_model import StochasticMLPRewardModel

from s2p.losses.loss_function_base import LossFunctionBase

class StochasticDistillationLoss(LossFunctionBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

    def _advance_imagination(self, dynamics_distribution):
        """
        Take one step of an imagination rollout from the dynamics model's prediction,
        per `cfg.imagination_rollout_method`.

        'sample' — an rsample, so the rollout is a trajectory drawn from the model and each
                   imagination step compounds the model's own predictive noise.
        'mean'   — the distribution's mean, which is what the planner chains when the
                   dynamics model runs with `sample_mean` set. Matching it here makes the
                   multi-step losses train on the same rollout the planner will execute.

        Note this is deliberately *not* `dynamics_model.dynamics()`: that follows the
        model's `sample_mean`, which is a statement about what consumers are handed, and
        this loss should not change meaning when the planner is reconfigured.
        """
        if self.cfg.imagination_rollout_method == "sample":
            return dynamics_distribution.rsample()
        if self.cfg.imagination_rollout_method == "mean":
            return dynamics_distribution.mean
        raise ValueError(
            f"{self.cfg.imagination_rollout_method} is not a valid choice for "
            "`imagination_rollout_method`. Please choose one of 'sample' or 'mean'."
        )

    def __call__(self, batch, model:AgentModel):

        T, B = batch.shape

        # Check that the model is a valid choice for this loss
        assert isinstance(model.encoder_model, TeacherStudentEncoderModel)
        assert (
            isinstance(model.encoder_model.student_encoder, torch.nn.Module)
            and isinstance(model.dynamics_model, StochasticMLPDynamicsModel)
            and isinstance(model.reward_model, StochasticMLPRewardModel)
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
        zbar_distribution = model.encoder_model.student_encoder.encode_distribution(teacher_latent) # [T, B] of distributions
        sample_zbar = zbar_distribution.rsample()
        # Get the target encodings, if requested. Sampled from the EMA distribution rather
        # than taken from `encode_target`, which follows `cfg.sample_mean`: what this loss
        # regresses its dynamics onto should not change when the planner is switched between
        # sampled and mean rollouts.
        if self.cfg.use_target_encoder:
            target_zbar_distribution = model.encoder_model.student_encoder.encode_target_distribution(
                teacher_latent)

        # Create useful quantities
        a = batch["action"][1:]
        r = batch["reward"][1:]
        zbar = sample_zbar[:-1]

        # What dynamics targets are we trying to hit? The output from encoder or target encoder?
        target_distribution = (
            target_zbar_distribution if self.cfg.use_target_encoder else zbar_distribution
        )
        # What dynamics targets are we trying to hit? The sample or the mean?
        if self.cfg.dynamics_target == "sample":
            zbar_prime = (
                target_distribution.rsample()[1:] if self.cfg.use_target_encoder
                else sample_zbar[1:]
            )
        elif self.cfg.dynamics_target == "mean":
            zbar_prime = target_distribution.mean[1:]
        else:
            raise ValueError(
                f"{self.cfg.dynamics_target} is not a valid choice for `dynamics_target`. "
                "Please choose one of 'sample' or 'mean'."
            )

        zbar_prime_prior = model.dynamics_model.dynamics_distribution(zbar, a)
        r_distribution = model.reward_model.reward_distribution(zbar, a)

        # KL Loss: Get the Student's encoder distribution & the prior we are regularizing towards
        #
        # `stop_kl_loss` ablates the KL term's *gradient into the encoder* rather than the term
        # itself: the loss is still computed and still trains the dynamics model (in `dynamics`
        # mode the prior is its prediction), but every encoder-produced quantity it touches is a
        # constant. That means detaching both sides -- the posterior's parameters and, in
        # `dynamics` mode, the `zbar` the prior is conditioned on -- since either one alone leaves
        # a path back to the encoder.
        kl_zbar_distribution = (
            Independent(Normal(zbar_distribution.mean.detach(), zbar_distribution.stddev.detach()), 1)
            if self.cfg.stop_kl_loss else zbar_distribution
        )
        if self.cfg.kl_mode == "standard":
            # Regularize the encoder distribution towards a standard normal prior
            kl_posterior = kl_zbar_distribution
            kl_prior = Independent(Normal(torch.zeros_like(sample_zbar), torch.ones_like(sample_zbar)), 1) # [T, B] of distributions
        elif self.cfg.kl_mode == "dynamics":
            # Regularize the encoder's distribution over the next latent towards the dynamics model's prediction
            kl_posterior = Independent(Normal(kl_zbar_distribution.mean[1:], kl_zbar_distribution.stddev[1:]), 1) # [T-1, B] of distributions
            # dynamics prediction over the next latent, [T-1, B] of distributions
            kl_prior = (
                model.dynamics_model.dynamics_distribution(zbar.detach(), a)
                if self.cfg.stop_kl_loss else zbar_prime_prior
            )
        else:
            raise ValueError(
                f"{self.cfg.kl_mode} is not a valid choice for `kl_mode`."
                "Please choose one of 'dynamics' or 'standard'."
            )
        zbar_kl = torch.mean(kl_divergence(kl_posterior, kl_prior)/sample_zbar.shape[-1]) # mean over [T, B] of distributions

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
        zbar_prime_ll = torch.tensor(0.0, device=a_.device)
        lam_ = lam
        rho_ = 1.0
        for _ in range(self.cfg.latent_dynamics_imagination_horizon):
            zbar_prime_prior_ = model.dynamics_model.dynamics_distribution(zbar_, a_)
            l = rho_ * lam_ * zbar_prime_prior_.log_prob(zbar_prime_) / zbar_prime.shape[-1]
            zbar_prime_ll += torch.mean(l)
            # Advance the imagined latent, reusing the distribution the log-prob above was
            # taken from rather than running the dynamics model a second time.
            zbar_ = self._advance_imagination(zbar_prime_prior_)[:-1]
            a_ = a_[1:]
            zbar_prime_ = zbar_prime_[1:]
            lam_ = lam_[1:]
            rho_ = rho_*self.cfg.rho
        if self.cfg.latent_dynamics_imagination_horizon != 0: zbar_prime_ll /= self.cfg.latent_dynamics_imagination_horizon

        # Reward Loss: Get the Student's distribution over rewards
        # For multi-step loss, r_ll is the ll
        # `stop_reward_loss` detaches the starting latent, so reward error trains the reward head
        # (and, past the first imagined step, the dynamics model) but not the encoder.
        zbar_ = zbar.detach() if self.cfg.stop_reward_loss else zbar # Do let the reward on reward error flow to the encoder
        a_ = a
        r_ = r
        r_ll = torch.tensor(0.0, device=a_.device)
        lam_ = lam
        rho_ = 1.0
        for _ in range(self.cfg.latent_rewards_imagination_horizon):
            r_distribution_ = model.reward_model.reward_distribution(zbar_, a_)
            l = rho_*lam_*r_distribution_.log_prob(r_)
            r_ll += torch.mean(l)
            zbar_ = self._advance_imagination(
                model.dynamics_model.dynamics_distribution(zbar_, a_))[:-1]
            a_ = a_[1:]
            r_ = r_[1:]
            lam_ = lam_[1:]
            rho_ = rho_ * self.cfg.rho
        if self.cfg.latent_rewards_imagination_horizon != 0: r_ll /= self.cfg.latent_rewards_imagination_horizon
        
        loss = torch.mean(
            -self.cfg.dynamics_ll_coefficient*zbar_prime_ll
            -self.cfg.reward_ll_coefficient*r_ll
            +self.cfg.kl_coefficient*zbar_kl
        )
        
        # Sum them up
        return loss, {
            # Loss but as a logged metric
            "loss": loss.detach(),
            # Log-likelihood components of the loss
            "dynamics_log_likelihood": zbar_prime_ll.detach(),
            "reward_log_likelihood": r_ll.detach(),
            "encoder_kl_divergence": zbar_kl.detach(),
            # Variance of the output gaussian (should not immediately go to zero)
            "encoder_distribution_variance": torch.mean(zbar_distribution.variance).detach(),
            "dynamics_distribution_variance": torch.mean(zbar_prime_prior.variance).detach(),
            "reward_distribution_variance": torch.mean(r_distribution.variance).detach(),
            # Variance of the output gaussian means across all [T, B] entries in the batch (should never go to zero)
            "dynamics_output_variance": torch.mean(zbar_prime_prior.mean.var(dim=(0,1))).detach(),
            "reward_output_variance": torch.mean(r_distribution.mean.var(dim=(0,1))).detach(),
            "encoder_output_variance": torch.mean(zbar_distribution.mean.var(dim=(0,1))).detach(),
            # Log-probability of the means
            "ll_mean_zbar_prime": zbar_prime_prior.log_prob(torch.mean(zbar_prime, dim=(0,1)).repeat(*zbar_prime.shape[:2], 1))[0,0].detach(),
            "ll_mean_reward": r_distribution.log_prob(torch.mean(batch["reward"][1:], dim=(0,1)).repeat(*batch["reward"][1:].shape[:2], 1))[0,0].detach(),
            # MSE
            "reward_mse": torch.mean((r_distribution.mean - batch["reward"][1:]) ** 2).detach(),
            "dynamics_mse": torch.mean((zbar_prime_prior.mean - zbar_prime) ** 2).detach(),
            "dynamics_mse_to_mean": torch.mean(
                (zbar_prime_prior.mean - target_distribution.mean[1:]) ** 2).detach()
        }