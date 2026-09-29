import torch
from torch.distributions import Independent, Normal, kl_divergence

from s2p.models.plannable_model_with_value_base import PlannableModelWithValueBase
from s2p.models.teacher_student_world_model import TeacherStudentWorldModel

from s2p.losses.loss_function_base import LossFunctionBase

class TeacherStudentValueLoss(LossFunctionBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
    
    def __call__(self, batch, model:PlannableModelWithValueBase):
        T, B = batch.shape
        # Check that the model is a valid choice for this loss
        # TODO: Come up with a better way to determine if a model has a policy and Q-function
        assert (
            isinstance(model, TeacherStudentWorldModel)
            and isinstance(model.student, PlannableModelWithValueBase)
            and isinstance(model.teacher, PlannableModelWithValueBase)
        )

        # Assert that trajectories are long enough for the desired latent imagination horizons
        assert self.cfg.latent_value_imagination_horizon < batch.shape[0]

        # Lambda value for weighting more distant losses less
        t = torch.arange(T-1, dtype=torch.float32, device=batch.device).view(T-1, 1)  # 0..T-1
        lam = (self.cfg.lam ** t).expand(T-1, B)  # shape [T,B,N]

        # Encode observations to Teacher latent space 
        teacher_state = model.teacher.encode(batch["obs"])[:-1] # [T, B, Z]

        # Get the target values
        value_target = model.teacher.value(teacher_state).detach().squeeze(-1)

        # Get the student state
        student_state = model.student.encode(teacher_state)
        action = batch["action"][1:]

        # Value Loss
        state_ = student_state
        a_ = action
        value_target_ = value_target
        value_loss = torch.tensor(0.0, device=student_state.device)
        lam_ = lam
        rho_ = 1.0
        for _ in range(self.cfg.latent_value_imagination_horizon):
            value_prediction = model.student.value(state_.detach()).squeeze(-1)
            l = rho_*lam_*(value_target_ - value_prediction)**2
            value_loss += torch.mean(l)
            # Increment the imagined state
            state_ = model.dynamics(state_, a_)[:-1]
            value_target_ = value_target_[1:]
            a_ = a_[1:]
            lam_ = lam_[1:]
            rho_ = rho_ * self.cfg.rho
        if self.cfg.latent_value_imagination_horizon != 0: value_loss /= self.cfg.latent_value_imagination_horizon
        
        loss = value_loss
        
        # Sum them up
        return loss, {
            # Loss but as a logged metric
            "loss": loss.detach(),
        }