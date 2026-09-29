# Squeeze-to-Plan

## Note for ICLR Reviewers
All configs referenced below are in `s2p/configs/`.

**Training S2P.** The training configs used for S2P in the paper, one per task:
- `train_visual_online_deterministic_lift_peg.yaml`
- `train_visual_online_deterministic_push_cube.yaml`
- `train_visual_online_deterministic_pick_cube.yaml`

**Evaluation.** The evaluation configs used for each method follow the pattern `evaluate_<method>_<task>.yaml`, with `<task>` one of `lift_peg`, `push_cube`, `pick_cube`:

| Method | Evaluation configs (lift_peg / push_cube / pick_cube) |
|---|---|
| `s2p` | `evaluate_s2p_lift_peg.yaml` / `evaluate_s2p_push_cube.yaml` / `evaluate_s2p_pick_cube.yaml` |
| `tdmpc2` | `evaluate_tdmpc2_lift_peg.yaml` / `evaluate_tdmpc2_push_cube.yaml` / `evaluate_tdmpc2_pick_cube.yaml` |
| `dino_wm` | `evaluate_dino_wm_lift_peg.yaml` / `evaluate_dino_wm_push_cube.yaml` / `evaluate_dino_wm_pick_cube.yaml` |
| `dino_bisim` | `evaluate_dino_bisim_lift_peg.yaml` / `evaluate_dino_bisim_push_cube.yaml` / `evaluate_dino_bisim_pick_cube.yaml` |
| `tc_wm` | `evaluate_tc_wm_lift_peg.yaml` / `evaluate_tc_wm_push_cube.yaml` / `evaluate_tc_wm_pick_cube.yaml` |
| `sparse_imagination` | `evaluate_sparse_imagination_lift_peg.yaml` / `evaluate_sparse_imagination_push_cube.yaml` / `evaluate_sparse_imagination_pick_cube.yaml` |

To train an agent with our method (S2P), first check that you have the relevant additional data required (e.g. offline datasets, pretrained checkpoints). These will be released with the de-anonymized version of this code.

Training runs with S2P are managed using Hydra configs. For an example, take a look at: `s2p/configs/visual_offline_privileged_training_nonlinear_encoder.yaml`. The configs will specify the following fields:
- `runner`: Either `training` or `evaluation`, representing whether we want to run training or just a single evaluation.
- `trainer`: Either `online` or `offline`, representing how we want to run training.
- `task`: Configuration for a `Task` object, specifying a task which the robot can be trained to perform. See `s2p/tasks` for the available `Task` classes.
- `agent`: Configuration for an `Agent` object, which contains some combination of an `Encoder` model, a `Dynamics` model, a `Reward` model, a `Policy` model, and a `value` model, representing an agent which can act in the environment. See `s2p/models` for the list of available classes which can be used for these components.
- `planner`: Configuration for a `Planner` object, which can take an agent containing an `Encoder` model, `Dynamics` model, `Reward` model and `Value` model and search over possible action sequences to find the best one. See `s2p/planners` for a list of currently supported classes of planners.
- `evaluators`: A dictionary of configurations for `Evaluator` classes, which run different types of evaluations on the model during training. See `s2p/evaluators` for a list of currently supported evaluators.
- `loss_functions`: A dictionary containing configurations for `LossFunction` classes. See `s2p/losses` for a list of currently supported classes of loss functions.
- `loss_functions`: A dictionary containing configurations for `LossFunction` classes. See `s2p/losses` for a list of currently supported classes of loss functions.
- `loss_function_weights`: A dictionary containing weights to use for summing up the `loss_functions`.
- `latent_dimension`: The dimensionality of the student latent space.
- `device`: Name of the torch device on which to run training.
- `seed`: Seed for the training run.
- `data_dir`: Path to the directory where offline datasets are kept. This is used by the various `s2p/tasks` classes to locate their offline datasets for offline training.
- `checkpoint_dir`: Path to the directory where pretrained checkpoints are kept. This is used by the various `s2p/models` classes which rely on pre-trained weights (e.g. DINO).
- `output_dir`: Path to the folder where training outputs (checkpoints, hydra outputs, wandb logs) should be written to.
