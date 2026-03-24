from dataclasses import dataclass


# source: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax_scan.py
@dataclass
class PPOArgs:
    # the name of this experiment
    exp_name: str = "brittle_star_ppo"

    # seed of the experiment
    seed: int = 1

    # if toggled, `torch.backends.cudnn.deterministic=False`
    torch_deterministic: bool = True

    # if toggled, cuda will be enabled by default
    cuda: bool = True

    # if toggled, this experiment will be tracked with Weights and Biases
    track: bool = False

    # the wandb's project name
    wandb_project_name: str = "PPO-Modularity"

    # the entity (team) of wandb's project
    wandb_entity: str | None = None

    # whether to capture videos of the agent performances (check out `videos` folder)
    capture_video: bool = False

    # whether to save model into the `runs/{run_name}` folder
    save_model: bool = False

    # whether to upload the saved model to huggingface
    upload_model: bool = False

    # the user or org name of the model repository from the Hugging Face Hub
    hf_entity: str = ""

    # ==== Algorithm specific dataclasses ====
    # the id of the environment
    env_id: str = "" # todo

    # total timesteps of the experiments
    total_timesteps: int = 10000000

    # the learning rate of the optimizer
    learning_rate: float = 2.5e-4

    # the number of parallel game environments
    num_envs: int = 16

    # the number of steps to run in each environment per policy rollout
    num_steps: int = 128

    # Toggle learning rate annealing for policy and value networks
    anneal_lr: bool = True

    # the discount factor gamma
    gamma: float = 0.99

    # the lambda for the general advantage estimation
    gae_lambda: float = 0.95

    # the number of mini-batches
    num_minibatches: int = 4

    # the K epochs to update the policy
    update_epochs: int = 4

    # Toggles advantages normalization
    norm_adv: bool = True

    # the surrogate clipping coefficient
    clip_coef: float = 0.1

    # Toggles whether or not to use a clipped loss for the value function, as per the paper.
    clip_vloss: bool = True

    # coefficient of the entropy
    ent_coef: float = 0.01

    # coefficient of the value function
    vf_coef: float = 0.5

    # the maximum norm for the gradient clipping
    max_grad_norm: float = 0.5

    # the target KL divergence threshold
    target_kl: float | None = None

    # ==== to be filled in runtime ====
    # the batch size (computed in runtime)
    batch_size: int = 0

    # the mini-batch size (computed in runtime)
    minibatch_size: int = 0

    # the number of iterations (computed in runtime)
    num_iterations: int = 0
