## Default envconfig
task: Task = Task.DIRECTED_LOCOMOTION
simulation_time: float = 5.0
num_physics_steps_per_control_step: int = 10
time_scale: int = 2
camera_ids: list[int] = field(default_factory=lambda: [0, 1])
render_size: tuple[int, int] = (480, 640)
joint_randomization_noise_scale: float = 0.0
target_distance: float = 3.0
light_perlin_noise_scale: int = 0


## Default ppoargs
seed: int = 1
torch_deterministic: bool = True
cuda: bool = True
track: bool = False
checkpoint_frequency: int = 100
learning_rate: float = 2.5e-4
anneal_lr: bool = True
gamma: float = 0.99
gae_lambda: float = 0.95
num_minibatches: int = 4
update_epochs: int = 4
norm_adv: bool = True
clip_coef: float = 0.1
clip_vloss: bool = True
ent_coef: float = 0.01
vf_coef: float = 0.5
max_grad_norm: float = 0.5
target_kl: float | None = None
batch_size: int = 0
minibatch_size: int = 0
num_iterations: int = 0

## Used config file:
num_envs: 32
num_steps: 32
total_timesteps: 102400

## Arena config:
size: tuple[float, float] = (10.0, 5.0)
sand_ground_color: bool = True
attach_target: bool = True
wall_height: float = 1.5
wall_thickness: float = 0.1

## Morphology:
num_arms: int = 5
num_segments_per_arm: int = 4
use_p_control: bool = True
use_torque_control: bool = False

## MLPs:
### Sensor & Feature_extractor:
class GenericDenseLayersWithActivation(nn.Module):
    layer_sizes: Sequence[int] = field(default_factory=lambda: [64, 64])
    activation: Callable = nn.tanh

    @nn.compact
    def __call__(self, x):
        for size in self.layer_sizes:
            x = nn.Dense(size, kernel_init=orthogonal(jnp.sqrt(2)))(x)
            x = self.activation(x)
        return x

### Actor:
class Actor(nn.Module):
    action_dim: int
    @nn.compact
    def __call__(self, x):
        mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        return mean, log_std

### Critic:
class OneDenseLayerMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)

### Observations:
