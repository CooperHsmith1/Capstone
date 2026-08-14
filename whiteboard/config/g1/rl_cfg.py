"""RL configurations for Unitree G1 tasks."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO runner for the G1 velocity (locomotion) task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_velocity",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def unitree_g1_drawing_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO runner for the G1 whiteboard drawing task.

  Key differences from the locomotion runner:

  Network:
    Smaller hidden dims (256, 128, 64) — the drawing observation space is
    much smaller than locomotion (no height scan, no foot sensors).  A large
    network overfits early and slows convergence.

  Action std:
    Lower init_std (0.5 vs 1.0) — the arm needs fine-grained control from
    the start.  High initial noise causes the arm to flail and makes the
    pen-tip reward signal too sparse to bootstrap from.

  Entropy:
    Lower entropy_coef (0.005 vs 0.01) — we want the policy to exploit the
    dense pen-tracking reward quickly rather than explore widely.  The
    command resampling already provides sufficient diversity.

  Rollout length:
    Longer num_steps_per_env (48 vs 24) — the arm needs several seconds of
    context to learn to hold the pen on the board while tracking a moving
    target.  Longer rollouts give the value function a better signal.

  Learning rate / schedule:
    Kept at 1e-3 adaptive — works well for both tasks.

  Discount:
    gamma=0.99 (same) — the 20 s episode with decimation=4 and dt=0.005
    gives ~1000 steps; gamma=0.99 discounts the horizon to ~100 steps,
    which is appropriate for a reaching task.

  Iterations:
    Fewer max_iterations (15_000) — the task is simpler than full locomotion
    so it converges faster.  Increase to 30_000 when you add path-following.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        # Lower initial std: arm control requires precision from early in
        # training.  Too-high noise makes the pen-tracking reward too sparse.
        "init_std": 0.5,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      # Lower entropy: exploit the dense pen-tracking reward rather than
      # exploring.  The draw_target command resampling drives diversity.
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_drawing",
    save_interval=50,
    # Longer rollouts: the arm needs several seconds of context to learn
    # to hold the pen on the board and track a target smoothly.
    num_steps_per_env=48,
    max_iterations=15_000,
  )
