"""Task registration for the Unitree G1 whiteboard drawing task.

Registers the ground-truth task plus one variant per state estimator, so the
comparative filtering study of report Section 4.3 can be run from the CLI
without editing any config:

    uv run train Mjlab-Drawing-Flat-Unitree-G1     --env.scene.num-envs 4096
    uv run train Mjlab-Drawing-Flat-Unitree-G1-PF  --env.scene.num-envs 4096
    uv run train Mjlab-Drawing-Flat-Unitree-G1-EKF --env.scene.num-envs 4096
    uv run train Mjlab-Drawing-Flat-Unitree-G1-UKF --env.scene.num-envs 4096

The base task feeds the policy the true pen position. The three filter variants
feed it a filtered estimate derived from a noisy synthetic sensor, which is the
configuration that can actually transfer to hardware. Training the base task
and one filter variant and comparing them is the sim-to-real experiment.
"""


def _register() -> None:
  # Local imports to avoid a circular import at module load time.
  from mjlab.tasks.registry import register_mjlab_task
  from mjlab.tasks.whiteboard.mdp.state_estimation import StateEstimationCfg

  from .env_cfgs import unitree_g1_drawing_env_cfg
  from .rl_cfg import unitree_g1_drawing_ppo_runner_cfg

  register_mjlab_task(
    task_id="Mjlab-Drawing-Flat-Unitree-G1",
    env_cfg=unitree_g1_drawing_env_cfg(),
    play_env_cfg=unitree_g1_drawing_env_cfg(play=True),
    rl_cfg=unitree_g1_drawing_ppo_runner_cfg(),
  )

  # All three variants share ONE sensor model. This matters: if the particle
  # filter variant ran with outliers enabled and the Kalman variants did not,
  # the three task ids would not be comparable and the PF would look better
  # purely because it was scored on a different measurement stream. The outlier
  # rate is on for all of them because that is the regime the report argues a
  # real depth/vision pen tracker actually lives in.
  filters = {
    "PF": StateEstimationCfg(
      filter_type="particle",
      num_particles=512,
      proposal="optimal",
      sensor_noise_std=0.01,
      outlier_rate=0.05,
      outlier_scale=25.0,
    ),
    "EKF": StateEstimationCfg(
      filter_type="ekf",
      sensor_noise_std=0.01,
      outlier_rate=0.05,
      outlier_scale=25.0,
    ),
    "UKF": StateEstimationCfg(
      filter_type="ukf",
      sensor_noise_std=0.01,
      outlier_rate=0.05,
      outlier_scale=25.0,
    ),
  }

  for suffix, est_cfg in filters.items():
    register_mjlab_task(
      task_id=f"Mjlab-Drawing-Flat-Unitree-G1-{suffix}",
      env_cfg=unitree_g1_drawing_env_cfg(state_estimation=est_cfg),
      play_env_cfg=unitree_g1_drawing_env_cfg(play=True, state_estimation=est_cfg),
      rl_cfg=unitree_g1_drawing_ppo_runner_cfg(),
    )


_register()
