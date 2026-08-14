def _register() -> None:
    # Local imports to avoid circular import at module load time.
    from mjlab.tasks.registry import register_mjlab_task
    from .env_cfgs import (
        unitree_g1_drawing_env_cfg,
    )
    from .rl_cfg import (
        unitree_g1_drawing_ppo_runner_cfg,
    )

    register_mjlab_task(
        task_id="Mjlab-Drawing-Flat-Unitree-G1",
        env_cfg=unitree_g1_drawing_env_cfg(),
        play_env_cfg=unitree_g1_drawing_env_cfg(play=True),
        rl_cfg=unitree_g1_drawing_ppo_runner_cfg(),
    )


_register()
