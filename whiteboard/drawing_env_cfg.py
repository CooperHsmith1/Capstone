"""Whiteboard drawing task configuration for Unitree G1."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from .mdp.rewards import (
    pen_contact_reward,
    pen_tracking_reward,
    smooth_pen_motion_reward,
    upright_reward,
)
from .mdp.draw_target_cmd import DrawTargetCommandCfg
from .mdp.observations import site_position

if TYPE_CHECKING:
    import mujoco
    from mjlab.envs import ManagerBasedRlEnv

# ---------------------------------------------------------------------------
# Shared SceneEntityCfg objects
# ---------------------------------------------------------------------------

_PEN_TIP_CFG = SceneEntityCfg("robot", site_names=("pen_tip",))

_RIGHT_ARM_CFG = SceneEntityCfg(
    "robot",
    joint_names=(
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ),
)

_TORSO_CFG = SceneEntityCfg("robot", body_names=("torso_link",))


# ---------------------------------------------------------------------------
# Helpers for resolving mjlab internals at runtime
# ---------------------------------------------------------------------------

def _get_mj_model(env: ManagerBasedRlEnv) -> mujoco.MjModel:
    """Return the MjModel from an env, trying known mjlab attribute paths.

    mjlab has changed the path across versions:
      ≥ 1.x  env.scene.physics.model
      0.x    env.sim.model
    Raises AttributeError with a clear message if nothing works.
    """
    for path in (
        ("scene", "physics", "model"),
        ("sim", "model"),
        ("physics", "model"),
    ):
        obj = env
        try:
            for attr in path:
                obj = getattr(obj, attr)
            return obj  # type: ignore[return-value]
        except AttributeError:
            continue
    raise AttributeError(
        "Cannot locate MjModel on ManagerBasedRlEnv. "
        "Tried: env.scene.physics.model, env.sim.model, env.physics.model. "
        "Check your mjlab version and update _get_mj_model accordingly."
    )


def _get_mj_data(env: ManagerBasedRlEnv) -> mujoco.MjData:
    """Return the live MjData from an env, trying known mjlab attribute paths."""
    for path in (
        ("scene", "physics", "data"),
        ("sim", "data"),
        ("physics", "data"),
    ):
        obj = env
        try:
            for attr in path:
                obj = getattr(obj, attr)
            return obj  # type: ignore[return-value]
        except AttributeError:
            continue
    raise AttributeError(
        "Cannot locate MjData on ManagerBasedRlEnv. "
        "Tried: env.scene.physics.data, env.sim.data, env.physics.data. "
        "Check your mjlab version and update _get_mj_data accordingly."
    )


def _get_mjr_context(env: ManagerBasedRlEnv) -> mujoco.MjrContext | None:
    """Return the MjrContext if the viewer is open, or None otherwise."""
    for path in (
        ("_renderer", "mjr_context"),
        ("renderer", "mjr_context"),
        ("viewer", "mjr_context"),
        ("viewer", "ctx"),
    ):
        obj = env
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if obj is not None:
                return obj  # type: ignore[return-value]
        except AttributeError:
            continue
    return None


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def make_drawing_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create whiteboard drawing environment configuration."""

    actor_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        ),
        "actions": ObservationTermCfg(
            func=mdp.last_action,
        ),
        "pen_pos": ObservationTermCfg(
            func=site_position,
            params={"asset_cfg": _PEN_TIP_CFG},
        ),
        "target_pos": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "draw_target"},
        ),
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ),
            scale=0.25,
            use_default_offset=True,
        )
    }

    commands: dict[str, CommandTermCfg] = {
        "draw_target": DrawTargetCommandCfg(
            resampling_time_range=(3.0, 6.0),
            board_face_x=0.63,
            board_y_range=(-0.35, 0.35),
            board_z_range=(0.80, 1.40),
        ),
    }

    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.05, 0.05),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        # NOTE: NO reset event for the whiteboard. It is a fixed-base MOCAP body
        # (mjlab wraps it as whiteboard/mocap_base, mocap="true") placed at its
        # init_state pos (0.65, 0, 1.10) at build time. A mocap body is kinematic
        # and stays put on its own. Running reset_root_state_uniform on it injects
        # root velocity it cannot hold, which launches it ("fly away"). So we
        # simply do not reset it.
    }

    rewards = {
        "pen_tracking": RewardTermCfg(
            func=pen_tracking_reward,
            weight=10.0,
            params={"command_name": "draw_target", "asset_cfg": _PEN_TIP_CFG},
        ),
        "pen_contact": RewardTermCfg(
            func=pen_contact_reward,
            weight=2.0,
            params={"whiteboard_x": 0.63, "asset_cfg": _PEN_TIP_CFG},
        ),
        "smooth_pen_motion": RewardTermCfg(
            func=smooth_pen_motion_reward,
            weight=-0.01,          # was -0.5
            params={"asset_cfg": _RIGHT_ARM_CFG},
        ),
        "upright": RewardTermCfg(
            func=upright_reward,
            weight=1.0,            # was 2.0
            params={"asset_cfg": _TORSO_CFG},
        ),
        "action_rate_l2": RewardTermCfg(
            func=mdp.action_rate_l2,
            weight=-0.005,         # was -0.05
        ),
        "dof_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-0.05,          # was -1.0
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(
            func=mdp.time_out,
            time_out=True,
        ),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(60.0)},
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=1,
            extent=2.0,
            # terrain=TerrainEntityCfg(terrain_type="plane"),
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(
            # Track the ROBOT torso so the robot is always centered. The board
            # is static (jointless, cannot move) and sits ~0.65 m in front, so
            # a pulled-back side view keeps both in frame. (Earlier the camera
            # appeared to "follow the board" — that was the tracking target;
            # the board itself never moves.)
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            distance=3.5,
            elevation=-10.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=8192,
            njmax=1000,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=20,
                ls_iterations=40,
            ),
        ),
        decimation=4,
        episode_length_s=20.0,
    )


def make_drawing_env_with_canvas(device: str = "cpu"):
    """Build the env and wire DrawingCanvas into its post-physics callback.

    Args:
        device: PyTorch device string passed to ManagerBasedRlEnv, e.g.
                ``"cpu"`` or ``"cuda:0"``.

    Returns:
        env:    The constructed ManagerBasedRlEnv, ready to step.
        canvas: The DrawingCanvas instance.  Call ``canvas.reset()`` on
                episode boundaries if you want a fresh board each episode.

    Example::

        env, canvas = make_drawing_env_with_canvas()
        obs, _ = env.reset()
        for _ in range(1000):
            action = policy(obs)
            obs, reward, done, _, _ = env.step(action)
            if done:
                canvas.reset()
    """
    from mjlab.envs import ManagerBasedRlEnv
    from .mdp.DrawingCanvas import DrawingCanvas

    cfg = make_drawing_env_cfg()
    env = ManagerBasedRlEnv(cfg, device=device)

    # Resolve MjModel via the version-safe helper.
    canvas = DrawingCanvas(_get_mj_model(env))

    _original_post = getattr(env, "_post_physics_step", None)

    def _drawing_post_physics() -> None:
        canvas.step(_get_mj_data(env))
        ctx = _get_mjr_context(env)
        if ctx is not None:
            canvas.upload_texture(_get_mj_model(env), ctx)
        if _original_post is not None:
            _original_post()

    env._post_physics_step = _drawing_post_physics  # type: ignore[method-assign]

    return env, canvas