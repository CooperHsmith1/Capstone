"""Whiteboard drawing task configuration for the Unitree G1.

Builds a ``ManagerBasedRlEnvCfg`` for a fixed-base (by default) G1 holding a pen
in front of a whiteboard, rewarded for tracking randomly sampled target points
on the board face.

Optionally routes the pen-tip observation through a probabilistic state
estimator (particle filter, EKF or UKF) instead of feeding the policy ground
truth -- see ``mdp/state_estimation/`` and Section 4 of the interim report.
"""

from __future__ import annotations

import math
from copy import deepcopy

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
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from .mdp.assets import (
    PEN_TIP_SITE,
    get_g1_whiteboard_robot_cfg,
    whiteboard_spec_fn,
)
from .mdp.board import (
    BOARD_FACE_X,
    TARGET_Y_RANGE,
    TARGET_Z_RANGE,
    validate as validate_board_geometry,
)
from .mdp.draw_target_cmd import DrawTargetCommandCfg
from .mdp.observations import (
    estimated_pen_position,
    estimated_pen_velocity,
    pen_position_uncertainty,
    site_position,
)
from .mdp.rewards import (
    pen_contact_reward,
    pen_tracking_reward,
    smooth_pen_motion_reward,
    upright_reward,
)
from .mdp.state_estimation import StateEstimationCfg, reset_state_estimator

validate_board_geometry()

# ---------------------------------------------------------------------------
# Shared SceneEntityCfg objects
# ---------------------------------------------------------------------------

_PEN_TIP_CFG = SceneEntityCfg("robot", site_names=(PEN_TIP_SITE,))

RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

_RIGHT_ARM_CFG = SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS)
_TORSO_CFG = SceneEntityCfg("robot", body_names=("torso_link",))


def make_drawing_env_cfg(
    num_envs: int = 4096,
    fixed_base: bool = True,
    state_estimation: StateEstimationCfg | None = None,
) -> ManagerBasedRlEnvCfg:
    """Create the whiteboard drawing environment configuration.

    Args:
        num_envs: Parallel environments. Thousands on GPU; drop to 32-64 on CPU
            while debugging reward shaping.
        fixed_base: Bolt the robot to the world, removing the balance problem.
            Strongly recommended while the reaching behaviour is still being
            learned. Note that with ``fixed_base=True`` the ``upright`` reward
            and ``fell_over`` termination cannot fire, because the torso cannot
            tip -- they are kept registered so the same config works in both
            modes, but they contribute nothing here.
        state_estimation: If given, the policy sees a *filtered* pen state
            derived from a noisy synthetic sensor instead of ground truth.
            ``None`` keeps ground truth, which trains faster but gives the
            policy information no real robot has.
    """
    filtered = state_estimation is not None
    est_params = {"estimator_cfg": state_estimation} if filtered else {}

    # -- observations -------------------------------------------------------
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
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.1, n_max=0.1)
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "pen_pos": (
            ObservationTermCfg(func=estimated_pen_position, params=est_params)
            if filtered
            else ObservationTermCfg(
                func=site_position, params={"asset_cfg": _PEN_TIP_CFG}
            )
        ),
        "target_pos": ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "draw_target"}
        ),
    }

    if filtered:
        # Velocity and uncertainty are free once a filter is running, and the
        # uncertainty channel is what lets the policy learn to slow down when
        # the estimate is poor rather than committing to a bad stroke.
        actor_terms["pen_vel_est"] = ObservationTermCfg(
            func=estimated_pen_velocity, params=est_params
        )
        actor_terms["pen_pos_std"] = ObservationTermCfg(
            func=pen_position_uncertainty, params=est_params
        )

    # deepcopy, NOT the same dict object. Sharing one dict between the actor and
    # critic groups means any later edit to one group silently edits the other,
    # and a `del cfg.observations["actor"].terms[...]` raises KeyError on the
    # second delete. The critic also gets clean observations by design: it is
    # only used at training time, so feeding it uncorrupted state is legitimate
    # asymmetric-information training and gives a lower-variance value estimate.
    observations = {
        "actor": ObservationGroupCfg(
            terms=deepcopy(actor_terms),
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            terms=deepcopy(actor_terms),
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    # -- actions ------------------------------------------------------------
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=RIGHT_ARM_JOINTS,
            scale=0.25,
            use_default_offset=True,
        )
    }

    # -- commands -----------------------------------------------------------
    commands: dict[str, CommandTermCfg] = {
        "draw_target": DrawTargetCommandCfg(
            resampling_time_range=(4.0, 8.0),
            board_face_x=BOARD_FACE_X,
            board_y_range=TARGET_Y_RANGE,
            board_z_range=TARGET_Z_RANGE,
        ),
    }

    # -- events -------------------------------------------------------------
    events: dict[str, EventTermCfg] = {
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.05, 0.05),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
    }

    # There is deliberately no reset event for the whiteboard: it is a static
    # jointless body, and running reset_root_state_uniform on it would write a
    # root velocity it has no degrees of freedom to hold.
    #
    # There is also no reset_base event. With fixed_base=True the robot has no
    # free joint, so reset_root_state_uniform would have no root state to write
    # and fails. The floating-base case re-adds it below.
    if not fixed_base:
        events["reset_base"] = EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                               "yaw": (0.0, 0.0)},
                "velocity_range": {},
            },
        )

    if filtered:
        # Without this the filter carries its belief across the episode
        # boundary and spends the first steps of each episode chasing a pen
        # that teleported.
        events["reset_state_estimator"] = EventTermCfg(
            func=reset_state_estimator, mode="reset", params={}
        )

    # -- rewards ------------------------------------------------------------
    # These weights are the tuned values. Do not rebuild these terms in a
    # downstream config: reassigning a RewardTermCfg there silently reverts the
    # tuning. Override the .weight field instead.
    rewards = {
        "pen_tracking": RewardTermCfg(
            func=pen_tracking_reward,
            weight=10.0,
            params={"command_name": "draw_target", "asset_cfg": _PEN_TIP_CFG},
        ),
        "pen_contact": RewardTermCfg(
            func=pen_contact_reward,
            weight=2.0,
            params={"whiteboard_x": BOARD_FACE_X, "asset_cfg": _PEN_TIP_CFG},
        ),
        "smooth_pen_motion": RewardTermCfg(
            func=smooth_pen_motion_reward,
            weight=-0.01,
            params={"asset_cfg": _RIGHT_ARM_CFG},
        ),
        "upright": RewardTermCfg(
            func=upright_reward, weight=1.0, params={"asset_cfg": _TORSO_CFG}
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.005),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-0.05),
    }

    # -- terminations -------------------------------------------------------
    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    }
    if not fixed_base:
        terminations["fell_over"] = TerminationTermCfg(
            func=mdp.bad_orientation, params={"limit_angle": math.radians(60.0)}
        )

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=num_envs,
            extent=2.0,
            entities={"robot": get_g1_whiteboard_robot_cfg(fixed_base=fixed_base)},
            spec_fn=whiteboard_spec_fn,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            distance=3.5,
            elevation=-10.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            njmax=300,
            mujoco=MujocoCfg(timestep=0.005, iterations=20, ls_iterations=40),
        ),
        decimation=4,
        episode_length_s=20.0,
    )
