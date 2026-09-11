from __future__ import annotations

import mujoco

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_ARTICULATION,
  KNEES_BENT_KEYFRAME,
  get_spec,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  action_rate_l2,
  joint_pos_limits,
  joint_pos_rel,
  joint_vel_rel,
  last_action,
)
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.events import reset_joints_by_offset
from mjlab.envs.mdp.terminations import time_out
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.whiteboard import mdp
from mjlab.tasks.whiteboard.mdp.commands import CircleDrawingCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

_PEN_TIP_SITE = "pen_tip"
_PEN_PARENT_BODY = "right_wrist_yaw_link"
_RIGHT_ARM_JOINTS = (
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)


def _get_g1_whiteboard_spec() -> mujoco.MjSpec:
  spec = get_spec()
  parent = next((body for body in spec.bodies if body.name == _PEN_PARENT_BODY), None)
  if parent is None:
    raise ValueError(f"Cannot attach pen: G1 body '{_PEN_PARENT_BODY}' is missing")
  pen = parent.add_body(name="pen", pos=(0.08, 0.0, 0.0))
  pen.add_geom(
    name="pen_body",
    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    fromto=(0.0, 0.0, 0.0, 0.16, 0.0, 0.0),
    size=(0.008, 0.0, 0.0),
    mass=0.02,
    rgba=(0.15, 0.15, 0.18, 1.0),
    contype=0,
    conaffinity=0,
  )
  pen.add_site(
    name=_PEN_TIP_SITE,
    pos=(0.16, 0.0, 0.0),
    size=(0.006, 0.0, 0.0),
    rgba=(0.9, 0.1, 0.1, 1.0),
  )
  for joint in spec.joints:
    if joint.name == "floating_base_joint":
      spec.delete(joint)
      break
  return spec


def _add_whiteboard(spec: mujoco.MjSpec) -> None:
  board = spec.worldbody.add_body(name="whiteboard", pos=(0.56, 0.0, 1.1))
  board.add_geom(
    name="whiteboard_surface",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.01, 0.42, 0.36),
    rgba=(0.95, 0.95, 0.95, 1.0),
    friction=(0.8, 0.01, 0.001),
  )


def whiteboard_circle_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  pen_cfg = SceneEntityCfg("robot", site_names=(_PEN_TIP_SITE,))
  actor_terms = {
    "joint_pos": ObservationTermCfg(func=joint_pos_rel, noise=None),
    "joint_vel": ObservationTermCfg(func=joint_vel_rel, noise=None),
    "circle_target": ObservationTermCfg(
      func=mdp.circle_target_in_base,
      params={"command_name": "circle", "asset_cfg": pen_cfg},
      history_length=2,
      delay_min_lag=0,
      delay_max_lag=2,
    ),
    "actions": ObservationTermCfg(func=last_action),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=not play),
    "critic": ObservationGroupCfg({**actor_terms}, enable_corruption=False),
  }
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=_RIGHT_ARM_JOINTS,
      scale=0.25,
      use_default_offset=True,
    )
  }
  events = {
    "reset_robot_joints": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    )
  }
  rewards = {
    "circle_tracking": RewardTermCfg(
      func=mdp.circle_tracking,
      weight=2.0,
      params={"command_name": "circle", "asset_cfg": pen_cfg, "std": 0.04},
    ),
    "board_contact": RewardTermCfg(
      func=mdp.whiteboard_contact,
      weight=0.5,
      params={"asset_cfg": pen_cfg, "board_axis": 0, "board_position": 0.55},
    ),
    "circle_completion": RewardTermCfg(
      func=mdp.circle_completion,
      weight=1.0,
      params={"command_name": "circle"},
    ),
    "action_rate_l2": RewardTermCfg(func=action_rate_l2, weight=-0.01),
    "joint_pos_limits": RewardTermCfg(
      func=joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=_RIGHT_ARM_JOINTS)},
    ),
  }
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={
        "robot": EntityCfg(
          init_state=KNEES_BENT_KEYFRAME,
          spec_fn=_get_g1_whiteboard_spec,
          articulation=G1_ARTICULATION,
        ),
      },
      num_envs=1,
      env_spacing=1.0,
      spec_fn=_add_whiteboard,
    ),
    observations=observations,
    actions=actions,
    commands={
      "circle": CircleDrawingCommandCfg(
        resampling_time_range=(1000.0, 1000.0),
        center=(0.54, 0.0, 1.1),
        plane_axes=(1, 2),
        normal_axis=0,
        surface_offset=0.0,
        debug_vis=False,
      )
    },
    events=events,
    rewards=rewards,
    terminations={"time_out": TerminationTermCfg(func=time_out, time_out=True)},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="arm",
      distance=1.2,
      elevation=-10.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.005, iterations=10)),
    decimation=4,
    episode_length_s=20.0,
  )
