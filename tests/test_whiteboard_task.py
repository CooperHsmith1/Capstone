from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from mjlab.tasks.whiteboard.config.g1.env_cfgs import unitree_g1_drawing_env_cfg
from mjlab.tasks.whiteboard.drawing_env_cfg import make_drawing_env_cfg
from mjlab.tasks.whiteboard.mdp.assets import (
  PEN_TIP_GEOM,
  PEN_TIP_RADIUS,
  get_g1_whiteboard_spec,
  whiteboard_spec_fn,
)
from mjlab.tasks.whiteboard.mdp.draw_target_cmd import gerono_figure_eight
from mjlab.tasks.whiteboard.mdp.observations import pen_board_contact_force
from mjlab.tasks.whiteboard.mdp.rewards import (
  pen_contact_force_penalty,
  pen_contact_reward,
)


def test_gerono_path_crosses_center_and_traces_two_lobes() -> None:
  phase = torch.tensor([0.0, torch.pi / 2, torch.pi, 3 * torch.pi / 2])
  points = gerono_figure_eight(phase, center_y=0.0, center_z=0.6, radius=0.15)

  expected = torch.tensor(
    [
      [0.0, 0.6],
      [0.15, 0.6],
      [0.0, 0.6],
      [-0.15, 0.6],
    ]
  )
  torch.testing.assert_close(points, expected)

  lobe = gerono_figure_eight(
    torch.tensor([torch.pi / 4]), center_y=0.0, center_z=0.6, radius=0.15
  )
  torch.testing.assert_close(
    lobe,
    torch.tensor([[0.15 / np.sqrt(2), 0.675]], dtype=torch.float32),
  )


def test_pen_tip_collision_is_limited_to_the_board() -> None:
  spec = get_g1_whiteboard_spec(fixed_base=False)
  whiteboard_spec_fn(spec)
  model = spec.compile()

  tip_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PEN_TIP_GEOM)
  board_geom_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_GEOM, "whiteboard_surface"
  )
  assert tip_geom_id >= 0
  assert board_geom_id >= 0
  tip_contact_radius = model.geom_size[tip_geom_id, 0] + model.geom_margin[tip_geom_id]
  assert tip_contact_radius == PEN_TIP_RADIUS

  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(
    model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
  )
  mujoco.mj_forward(model, data)

  free_joint_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
  )
  root_x_qpos = model.jnt_qposadr[free_joint_id]
  tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pen_tip")
  tip_x = data.site_xpos[tip_site_id, 0]
  data.qpos[root_x_qpos] += 0.33 - (tip_x + tip_contact_radius) + 0.001
  mujoco.mj_forward(model, data)

  contacts = {
    frozenset((data.contact[i].geom1, data.contact[i].geom2)) for i in range(data.ncon)
  }
  assert frozenset((tip_geom_id, board_geom_id)) in contacts
  assert model.geom_contype[tip_geom_id] == 2
  assert model.geom_conaffinity[tip_geom_id] == 0


def test_registered_training_and_play_share_one_complete_figure_eight() -> None:
  train_cfg = unitree_g1_drawing_env_cfg(num_envs=1)
  play_cfg = unitree_g1_drawing_env_cfg(play=True, num_envs=1)
  train_command = train_cfg.commands["draw_target"]
  play_command = play_cfg.commands["draw_target"]

  assert train_command.angular_speed == play_command.angular_speed
  assert train_command.radius == play_command.radius
  assert train_command.approach_duration_s == play_command.approach_duration_s
  assert train_command.resampling_time_range == play_command.resampling_time_range
  assert (
    train_command.approach_duration_s + 2 * np.pi / train_command.angular_speed
    <= train_cfg.episode_length_s
  )

  pen_sensor = next(
    sensor for sensor in train_cfg.scene.sensors if sensor.name == "pen_board_contact"
  )
  assert pen_sensor.primary.pattern == PEN_TIP_GEOM
  assert pen_sensor.secondary is not None
  assert pen_sensor.secondary.pattern == "whiteboard_surface"


def test_contact_observation_and_force_penalty_use_measured_contact() -> None:
  sensor = SimpleNamespace(
    data=SimpleNamespace(
      found=torch.tensor([[1], [0]]),
      force=torch.tensor([[[10.0, 3.0, 0.0]], [[0.0, 0.0, 0.0]]]),
    )
  )
  env = SimpleNamespace(scene={"pen_board_contact": sensor})

  force_observation = pen_board_contact_force(env, "pen_board_contact")
  torch.testing.assert_close(
    force_observation,
    torch.tensor([[np.log1p(10.0)], [0.0]], dtype=torch.float32),
  )
  torch.testing.assert_close(
    pen_contact_reward(env, "pen_board_contact"),
    torch.tensor([1.0, 0.0]),
  )
  torch.testing.assert_close(
    pen_contact_force_penalty(env, "pen_board_contact"),
    torch.tensor([1.0, 0.0]),
  )


def test_factory_keeps_the_robot_floating_for_walk_to_board() -> None:
  cfg = make_drawing_env_cfg(num_envs=1)
  robot_spec = cfg.scene.entities["robot"].spec_fn()

  assert any(joint.name == "floating_base_joint" for joint in robot_spec.joints)
  assert "pen_board_force" in cfg.observations["actor"].terms
  assert cfg.episode_length_s == 24.0


def test_pen_tip_registers_contact_with_board_at_runtime() -> None:
  import torch

  from mjlab.envs import ManagerBasedRlEnv

  env = ManagerBasedRlEnv(make_drawing_env_cfg(num_envs=1), device="cpu")
  robot = env.scene["robot"]
  sensor = env.scene["pen_board_contact"]
  tip = robot.site_names.index("pen_tip")
  action = torch.zeros(1, env.action_manager.total_action_dim)
  env.reset()
  touched = False
  for _ in range(30):
    pose = robot.data.root_link_pose_w.clone()
    pose[:, 0] += 0.004
    robot.write_root_link_pose_to_sim(pose)
    env.step(action)
    touched |= bool(sensor.data.found.any())
  assert touched, robot.data.site_pos_w[0, tip]
  env.close()


def test_ink_trace_records_marks_only_on_board_contact() -> None:
  import torch

  from mjlab.envs import ManagerBasedRlEnv

  env = ManagerBasedRlEnv(make_drawing_env_cfg(num_envs=1), device="cpu")
  cmd = env.command_manager.get_term("draw_target")
  action = torch.zeros(1, env.action_manager.total_action_dim)
  env.reset()
  env.step(action)
  assert int(cmd.ink_count[0]) == 0
  robot = env.scene["robot"]
  spheres = []

  class _Vis:
    def get_env_indices(self, num_envs):
      return range(num_envs)

    def add_sphere(self, center, radius, color, label=None):
      spheres.append(center)

  cmd.cfg.debug_vis = True
  mark = None
  for _ in range(12):
    pose = robot.data.root_link_pose_w.clone()
    pose[:, 0] += 0.01
    robot.write_root_link_pose_to_sim(pose)
    env.step(action)
    # Leaning into the board ends the episode, which clears the trace.
    if mark is None and int(cmd.ink_count[0]) > 0:
      mark = cmd.ink_points[0, 0].tolist()
      cmd.debug_vis(_Vis())  # type: ignore[arg-type]
  assert mark is not None
  assert abs(mark[0]) < 0.5 and 0.2 < mark[1] < 1.0
  assert spheres and all(hasattr(c, "copy") for c in spheres)
  env.reset()
  assert int(cmd.ink_count[0]) == 0
  env.close()
