from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mjlab.tasks.whiteboard.drawing_env_cfg import make_drawing_env_cfg
from mjlab.tasks.whiteboard.mdp.board import BOARD_CENTRE_Y, BOARD_CENTRE_Z
from mjlab.tasks.whiteboard.mdp.draw_target_cmd import DrawTargetCommandCfg
from mjlab.tasks.whiteboard.mdp.gcode import fit_to_board, load_gcode_path, parse_gcode

SAMPLE = Path(__file__).parents[1] / "examples" / "square_circle.gcode"


def test_parse_marks_travel_and_drawing_moves() -> None:
  vertices = parse_gcode("G0 Z5\nG0 X10 Y0\nG1 Z0\nG1 X20 Y0\nG1 X20 Y10 ; side\n")
  assert [v[2] for v in vertices] == [False, False, True, True]


def test_arc_stays_on_circle() -> None:
  vertices = parse_gcode("G1 Z0\nG0 X0 Y0\nG2 X0 Y0 I10 J0\n")
  points = np.array([(v[0], v[1]) for v in vertices[1:]])
  radius = np.linalg.norm(points - np.array([10.0, 0.0]), axis=1)
  np.testing.assert_allclose(radius, 10.0, atol=1e-6)


def test_fit_scales_and_centres_on_board() -> None:
  path = load_gcode_path(SAMPLE, max_size=(0.4, 0.3))
  assert path.pen_down[0]
  drawn = path.points[path.pen_down]
  size = drawn.max(axis=0) - drawn.min(axis=0)
  assert size[0] <= 0.4 + 1e-6 and size[1] <= 0.3 + 1e-6
  np.testing.assert_allclose(
    drawn.mean(axis=0) - np.array([BOARD_CENTRE_Y, BOARD_CENTRE_Z]), 0.0, atol=0.1
  )
  gaps = np.linalg.norm(np.diff(path.points, axis=0), axis=1)
  assert gaps.max() <= path.spacing * 1.05
  assert not path.pen_down.all()  # includes a pen-up travel between shapes


def test_gcode_without_drawing_is_rejected() -> None:
  with pytest.raises(ValueError):
    fit_to_board(parse_gcode("G0 X1 Y1\nG0 X2 Y2\n"))


def test_gcode_env_cfg_gates_contact_and_adds_progress_reward() -> None:
  cfg = make_drawing_env_cfg(num_envs=2, gcode_path=str(SAMPLE))
  command = cfg.commands["draw_target"]
  assert isinstance(command, DrawTargetCommandCfg)
  assert command.gcode_path == str(SAMPLE)
  assert cfg.rewards["pen_contact"].params["command_name"] == "draw_target"
  assert "path_progress" in cfg.rewards
  assert cfg.episode_length_s > 24.0 - 1e-6
