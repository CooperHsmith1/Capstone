"""Scene assets for the whiteboard drawing task: the pen and the board.

The original task config assumed a hand-edited ``g1_29dof_whiteboard.xml`` and a
``get_g1_robot_fixed_base_cfg`` helper in mjlab's asset zoo. Neither exists:
the XML was never in the repository, and mjlab (as of 1.6.0) exports only
``get_g1_robot_cfg``. The task therefore could not be registered at all.

This module removes both dependencies by building the extra geometry
programmatically from mjlab's stock G1 spec:

  * :func:`get_g1_whiteboard_robot_cfg` -- the G1 with a pen welded to the right
    wrist and a ``pen_tip`` site at its writing end. Optionally fixed-base.
  * :func:`whiteboard_spec_fn` -- a ``SceneCfg.spec_fn`` callback that adds the
    board itself, along with the material and texture the live drawing canvas
    paints into.

Doing it in code rather than in a checked-in XML means the pen and the board
cannot drift out of sync with ``board.py``: every position here is derived from
those same constants, so moving the board is a one-line change.
"""

from __future__ import annotations

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    G1_ARTICULATION,
    KNEES_BENT_KEYFRAME,
    FULL_COLLISION,
    get_spec,
)
from mjlab.entity import EntityCfg

from .board import (
    BOARD_CENTRE_X,
    BOARD_CENTRE_Y,
    BOARD_CENTRE_Z,
    BOARD_HALF_DEPTH,
    BOARD_HALF_HEIGHT,
    BOARD_HALF_WIDTH,
)

__all__ = [
    "PEN_TIP_SITE",
    "PEN_PARENT_BODY",
    "BOARD_BODY",
    "BOARD_GEOM",
    "BOARD_MATERIAL",
    "BOARD_TEXTURE",
    "CANVAS_PIXELS",
    "add_pen",
    "make_fixed_base",
    "get_g1_whiteboard_spec",
    "get_g1_whiteboard_robot_cfg",
    "whiteboard_spec_fn",
]

# Names referenced elsewhere (rewards, observations, canvas). Import these
# rather than retyping the strings.
PEN_TIP_SITE = "pen_tip"
PEN_BODY = "pen"
PEN_PARENT_BODY = "right_wrist_yaw_link"
BOARD_BODY = "whiteboard"
BOARD_GEOM = "whiteboard_surface"
BOARD_MATERIAL = "whiteboard_mat"
BOARD_TEXTURE = "whiteboard_tex"

# Canvas resolution. Must be a power of two for MuJoCo texture upload to be
# well behaved across drivers.
CANVAS_PIXELS = 512

# Pen geometry: a short capsule sticking forward (+X in the wrist frame) out of
# the palm, with the writing tip at its far end.
PEN_LENGTH = 0.16
PEN_RADIUS = 0.008
PEN_OFFSET = (0.06, 0.0, 0.0)

# The free joint mjlab's G1 uses for its floating base.
FREE_JOINT_NAME = "floating_base_joint"


def add_pen(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Weld a pen to the G1's right wrist and add the ``pen_tip`` site.

    The pen is a rigid child body of the wrist -- no joint -- so it is welded in
    the mechanical sense and adds no degrees of freedom. The site sits at the
    writing end, which is what every reward, observation and the drawing canvas
    reads.

    Raises:
        ValueError: If the expected wrist body is missing, which would mean the
            G1 model changed and the attachment point needs revisiting.
    """
    parent = None
    for body in spec.bodies:
        if body.name == PEN_PARENT_BODY:
            parent = body
            break
    if parent is None:
        available = [b.name for b in spec.bodies if "wrist" in b.name]
        raise ValueError(
            f"Cannot attach the pen: body {PEN_PARENT_BODY!r} not found in the "
            f"G1 spec. Wrist bodies present: {available}. The asset_zoo model "
            f"has probably changed; update PEN_PARENT_BODY."
        )

    pen = parent.add_body(name=PEN_BODY, pos=PEN_OFFSET)

    # Capsule from the wrist outward along +X.
    pen.add_geom(
        name="pen_body",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0.0, 0.0, 0.0, PEN_LENGTH, 0.0, 0.0],
        size=[PEN_RADIUS, 0.0, 0.0],
        rgba=[0.15, 0.15, 0.18, 1.0],
        mass=0.02,
        contype=0,
        conaffinity=0,
    )

    # The tip site. contype/conaffinity are irrelevant for sites; the contact
    # reward works off the geometric distance to the board plane instead of a
    # physical contact, which keeps the reward smooth and differentiable.
    pen.add_site(
        name=PEN_TIP_SITE,
        pos=[PEN_LENGTH, 0.0, 0.0],
        size=[0.006, 0.0, 0.0],
        rgba=[0.9, 0.2, 0.2, 1.0],
    )
    return spec


def make_fixed_base(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Remove the floating base so the robot is bolted to the world.

    Fixed-base training isolates the arm: it removes the balance problem
    entirely, which makes the reaching task learnable much faster. Be aware of
    what this costs -- with no floating base the ``upright`` reward and the
    ``fell_over`` termination are both inert, because the torso cannot tip. If
    the report is going to discuss humanoid balance as a coupled control
    problem, that discussion belongs to the floating-base configuration, not
    this one.
    """
    for joint in spec.joints:
        if joint.name == FREE_JOINT_NAME:
            # Removal goes through the spec, not the element: MjsJoint has no
            # delete() of its own in mujoco 3.11.
            spec.delete(joint)
            return spec
    # Not finding it is not fatal -- some model versions may already be fixed.
    return spec


def get_g1_whiteboard_spec(fixed_base: bool = True) -> mujoco.MjSpec:
    """Return the G1 spec with the pen attached, optionally fixed-base."""
    spec = get_spec()
    add_pen(spec)
    if fixed_base:
        make_fixed_base(spec)
    return spec


def get_g1_whiteboard_robot_cfg(fixed_base: bool = True) -> EntityCfg:
    """G1 entity config for the drawing task.

    Mirrors mjlab's ``get_g1_robot_cfg`` but swaps in the pen-equipped spec.
    A fresh instance is returned each call so the config can be safely mutated
    by callers.
    """
    return EntityCfg(
        init_state=KNEES_BENT_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=lambda: get_g1_whiteboard_spec(fixed_base=fixed_base),
        articulation=G1_ARTICULATION,
    )


def _blank_canvas(size: int = CANVAS_PIXELS) -> np.ndarray:
    """A plain off-white board with a faint border, as RGB uint8."""
    img = np.full((size, size, 3), 247, dtype=np.uint8)
    edge = max(2, size // 128)
    img[:edge, :, :] = 200
    img[-edge:, :, :] = 200
    img[:, :edge, :] = 200
    img[:, -edge:, :] = 200
    return img


def whiteboard_spec_fn(spec: mujoco.MjSpec) -> None:
    """``SceneCfg.spec_fn`` callback that adds the whiteboard to the scene.

    Adds a static box at the board position from ``board.py``, backed by a
    writable texture so :class:`DrawingCanvas` has somewhere to paint. The body
    carries no joint, so it is static geometry: it cannot be pushed, and it must
    never be passed to a reset event that writes root velocity.
    """
    # Texture the canvas paints into. Created as 2d so the whole image maps to
    # the board face.
    spec.add_texture(
        name=BOARD_TEXTURE,
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        width=CANVAS_PIXELS,
        height=CANVAS_PIXELS,
        data=_blank_canvas().flatten().tolist(),
    )
    # add_material returns the element; its ``textures`` field is a fixed-length
    # vector with one slot per texture role. The colour texture goes in the RGB
    # slot -- NOT slot 0, which is the user role and stays empty.
    material = spec.add_material(name=BOARD_MATERIAL)
    material.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = BOARD_TEXTURE

    board = spec.worldbody.add_body(
        name=BOARD_BODY,
        pos=[BOARD_CENTRE_X, BOARD_CENTRE_Y, BOARD_CENTRE_Z],
    )
    board.add_geom(
        name=BOARD_GEOM,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[BOARD_HALF_DEPTH, BOARD_HALF_WIDTH, BOARD_HALF_HEIGHT],
        material=BOARD_MATERIAL,
        rgba=[1.0, 1.0, 1.0, 1.0],
        contype=1,
        conaffinity=1,
    )
