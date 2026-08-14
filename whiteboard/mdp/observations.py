from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

from .cfg_utils import resolve_first_site_id

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def foot_height(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Per-foot vertical clearance above terrain.

    Returns:
        Tensor of shape [B, F] where F is the number of frames (feet).
    """
    sensor = env.scene[sensor_name]
    assert isinstance(sensor, TerrainHeightSensor), (
        f"foot_height requires a TerrainHeightSensor, got {type(sensor).__name__}"
    )
    return sensor.data.heights


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    current_air_time = sensor_data.current_air_time
    assert current_air_time is not None
    return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    assert sensor_data.found is not None
    return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    assert sensor_data.force is not None
    forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
    return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def site_position(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """ENV-LOCAL position of a single site on a robot asset.

    ``asset_cfg`` must be constructed with ``site_names`` so that
    ``asset_cfg.site_ids`` is resolved before this function is called.
    Returns a tensor of shape ``[B, 3]``.

    NOTE: site_pos_w is the ABSOLUTE world position, which includes each env's
    tiled spawn offset (env_origins). We subtract env_origins so the observation
    is in the env-local frame — consistent with the draw_target command and the
    pen_tracking reward. Without this, the policy sees pen positions tens of
    metres away for far-tiled envs and can never learn to reach the board.

    Example::

        "pen_pos": ObservationTermCfg(
            func=site_position,
            params={"asset_cfg": SceneEntityCfg("robot", site_names=("pen_tip",))},
        )
    """
    asset: Entity = env.scene[asset_cfg.name]
    idx = resolve_first_site_id(asset_cfg)  # int — never a slice
    pos_w = asset.data.site_pos_w[:, idx, :]          # [B, 3] absolute world
    return pos_w - env.scene.env_origins             # [B, 3] env-local