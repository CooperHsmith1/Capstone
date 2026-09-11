"""Shared utilities for resolving SceneEntityCfg index fields.

``SceneEntityCfg`` stores ``site_ids``, ``body_ids``, and ``joint_ids`` as
``list[int] | slice``.  Pylance (correctly) flags ``ids[0]`` when the type
includes ``slice`` because ``slice.__getitem__`` is not defined.  This module
provides a single canonical resolver used across the drawing task package so
the fix lives in one place.
"""

from __future__ import annotations

from mjlab.managers.scene_entity_config import SceneEntityCfg


def resolve_first_id(ids: "list[int] | slice") -> int:
  """Return the first element of a ``list[int] | slice`` index field.

  Args:
      ids: The raw value of a ``SceneEntityCfg`` index attribute such as
           ``site_ids``, ``body_ids``, or ``joint_ids``.

  Returns:
      The first resolved integer index.

  Raises:
      ValueError: If the list is empty or the slice resolves to no indices.
  """
  if isinstance(ids, slice):
    # slice.indices(length) needs a concrete length.  We use a large
    # sentinel (2**16) which is safely larger than any real site count.
    indices = range(*ids.indices(2**16))
    if not indices:
      raise ValueError(
        "SceneEntityCfg index slice resolves to an empty range. "
        "Check that site_names / body_names / joint_names were set."
      )
    return indices[0]
  if not ids:
    raise ValueError(
      "SceneEntityCfg index list is empty. "
      "Did you forget to pass site_names / body_names / joint_names?"
    )
  return ids[0]


def resolve_first_site_id(asset_cfg: SceneEntityCfg) -> int:
  """Convenience wrapper: resolve the first site index from a SceneEntityCfg."""
  return resolve_first_id(asset_cfg.site_ids)
