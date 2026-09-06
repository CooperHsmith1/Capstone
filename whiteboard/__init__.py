"""Whiteboard drawing task for Unitree G1.

Importing this package registers the G1 drawing task with mjlab.

The registration import is guarded so that ``import whiteboard`` still succeeds
on a machine without mjlab installed. That matters for two things this project
actually relies on:

  * ``python -m mdp.state_estimation.benchmark`` -- generating the filter
    comparison table for the report on a laptop, with no simulator present.
  * ``pytest`` -- the state-estimation suite is deliberately mjlab-free, but
    pytest reconstructs the full dotted module path when collecting ``tests/``,
    which imports this file. Without the guard every test errors during
    collection with a misleading "No module named 'mjlab'".

Only a missing *mjlab* is tolerated. Any other ImportError -- a genuine bug in
the task configuration, a typo in a submodule, a broken dependency -- is
re-raised, so a real problem is never masked by this guard.
"""

MJLAB_AVAILABLE: bool
"""True if the mjlab task registration completed successfully."""

_IMPORT_ERROR: "ImportError | None" = None

try:
    from .config.g1 import *  # noqa: F401,F403

    MJLAB_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - depends on install environment
    if "mjlab" not in str(exc):
        raise
    MJLAB_AVAILABLE = False
    _IMPORT_ERROR = exc


def require_mjlab() -> None:
    """Raise a clear error if the task registry could not be loaded.

    Call this from any entry point that genuinely needs the simulator, so the
    failure names the real cause instead of surfacing later as a missing task id.
    """
    if not MJLAB_AVAILABLE:
        raise ImportError(
            "The G1 whiteboard task could not be registered because mjlab is "
            "not installed in this environment. Install mjlab, or use the "
            "mjlab-free entry points (mdp.state_estimation.benchmark and the "
            f"tests/ suite). Original error: {_IMPORT_ERROR}"
        )
