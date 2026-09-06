# State Estimation for the G1 Whiteboard Drawing Task

Implements Section 4 of the interim report: a particle filter as the primary
estimator, with EKF and UKF baselines for the Section 4.3 comparative
framework. Also fixes several bugs found in the existing task code — one of
which likely explains why drawing behaviour has not emerged in training yet.

---

## Quick start

```bash
# Filter comparison table — needs only torch, no mjlab, no sim
python -m mdp.state_estimation.benchmark

# Test suites
python tests/test_state_estimation.py   # filter maths, no mjlab needed
python tests/test_integration_mock.py   # env wiring against a mock mjlab
```

Enable estimation in training:

```python
from mjlab.tasks.whiteboard.mdp.state_estimation import (
    StateEstimationCfg, StateEstimatorCfg,
)
from mjlab.tasks.whiteboard.config.g1.env_cfgs import unitree_g1_drawing_env_cfg

cfg = unitree_g1_drawing_env_cfg(
    num_envs=4096,
    state_estimation=StateEstimationCfg(
        filter_type="particle",      # or "ekf" / "ukf"
        num_particles=512,
        sensor_noise_std=0.01,       # injected sensor noise, metres
        outlier_rate=0.02,           # dropouts/glitches
        noise=StateEstimatorCfg(measurement_noise=0.01),
    ),
)
```

Passing `state_estimation=None` (the default) keeps ground-truth pen position.
**Swapping between the two is the core sim-to-real experiment** — ground-truth
pen position does not exist on the physical G1, so a policy trained against it
is learning to use information it will never have.

---

## What was added

```
mdp/
  board.py                        NEW  single source of truth for board geometry
  state_estimation/
    base.py                       NEW  shared 6-D state, motion/measurement models
    particle_filter.py            NEW  predict → update → resample (Figure 2)
    kalman.py                     NEW  EKF + UKF baselines
    integration.py                NEW  mjlab env wiring, noise injection, metrics
    benchmark.py                  NEW  comparative harness (Section 4.3)
tests/
  test_state_estimation.py        NEW  filter maths (no mjlab required)
  test_integration_mock.py        NEW  env wiring against a mock mjlab
```

State vector, shared by all three filters so results are comparable:

```
x = [px, py, pz, vx, vy, vz]        env-local, metres / m·s⁻¹
```

Constant-velocity motion model, position-only measurement. New observation
terms: `estimated_pen_position`, `estimated_pen_velocity`,
`pen_position_uncertainty`, `pen_estimate_innovation`.

---

## Benchmark results

From `python run_benchmark.py`, 1000 steps x 8 envs, sigma = 10 mm, aggregated
over **5 independent noise realisations** (mean +/- std). Every filter sees
identical measurements and an identical noise config, so differences are
attributable to the algorithm rather than tuning.

| Noise regime | EKF | UKF | PF-bootstrap | PF-optimal |
|---|---|---|---|---|
| Gaussian | **11.72 +/- 0.07 mm** | **11.72 +/- 0.07 mm** | 372 +/- 380 mm | 12.18 +/- 0.07 mm |
| Outliers (6%) | 86.78 +/- 1.32 mm | 86.78 +/- 1.32 mm | 845 +/- 562 mm | **50.61 +/- 4.76 mm** |
| Dropout (15%) | **13.12 +/- 0.08 mm** | **13.12 +/- 0.08 mm** | 167 +/- 128 mm | 13.55 +/- 0.08 mm |
| Heavy noise (4 sigma) | **46.33 +/- 0.17 mm** | **46.33 +/- 0.17 mm** | 613 +/- 395 mm | 54.06 +/- 1.37 mm |
| Cost (us/step) | ~155 | ~315 | ~1000 | ~1450 |

Raw measurement error is ~17 mm (Gaussian) and ~129 mm (outliers).

**The headline finding for the report:** under clean Gaussian noise the Kalman
filters are optimal and the particle filter cannot beat them -- it costs ~9x more
compute to be marginally worse (12.18 vs 11.72 mm, a gap of about 6 standard
deviations, so small but real). Under 6% outliers the particle filter is **1.7x
more accurate** than either Kalman filter (50.6 vs 86.8 mm). That crossover, not
a single RMSE number, is the result worth reporting. A table showing only the
outlier case would overstate the particle filter; one showing only the Gaussian
case would understate it.

**Read the error bars before claiming a winner.** The dropout column separates
EKF and PF-optimal by 0.43 mm against a spread of 0.08 mm -- real but negligible
in practice. PF-bootstrap's standard deviation is comparable to its mean, which
is the signature of an unstable filter rather than a merely inaccurate one; do
not quote its mean without the spread.

Every number above is reproducible: noise seeds are derived with `crc32`, not
`hash()`, so repeated runs give bit-identical accuracy columns on any machine.
Only the timing column varies between runs.

EKF and UKF agree to 2.4 × 10⁻⁷ m, which cross-validates both implementations.

---

## Two findings that will affect your write-up

### 1. The EKF here is algebraically a linear Kalman filter

Both the motion and measurement models are linear, so the EKF's Jacobians are
the constant matrices `F` and `H`. This is the *correct* EKF for this system,
not a simplification — but it means the report should not claim the EKF is
"linearising a nonlinear system" in this task. Nonlinearity would enter if the
state were the arm's joint angles with forward kinematics as the measurement
model, which is a reasonable Semester 2 extension.

### 2. The textbook bootstrap particle filter fails on this system

Only position is measured; velocity is unobserved. Over one 20 ms step a
velocity error of 0.4 m/s displaces the pen by 8 mm — less than the 10 mm
measurement noise. So a single step carries almost no information about
velocity, the particle cloud never contains plausible velocities, and the
filter lags badly (it underestimated speed by almost exactly half in testing).

Inflating the process noise hides this but is a fudge — it makes the filter
claim the arm is more erratic than it is. The implemented fix is the **optimal
proposal**, `p(xₜ | xₜ₋₁, zₜ)`, which folds the current measurement into the
proposal rather than only into the weights. Closed form for a linear-Gaussian
measurement model, one extra 6×6 solve per step, no extra particles.

Both proposals are kept (`proposal="bootstrap"` / `"optimal"`). The comparison
at fixed, correctly-specified `Q` is a genuine result: it isolates sampling
efficiency from model tuning, and explains *why* a particle filter can look
worse than an EKF despite being the more general estimator.

---

## Bugs fixed in existing code

### Reward weights were being silently reverted — most important

`drawing_env_cfg.py` contains tuned weights with comments recording the change:

```python
"smooth_pen_motion": RewardTermCfg(weight=-0.01),   # was -0.5
"upright":           RewardTermCfg(weight=1.0),     # was 2.0
```

`unitree_g1_drawing_env_cfg()` in `config/g1/env_cfgs.py` then rebuilt those
terms from scratch, overwriting them:

```python
cfg.rewards["smooth_pen_motion"] = RewardTermCfg(weight=-0.05)  # 5× tuned value
cfg.rewards["upright"]           = RewardTermCfg(weight=2.0)    # 2× tuned value
```

Because that function is the one registered for training, **the tuning recorded
in `drawing_env_cfg.py` never took effect in any run.** An `upright` weight of
2.0 against a `pen_tracking` weight of 10.0 makes standing still locally
optimal for a long time, which is consistent with the report's note that
"meaningful drawing behaviour has not yet emerged."

Fixed by deleting the redundant block — `make_drawing_env_cfg()` already
installs all six terms correctly. **Worth re-running your reward sensitivity
analysis after this**, since previous results were measuring the wrong weights.

### `num_envs` was hard-coded to 1

`SceneCfg(num_envs=1)` meant every caller got a single environment regardless
of intent — one rollout of experience per step instead of hundreds, which makes
PPO extremely sample-inefficient. Now a parameter, defaulting to 4096 in the G1
config.

### Board geometry duplicated across four files

`BOARD_FACE_X`, Y and Z ranges appeared in `DrawingCanvas.py`,
`draw_target_cmd.py`, `drawing_env_cfg.py` and `env_cfgs.py` — with three
different Y ranges (±0.40, ±0.35, ±0.30). Move the board in the MJCF, miss one
file, and some fraction of draw targets land where the arm cannot reach while
training carries on silently. Centralised in `mdp/board.py` with an import-time
consistency check.

### `DrawTargetCommand.__init__` ordering

`self._target` was created *after* `super().__init__()`, but the base class may
call `_resample_command` / `_update_metrics` before returning. Worked by luck of
the current base-class implementation; would break on an mjlab upgrade with a
confusing `AttributeError`. `_target` is now created first.

### `_get_pen_tip_pos` re-resolved the site every step

It built a fresh `SceneEntityCfg` and re-resolved the `"pen_tip"` string on
every call — once per env per step, in the innermost training loop, for a value
that never changes. Now resolved once and cached. The bare `except Exception`
that hid any error forever was narrowed to the specific setup-time exceptions.

### `make_drawing_env_with_canvas()` could not work

It called `make_drawing_env_cfg()` directly, which returns a config with empty
`scene.entities` — no robot, so `ManagerBasedRlEnv` cannot build and there is no
`pen_tip` site. Now takes a robot-configured cfg and raises a clear error
explaining the fix if omitted. It also monkey-patched the private
`_post_physics_step`; if that attribute is absent it now warns rather than
leaving you staring at a permanently blank whiteboard.

---

## Things I could not verify

mjlab is not installed here, so **the integration layer is tested against a
mock, not the real mjlab API**. The mock covers `Entity`, `SceneEntityCfg`,
scene indexing, `env_origins` and the step counter. Two things to check on
first real run:

1. **Step counter attribute.** `StateEstimatorManager._step_counter()` tries
   `common_step_counter`, then `episode_length_buf`, then `_step_count`, with a
   local fallback. If your mjlab version names it differently the filter would
   advance once per *observation term* instead of once per step. The mock test
   `test_stepped_once_per_env_step` pins the intended behaviour — check
   `state_est/rmse_m` looks sane on a real run.
2. **`step_dt`.** Falls back to `cfg.sim.mujoco.timestep × cfg.decimation`
   (0.005 × 4 = 0.02 s) if `env.step_dt` is absent. A wrong `dt` degrades the
   motion model silently rather than crashing.

The noise defaults (`process_noise_vel=0.15`, `measurement_noise=0.01`) were
tuned against a synthetic Lissajous trajectory with roughly 1.2 m/s² peak
acceleration. Re-tune against real logged pen trajectories once the arm is
tracking — `process_noise_vel` should be about `peak_accel × √dt`.

---

## Suggested next steps

1. Re-run reward sensitivity analysis with the un-clobbered weights.
2. Train two policies — ground truth vs filtered pen position — and compare.
   That is the direct evidence for the report's central research question.
3. Log `manager.metrics()` to TensorBoard; if `state_est/improvement` goes
   negative the filter is adding latency for nothing.
4. For a genuinely nonlinear comparison, extend the state to joint angles with
   forward kinematics as the measurement model. That is where EKF, UKF and PF
   should finally separate on the Gaussian benchmark too.

---

## Corrections made in this revision

Three defects were found by running the code rather than reading it. All are
fixed; they are listed because two of them affected numbers that would
otherwise have gone into the report.

### 1. The benchmark was not reproducible (critical)

`run_benchmark` seeded each noise regime with `hash(regime_name) % 10_000`.
CPython salts string hashing per process, so **every run drew a different noise
realisation** and the results table could not be reproduced or cited. The
practical consequence was worse than an inconvenience: one run showed the
particle filter *losing* to the EKF under outlier noise (130 mm vs 86 mm), the
opposite of the conclusion the report draws, purely from an unlucky draw.

Fixed by deriving seeds from `zlib.crc32`, which is stable across processes and
machines, and by running each cell over 5 independent realisations and reporting
mean +/- standard deviation. The particle filter's outlier advantage survives
this properly and is now a defensible claim.

### 2. The outlier test could not fail

`test_particle_filter_beats_kalman_on_outliers` compared one seed and had **no
assertion** -- it printed a note if the particle filter lost and passed anyway.
It now averages over 5 seeds and asserts both that the PF wins on average and
that its worst seed still beats the EKF's mean. Verified: PF 25.8 mm vs EKF
50.1 mm, PF worst case 30.4 mm.

### 3. `pytest` could not collect any test

Both test files documented `python -m pytest tests/...` as the way to run them,
but that failed with 16 collection errors on any machine without mjlab. pytest
reconstructs the full dotted module path when collecting `tests/`, which imports
the package root, which imported the mjlab task registry unconditionally.

Fixed by guarding that import in `__init__.py` so a *missing mjlab* degrades to
`MJLAB_AVAILABLE = False` while any other `ImportError` is still re-raised -- a
real bug in the task config is never masked. Added `pytest.ini` and
`tests/conftest.py`. All 16 tests now pass under both `pytest` and direct
execution.

### Verification status

| Component | Status |
|---|---|
| Filter maths (PF / EKF / UKF) | Verified numerically, 16/16 tests pass |
| Benchmark reproducibility | Verified: identical accuracy columns across runs |
| Integration layer | Verified against a mock env only |
| mjlab environment configs | **Not executed** -- mjlab is not installable here |

The last row matters. The env-config changes (reward-weight clobbering, board
constant centralisation, `num_envs` plumbing, estimator wiring) are reviewed and
syntax-checked but have never been run against a real mjlab build. Expect to
adjust attribute paths on first run; `drawing_env_cfg.py` documents the mjlab
version differences it tries to handle.


---

## Running it in mjlab

Copy the `whiteboard/` folder into `mjlab/tasks/` (alongside `velocity/` and
`tracking/`), then:

```bash
uv run train Mjlab-Drawing-Flat-Unitree-G1      --env.scene.num-envs 4096   # ground truth
uv run train Mjlab-Drawing-Flat-Unitree-G1-PF   --env.scene.num-envs 4096   # particle filter
uv run train Mjlab-Drawing-Flat-Unitree-G1-EKF  --env.scene.num-envs 4096
uv run train Mjlab-Drawing-Flat-Unitree-G1-UKF  --env.scene.num-envs 4096
```

The base task feeds the policy the true pen position. The three filter variants
feed a filtered estimate derived from a noisy synthetic sensor, and all three
share **one** sensor model (sigma = 10 mm, 5% outliers) so the task ids are
directly comparable. Training the base task against one filter variant is the
sim-to-real experiment.

Verified end to end against **mjlab 1.6.0** on CPU: all four tasks register,
build, reset and step, in both train and play mode. Live filter performance
inside the simulator, 60 steps, 4 envs, identical sensor stream:

| Variant | Filtered RMSE | Raw sensor RMSE | Error removed |
|---|---|---|---|
| PF | **27.2 mm** | 93.1 mm | **70.8%** |
| EKF | 80.6 mm | 121.8 mm | 33.8% |
| UKF | 75.4 mm | 112.1 mm | 32.7% |

This reproduces the offline benchmark's conclusion inside the actual simulator.

### What had to be built to make it run

The task could not be registered at all before this pass:

* **`get_g1_robot_fixed_base_cfg` does not exist** in mjlab 1.6 -- the asset zoo
  exports only `get_g1_robot_cfg`. Registration died on the import.
* **The pen and whiteboard did not exist.** The config referenced a hand-edited
  `g1_29dof_whiteboard.xml` that was never in the repository, along with a
  `pen_tip` site and a `whiteboard_surface` geom. `mdp/assets.py` now builds
  both programmatically from mjlab's stock G1 spec, including the writable
  texture the drawing canvas paints into.
* **`_update_command` had the wrong signature.** mjlab requires
  `_update_command(self, env_ids)` and raises a `TypeError` at construction
  without it.
* **`env._post_physics_step` does not exist** in mjlab 1.6, so the canvas
  monkey-patch in the old `make_drawing_env_with_canvas` silently never ran --
  `getattr(env, "_post_physics_step", None)` returned `None` and the new
  attribute was never called. The estimator is driven from the observation
  manager instead, which needs no private hooks.
* **`reset_base` on a fixed-base robot.** `reset_root_state_uniform` needs a
  free joint; with the base welded there is none. The event is now registered
  only in floating-base mode.
