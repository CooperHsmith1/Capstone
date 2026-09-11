"""Batched particle filter for pen-tip state estimation.

Implements the predict -> update -> resample recursion described in Section 4.2
of the interim report, vectorised over both environments and particles so it
runs inside an mjlab rollout without a Python loop.

Tensor layout throughout:

    particles     [B, N, 6]   N hypotheses per environment
    log_weights   [B, N]      kept in log space for numerical stability

Working in log space matters more than it looks. With N=512 particles and a
measurement noise of 1 cm, a particle 10 cm off the measurement has a
likelihood around exp(-50) which underflows float32 to exactly zero. In linear
space that silently turns into a division by zero the moment *every* particle
is off (which is exactly what happens during a fast arm swing). In log space
the same situation degrades gracefully into a uniform-weight posterior.

PROPOSAL DISTRIBUTIONS
----------------------
Two proposals are provided, and which one you use materially changes the
result:

``"bootstrap"`` propagates particles through the motion model and *then*
weights them by the measurement -- the textbook formulation, and the one drawn
in Figure 2 of the interim report. It has a known weakness on this exact
system. Only position is measured; velocity is unobserved and is inferred
purely from which particles happen to predict the next position well. Over one
20 ms step, a velocity error of 0.4 m/s displaces the pen by 8 mm, which is
smaller than the 10 mm measurement noise -- so a single step carries almost no
information about velocity. With a physically-correct (and therefore narrow)
process noise, the ensemble never contains particles with plausible velocities
and the filter lags badly, underestimating speed by roughly half. Inflating the
process noise hides this but is a fudge: it makes the filter claim the arm is
more erratic than it really is.

``"optimal"`` (the default) samples from p(x_t | x_{t-1}, z_t) instead -- it
folds the current measurement into the proposal rather than only into the
weights. For a linear-Gaussian measurement model this has a closed form
(Doucet et al., 2000), so it costs one extra 6x6 solve per timestep and no
extra particles. It keeps the process noise physically honest while placing
particles where the measurement says they should be.

Keep both. The bootstrap-vs-optimal comparison at a fixed, correctly-specified
Q is a genuine result for the report: it isolates sampling efficiency from
model tuning, and it explains *why* a particle filter can look worse than an
EKF on a benchmark despite being the more general estimator.
"""

from __future__ import annotations

import torch

from .base import (
  MEAS_DIM,
  POS_SLICE,
  STATE_DIM,
  VEL_SLICE,
  StateEstimator,
  StateEstimatorCfg,
  constant_velocity_transition,
)


class ParticleFilter(StateEstimator):
  """Sequential Monte Carlo estimator with systematic resampling.

  Args:
      num_envs: Batch size B.
      cfg: Shared noise configuration.
      num_particles: Ensemble size N per environment. 256-512 is ample for a
          6-D state; cost scales linearly in N.
      resample_threshold: Resample when the effective sample size falls below
          this fraction of N. 0.5 is the standard choice -- resampling every
          step throws away diversity for no benefit, and never resampling
          lets a single particle absorb all the weight (degeneracy).
      roughening: Std dev of jitter added to positions after resampling, as a
          fraction of the initial position spread. Without it, resampling
          duplicates particles exactly and the ensemble collapses to a handful
          of distinct values within a few dozen steps (sample impoverishment).
      proposal: ``"optimal"`` or ``"bootstrap"``. See the module docstring --
          this choice matters more than the particle count on this system.
      device: Torch device.
      seed: Optional RNG seed for reproducible benchmark runs.
  """

  def __init__(
    self,
    num_envs: int,
    cfg: StateEstimatorCfg,
    num_particles: int = 512,
    resample_threshold: float = 0.5,
    roughening: float = 0.05,
    proposal: str = "optimal",
    device: torch.device | str = "cpu",
    seed: int | None = None,
  ) -> None:
    super().__init__(num_envs, cfg, device)

    if num_particles < 2:
      raise ValueError("num_particles must be >= 2.")
    if proposal not in ("optimal", "bootstrap"):
      raise ValueError(f"proposal must be 'optimal' or 'bootstrap', got {proposal!r}.")
    self.num_particles = int(num_particles)
    self.resample_threshold = float(resample_threshold)
    self.roughening = float(roughening)
    self.proposal = proposal

    # The optimal proposal needs the timestep from predict() while doing its
    # work in update(); cached here between the two calls.
    self._pending_dt: float | None = None

    self._generator: torch.Generator | None = None
    if seed is not None:
      self._generator = torch.Generator(device=self.device)
      self._generator.manual_seed(int(seed))

    self._particles = torch.zeros(
      self.num_envs, self.num_particles, STATE_DIM, device=self.device
    )
    # Uniform prior: log(1/N) for every particle.
    self._log_weights = torch.full(
      (self.num_envs, self.num_particles),
      -torch.log(torch.tensor(float(self.num_particles))).item(),
      device=self.device,
    )

    # Diagnostics.
    self._resample_count = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.long
    )
    self._last_ess = torch.full(
      (self.num_envs,), float(self.num_particles), device=self.device
    )

    self.reset()

  # ------------------------------------------------------------------
  # Identity and estimates
  # ------------------------------------------------------------------

  @property
  def name(self) -> str:
    return f"ParticleFilter(N={self.num_particles})"

  @property
  def particles(self) -> torch.Tensor:
    """Particle ensemble, shape [B, N, 6]."""
    return self._particles

  @property
  def weights(self) -> torch.Tensor:
    """Normalised linear weights, shape [B, N]."""
    return torch.exp(self._log_weights)

  @property
  def state(self) -> torch.Tensor:
    """Weighted-mean (MMSE) state estimate, shape [B, 6]."""
    w = self.weights.unsqueeze(-1)  # [B, N, 1]
    return torch.sum(w * self._particles, dim=1)

  @property
  def map_state(self) -> torch.Tensor:
    """Maximum-a-posteriori state (single highest-weight particle), [B, 6].

    Preferred over the weighted mean when the posterior is multimodal: the
    mean of a two-cluster posterior sits in the empty space between the
    clusters, which is the one place the pen certainly is not.
    """
    idx = torch.argmax(self._log_weights, dim=1)  # [B]
    return self._particles[torch.arange(self.num_envs, device=self.device), idx]

  @property
  def covariance(self) -> torch.Tensor:
    """Weighted sample covariance, shape [B, 6, 6]."""
    mean = self.state.unsqueeze(1)  # [B, 1, 6]
    delta = self._particles - mean  # [B, N, 6]
    w = self.weights.unsqueeze(-1)  # [B, N, 1]
    weighted = w * delta  # [B, N, 6]
    cov = torch.einsum("bni,bnj->bij", weighted, delta)
    # Guard against a singular covariance when the ensemble has collapsed.
    eye = torch.eye(STATE_DIM, device=self.device).expand_as(cov)
    return cov + eye * 1e-9

  @property
  def effective_sample_size(self) -> torch.Tensor:
    """ESS = 1 / sum(w^2), shape [B]. Ranges from 1 (degenerate) to N."""
    w = self.weights
    return 1.0 / torch.clamp(torch.sum(w * w, dim=1), min=1e-12)

  @property
  def resample_count(self) -> torch.Tensor:
    """Number of resampling events per env since reset, shape [B]."""
    return self._resample_count

  # ------------------------------------------------------------------
  # 1. PREDICT
  # ------------------------------------------------------------------

  def _predict(self, dt: float) -> None:
    """Propagate particles through the motion model.

    Under the bootstrap proposal this does the full prediction. Under the
    optimal proposal it only applies the deterministic drift and defers the
    stochastic part to ``_update``, where the measurement is available to
    steer it.
    """
    F = constant_velocity_transition(dt, self.device)  # [6, 6]
    # [B, N, 6] @ [6, 6]^T -> [B, N, 6]
    self._particles = self._particles @ F.T

    if self.proposal == "bootstrap":
      self._particles = self._particles + self._process_noise(dt)
    else:
      self._pending_dt = dt

  def _process_noise(self, dt: float) -> torch.Tensor:
    """Sample process noise w ~ N(0, Q), shape [B, N, 6]."""
    noise = self._randn_like(self._particles)
    noise[..., POS_SLICE] *= self.cfg.process_noise_pos
    # Velocity noise scales with sqrt(dt) so the diffusion is consistent
    # across different control rates rather than depending on the timestep.
    noise[..., VEL_SLICE] *= self.cfg.process_noise_vel * (dt**0.5)
    return noise

  def _proposal_moments(self, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (Sigma, S) for the optimal proposal at this timestep.

    Sigma = (Q^-1 + H^T R^-1 H)^-1     proposal covariance,      [6, 6]
    S     = H Q H^T + R                predictive meas. cov,     [3, 3]

    Both are constant given dt, so they are computed once per step rather
    than once per particle.
    """
    Q = self._Q.clone()
    # Position process noise is deliberately tiny, which makes Q^-1 huge and
    # the solve ill-conditioned in float32. Floor it: this is a numerical
    # guard, not a model change, and it is far below the measurement noise.
    q_diag = torch.diagonal(Q).clamp(min=1e-10)
    Q = torch.diag(q_diag)

    Q_inv = torch.diag(1.0 / q_diag)
    R_inv = torch.linalg.inv(self._R)
    info = Q_inv + self._H.T @ R_inv @ self._H
    Sigma = torch.linalg.inv(info)
    Sigma = 0.5 * (Sigma + Sigma.T)  # enforce symmetry after inversion

    S = self._H @ Q @ self._H.T + self._R
    return Sigma, S

  # ------------------------------------------------------------------
  # 2. UPDATE
  # ------------------------------------------------------------------

  def _update(self, measurement: torch.Tensor) -> None:
    """Reweight particles by the likelihood of the observed position."""
    if self.proposal == "optimal":
      log_lik = self._update_optimal(measurement)
    else:
      log_lik = self._update_bootstrap(measurement)

    self._log_weights = self._log_weights + log_lik
    self._normalise_log_weights()

    self._last_ess = self.effective_sample_size
    self._maybe_resample()

  def _update_bootstrap(self, measurement: torch.Tensor) -> torch.Tensor:
    """Standard bootstrap weighting: log p(z | x). Returns [B, N]."""
    pred_pos = self._particles[..., POS_SLICE]  # [B, N, 3]
    residual = measurement.unsqueeze(1) - pred_pos  # [B, N, 3]

    # Gaussian log-likelihood up to an additive constant. The constant is
    # dropped because normalisation cancels it.
    var = self.cfg.measurement_noise**2
    return -0.5 * torch.sum(residual * residual, dim=-1) / var

  def _update_optimal(self, measurement: torch.Tensor) -> torch.Tensor:
    """Sample from p(x_t | x_{t-1}, z_t) and weight by p(z_t | x_{t-1}).

    ``self._particles`` currently holds the deterministic drift F x_{t-1}
    (the stochastic part was deferred by ``_predict``). Returns [B, N].
    """
    dt = self._pending_dt if self._pending_dt is not None else 0.0
    self._pending_dt = None

    Sigma, S = self._proposal_moments(dt)
    Q = torch.diag(torch.diagonal(self._Q).clamp(min=1e-10))
    Q_inv = torch.diag(1.0 / torch.diagonal(Q))
    R_inv = torch.linalg.inv(self._R)

    drift = self._particles  # [B, N, 6] = F x
    z = measurement.unsqueeze(1)  # [B, 1, 3]

    # Proposal mean: Sigma (Q^-1 F x + H^T R^-1 z), per particle.
    term_prior = drift @ Q_inv.T  # [B, N, 6]
    term_meas = (z @ R_inv.T) @ self._H  # [B, 1, 6]
    mean = (term_prior + term_meas) @ Sigma.T  # [B, N, 6]

    # Sample from N(mean, Sigma) via a Cholesky factor of Sigma.
    L = torch.linalg.cholesky(Sigma + torch.eye(STATE_DIM, device=self.device) * 1e-12)
    eps = self._randn_like(drift)  # [B, N, 6]
    self._particles = mean + eps @ L.T

    # Incremental weight: log N(z; H F x_{t-1}, S).
    pred_meas = drift[..., POS_SLICE]  # [B, N, 3] = H F x
    residual = z - pred_meas  # [B, N, 3]
    S_inv = torch.linalg.inv(S)
    quad = torch.einsum("bni,ij,bnj->bn", residual, S_inv, residual)
    return -0.5 * quad

  def _normalise_log_weights(self) -> None:
    """Normalise log-weights via log-sum-exp, with a degeneracy fallback."""
    self._log_weights = self._log_weights - torch.logsumexp(
      self._log_weights, dim=1, keepdim=True
    )
    # If every particle underflowed, logsumexp returns -inf and the whole
    # row becomes NaN. Reset those rows to a uniform prior rather than
    # propagating NaN into the policy observation.
    bad = ~torch.isfinite(self._log_weights).all(dim=1)
    if bool(bad.any()):
      uniform = -torch.log(torch.tensor(float(self.num_particles)))
      self._log_weights[bad] = uniform.to(self.device)

  # ------------------------------------------------------------------
  # 3. RESAMPLE
  # ------------------------------------------------------------------

  def _maybe_resample(self) -> None:
    """Resample the environments whose ESS has fallen below threshold."""
    need = self._last_ess < (self.resample_threshold * self.num_particles)
    if not bool(need.any()):
      return
    env_ids = torch.nonzero(need, as_tuple=False).squeeze(-1)
    self._resample_idx(env_ids)
    self._resample_count[env_ids] += 1

  def _resample_idx(self, env_ids: torch.Tensor) -> None:
    """Systematic resampling for the selected environments.

    Systematic (rather than multinomial) resampling is used because it has
    lower variance for the same cost: it draws a single uniform offset and
    then steps through the cumulative distribution in equal strides, so a
    particle with weight w is guaranteed floor(N*w) or ceil(N*w) copies
    instead of a binomially-distributed count.
    """
    k = env_ids.shape[0]
    weights = torch.exp(self._log_weights[env_ids])  # [K, N]
    cumsum = torch.cumsum(weights, dim=1)
    cumsum = cumsum / cumsum[:, -1:].clamp(min=1e-12)  # guard rounding

    # One random offset per env, then N equally spaced positions.
    offset = self._rand((k, 1))
    strides = torch.arange(self.num_particles, device=self.device).float()
    positions = (offset + strides.unsqueeze(0)) / self.num_particles  # [K, N]

    idx = torch.searchsorted(cumsum, positions.contiguous())
    idx = torch.clamp(idx, max=self.num_particles - 1)  # [K, N]

    resampled = torch.gather(
      self._particles[env_ids],
      dim=1,
      index=idx.unsqueeze(-1).expand(-1, -1, STATE_DIM),
    )

    # Roughening: jitter duplicated particles so the ensemble keeps its
    # diversity. Without this the filter reports an ever-shrinking
    # covariance while its actual error grows -- confidently wrong.
    if self.roughening > 0.0:
      jitter = self._randn_like(resampled)
      jitter[..., POS_SLICE] *= self.roughening * self.cfg.initial_pos_std
      jitter[..., VEL_SLICE] *= self.roughening * self.cfg.initial_vel_std
      resampled = resampled + jitter

    self._particles[env_ids] = resampled
    self._log_weights[env_ids] = -torch.log(
      torch.tensor(float(self.num_particles), device=self.device)
    )

  # ------------------------------------------------------------------
  # Reset
  # ------------------------------------------------------------------

  def _reset_idx(self, env_ids: torch.Tensor, measurement: torch.Tensor | None) -> None:
    k = env_ids.shape[0]
    particles = torch.zeros(k, self.num_particles, STATE_DIM, device=self.device)

    if measurement is not None:
      centre = measurement.unsqueeze(1)  # [K, 1, 3]
    else:
      centre = torch.zeros(k, 1, MEAS_DIM, device=self.device)

    spread = self._randn((k, self.num_particles, STATE_DIM))
    particles[..., POS_SLICE] = (
      centre + spread[..., POS_SLICE] * self.cfg.initial_pos_std
    )
    particles[..., VEL_SLICE] = spread[..., VEL_SLICE] * self.cfg.initial_vel_std

    self._particles[env_ids] = particles
    self._log_weights[env_ids] = -torch.log(
      torch.tensor(float(self.num_particles), device=self.device)
    )
    self._resample_count[env_ids] = 0
    self._last_ess[env_ids] = float(self.num_particles)

  # ------------------------------------------------------------------
  # RNG helpers (respect the optional seeded generator)
  # ------------------------------------------------------------------

  def _randn(self, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randn(*shape, device=self.device, generator=self._generator)

  def _randn_like(self, ref: torch.Tensor) -> torch.Tensor:
    return torch.randn(ref.shape, device=self.device, generator=self._generator)

  def _rand(self, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.rand(*shape, device=self.device, generator=self._generator)
