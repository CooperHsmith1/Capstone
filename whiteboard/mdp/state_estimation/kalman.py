"""Extended and Unscented Kalman filters for the comparative framework.

These are the baselines the particle filter is benchmarked against in Section
4.3 of the interim report. Both are batched over environments in exactly the
same way as the particle filter so the three are drop-in interchangeable.

Read the "NOTE ON THE EKF" docstring in ``base.py`` before interpreting any
benchmark numbers -- with this system's linear models the EKF is algebraically
a linear Kalman filter, and that is the correct implementation rather than a
simplification.
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


class ExtendedKalmanFilter(StateEstimator):
    """EKF over the shared 6-D constant-velocity pen model.

    Args:
        num_envs: Batch size B.
        cfg: Shared noise configuration.
        device: Torch device.
    """

    def __init__(
        self,
        num_envs: int,
        cfg: StateEstimatorCfg,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(num_envs, cfg, device)
        self._mean = torch.zeros(self.num_envs, STATE_DIM, device=self.device)
        self._cov = torch.zeros(
            self.num_envs, STATE_DIM, STATE_DIM, device=self.device
        )
        self.reset()

    @property
    def name(self) -> str:
        return "EKF"

    @property
    def state(self) -> torch.Tensor:
        return self._mean

    @property
    def covariance(self) -> torch.Tensor:
        return self._cov

    def _predict(self, dt: float) -> None:
        F = constant_velocity_transition(dt, self.device)     # [6, 6]
        self._mean = self._mean @ F.T                         # [B, 6]
        # P = F P F^T + Q, batched.
        self._cov = F @ self._cov @ F.T + self._Q

    def _update(self, measurement: torch.Tensor) -> None:
        H = self._H                                           # [3, 6]
        # Innovation covariance S = H P H^T + R.
        S = H @ self._cov @ H.T + self._R                     # [B, 3, 3]
        # Kalman gain K = P H^T S^-1, solved rather than inverted for stability.
        PHt = self._cov @ H.T                                 # [B, 6, 3]
        K = torch.linalg.solve(S.transpose(-1, -2), PHt.transpose(-1, -2))
        K = K.transpose(-1, -2)                               # [B, 6, 3]

        innovation = (measurement - self._mean[:, POS_SLICE]).unsqueeze(-1)
        self._mean = self._mean + (K @ innovation).squeeze(-1)

        # Joseph form: numerically stable and keeps P symmetric positive
        # definite even with an imperfect gain, which the plain (I - KH)P form
        # does not once round-off accumulates over a 1000-step episode.
        eye = torch.eye(STATE_DIM, device=self.device).expand_as(self._cov)
        IKH = eye - K @ H
        self._cov = IKH @ self._cov @ IKH.transpose(-1, -2) + K @ self._R @ K.transpose(-1, -2)

    def _reset_idx(
        self, env_ids: torch.Tensor, measurement: torch.Tensor | None
    ) -> None:
        k = env_ids.shape[0]
        mean = torch.zeros(k, STATE_DIM, device=self.device)
        if measurement is not None:
            mean[:, POS_SLICE] = measurement
        self._mean[env_ids] = mean

        diag = torch.tensor(
            [self.cfg.initial_pos_std**2] * 3 + [self.cfg.initial_vel_std**2] * 3,
            device=self.device,
        )
        self._cov[env_ids] = torch.diag(diag).expand(k, STATE_DIM, STATE_DIM)


class UnscentedKalmanFilter(StateEstimator):
    """UKF over the shared 6-D constant-velocity pen model.

    Propagates 2n+1 = 13 sigma points through the models rather than
    linearising. For this linear system that yields the same answer as the EKF
    to numerical precision -- which is itself a useful benchmark result, since
    it confirms both implementations are correct before non-Gaussian noise is
    introduced.

    Args:
        num_envs: Batch size B.
        cfg: Shared noise configuration.
        alpha: Spread of the sigma points around the mean.

            DO NOT set this to the textbook 1e-3 here. That value is quoted for
            float64 implementations. The mean-point weight is
            lambda / (n + lambda) with lambda = alpha^2 (n + kappa) - n, so at
            alpha=1e-3 and n=6 you get lambda = -5.999994 and a mean weight of
            about -1e6, with the remaining weights near +8e4. Those cancel to
            the right answer in float64 but lose every significant digit in the
            float32 that mjlab runs in -- measured RMSE degrades from 0.0069 m
            to 0.024 m, i.e. worse than not filtering at all.

            alpha=1.0 with kappa=0 gives lambda=0 and uniform weights of 1/(2n),
            which is perfectly conditioned. Anything from 0.1 upward reproduces
            the EKF to six decimal places on this linear system.
        beta: Prior knowledge of the distribution; 2.0 is optimal for Gaussians.
        kappa: Secondary scaling, conventionally 0 or 3-n.
        device: Torch device.
    """

    def __init__(
        self,
        num_envs: int,
        cfg: StateEstimatorCfg,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(num_envs, cfg, device)

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)
        n = STATE_DIM
        self._lambda = self.alpha**2 * (n + self.kappa) - n

        denom = n + self._lambda
        if abs(denom) < 1e-2:
            raise ValueError(
                f"UKF sigma-point scaling is numerically unusable: "
                f"n + lambda = {denom:.2e} with alpha={self.alpha}, "
                f"kappa={self.kappa}. This yields sigma-point weights of order "
                f"{1.0 / max(abs(denom), 1e-12):.0e}, which lose all precision "
                f"in float32. Use alpha >= 0.1 (alpha=1.0 is the safe default)."
            )
        self._num_sigma = 2 * n + 1

        # Sigma-point weights: index 0 is the mean point, the rest are paired.
        wm = torch.full(
            (self._num_sigma,),
            1.0 / (2.0 * (n + self._lambda)),
            device=self.device,
        )
        wc = wm.clone()
        wm[0] = self._lambda / (n + self._lambda)
        wc[0] = wm[0] + (1.0 - self.alpha**2 + self.beta)
        self._wm = wm
        self._wc = wc

        self._mean = torch.zeros(self.num_envs, STATE_DIM, device=self.device)
        self._cov = torch.zeros(
            self.num_envs, STATE_DIM, STATE_DIM, device=self.device
        )
        self.reset()

    @property
    def name(self) -> str:
        return "UKF"

    @property
    def state(self) -> torch.Tensor:
        return self._mean

    @property
    def covariance(self) -> torch.Tensor:
        return self._cov

    def _sigma_points(self) -> torch.Tensor:
        """Return sigma points, shape [B, 2n+1, 6]."""
        n = STATE_DIM
        scale = n + self._lambda

        # Symmetrise and floor the eigenvalues before factorising: Cholesky
        # fails outright on a matrix that has drifted a hair non-PSD, and that
        # drift is inevitable over thousands of steps in float32.
        cov = 0.5 * (self._cov + self._cov.transpose(-1, -2))
        eye = torch.eye(n, device=self.device).expand_as(cov)
        cov = cov + eye * 1e-9

        try:
            sqrt_cov = torch.linalg.cholesky(scale * cov)
        except RuntimeError:
            # Fall back to an eigendecomposition, clamping negative eigenvalues.
            evals, evecs = torch.linalg.eigh(scale * cov)
            evals = torch.clamp(evals, min=1e-12)
            sqrt_cov = evecs @ torch.diag_embed(torch.sqrt(evals))

        mean = self._mean.unsqueeze(1)                        # [B, 1, 6]
        # Columns of the Cholesky factor are the offsets.
        offsets = sqrt_cov.transpose(-1, -2)                  # [B, 6, 6]
        return torch.cat([mean, mean + offsets, mean - offsets], dim=1)

    def _predict(self, dt: float) -> None:
        F = constant_velocity_transition(dt, self.device)
        sigmas = self._sigma_points()                         # [B, S, 6]
        propagated = sigmas @ F.T                             # [B, S, 6]

        wm = self._wm.view(1, -1, 1)
        self._mean = torch.sum(wm * propagated, dim=1)        # [B, 6]

        delta = propagated - self._mean.unsqueeze(1)          # [B, S, 6]
        wc = self._wc.view(1, -1, 1)
        self._cov = torch.einsum("bsi,bsj->bij", wc * delta, delta) + self._Q

    def _update(self, measurement: torch.Tensor) -> None:
        sigmas = self._sigma_points()                         # [B, S, 6]
        meas_sigmas = sigmas[..., POS_SLICE]                  # [B, S, 3]

        wm = self._wm.view(1, -1, 1)
        wc = self._wc.view(1, -1, 1)

        z_pred = torch.sum(wm * meas_sigmas, dim=1)           # [B, 3]
        dz = meas_sigmas - z_pred.unsqueeze(1)                # [B, S, 3]
        dx = sigmas - self._mean.unsqueeze(1)                 # [B, S, 6]

        S = torch.einsum("bsi,bsj->bij", wc * dz, dz) + self._R      # [B, 3, 3]
        cross = torch.einsum("bsi,bsj->bij", wc * dx, dz)            # [B, 6, 3]

        K = torch.linalg.solve(S.transpose(-1, -2), cross.transpose(-1, -2))
        K = K.transpose(-1, -2)                                      # [B, 6, 3]

        innovation = (measurement - z_pred).unsqueeze(-1)            # [B, 3, 1]
        self._mean = self._mean + (K @ innovation).squeeze(-1)
        self._cov = self._cov - K @ S @ K.transpose(-1, -2)
        # Re-symmetrise: the subtraction above is where asymmetry creeps in.
        self._cov = 0.5 * (self._cov + self._cov.transpose(-1, -2))

    def _reset_idx(
        self, env_ids: torch.Tensor, measurement: torch.Tensor | None
    ) -> None:
        k = env_ids.shape[0]
        mean = torch.zeros(k, STATE_DIM, device=self.device)
        if measurement is not None:
            mean[:, POS_SLICE] = measurement
        self._mean[env_ids] = mean

        diag = torch.tensor(
            [self.cfg.initial_pos_std**2] * 3 + [self.cfg.initial_vel_std**2] * 3,
            device=self.device,
        )
        self._cov[env_ids] = torch.diag(diag).expand(k, STATE_DIM, STATE_DIM)
