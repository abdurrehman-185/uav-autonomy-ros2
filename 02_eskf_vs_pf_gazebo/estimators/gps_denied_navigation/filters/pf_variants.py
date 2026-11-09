"""Remedy estimators for the bootstrap PF, kept separate so the baseline is unchanged.

RoughenedParticleFilter
    Bootstrap PF plus the roughening step of Gordon, Salmond and Smith (1993):
    after each resampling, every state dimension j receives independent Gaussian
    jitter with SD = K * E_j * N^(-1/d), where E_j is the sample range and d = 15.
    K is a tuning constant; the study evaluates K = 0.2, 0.05 and 0.01.

MarginalizedParticleFilter
    Rao-Blackwellized filter (Schon, Gustafsson and Nordlund 2005). Particles sample
    only the nonlinear part (attitude q, gyro bias b_g). Conditioned on the attitude
    history, position, velocity and accelerometer bias are linear-Gaussian, so each
    particle carries a Kalman mean m (9) and covariance C (9x9) for them:
        x+ = A x + u + noise,  z = H x + n
    It uses the same strapdown step and the same noise model as the EKF and the PF.
"""
import numpy as np

from .base_filter import NavigationState
from .particle_filter import ParticleFilter, normalize_log_weights, systematic_resample
from .quaternion import conjugate, exp_rotation, log_rotation, multiply, normalize, weighted_mean, rotate

STATE_DIMENSION = 15


class RoughenedParticleFilter(ParticleFilter):
    roughening_k = 0.2

    def resample(self):
        super().resample()
        n = len(self.weights)
        scale = self.roughening_k*n**(-1./STATE_DIMENSION)
        for name in ("position", "velocity", "accel_bias", "gyro_bias"):
            values = getattr(self, name)
            span = values.max(axis=0)-values.min(axis=0)
            setattr(self, name, values+self.rng.normal(size=values.shape)*scale*span)
        mean = weighted_mean(self.quaternion, self.weights)
        errors = log_rotation(multiply(conjugate(mean), self.quaternion))
        span = errors.max(axis=0)-errors.min(axis=0)
        jitter = self.rng.normal(size=errors.shape)*scale*span
        self.quaternion = normalize(multiply(self.quaternion, exp_rotation(jitter)))


class RoughenedK005(RoughenedParticleFilter):
    roughening_k = 0.05


class RoughenedK001(RoughenedParticleFilter):
    roughening_k = 0.01


class MarginalizedParticleFilter:
    uses_gps = True

    def __init__(self, initial_state, imu_config, gps_config, filter_config, particle_config, rng):
        self.imu_config, self.gps_config = imu_config, gps_config
        self.config, self.particle_config, self.rng = filter_config, particle_config, rng
        n = particle_config.particles
        draws = rng.normal(size=(n, 6))
        attitude = draws[:, :3]*filter_config.initial_attitude_std
        self.quaternion = normalize(multiply(initial_state.quaternion, exp_rotation(attitude)))
        self.gyro_bias = initial_state.gyro_bias+draws[:, 3:6]*filter_config.initial_gyro_bias_std
        self.mean = np.tile(np.r_[initial_state.position, initial_state.velocity, initial_state.accel_bias], (n, 1))
        prior = np.repeat([filter_config.initial_position_std, filter_config.initial_velocity_std,
                           filter_config.initial_accel_bias_std], 3)**2
        self.covariance = np.tile(np.diag(prior), (n, 1, 1))
        self.weights = np.full(n, 1/n)
        self.last_ess = float(n)
        self.resampling_count = 0
        self.degeneracy_count = 0
        self.minimum_ess = float(n)
        self.lineages = np.arange(n)
        self.unique_ancestor_count = n
        self.resampled_last_update = False
        self._posterior = None

    @property
    def effective_sample_size(self):
        return float(1/np.sum(self.weights**2))

    def predict(self, measurement):
        n, c, dt = len(self.weights), self.imu_config, measurement.dt
        k = np.sqrt(self.config.process_noise_scale)
        rate = measurement.angular_velocity-self.gyro_bias+self.rng.normal(0, c.gyro_noise_std*k, (n, 3))
        q_mid = normalize(multiply(self.quaternion, exp_rotation(rate*dt/2)))
        # Columns of R(q_mid), vectorized over particles: R has shape (n, 3, 3).
        R = np.stack([rotate(q_mid, axis) for axis in np.eye(3)], axis=-1)
        world = np.einsum("nij,j->ni", R, measurement.acceleration)+np.array([0., 0., -c.gravity])
        A = np.tile(np.eye(9), (n, 1, 1))
        A[:, 0:3, 3:6] = np.eye(3)*dt
        A[:, 0:3, 6:9] = -.5*R*dt**2
        A[:, 3:6, 6:9] = -R*dt
        m = self.mean
        new = np.empty_like(m)
        new[:, 0:3] = m[:, 0:3]+m[:, 3:6]*dt-.5*dt**2*np.einsum("nij,nj->ni", R, m[:, 6:9])+.5*world*dt**2
        new[:, 3:6] = m[:, 3:6]-dt*np.einsum("nij,nj->ni", R, m[:, 6:9])+world*dt
        new[:, 6:9] = m[:, 6:9]
        self.mean = new
        C = A @ self.covariance @ A.transpose(0, 2, 1)
        # Accelerometer noise: G = [-0.5 R dt^2; -R dt; 0]. R R^T = I reduces G G^T to scalar blocks.
        var = (c.accel_noise_std*k)**2
        identity = np.eye(3)
        C[:, 0:3, 0:3] += var*.25*dt**4*identity
        C[:, 0:3, 3:6] += var*.5*dt**3*identity
        C[:, 3:6, 0:3] += var*.5*dt**3*identity
        C[:, 3:6, 3:6] += var*dt**2*identity
        C[:, 6:9, 6:9] += (c.accel_bias_random_walk*k)**2*dt*identity
        self.covariance = C
        self.quaternion = normalize(multiply(self.quaternion, exp_rotation(rate*dt)))
        self.gyro_bias = self.gyro_bias+self.rng.normal(0, c.gyro_bias_random_walk*k*np.sqrt(dt), (n, 3))
        self._posterior = None

    def update(self, measurement):
        self.resampled_last_update = False
        if measurement.available:
            if measurement.velocity is not None:
                raise NotImplementedError("Position-only GNSS is used in this study")
            variance = max(self.gps_config.position_noise_std**2*self.config.measurement_noise_scale,
                           self.config.measurement_variance_floor)
            y = measurement.position-self.mean[:, :3]
            S = self.covariance[:, :3, :3]+variance*np.eye(3)
            solved = np.linalg.solve(S, y[..., None])[..., 0]
            _, logdet = np.linalg.slogdet(S)
            with np.errstate(divide="ignore"):
                log_weights = np.log(self.weights)-.5*np.sum(y*solved, axis=1)-.5*logdet
            self.weights = normalize_log_weights(log_weights)
            # K = C H^T S^-1; x+ = x + K y; Joseph form for C.
            CHt = self.covariance[:, :, :3]
            K = np.linalg.solve(S, CHt.transpose(0, 2, 1)).transpose(0, 2, 1)
            self.mean = self.mean+np.einsum("nij,nj->ni", K, y)
            I_KH = np.tile(np.eye(9), (len(y), 1, 1))
            I_KH[:, :, :3] -= K
            C = I_KH @ self.covariance @ I_KH.transpose(0, 2, 1)+variance*(K @ K.transpose(0, 2, 1))
            self.covariance = .5*(C+C.transpose(0, 2, 1))
        self.last_ess = self.effective_sample_size
        self._snapshot()
        if measurement.available:
            self.minimum_ess = min(self.minimum_ess, self.last_ess)
            n = len(self.weights)
            if self.last_ess < self.particle_config.degeneracy_threshold*n:
                self.degeneracy_count += 1
            if self.last_ess < self.particle_config.resample_threshold*n:
                self.resample()

    def resample(self):
        indices = systematic_resample(self.weights, self.rng)
        self.quaternion = self.quaternion[indices].copy()
        self.gyro_bias = self.gyro_bias[indices].copy()
        self.mean = self.mean[indices].copy()
        self.covariance = self.covariance[indices].copy()
        self.lineages = self.lineages[indices]
        self.unique_ancestor_count = int(len(np.unique(self.lineages)))
        self.weights = np.full(len(self.weights), 1/len(self.weights))
        self.resampling_count += 1
        self.resampled_last_update = True

    def _snapshot(self):
        self._posterior = (self.weights.copy(), self.quaternion, self.gyro_bias, self.mean, self.covariance)

    def diagnostics(self):
        """Mixture moments of the marginal posterior (law of total variance)."""
        if self._posterior is None:
            self._snapshot()
        w, q, bg, m, C = self._posterior
        mean = w @ m
        delta = m-mean
        total = np.einsum("n,nij->ij", w, C)+(delta.T*w) @ delta
        spread = np.sqrt(np.maximum(0, np.diag(total)))
        rotation = log_rotation(multiply(conjugate(weighted_mean(q, w)), q))
        bias_mean = w @ bg
        return {"ess": float(1/np.sum(w**2)), "weight_min": float(w.min()), "weight_max": float(w.max()),
                "weight_mean": float(w.mean()), "position_std": spread[:3], "velocity_std": spread[3:6],
                "accel_bias_std": spread[6:9],
                "attitude_std": np.sqrt(np.maximum(0, w @ (rotation**2))),
                "gyro_bias_std": np.sqrt(np.maximum(0, w @ ((bg-bias_mean)**2))),
                "position_covariance": total[:3, :3]}

    def estimate(self):
        mean = self.weights @ self.mean
        return NavigationState(mean[:3], mean[3:6], weighted_mean(self.quaternion, self.weights), mean[6:9],
                               self.weights @ self.gyro_bias)

    def position_covariance(self):
        return self.diagnostics()["position_covariance"]
