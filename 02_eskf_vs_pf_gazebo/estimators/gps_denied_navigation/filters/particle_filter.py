"""Vectorized bootstrap SMC with bias states and quaternion attitude.

Diagnostics are computed lazily from a reference to the pre-resampling cloud so
that observation cost is excluded from the timed predict/update region. The
estimator arithmetic is unchanged by observation (verified bitwise in tests).
"""
import numpy as np
from .base_filter import NavigationState, prior_std, propagate
from .quaternion import normalize, multiply, exp_rotation, weighted_mean, log_rotation, conjugate


def systematic_resample(weights, rng):
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(weights) == 0 or np.any(weights < 0) or not np.all(np.isfinite(weights)) or not np.isclose(weights.sum(), 1):
        raise ValueError("Resampling requires normalized finite nonnegative weights")
    positions = (rng.random()+np.arange(len(weights)))/len(weights)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def normalize_log_weights(log_weights):
    """Max-subtracted exponentiation; returns normalized weights.

    Raises FloatingPointError when every weight is -inf/NaN (total support loss).
    """
    maximum = np.max(log_weights)
    if not np.isfinite(maximum):
        raise FloatingPointError("PF likelihood normalization failed")
    weights = np.exp(log_weights-maximum)
    return weights/weights.sum()


def cloud_statistics(weights, position, velocity, quaternion, accel_bias, gyro_bias):
    """Weighted moments of a particle cloud (read-only)."""
    w = weights
    p, v = w @ position, w @ velocity
    delta = position-p
    mean_q = weighted_mean(quaternion, w)
    rotation_error = log_rotation(multiply(conjugate(mean_q), quaternion))
    ba, bg = w @ accel_bias, w @ gyro_bias
    return {
        "ess": float(1/np.sum(w**2)),
        "weight_min": float(w.min()), "weight_max": float(w.max()),
        "weight_mean": float(w.mean()),
        "position_std": np.sqrt(np.maximum(0, w @ ((position-p)**2))),
        "velocity_std": np.sqrt(np.maximum(0, w @ ((velocity-v)**2))),
        "attitude_std": np.sqrt(np.maximum(0, w @ (rotation_error**2))),
        "accel_bias_std": np.sqrt(np.maximum(0, w @ ((accel_bias-ba)**2))),
        "gyro_bias_std": np.sqrt(np.maximum(0, w @ ((gyro_bias-bg)**2))),
        "position_covariance": (delta.T*w) @ delta,
    }


class ParticleFilter:
    uses_gps = True

    def __init__(self, initial_state, imu_config, gps_config, filter_config, particle_config, rng):
        self.imu_config, self.gps_config = imu_config, gps_config
        self.config, self.particle_config, self.rng = filter_config, particle_config, rng
        n = particle_config.particles
        errors = rng.normal(size=(n,15))*prior_std(filter_config)
        self.position = initial_state.position+errors[:,:3]
        self.velocity = initial_state.velocity+errors[:,3:6]
        self.quaternion = normalize(multiply(initial_state.quaternion, exp_rotation(errors[:,6:9])))
        self.accel_bias = initial_state.accel_bias+errors[:,9:12]
        self.gyro_bias = initial_state.gyro_bias+errors[:,12:15]
        self.weights = np.full(n, 1/n)
        self.last_ess = float(n)  # ESS before any resampling, for diagnostics
        self.resampling_count = 0
        self.degeneracy_count = 0
        self.minimum_ess = float(n)
        self.lineages = np.arange(n)
        self.unique_ancestor_count = n
        self.resampled_last_update = False
        self._posterior = None

    def _snapshot(self):
        # References only: resampling and predict() assign NEW arrays, so these
        # remain the pre-resampling cloud until the next predict() call.
        self._posterior = (self.weights.copy(), self.position, self.velocity, self.quaternion,
                           self.accel_bias, self.gyro_bias)

    def diagnostics(self):
        """Weighted posterior moments before resampling (current cloud if none yet)."""
        if self._posterior is None:
            self._snapshot()
        return cloud_statistics(*self._posterior)

    @property
    def last_diagnostics(self):
        return self.diagnostics()

    @property
    def effective_sample_size(self):
        return float(1/np.sum(self.weights**2))

    def predict(self, measurement):
        n, c, dt = len(self.weights), self.imu_config, measurement.dt
        k = np.sqrt(self.config.process_noise_scale)   # SD scale; exactly 1.0 by default
        force = measurement.acceleration-self.accel_bias + self.rng.normal(0,c.accel_noise_std*k,(n,3))
        rate = measurement.angular_velocity-self.gyro_bias + self.rng.normal(0,c.gyro_noise_std*k,(n,3))
        self.position, self.velocity, self.quaternion = propagate(self.position, self.velocity,
            self.quaternion, force, rate, dt, c.gravity)
        # Non-in-place updates keep any posterior snapshot immutable.
        self.accel_bias = self.accel_bias + self.rng.normal(0,c.accel_bias_random_walk*k*np.sqrt(dt),(n,3))
        self.gyro_bias = self.gyro_bias + self.rng.normal(0,c.gyro_bias_random_walk*k*np.sqrt(dt),(n,3))
        self._posterior = None

    def log_likelihood(self, measurement):
        """Gaussian GPS log-likelihood per particle, up to a common constant."""
        scale = self.config.measurement_noise_scale
        variance = max(self.gps_config.position_noise_std**2*scale, self.config.measurement_variance_floor)
        log_likelihood = -.5*np.sum((self.position-measurement.position)**2,axis=1)/variance
        if measurement.velocity is not None:
            variance_v = max(self.gps_config.velocity_noise_std**2*scale, self.config.measurement_variance_floor)
            log_likelihood -= .5*np.sum((self.velocity-measurement.velocity)**2,axis=1)/variance_v
        return log_likelihood

    def update(self, measurement):
        self.resampled_last_update = False
        if not measurement.available:
            # Without an observation ESS is that of the current (unchanged) weights.
            self.last_ess = self.effective_sample_size
            self._snapshot()
            return
        with np.errstate(divide="ignore"):
            log_weights = np.log(self.weights)+self.log_likelihood(measurement)
        weights = normalize_log_weights(log_weights)
        self.weights = weights
        self.last_ess = self.effective_sample_size
        self._snapshot()
        self.minimum_ess = min(self.minimum_ess, self.last_ess)
        if self.last_ess < self.particle_config.degeneracy_threshold*len(weights):
            self.degeneracy_count += 1
        if self.last_ess < self.particle_config.resample_threshold*len(weights):
            self.resample()

    def resample(self):
        indices = systematic_resample(self.weights, self.rng)
        for name in ("position", "velocity", "quaternion", "accel_bias", "gyro_bias"):
            setattr(self, name, getattr(self, name)[indices].copy())
        self.lineages = self.lineages[indices]
        self.unique_ancestor_count = int(len(np.unique(self.lineages)))
        self.weights = np.full(len(self.weights), 1/len(self.weights))
        self.resampling_count += 1
        self.resampled_last_update = True

    def estimate(self):
        return NavigationState(self.weights @ self.position, self.weights @ self.velocity,
            weighted_mean(self.quaternion, self.weights), self.weights @ self.accel_bias,
            self.weights @ self.gyro_bias)

    def position_covariance(self):
        delta = self.position-self.weights @ self.position
        return (delta.T*self.weights) @ delta
