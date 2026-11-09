"""15-error-state inertial/GPS EKF, right multiplicative attitude errors."""
import numpy as np
from .base_filter import prior_std, propagate
from .quaternion import matrix, skew, multiply, exp_rotation, normalize, right_jacobian


def linearized_dynamics(quaternion, force, rate):
    R = matrix(quaternion)
    F = np.zeros((15, 15))
    F[0:3, 3:6] = np.eye(3)
    F[3:6, 6:9] = -R @ skew(force)
    F[3:6, 9:12] = -R
    F[6:9, 6:9] = -skew(rate)
    F[6:9, 12:15] = -np.eye(3)
    G = np.zeros((15, 12))
    G[3:6, 0:3] = -R
    G[6:9, 3:6] = -np.eye(3)
    G[9:12, 6:9] = np.eye(3)
    G[12:15, 9:12] = np.eye(3)
    return F, G


def discrete_model(quaternion, force, rate, dt):
    """Exact first-order Jacobians of the implemented midpoint discrete map."""
    qmid = normalize(multiply(quaternion, exp_rotation(rate*dt/2)))
    R = matrix(qmid)
    Ehalf = matrix(exp_rotation(-rate*dt/2))
    Efull = matrix(exp_rotation(-rate*dt))
    Jrhalf = right_jacobian(rate*dt/2)
    Jrfull = right_jacobian(rate*dt)
    Dtheta = -R @ skew(force) @ Ehalf
    Dgyro = .5*R @ skew(force) @ Jrhalf*dt
    Phi = np.eye(15)
    Phi[:3,3:6] = np.eye(3)*dt
    Phi[:3,6:9] = .5*Dtheta*dt**2
    Phi[:3,9:12] = -.5*R*dt**2
    Phi[:3,12:15] = .5*Dgyro*dt**2
    Phi[3:6,6:9] = Dtheta*dt
    Phi[3:6,9:12] = -R*dt
    Phi[3:6,12:15] = Dgyro*dt
    Phi[6:9,6:9] = Efull
    Phi[6:9,12:15] = -Jrfull*dt
    B = np.zeros((15,12))
    B[:3,:3] = -.5*R*dt**2
    B[3:6,:3] = -R*dt
    B[:3,3:6] = .5*Dgyro*dt**2
    B[3:6,3:6] = Dgyro*dt
    B[6:9,3:6] = -Jrfull*dt
    B[9:12,6:9] = np.eye(3)
    B[12:15,9:12] = np.eye(3)
    return Phi,B


def process_covariance(B, imu_config, dt):
    variance = np.repeat([imu_config.accel_noise_std**2,imu_config.gyro_noise_std**2,
        imu_config.accel_bias_random_walk**2*dt,imu_config.gyro_bias_random_walk**2*dt],3)
    L = B*np.sqrt(variance)
    return L @ L.T


class ExtendedKalmanFilter:
    uses_gps = True

    def __init__(self, initial_state, imu_config, gps_config, filter_config):
        self.state = initial_state.copy()
        self.imu_config, self.gps_config, self.config = imu_config, gps_config, filter_config
        self.P = np.diag(prior_std(filter_config)**2)
        self.last_innovation = None
        self.last_innovation_covariance = None
        self.last_gain = None
        self.last_update_applied = False

    def predict(self, measurement):
        s, dt = self.state, measurement.dt
        force = measurement.acceleration-s.accel_bias
        rate = measurement.angular_velocity-s.gyro_bias
        Phi,B = discrete_model(s.quaternion,force,rate,dt)
        Q = process_covariance(B,self.imu_config,dt)*self.config.process_noise_scale
        self.P = Phi @ self.P @ Phi.T + Q
        self.P = .5*(self.P+self.P.T)
        s.position, s.velocity, s.quaternion = propagate(s.position, s.velocity, s.quaternion,
            force, rate, dt, self.imu_config.gravity)
        if not np.all(np.isfinite(self.P)):
            raise FloatingPointError("Nonfinite EKF covariance")

    def update(self, measurement):
        self.last_update_applied = False
        if not measurement.available:
            return
        s = self.state
        has_velocity = measurement.velocity is not None
        size = 6 if has_velocity else 3
        H = np.zeros((size, 15))
        H[:3, :3] = np.eye(3)
        innovation = measurement.position-s.position
        variance = [self.gps_config.position_noise_std**2]*3
        if has_velocity:
            H[3:6, 3:6] = np.eye(3)
            innovation = np.concatenate((innovation, measurement.velocity-s.velocity))
            variance += [self.gps_config.velocity_noise_std**2]*3
        R = np.diag(np.maximum(np.asarray(variance)*self.config.measurement_noise_scale, self.config.measurement_variance_floor))
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        correction = K @ innovation
        I_KH = np.eye(15)-K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        s.position += correction[:3]
        s.velocity += correction[3:6]
        s.quaternion = normalize(multiply(s.quaternion, exp_rotation(correction[6:9])))
        s.accel_bias += correction[9:12]
        s.gyro_bias += correction[12:15]
        reset = np.eye(15)
        reset[6:9, 6:9] = right_jacobian(correction[6:9])
        self.P = reset @ self.P @ reset.T
        self.P = .5*(self.P+self.P.T)
        self.last_innovation = innovation.copy()
        self.last_innovation_covariance = S
        self.last_gain = K
        self.last_update_applied = True
        self.validate_covariance()

    def validate_covariance(self):
        if not np.all(np.isfinite(self.P)) or np.linalg.eigvalsh(self.P)[0] < -self.config.covariance_negative_tolerance:
            raise FloatingPointError("EKF covariance is not positive semidefinite")

    def estimate(self):
        return self.state.copy()

    def position_covariance(self):
        return self.P[:3, :3].copy()
