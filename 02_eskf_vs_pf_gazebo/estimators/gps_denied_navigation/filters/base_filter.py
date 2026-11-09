from dataclasses import dataclass
from typing import Protocol
import numpy as np
from .quaternion import normalize, multiply, exp_rotation, rotate


@dataclass
class NavigationState:
    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray
    accel_bias: np.ndarray
    gyro_bias: np.ndarray

    def copy(self):
        return NavigationState(*(np.array(v, dtype=float, copy=True) for v in
            (self.position, self.velocity, self.quaternion, self.accel_bias, self.gyro_bias)))


class NavigationFilter(Protocol):
    def predict(self, measurement): ...
    def update(self, measurement): ...
    def estimate(self) -> NavigationState: ...


def propagate(position, velocity, quaternion, force, rate, dt, gravity):
    """Constant measured body force/rate over interval, midpoint rotation."""
    qmid = normalize(multiply(quaternion, exp_rotation(rate*dt/2)))
    acceleration = rotate(qmid, force) + np.array([0., 0., -gravity])
    new_position = position + velocity*dt + .5*acceleration*dt**2
    new_velocity = velocity + acceleration*dt
    new_quaternion = normalize(multiply(quaternion, exp_rotation(rate*dt)))
    return new_position, new_velocity, new_quaternion


def prior_std(config):
    return np.repeat([config.initial_position_std, config.initial_velocity_std,
        config.initial_attitude_std, config.initial_accel_bias_std, config.initial_gyro_bias_std], 3)
