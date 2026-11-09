from .base_filter import propagate


class DeadReckoning:
    """Strapdown inertial integration only; GPS packets are ignored after initialization."""
    uses_gps = False

    def __init__(self, initial_state, gravity):
        self.state = initial_state.copy()
        self.gravity = gravity

    def predict(self, measurement):
        s = self.state
        s.position, s.velocity, s.quaternion = propagate(s.position, s.velocity, s.quaternion,
            measurement.acceleration-s.accel_bias, measurement.angular_velocity-s.gyro_bias,
            measurement.dt, self.gravity)

    def update(self, measurement):
        # Raw inertial baseline intentionally does not use GPS after initialization.
        return None

    def estimate(self):
        return self.state.copy()
