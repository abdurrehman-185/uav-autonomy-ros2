from dataclasses import dataclass
import numpy as np


def readonly(value):
    out = np.array(value, dtype=float, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class IMUMeasurement:
    time: float  # midpoint of measurement integration interval
    dt: float
    acceleration: np.ndarray  # body specific force, m/s²
    angular_velocity: np.ndarray  # body rad/s

    def __post_init__(self):
        for name in ("acceleration", "angular_velocity"):
            value = readonly(getattr(self, name))
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"Invalid IMU {name}")
            object.__setattr__(self, name, value)
        if not np.isfinite(self.time) or not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Invalid IMU timestamp or interval")


@dataclass(frozen=True)
class GPSMeasurement:
    time: float
    available: bool
    position: np.ndarray | None = None
    velocity: np.ndarray | None = None
    reason: str = "available"

    def __post_init__(self):
        if self.available != (self.position is not None):
            raise ValueError("Unavailable GPS must not contain a position")
        if not self.available and self.velocity is not None:
            raise ValueError("Unavailable GPS must not contain a velocity")
        for name in ("position", "velocity"):
            value = getattr(self, name)
            if value is not None:
                value = readonly(value)
                if value.shape != (3,) or not np.all(np.isfinite(value)):
                    raise ValueError(f"Invalid GPS {name}")
                object.__setattr__(self, name, value)
