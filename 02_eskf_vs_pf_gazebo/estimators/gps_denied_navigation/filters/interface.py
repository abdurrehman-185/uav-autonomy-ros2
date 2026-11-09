"""Truth-isolating estimator API.

Estimators see exactly two kinds of objects after construction:

* ``EstimatorInput`` - built by the runner from the immutable ``SensorDataset``
  only (an IMU packet for the interval ending at ``time`` and the GPS packet at
  ``time``, which may be unavailable). It has no field that can carry truth.
* their own initial ``NavigationState`` prior and sensor/filter configuration.

``EstimatorOutput`` is what the estimator reports back. Ground truth is joined
with outputs only in evaluation code (``metrics``, ``analysis``, GUI).

The import-boundary test (tests/test_truth_isolation.py) forbids the filters
package from importing simulation, experiments, metrics, analysis, GUI or app
modules, so truth types cannot even be referenced here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from ..sensors.sensor_models import GPSMeasurement, IMUMeasurement
from .base_filter import NavigationState
from .quaternion import to_rpy


@dataclass(frozen=True)
class EstimatorInput:
    """One estimator epoch: optional IMU interval ending now, and the GPS packet now."""
    index: int
    time: float
    imu: Optional[IMUMeasurement]
    gps: GPSMeasurement

    def __post_init__(self):
        if self.imu is not None and not isinstance(self.imu, IMUMeasurement):
            raise TypeError("EstimatorInput.imu must be an IMUMeasurement or None")
        if not isinstance(self.gps, GPSMeasurement):
            raise TypeError("EstimatorInput.gps must be a GPSMeasurement")


@dataclass(frozen=True)
class EstimatorOutput:
    """Estimate after processing one EstimatorInput."""
    time: float
    position: np.ndarray
    velocity: np.ndarray
    attitude_rpy: np.ndarray
    accel_bias: np.ndarray
    gyro_bias: np.ndarray
    gps_used: bool
    position_covariance: Optional[np.ndarray] = None
    diagnostics: dict = field(default_factory=dict)


@runtime_checkable
class Estimator(Protocol):
    def predict(self, measurement: IMUMeasurement) -> None: ...
    def update(self, measurement: GPSMeasurement) -> None: ...
    def estimate(self) -> NavigationState: ...


def step(estimator: Estimator, packet: EstimatorInput) -> EstimatorOutput:
    """Process one epoch with any estimator (DR, EKF, PF, or a plugin).

    This is the public entry point for external drivers such as real-log
    replay. The benchmark runner calls predict/update separately only so that
    it can time them individually and observe the prior between them.
    """
    if packet.imu is not None:
        estimator.predict(packet.imu)
    estimator.update(packet.gps)
    state = estimator.estimate()
    covariance = None
    if hasattr(estimator, "position_covariance"):
        covariance = np.asarray(estimator.position_covariance(), dtype=float)
    return EstimatorOutput(packet.time, state.position, state.velocity, to_rpy(state.quaternion),
                           state.accel_bias, state.gyro_bias,
                           bool(packet.gps.available and getattr(estimator, "uses_gps", True)),
                           covariance)
