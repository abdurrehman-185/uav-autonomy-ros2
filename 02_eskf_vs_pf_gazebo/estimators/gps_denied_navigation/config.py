"""Strict, immutable, SI-unit configuration for simulation and experiments."""
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import math
import yaml


def positive(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")


def vector(name, value):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{name} must contain three finite numbers")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError(f"{name} must contain three finite numbers")
    return tuple(float(x) for x in value)


@dataclass(frozen=True)
class SimulationConfig:
    duration: float = 20.0
    dt: float = 0.01
    seed: int = 42

    def __post_init__(self):
        positive("duration", self.duration)
        positive("dt", self.dt)
        if self.dt > self.duration:
            raise ValueError("dt must not exceed duration")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


@dataclass(frozen=True)
class TrajectoryConfig:
    type: str = "figure_eight"
    initial_position: tuple = (0.0, 0.0, 20.0)
    initial_velocity: tuple | None = None
    attitude_rpy: tuple = (0.0, 0.0, 0.0)
    speed: float = 5.0
    radius: float = 20.0
    vertical_amplitude: float = 5.0
    helix_climb_per_turn: float = 10.0
    waypoint_segment_duration: float = 10.0
    waypoint_offsets: tuple = ((0., 0., 0.), (20., 0., 5.), (20., 20., 0.), (0., 20., -5.))
    random_harmonics: int = 4
    attitude_mode: str = "fixed"      # fixed | thrust_aligned (quadrotor differential flatness)
    yaw_mode: str = "fixed"           # fixed (attitude_rpy yaw) | velocity (heading follows track)

    def __post_init__(self):
        if self.attitude_mode not in {"fixed", "thrust_aligned"}:
            raise ValueError(f"Unknown attitude_mode: {self.attitude_mode}")
        if self.yaw_mode not in {"fixed", "velocity"}:
            raise ValueError(f"Unknown yaw_mode: {self.yaw_mode}")
        if self.attitude_mode == "fixed" and self.yaw_mode != "fixed":
            raise ValueError("yaw_mode='velocity' requires attitude_mode='thrust_aligned'")
        if self.type not in {"straight", "circular", "figure_eight", "helical", "waypoint", "random_smooth"}:
            raise ValueError(f"Unknown trajectory type: {self.type}")
        for name in ("initial_position", "attitude_rpy"):
            object.__setattr__(self, name, vector(name, getattr(self, name)))
        if self.initial_velocity is not None:
            object.__setattr__(self, "initial_velocity", vector("initial_velocity", self.initial_velocity))
            if self.type != "straight":
                raise ValueError("initial_velocity is supported only for straight trajectories")
        for name in ("speed", "radius", "waypoint_segment_duration"):
            positive(name, getattr(self, name))
        for name in ("vertical_amplitude", "helix_climb_per_turn"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if isinstance(self.random_harmonics, bool) or not isinstance(self.random_harmonics, int) or self.random_harmonics < 1:
            raise ValueError("random_harmonics must be a positive integer")
        if not isinstance(self.waypoint_offsets, (tuple, list)) or len(self.waypoint_offsets) < 2:
            raise ValueError("At least two waypoint offsets are required")
        offsets = tuple(vector("waypoint offset", p) for p in self.waypoint_offsets)
        if offsets[0] != (0., 0., 0.):
            raise ValueError("First waypoint offset must be zero to preserve initial_position")
        object.__setattr__(self, "waypoint_offsets", offsets)


def nonnegative(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def integer(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class IMUConfig:
    frequency: float = 100.0
    gravity: float = 9.80665
    accel_noise_std: float = 0.05
    gyro_noise_std: float = 0.002
    accel_bias: tuple = (0.02, -0.01, 0.015)
    gyro_bias: tuple = (0.0002, -0.0001, 0.0003)
    accel_bias_random_walk: float = 0.001
    gyro_bias_random_walk: float = 0.00001

    def __post_init__(self):
        for name in ("frequency", "gravity"):
            positive(name, getattr(self, name))
        for name in ("accel_noise_std", "gyro_noise_std", "accel_bias_random_walk", "gyro_bias_random_walk"):
            nonnegative(name, getattr(self, name))
        for name in ("accel_bias", "gyro_bias"):
            object.__setattr__(self, name, vector(name, getattr(self, name)))


@dataclass(frozen=True)
class GPSConfig:
    frequency: float = 5.0
    position_noise_std: float = 1.5
    velocity_noise_std: float = 0.2
    measure_velocity: bool = False
    outage_start: float = 5.0
    outage_duration: float = 10.0
    outages: tuple = ()
    periodic_start: float = 0.0
    periodic_period: float = 0.0
    periodic_duration: float = 0.0

    def __post_init__(self):
        positive("gps.frequency", self.frequency)
        for name in ("position_noise_std", "velocity_noise_std", "outage_start", "outage_duration",
                     "periodic_start", "periodic_period", "periodic_duration"):
            nonnegative(name, getattr(self, name))
        if not isinstance(self.measure_velocity, bool):
            raise ValueError("measure_velocity must be boolean")
        if self.periodic_duration > self.periodic_period:
            raise ValueError("periodic_duration must not exceed periodic_period")
        intervals = []
        if not isinstance(self.outages, (list, tuple)):
            raise ValueError("outages must be a sequence of [start, end] pairs")
        for pair in self.outages:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("Each outage must be [start, end]")
            start, end = pair
            nonnegative("outage start", start)
            positive("outage end", end)
            if end <= start:
                raise ValueError("Outage end must exceed start")
            intervals.append((float(start), float(end)))
        object.__setattr__(self, "outages", tuple(intervals))


@dataclass(frozen=True)
class FilterConfig:
    initialize_from_truth: bool = True
    initial_position: tuple = (0.0, 0.0, 20.0)
    initial_velocity: tuple = (0.0, 0.0, 0.0)
    initial_attitude_rpy: tuple = (0.0, 0.0, 0.0)
    initial_accel_bias: tuple = (0.0, 0.0, 0.0)
    initial_gyro_bias: tuple = (0.0, 0.0, 0.0)
    initial_position_std: float = 1.0
    initial_velocity_std: float = 0.3
    initial_attitude_std: float = 0.01
    initial_accel_bias_std: float = 0.05
    initial_gyro_bias_std: float = 0.001
    measurement_variance_floor: float = 1e-12
    covariance_negative_tolerance: float = 1e-8
    # Filter-ASSUMED noise relative to the simulated sensor model (model mismatch
    # studies). Applied identically to EKF (Q, R) and PF (process draws, likelihood).
    process_noise_scale: float = 1.0      # Q_filter = scale * Q_sensor-model
    measurement_noise_scale: float = 1.0  # R_filter = scale * R_sensor-model

    def __post_init__(self):
        positive("process_noise_scale", self.process_noise_scale)
        positive("measurement_noise_scale", self.measurement_noise_scale)
        if not isinstance(self.initialize_from_truth, bool):
            raise ValueError("initialize_from_truth must be boolean")
        for name in ("initial_position", "initial_velocity", "initial_attitude_rpy", "initial_accel_bias", "initial_gyro_bias"):
            object.__setattr__(self, name, vector(name, getattr(self, name)))
        for name in ("initial_position_std", "initial_velocity_std", "initial_attitude_std",
                     "initial_accel_bias_std", "initial_gyro_bias_std", "covariance_negative_tolerance"):
            nonnegative(name, getattr(self, name))
        positive("measurement_variance_floor", self.measurement_variance_floor)


@dataclass(frozen=True)
class ParticleFilterConfig:
    particles: int = 1000
    resample_threshold: float = 0.5
    degeneracy_threshold: float = 0.02

    def __post_init__(self):
        integer("particles", self.particles)
        positive("resample_threshold", self.resample_threshold)
        nonnegative("degeneracy_threshold", self.degeneracy_threshold)
        if self.resample_threshold > 1 or self.degeneracy_threshold > 1:
            raise ValueError("ESS thresholds are fractions in [0, 1]")


@dataclass(frozen=True)
class ExperimentConfig:
    monte_carlo_runs: int = 100
    base_seed: int = 42
    recovery_threshold: float = 5.0
    recovery_hold_seconds: float = 1.0
    sweep_recovery_seconds: float = 30.0
    divergence_threshold: float = 500.0
    bootstrap_samples: int = 2000
    outage_durations: tuple = (0., 10., 30., 60., 120., 180., 300.)
    particle_counts: tuple = (100, 250, 500, 1000, 2000, 5000)
    noise_scales: tuple = (0.25, 1., 3., 10.)
    output_directory: str = "results/experiments"
    figure_formats: tuple = ("png", "pdf", "svg")
    figure_dpi: int = 300

    def __post_init__(self):
        for name in ("monte_carlo_runs", "bootstrap_samples", "figure_dpi"):
            integer(name, getattr(self, name))
        integer("base_seed", self.base_seed, 0)
        for name in ("recovery_threshold", "divergence_threshold", "sweep_recovery_seconds"):
            positive(name, getattr(self, name))
        nonnegative("recovery_hold_seconds", self.recovery_hold_seconds)
        for name in ("outage_durations", "noise_scales", "particle_counts"):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError(f"{name} must be a nonempty sequence")
            for value in values:
                if name == "particle_counts":
                    integer(name, value)
                else:
                    nonnegative(name, value)
            object.__setattr__(self, name, tuple(values))
        if not isinstance(self.figure_formats, (list, tuple)) or not self.figure_formats or any(v not in {"png", "pdf", "svg"} for v in self.figure_formats):
            raise ValueError("figure_formats must contain png, pdf, or svg")
        object.__setattr__(self, "figure_formats", tuple(self.figure_formats))
        if not isinstance(self.output_directory, str) or not self.output_directory:
            raise ValueError("output_directory must be a nonempty path")


@dataclass(frozen=True)
class Config:
    simulation: SimulationConfig = SimulationConfig()
    trajectory: TrajectoryConfig = TrajectoryConfig()
    imu: IMUConfig = IMUConfig()
    gps: GPSConfig = GPSConfig()
    filters: FilterConfig = FilterConfig()
    particle_filter: ParticleFilterConfig = ParticleFilterConfig()
    experiment: ExperimentConfig = ExperimentConfig()

    def __post_init__(self):
        if self.gps.frequency > self.imu.frequency:
            raise ValueError("GPS frequency must not exceed IMU frequency")
        if 1/self.imu.frequency > self.simulation.duration:
            raise ValueError("Simulation must span at least one IMU interval")

    def to_dict(self):
        # YAML-safe lists rather than Python-specific tuple tags.
        def plain(x):
            if isinstance(x, dict):
                return {k: plain(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [plain(v) for v in x]
            return x
        return plain(asdict(self))


def load_config(path: str | Path) -> Config:
    with Path(path).open(encoding="utf-8") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid YAML configuration: {error}") from error
    return config_from_dict(data)


def config_from_dict(data) -> Config:
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping")
    sections = (("simulation", SimulationConfig), ("trajectory", TrajectoryConfig),
                ("imu", IMUConfig), ("gps", GPSConfig), ("filters", FilterConfig),
                ("particle_filter", ParticleFilterConfig), ("experiment", ExperimentConfig))
    extra = set(data) - {name for name, _ in sections}
    if extra:
        raise ValueError(f"Unknown configuration sections: {sorted(extra)}")
    values = {}
    for name, cls in sections:
        section = data.get(name, {})
        if not isinstance(section, dict):
            raise ValueError(f"{name} must be a mapping")
        extra = set(section) - {f.name for f in fields(cls)}
        if extra:
            raise ValueError(f"Unknown {name} options: {sorted(extra)}")
        values[name] = cls(**section)
    return Config(**values)
