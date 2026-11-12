"""Discrete PID controller and a simple rotating-joint physics simulator.

Joint model (one revolute joint, no gravity because the arm is planar):

    J * theta'' = torque - b * theta'

    J = inertia (kg m^2), b = viscous damping (N m s/rad)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


# ----------------------------------------------------------------------
# PID controller
# ----------------------------------------------------------------------
@dataclass
class PID:
    """Discrete PID controller.

    Two practical improvements over the textbook version:
    * derivative on measurement  -> no "derivative kick" when the target jumps
    * conditional integration     -> anti-windup when the output saturates
    """

    kp: float
    ki: float
    kd: float
    dt: float
    output_limit: float = float("inf")
    integral_limit: Optional[float] = None

    _integral: float = field(default=0.0, init=False, repr=False)
    _prev_measurement: Optional[float] = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_measurement = None

    def update(self, setpoint: float, measurement: float) -> float:
        error = setpoint - measurement

        # Derivative acts on the measurement, not on the error.
        if self._prev_measurement is None:
            d_meas = 0.0
        else:
            d_meas = (measurement - self._prev_measurement) / self.dt
        self._prev_measurement = measurement

        p_term = self.kp * error
        d_term = -self.kd * d_meas

        # Candidate integral (optionally clamped).
        integral = self._integral + error * self.dt
        if self.integral_limit is not None:
            integral = max(-self.integral_limit, min(self.integral_limit, integral))

        unsaturated = p_term + self.ki * integral + d_term
        output = max(-self.output_limit, min(self.output_limit, unsaturated))

        # Anti-windup: stop integrating while saturated and still pushing further in.
        saturated = output != unsaturated
        if not (saturated and error * unsaturated > 0):
            self._integral = integral

        return output


# ----------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------
@dataclass
class SimResult:
    t: np.ndarray
    theta: np.ndarray
    omega: np.ndarray
    torque: np.ndarray
    target: float
    theta0: float


def simulate_joint(
    target: float,
    kp: float,
    ki: float,
    kd: float,
    inertia: float = 0.05,
    damping: float = 0.1,
    torque_limit: float = 5.0,
    duration: float = 3.0,
    dt: float = 0.002,
    theta0: float = 0.0,
    omega0: float = 0.0,
    disturbance: float = 0.0,
) -> SimResult:
    """Simulate one joint driven by a PID controller toward ``target`` (rad).

    ``disturbance`` is a constant external torque (N m) acting on the joint,
    useful for showing why the integral term matters.
    """
    n = int(round(duration / dt))
    t = np.arange(n) * dt
    theta = np.zeros(n)
    omega = np.zeros(n)
    torque = np.zeros(n)

    pid = PID(kp, ki, kd, dt, output_limit=torque_limit)
    th, om = theta0, omega0

    for i in range(n):
        tau = pid.update(target, th)
        alpha = (tau + disturbance - damping * om) / inertia
        om += alpha * dt          # semi-implicit Euler: update velocity first,
        th += om * dt             # then position with the new velocity
        theta[i], omega[i], torque[i] = th, om, tau

    return SimResult(t, theta, omega, torque, target, theta0)


def step_metrics(res: SimResult, band: float = 0.02) -> Dict[str, Optional[float]]:
    """Classic step-response metrics, or None where they cannot be computed.

    overshoot_pct : peak beyond the target, as % of the step size
    rise_time     : time to go from 10 % to 90 % of the step
    settling_time : time after which the response stays within +-band of the target
    steady_error  : |target - final angle| in radians
    """
    step = res.target - res.theta0
    out: Dict[str, Optional[float]] = {
        "overshoot_pct": None,
        "rise_time": None,
        "settling_time": None,
        "steady_error": float(abs(res.target - res.theta[-1])),
    }
    if abs(step) < 1e-9:
        return out

    y = (res.theta - res.theta0) / step   # normalised: 0 -> start, 1 -> target
    t = res.t

    out["overshoot_pct"] = float(max(0.0, y.max() - 1.0) * 100.0)

    above10 = np.where(y >= 0.1)[0]
    above90 = np.where(y >= 0.9)[0]
    if len(above10) and len(above90):
        out["rise_time"] = float(t[above90[0]] - t[above10[0]])

    outside = np.where(np.abs(y - 1.0) > band)[0]
    if len(outside) == 0:
        out["settling_time"] = 0.0
    elif outside[-1] < len(t) - 1:
        out["settling_time"] = float(t[outside[-1] + 1])
    # else: still outside the band at the end -> never settled -> None

    return out


def simulate_joint_motion(
    start_angle: float,
    target_angle: float,
    kp: float,
    ki: float,
    kd: float,
    inertia: float = 0.5,
    damping: float = 0.1,
    torque_limit: float = 100.0,
    duration: float = 4.0,
    dt: float = 0.002,
) -> Dict[str, np.ndarray]:
    """Convenience wrapper: angles in radians in, plot-ready arrays (degrees) out.

    The defaults (inertia 0.5 kg m^2, high torque limit) are chosen so the
    "critically damped / underdamped / overdamped" presets behave as their
    names say.
    """
    res = simulate_joint(
        target_angle, kp, ki, kd, inertia=inertia, damping=damping,
        torque_limit=torque_limit, duration=duration, dt=dt, theta0=start_angle,
    )
    return {
        "time": res.t,
        "angle_deg": np.degrees(res.theta),
        "target_deg": np.full_like(res.t, np.degrees(target_angle)),
        "torque": res.torque,
    }


if __name__ == "__main__":
    res = simulate_joint(target=1.0, kp=20, ki=0.5, kd=2.5)
    print({k: (None if v is None else round(v, 4)) for k, v in step_metrics(res).items()})