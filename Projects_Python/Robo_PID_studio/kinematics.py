"""2-link planar arm kinematics.

Angle convention
----------------
theta1 : angle of link 1 measured from the +X axis (radians)
theta2 : angle of link 2 measured *relative to link 1* (radians)

The base of the arm sits at the origin (0, 0).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

Point = Tuple[float, float]


def forward_kinematics(theta1, theta2, l1: float, l2: float):
    """Return the elbow and end-effector positions.

    Works with single numbers *and* NumPy arrays, so a whole simulated
    trajectory can be converted in one call.

    Returns
    -------
    (x1, y1), (x2, y2) : elbow position, end-effector position
    """
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    return (x1, y1), (x2, y2)


def workspace_limits(l1: float, l2: float) -> Tuple[float, float]:
    """Return (min_reach, max_reach) of the arm."""
    return abs(l1 - l2), l1 + l2


def is_reachable(x: float, y: float, l1: float, l2: float) -> bool:
    """True if the point (x, y) lies inside the arm's workspace."""
    r = math.hypot(x, y)
    r_min, r_max = workspace_limits(l1, l2)
    return r_min - 1e-9 <= r <= r_max + 1e-9


def inverse_kinematics(
    x: float, y: float, l1: float, l2: float, elbow_up: bool = True
) -> Optional[Tuple[float, float]]:
    """Return (theta1, theta2) that place the end-effector at (x, y).

    Uses the law of cosines. ``elbow_up=True`` picks the solution with
    theta2 > 0 (elbow bent counter-clockwise); ``False`` picks theta2 < 0.
    Returns ``None`` when the target is outside the workspace.
    """
    if not is_reachable(x, y, l1, l2):
        return None

    cos_t2 = (x * x + y * y - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    cos_t2 = max(-1.0, min(1.0, cos_t2))  # guard against rounding error
    sin_t2 = math.sqrt(1.0 - cos_t2 * cos_t2)
    if not elbow_up:
        sin_t2 = -sin_t2

    theta2 = math.atan2(sin_t2, cos_t2)
    theta1 = math.atan2(y, x) - math.atan2(l2 * sin_t2, l1 + l2 * cos_t2)
    return theta1, theta2


# ----------------------------------------------------------------------
# 3-DOF articulated arm (base yaw + shoulder + elbow)
# theta0: base yaw from +X; theta1: shoulder elevation from horizontal;
# theta2: elbow angle relative to link 1 (theta2 > 0 puts the elbow BELOW
# the shoulder-target line, theta2 < 0 puts it ABOVE).
# ----------------------------------------------------------------------
def forward_kinematics_3d(theta0, theta1, theta2, l1: float, l2: float):
    """Return 3D coordinates of elbow and end-effector.

    Accepts scalars or 1D NumPy arrays for full-trajectory vectorization.
    Returns ((x1, y1, z1), (x2, y2, z2)).
    """
    r1 = l1 * np.cos(theta1)
    z1 = l1 * np.sin(theta1)
    x1 = r1 * np.cos(theta0)
    y1 = r1 * np.sin(theta0)

    r2 = r1 + l2 * np.cos(theta1 + theta2)
    z2 = z1 + l2 * np.sin(theta1 + theta2)
    x2 = r2 * np.cos(theta0)
    y2 = r2 * np.sin(theta0)

    return (x1, y1, z1), (x2, y2, z2)

def is_reachable_3d(x: float, y: float, z: float, l1: float, l2: float) -> bool:
    """True if 3D point (x, y, z) lies inside the arm's spherical workspace."""
    d = math.sqrt(x * x + y * y + z * z)
    r_min, r_max = workspace_limits(l1, l2)
    return r_min - 1e-9 <= d <= r_max + 1e-9

def inverse_kinematics_3d(
    x: float, y: float, z: float, l1: float, l2: float, elbow_up: bool = True
) -> Optional[Tuple[float, float, float]]:
    """Return (theta0, theta1, theta2) in radians to place end-effector at (x, y, z)."""
    if not is_reachable_3d(x, y, z, l1, l2):
        return None

    theta0 = math.atan2(y, x)
    r = math.hypot(x, y)

    cos_t2 = (r * r + z * z - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    cos_t2 = max(-1.0, min(1.0, cos_t2))
    sin_t2 = math.sqrt(1.0 - cos_t2 * cos_t2)
    if not elbow_up:
        sin_t2 = -sin_t2

    theta2 = math.atan2(sin_t2, cos_t2)
    theta1 = math.atan2(z, r) - math.atan2(l2 * sin_t2, l1 + l2 * cos_t2)
    return theta0, theta1, theta2


if __name__ == "__main__":
    # Quick self-check: IK followed by FK must return the original target.
    l1, l2 = 0.5, 0.4
    for target in [(0.6, 0.3), (-0.3, 0.5), (0.2, -0.6)]:
        for up in (True, False):
            t1, t2 = inverse_kinematics(*target, l1, l2, elbow_up=up)
            _, (fx, fy) = forward_kinematics(t1, t2, l1, l2)
            assert abs(fx - target[0]) < 1e-9 and abs(fy - target[1]) < 1e-9
    print("kinematics self-check passed")
    # 3D check
    for target in [(0.45, 0.35, 0.30), (-0.3, 0.2, -0.4)]:
        for up in (True, False):
            angles = inverse_kinematics_3d(*target, l1, l2, elbow_up=up)
            _, (fx, fy, fz) = forward_kinematics_3d(*angles, l1, l2)
            assert max(abs(fx - target[0]), abs(fy - target[1]), abs(fz - target[2])) < 1e-9
    print("3D kinematics self-check passed")