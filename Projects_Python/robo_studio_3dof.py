"""Robo PID Studio - 3-DOF edition (single file, no other project files needed).

A 3-joint articulated arm (base yaw + shoulder + elbow) with:
  * 3D inverse / forward kinematics
  * a discrete PID controller (derivative-on-measurement + anti-windup) per joint
  * a rotatable, animated 3D view with solid links
  * animated PID step responses, torque plots and CSV export

Run with:   streamlit run robo_studio_3dof.py
Needs:      pip install streamlit plotly numpy pandas
"""
from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DT = 0.002
N_FRAMES = 100

ARM_BLUE = "#4C78FF"
ARM_ORANGE = "#FF8A3D"
TRAIL = "#00CC96"
TARGET = "#FF4B4B"
JOINT_COLORS = ["#636EFA", "#EF553B", "#00CC96"]


# ======================================================================
# 1. Kinematics (3-DOF: base yaw, shoulder pitch, elbow pitch)
#    theta0: base yaw from +X      theta1: shoulder elevation from horizontal
#    theta2: elbow angle relative to link 1
#            (theta2 > 0 -> elbow BELOW the shoulder-target line,
#             theta2 < 0 -> elbow ABOVE it)
# ======================================================================
def forward_kinematics_3d(theta0, theta1, theta2, l1: float, l2: float):
    """Elbow and end-effector positions. Accepts scalars or NumPy arrays."""
    r1 = l1 * np.cos(theta1)
    z1 = l1 * np.sin(theta1)
    r2 = r1 + l2 * np.cos(theta1 + theta2)
    z2 = z1 + l2 * np.sin(theta1 + theta2)
    elbow = (r1 * np.cos(theta0), r1 * np.sin(theta0), z1)
    end = (r2 * np.cos(theta0), r2 * np.sin(theta0), z2)
    return elbow, end


def workspace_limits(l1: float, l2: float) -> Tuple[float, float]:
    return abs(l1 - l2), l1 + l2


def is_reachable_3d(x, y, z, l1, l2) -> bool:
    d = math.sqrt(x * x + y * y + z * z)
    r_min, r_max = workspace_limits(l1, l2)
    return r_min - 1e-9 <= d <= r_max + 1e-9


def inverse_kinematics_3d(x, y, z, l1, l2, theta2_positive=True):
    """Return (theta0, theta1, theta2) in radians, or None if unreachable."""
    if not is_reachable_3d(x, y, z, l1, l2):
        return None
    theta0 = math.atan2(y, x)
    r = math.hypot(x, y)
    cos_t2 = (r * r + z * z - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    cos_t2 = max(-1.0, min(1.0, cos_t2))
    sin_t2 = math.sqrt(1.0 - cos_t2 * cos_t2)
    if not theta2_positive:
        sin_t2 = -sin_t2
    theta2 = math.atan2(sin_t2, cos_t2)
    theta1 = math.atan2(z, r) - math.atan2(l2 * sin_t2, l1 + l2 * cos_t2)
    return theta0, theta1, theta2


# ======================================================================
# 2. PID controller + joint physics   (J * theta'' = torque - b * theta')
# ======================================================================
@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    dt: float
    output_limit: float = float("inf")
    _integral: float = field(default=0.0, init=False, repr=False)
    _prev: Optional[float] = field(default=None, init=False, repr=False)

    def update(self, setpoint: float, measurement: float) -> float:
        error = setpoint - measurement
        d_meas = 0.0 if self._prev is None else (measurement - self._prev) / self.dt
        self._prev = measurement                      # derivative on measurement
        integral = self._integral + error * self.dt
        unsat = self.kp * error + self.ki * integral - self.kd * d_meas
        out = max(-self.output_limit, min(self.output_limit, unsat))
        if not (out != unsat and error * unsat > 0):  # anti-windup
            self._integral = integral
        return out


@dataclass
class SimResult:
    t: np.ndarray
    theta: np.ndarray
    torque: np.ndarray
    target: float
    theta0: float


def simulate_joint(target, kp, ki, kd, inertia, damping, torque_limit,
                   duration, dt=DT, disturbance=0.0) -> SimResult:
    n = int(round(duration / dt))
    t = np.arange(n) * dt
    theta, torque = np.zeros(n), np.zeros(n)
    pid = PID(kp, ki, kd, dt, output_limit=torque_limit)
    th = om = 0.0
    for i in range(n):
        tau = pid.update(target, th)
        om += ((tau + disturbance - damping * om) / inertia) * dt
        th += om * dt
        theta[i], torque[i] = th, tau
    return SimResult(t, theta, torque, target, 0.0)


def step_metrics(res: SimResult, band: float = 0.02) -> Dict[str, Optional[float]]:
    step = res.target - res.theta0
    out = {"overshoot_pct": None, "rise_time": None, "settling_time": None,
           "steady_error": float(abs(res.target - res.theta[-1]))}
    if abs(step) < 1e-9:
        return out
    y = (res.theta - res.theta0) / step
    out["overshoot_pct"] = float(max(0.0, y.max() - 1.0) * 100.0)
    a10, a90 = np.where(y >= 0.1)[0], np.where(y >= 0.9)[0]
    if len(a10) and len(a90):
        out["rise_time"] = float(res.t[a90[0]] - res.t[a10[0]])
    outside = np.where(np.abs(y - 1.0) > band)[0]
    if len(outside) == 0:
        out["settling_time"] = 0.0
    elif outside[-1] < len(res.t) - 1:
        out["settling_time"] = float(res.t[outside[-1] + 1])
    return out


# ======================================================================
# 3. Figures
# ======================================================================
def sample_indices(n: int, n_frames: int = N_FRAMES) -> np.ndarray:
    return np.linspace(0, n - 1, min(n_frames, n)).astype(int)


def animation_controls(times, frame_ms: int, redraw: bool):
    """Play/Pause buttons and a time slider for a figure with frames "0".."n-1"."""
    anim = dict(frame=dict(duration=frame_ms, redraw=redraw), transition=dict(duration=0),
                fromcurrent=False, mode="immediate")
    stop = dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0), mode="immediate")
    menus = [dict(type="buttons", direction="left", showactive=False, x=0.0, y=-0.08,
                  xanchor="left", yanchor="top", pad=dict(t=0, r=8),
                  buttons=[dict(label="▶ Play / Replay", method="animate", args=[None, anim]),
                           dict(label="⏸ Pause", method="animate", args=[[None], stop])])]
    steps = [dict(method="animate", label=(f"{t:.1f}s" if k % 10 == 0 else ""),
                  args=[[str(k)], dict(mode="immediate", transition=dict(duration=0),
                                       frame=dict(duration=0, redraw=redraw))])
             for k, t in enumerate(times)]
    sliders = [dict(active=len(times) - 1, steps=steps, x=0.30, len=0.70, y=-0.08,
                    yanchor="top", pad=dict(t=0, b=0), currentvalue=dict(visible=False), ticklen=0)]
    return menus, sliders


def _cylinder(p0, p1, radius, n=16):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    d = d / max(np.linalg.norm(d), 1e-9)
    up = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(d, up)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.outer(np.cos(ang), u) + np.outer(np.sin(ang), v)
    pts = np.vstack([p0 + radius * ring, p1 + radius * ring])
    return pts[:, 0], pts[:, 1], pts[:, 2]


def _faces(n=16):
    i, j, k = [], [], []
    for a in range(n):
        b = (a + 1) % n
        i += [a, b]
        j += [b, n + b]
        k += [n + a, n + a]
    return i, j, k


def _mesh(p0, p1, radius, color):
    x, y, z = _cylinder(p0, p1, radius)
    i, j, k = _faces()
    return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, flatshading=True,
                     showlegend=False, hoverinfo="skip",
                     lighting=dict(ambient=0.55, diffuse=0.85, specular=0.35, roughness=0.5))


def build_arm_3d(elbow, end, times, reach, target, template, frame_ms):
    """elbow / end are ((x), (y), (z)) arrays already sampled to the frame count."""
    ex1, ey1, ez1 = elbow
    ex2, ey2, ez2 = end
    n = len(ex2)
    r = 0.03 * reach                  # link radius
    z0 = -0.28 * reach                # ground height
    bound = reach * 1.15
    ang = np.linspace(0, 2 * np.pi, 200)

    static = [
        go.Mesh3d(x=[-bound, bound, bound, -bound], y=[-bound, -bound, bound, bound],
                  z=[z0] * 4, i=[0, 0], j=[1, 2], k=[2, 3], color="gray", opacity=0.18,
                  showlegend=False, hoverinfo="skip"),
        go.Scatter3d(x=reach * np.cos(ang), y=reach * np.sin(ang), z=np.full_like(ang, z0),
                     mode="lines", line=dict(color="gray", width=3, dash="dot"),
                     hoverinfo="skip", showlegend=False),
        _mesh((0, 0, z0), (0, 0, z0 + 0.12 * r), 3.6 * r, "#5B606E"),       # base plate
        _mesh((0, 0, z0), (0, 0, -1.3 * r), 1.5 * r, "#7A7F8C"),            # pedestal
        _mesh((0, 0, -1.3 * r), (0, 0, 1.3 * r), 1.9 * r, "#B0B6C3"),       # turret
        go.Scatter3d(x=[target[0], target[0]], y=[target[1], target[1]], z=[z0, target[2]],
                     mode="lines+markers", line=dict(color=TARGET, width=3, dash="dash"),
                     marker=dict(size=[1, 8], color=TARGET, symbol="diamond"),
                     name="Target", showlegend=False),
    ]

    def pose(k):
        p0, p1, p2 = (0, 0, 0), (ex1[k], ey1[k], ez1[k]), (ex2[k], ey2[k], ez2[k])
        return [
            _mesh(p0, p1, r, ARM_BLUE),
            _mesh(p1, p2, 0.85 * r, ARM_ORANGE),
            go.Scatter3d(x=[0, p1[0], p2[0]], y=[0, p1[1], p2[1]], z=[0, p1[2], p2[2]],
                         mode="markers", marker=dict(size=[8, 8, 9], color=["white", "white", TRAIL]),
                         showlegend=False, hoverinfo="skip"),
            go.Scatter3d(x=ex2[: k + 1], y=ey2[: k + 1], z=ez2[: k + 1], mode="lines",
                         line=dict(color=TRAIL, width=6), showlegend=False, hoverinfo="skip"),
        ]

    dynamic = pose(n - 1)
    dyn_idx = list(range(len(static), len(static) + len(dynamic)))
    frames = []
    for k in range(n):
        p = pose(k)
        frames.append(go.Frame(name=str(k), traces=dyn_idx, data=[
            go.Mesh3d(x=p[0].x, y=p[0].y, z=p[0].z),
            go.Mesh3d(x=p[1].x, y=p[1].y, z=p[1].z),
            go.Scatter3d(x=p[2].x, y=p[2].y, z=p[2].z),
            go.Scatter3d(x=p[3].x, y=p[3].y, z=p[3].z),
        ]))

    fig = go.Figure(data=static + dynamic, frames=frames)
    menus, sliders = animation_controls(times, frame_ms, redraw=True)
    fig.update_layout(
        template=template, height=660, updatemenus=menus, sliders=sliders, showlegend=False,
        margin=dict(l=0, r=0, t=10, b=90), uirevision="arm3d",
        scene=dict(xaxis=dict(range=[-bound, bound], title="X (m)"),
                   yaxis=dict(range=[-bound, bound], title="Y (m)"),
                   zaxis=dict(range=[z0, bound], title="Z (m)"),
                   aspectmode="data", camera=dict(eye=dict(x=1.5, y=1.5, z=0.9))))
    return fig


def build_pid_figure(times, angles_deg, targets_deg, template, frame_ms):
    """Three joint angle curves that draw themselves; angles_deg is a list of 3 arrays."""
    n = len(times)
    names = ["Base yaw θ₀", "Shoulder θ₁", "Elbow θ₂"]
    traces = [go.Scatter(x=times, y=np.full(n, tg), mode="lines", showlegend=False,
                         line=dict(color=c, dash="dash"))
              for tg, c in zip(targets_deg, JOINT_COLORS)]
    traces += [go.Scatter(x=times, y=a, mode="lines", name=nm, line=dict(color=c, width=3))
               for a, nm, c in zip(angles_deg, names, JOINT_COLORS)]
    traces.append(go.Scatter(x=[times[-1]] * 3, y=[a[-1] for a in angles_deg], mode="markers",
                             showlegend=False,
                             marker=dict(size=11, color=JOINT_COLORS, line=dict(width=2, color="white"))))
    frames = [go.Frame(name=str(k), traces=[3, 4, 5, 6], data=[
        go.Scatter(x=times[: k + 1], y=a[: k + 1]) for a in angles_deg
    ] + [go.Scatter(x=[times[k]] * 3, y=[a[k] for a in angles_deg])]) for k in range(n)]
    fig = go.Figure(data=traces, frames=frames)
    menus, sliders = animation_controls(times, frame_ms, redraw=False)
    allv = np.concatenate([np.asarray(angles_deg).ravel(), np.asarray(targets_deg)])
    pad = 0.08 * max(allv.max() - allv.min(), 1.0)
    fig.update_layout(template=template, height=520, updatemenus=menus, sliders=sliders,
                      margin=dict(l=20, r=20, t=30, b=90), legend=dict(orientation="h", y=1.06),
                      xaxis=dict(title="Time (s)", range=[0, times[-1]]),
                      yaxis=dict(title="Joint angle (°)", range=[allv.min() - pad, allv.max() + pad]))
    return fig


def build_torque_figure(joints, torque_limit, template):
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Base torque", "Shoulder torque", "Elbow torque"))
    for col, (j, c) in enumerate(zip(joints, JOINT_COLORS), start=1):
        fig.add_trace(go.Scatter(x=j.t, y=j.torque, line=dict(color=c)), row=1, col=col)
        for lim in (torque_limit, -torque_limit):
            fig.add_hline(y=lim, line=dict(dash="dot", color="gray"), row=1, col=col)
        fig.update_xaxes(title_text="time (s)", row=1, col=col)
        fig.update_yaxes(title_text="N·m", row=1, col=col)
    fig.update_layout(template=template, height=330, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ======================================================================
# 4. Streamlit helpers (work on old and new Streamlit versions)
# ======================================================================
def show_plot(fig):
    if "width" in inspect.signature(st.plotly_chart).parameters:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def show_table(df):
    if "width" in inspect.signature(st.dataframe).parameters:
        st.dataframe(df, width="stretch")
    else:
        st.dataframe(df, use_container_width=True)


def fmt(value, unit, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f} {unit}"


PRESETS = {
    "Balanced": (20.0, 0.5, 2.0),
    "Underdamped (oscillates)": (40.0, 0.0, 0.5),
    "Sluggish": (3.0, 0.0, 1.5),
    "Heavy integral (Ki high)": (20.0, 40.0, 1.0),
}


def apply_preset():
    kp, ki, kd = PRESETS[st.session_state["preset"]]
    st.session_state.update(kp=kp, ki=ki, kd=kd)


# ======================================================================
# 5. App
# ======================================================================
def main():
    st.set_page_config(page_title="Robo PID Studio 3D", page_icon="🦾", layout="wide")
    st.session_state.setdefault("kp", 20.0)
    st.session_state.setdefault("ki", 0.5)
    st.session_state.setdefault("kd", 2.0)

    sb = st.sidebar
    sb.title("🦾 Robo PID Studio")

    sb.header("Arm & target")
    l1 = sb.slider("Link 1 length (m)", 0.2, 1.0, 0.5, 0.05)
    l2 = sb.slider("Link 2 length (m)", 0.2, 1.0, 0.4, 0.05)
    reach = l1 + l2
    tx = sb.slider("Target X (m)", -reach, reach, 0.45, 0.02)
    ty = sb.slider("Target Y (m)", -reach, reach, 0.35, 0.02)
    tz = sb.slider("Target Z (m)", -reach, reach, 0.30, 0.02)
    elbow_choice = sb.radio("Elbow configuration", ["Up (θ₂ < 0)", "Down (θ₂ > 0)"], horizontal=True)
    theta2_positive = elbow_choice.startswith("Down")

    sb.header("PID gains")
    sb.selectbox("Preset", list(PRESETS), key="preset", on_change=apply_preset)
    kp = sb.slider("Kp (proportional)", 0.0, 100.0, key="kp")
    ki = sb.slider("Ki (integral)", 0.0, 50.0, key="ki")
    kd = sb.slider("Kd (derivative)", 0.0, 20.0, key="kd")

    sb.header("Physics")
    inertia = sb.slider("Joint inertia (kg·m²)", 0.01, 0.5, 0.05, 0.01)
    damping = sb.slider("Joint friction (N·m·s/rad)", 0.0, 2.0, 0.1, 0.05)
    torque_limit = sb.slider("Motor torque limit (N·m)", 0.5, 20.0, 5.0, 0.5)
    disturbance = sb.slider("Constant disturbance torque (N·m)", -3.0, 3.0, 0.0, 0.1)
    duration = sb.slider("Simulation time (s)", 1.0, 8.0, 3.0, 0.5)

    sb.header("Display")
    theme = sb.radio("Theme", ["Dark", "Light"], horizontal=True)
    speed = sb.select_slider("Animation speed", options=[0.25, 0.5, 1.0, 2.0], value=1.0,
                             format_func=lambda s: f"{s}x")
    template = "plotly_dark" if theme == "Dark" else "plotly_white"

    st.title("Robo PID Studio")
    st.caption("3-DOF articulated arm · 3D inverse kinematics · PID joint control · drag the 3D view to rotate")

    ik = inverse_kinematics_3d(tx, ty, tz, l1, l2, theta2_positive)
    if ik is None:
        r_min, r_max = workspace_limits(l1, l2)
        st.error(f"Target ({tx:.2f}, {ty:.2f}, {tz:.2f}) is outside the workspace. The arm reaches "
                 f"{r_min:.2f} m to {r_max:.2f} m from the shoulder - move the target closer.")
        st.stop()
    targets = list(ik)

    kw = dict(kp=kp, ki=ki, kd=kd, inertia=inertia, damping=damping,
              torque_limit=torque_limit, duration=duration, disturbance=disturbance)
    joints = [simulate_joint(t, **kw) for t in targets]
    elbow, end = forward_kinematics_3d(joints[0].theta, joints[1].theta, joints[2].theta, l1, l2)

    idx = sample_indices(len(joints[0].t))
    times = joints[0].t[idx]
    frame_ms = max(15, int(duration * 1000 / len(idx) / speed))

    final_err = float(math.dist((end[0][-1], end[1][-1], end[2][-1]), (tx, ty, tz)))
    metrics = [step_metrics(j) for j in joints]
    cols = st.columns(5)
    cols[0].metric("Target (θ₀, θ₁, θ₂)", ", ".join(f"{np.degrees(t):.0f}°" for t in targets))
    cols[1].metric("3D position error", f"{final_err * 1000:.1f} mm")
    cols[2].metric("Base settling", fmt(metrics[0]["settling_time"], "s"))
    cols[3].metric("Shoulder settling", fmt(metrics[1]["settling_time"], "s"))
    cols[4].metric("Elbow settling", fmt(metrics[2]["settling_time"], "s"))

    tab3d, tabpid, tabdata = st.tabs(["🧊 3D Arm", "📈 PID Response", "🗂 Data"])

    with tab3d:
        fig = build_arm_3d(tuple(a[idx] for a in elbow), tuple(a[idx] for a in end), times,
                           reach, (tx, ty, tz), template, frame_ms)
        show_plot(fig)
        st.caption("Drag to rotate · scroll to zoom · right-drag to pan · press ▶ to animate "
                   "or drag the time slider to scrub through the motion.")

    with tabpid:
        angles = [np.degrees(j.theta)[idx] for j in joints]
        show_plot(build_pid_figure(times, angles, [float(np.degrees(t)) for t in targets],
                                   template, frame_ms))
        show_plot(build_torque_figure(joints, torque_limit, template))
        st.caption("Try the Underdamped preset, or add a disturbance torque and raise Ki to see "
                   "the integral term remove the steady-state error.")

    with tabdata:
        df = pd.DataFrame({
            "t": joints[0].t,
            "theta0_deg": np.degrees(joints[0].theta), "theta1_deg": np.degrees(joints[1].theta),
            "theta2_deg": np.degrees(joints[2].theta),
            "end_x": end[0], "end_y": end[1], "end_z": end[2],
        })
        show_table(df.iloc[::25].reset_index(drop=True))
        st.download_button("Download full simulation as CSV", df.to_csv(index=False),
                           file_name="robo_pid_3dof_simulation.csv", mime="text/csv")


if __name__ == "__main__":
    main()
