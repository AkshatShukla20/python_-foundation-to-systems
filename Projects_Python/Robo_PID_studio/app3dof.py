"""Robo PID Studio 3D: tune PID gains for a 3-DOF robot arm and animate it in 3D."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from kinematics import (
    forward_kinematics_3d,
    inverse_kinematics_3d,
    is_reachable_3d,
    workspace_limits,
)
from pid_controller import simulate_joint, step_metrics

DT = 0.002

st.set_page_config(page_title="Robo PID Studio 3D", page_icon="🦾", layout="wide")

PRESETS = {
    "Balanced": (20.0, 0.5, 2.0),
    "Underdamped (oscillates)": (40.0, 0.0, 0.5),
    "Sluggish": (3.0, 0.0, 1.5),
    "Heavy integral (Ki high)": (20.0, 40.0, 1.0),
}

st.session_state.setdefault("kp", 20.0)
st.session_state.setdefault("ki", 0.5)
st.session_state.setdefault("kd", 2.0)


def apply_preset():
    kp, ki, kd = PRESETS[st.session_state["preset"]]
    st.session_state.update(kp=kp, ki=ki, kd=kd)


# --- Sidebar Controls ---
sb = st.sidebar
sb.title("🦾 Robo PID Studio 3D")

sb.header("3-DOF Arm & Target")
l1 = sb.slider("Link 1 length (m)", 0.2, 1.0, 0.5, 0.05)
l2 = sb.slider("Link 2 length (m)", 0.2, 1.0, 0.4, 0.05)
reach = l1 + l2
tx = sb.slider("Target X (m)", -reach, reach, 0.45, 0.02)
ty = sb.slider("Target Y (m)", -reach, reach, 0.35, 0.02)
tz = sb.slider("Target Z (m)", -reach, reach, 0.30, 0.02)
elbow_choice = sb.radio("Elbow configuration", ["Up (θ₂ < 0)", "Down (θ₂ > 0)"])
theta2_positive = elbow_choice.startswith("Down")   # IK flag: True -> theta2 > 0

sb.header("PID Gains")
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

# --- Kinematics + 3-Joint Simulation ---
st.title("Robo PID Studio — 3D Articulated Arm")
st.caption("3-DOF robotic arm · 3D inverse kinematics · discrete PID control with anti-windup")

if not is_reachable_3d(tx, ty, tz, l1, l2):
    r_min, r_max = workspace_limits(l1, l2)
    st.error(
        f"Target ({tx:.2f}, {ty:.2f}, {tz:.2f}) is outside the 3D workspace. "
        f"Reachable distance from origin: {r_min:.2f} m to {r_max:.2f} m."
    )
    st.stop()

ik_sol = inverse_kinematics_3d(tx, ty, tz, l1, l2, theta2_positive)
if ik_sol is None:
    st.stop()
theta0_t, theta1_t, theta2_t = ik_sol

sim_kwargs = dict(
    kp=kp, ki=ki, kd=kd, inertia=inertia, damping=damping,
    torque_limit=torque_limit, duration=duration, dt=DT, disturbance=disturbance,
)
joint0 = simulate_joint(theta0_t, **sim_kwargs)
joint1 = simulate_joint(theta1_t, **sim_kwargs)
joint2 = simulate_joint(theta2_t, **sim_kwargs)

(elbow_x, elbow_y, elbow_z), (end_x, end_y, end_z) = forward_kinematics_3d(
    joint0.theta, joint1.theta, joint2.theta, l1, l2
)

final_err = float(np.sqrt((end_x[-1] - tx) ** 2 + (end_y[-1] - ty) ** 2 + (end_z[-1] - tz) ** 2))


def fmt(value, unit, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f} {unit}"


m0, m1, m2 = step_metrics(joint0), step_metrics(joint1), step_metrics(joint2)
cols = st.columns(5)
cols[0].metric(
    "Target (θ₀, θ₁, θ₂)",
    f"{np.degrees(theta0_t):.0f}°, {np.degrees(theta1_t):.0f}°, {np.degrees(theta2_t):.0f}°",
)
cols[1].metric("3D Position Error", f"{final_err * 1000:.1f} mm")
cols[2].metric("Base Yaw Settling", fmt(m0["settling_time"], "s"))
cols[3].metric("Shoulder Settling", fmt(m1["settling_time"], "s"))
cols[4].metric("Elbow Settling", fmt(m2["settling_time"], "s"))

tab_arm, tab_resp, tab_data = st.tabs(["🦾 3D Arm Animation", "📈 Joint Responses", "🗂 Data"])

# --- Tab 1: Interactive 3D Animation ---
with tab_arm:
    # Subsample to ~75 smooth animation frames
    step_stride = max(1, len(joint0.t) // 75)
    frame_indices = list(range(0, len(joint0.t), step_stride))
    if frame_indices[-1] != len(joint0.t) - 1:
        frame_indices.append(len(joint0.t) - 1)

    # Base pedestal + ground ring for 3D depth reference
    ring_ang = np.linspace(0, 2 * np.pi, 80)
    r_min, r_max = workspace_limits(l1, l2)

    frames = []
    slider_steps = []
    for n_frame, idx in enumerate(frame_indices):
        t_sec = joint0.t[idx]
        frames.append(
            go.Frame(
                name=f"{t_sec:.2f}",
                data=[
                    # Trace 0: End-effector 3D trajectory trail
                    go.Scatter3d(
                        x=end_x[: idx + 1],
                        y=end_y[: idx + 1],
                        z=end_z[: idx + 1],
                        mode="lines",
                        line=dict(color="#ff7f0e", width=4),
                        name="Trajectory",
                    ),
                    # Trace 1: Robot arm links (Base -> Shoulder -> Elbow -> Gripper)
                    go.Scatter3d(
                        x=[0.0, 0.0, elbow_x[idx], end_x[idx]],
                        y=[0.0, 0.0, elbow_y[idx], end_y[idx]],
                        z=[-0.15, 0.0, elbow_z[idx], end_z[idx]],
                        mode="lines+markers",
                        line=dict(color="#1f77b4", width=10),
                        marker=dict(size=[8, 7, 7, 6], color=["#444", "#1f77b4", "#1f77b4", "#00cc96"]),
                        name="Robotic Arm",
                    ),
                ],
            )
        )
        slider_steps.append(
            dict(
                args=[[f"{t_sec:.2f}"], dict(frame=dict(duration=30, redraw=True), mode="immediate")],
                label=(f"{t_sec:.1f}s" if n_frame % 8 == 0 else ""),
                method="animate",
            )
        )

    last_idx = frame_indices[-1]
    fig3d = go.Figure(
        data=[
            # Trace 0 (animated): Trajectory
            go.Scatter3d(
                x=end_x[: last_idx + 1],
                y=end_y[: last_idx + 1],
                z=end_z[: last_idx + 1],
                mode="lines",
                line=dict(color="#ff7f0e", width=4),
                name="Trajectory",
            ),
            # Trace 1 (animated): Arm links
            go.Scatter3d(
                x=[0.0, 0.0, elbow_x[last_idx], end_x[last_idx]],
                y=[0.0, 0.0, elbow_y[last_idx], end_y[last_idx]],
                z=[-0.15, 0.0, elbow_z[last_idx], end_z[last_idx]],
                mode="lines+markers",
                line=dict(color="#1f77b4", width=10),
                marker=dict(size=[8, 7, 7, 6], color=["#444", "#1f77b4", "#1f77b4", "#00cc96"]),
                name="Robotic Arm",
            ),
            # Trace 2 (static): Target marker
            go.Scatter3d(
                x=[tx], y=[ty], z=[tz],
                mode="markers",
                marker=dict(size=8, symbol="diamond", color="red"),
                name="Target (X, Y, Z)",
            ),
            # Trace 3 (static): XY workspace ring at z=0
            go.Scatter3d(
                x=r_max * np.cos(ring_ang),
                y=r_max * np.sin(ring_ang),
                z=np.zeros_like(ring_ang),
                mode="lines",
                line=dict(color="gray", width=2, dash="dash"),
                name="Max XY Reach",
            ),
        ],
        frames=frames,
    )

    bound = reach * 1.1
    fig3d.update_layout(
        height=620,
        margin=dict(l=0, r=0, t=10, b=10),
        scene=dict(
            xaxis=dict(range=[-bound, bound], title="X (m)"),
            yaxis=dict(range=[-bound, bound], title="Y (m)"),
            zaxis=dict(range=[-bound, bound], title="Z (m)"),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.45, y=1.45, z=1.1)),
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.02,
                y=0.95,
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, dict(frame=dict(duration=30, redraw=True), transition=dict(duration=0),
                                   fromcurrent=False, mode="immediate")],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                    ),
                ],
            )
        ],
        sliders=[dict(active=len(slider_steps) - 1, steps=slider_steps, currentvalue=dict(prefix="Time: "))],
    )
    st.plotly_chart(fig3d, width="stretch")
    st.caption("Click ▶ Play to animate the 3-DOF trajectory, drag to orbit in 3D, or scrub the timeline slider.")

# --- Tab 2: 3-Joint Step Responses ---
with tab_resp:
    fig = make_subplots(
        rows=2, cols=3, shared_xaxes=True, vertical_spacing=0.12,
        subplot_titles=(
            "Base Yaw (θ₀)", "Shoulder Pitch (θ₁)", "Elbow Pitch (θ₂)",
            "Base Torque", "Shoulder Torque", "Elbow Torque",
        ),
    )
    for col, joint in enumerate((joint0, joint1, joint2), start=1):
        fig.add_trace(
            go.Scatter(x=joint.t, y=np.degrees(joint.theta), line=dict(color="#1f77b4")),
            row=1, col=col,
        )
        fig.add_hline(y=np.degrees(joint.target), line=dict(dash="dash", color="red"), row=1, col=col)
        fig.add_trace(
            go.Scatter(x=joint.t, y=joint.torque, line=dict(color="#2ca02c")),
            row=2, col=col,
        )
        fig.add_hline(y=torque_limit, line=dict(dash="dot", color="gray"), row=2, col=col)
        fig.add_hline(y=-torque_limit, line=dict(dash="dot", color="gray"), row=2, col=col)
        fig.update_yaxes(title_text="deg", row=1, col=col)
        fig.update_yaxes(title_text="N·m", row=2, col=col)
        fig.update_xaxes(title_text="time (s)", row=2, col=col)

    fig.update_layout(height=600, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

# --- Tab 3: Raw Data ---
with tab_data:
    df = pd.DataFrame({
        "t": joint0.t,
        "theta0_deg": np.degrees(joint0.theta),
        "theta1_deg": np.degrees(joint1.theta),
        "theta2_deg": np.degrees(joint2.theta),
        "end_x": end_x,
        "end_y": end_y,
        "end_z": end_z,
    })
    st.dataframe(df.iloc[::25].reset_index(drop=True), width="stretch")
    st.download_button(
        "Download full 3D simulation CSV",
        df.to_csv(index=False),
        file_name="robo_pid_3d_simulation.csv",
        mime="text/csv",
    )