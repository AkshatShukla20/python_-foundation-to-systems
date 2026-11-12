"""Plotly figure builders for Robo PID Studio.

* build_arm_2d      : animated top-down arm with a growing end-effector trail
* build_arm_3d      : rotatable 3D arm (drag to rotate, scroll to zoom)
* build_pid_figure  : step-response curves that "draw themselves"

All builders take arrays that were already down-sampled to a manageable
number of animation frames (see ``sample_indices``).
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

N_FRAMES = 100

ARM_BLUE = "#4C78FF"
ARM_ORANGE = "#FF8A3D"
TRAIL = "#00CC96"
TARGET = "#FF4B4B"
J1_COLOR = "#636EFA"
J2_COLOR = "#EF553B"


def sample_indices(n: int, n_frames: int = N_FRAMES) -> np.ndarray:
    """Evenly spaced indices into a long simulation, always including the last one."""
    return np.linspace(0, n - 1, min(n_frames, n)).astype(int)


# ----------------------------------------------------------------------
# Animation controls (Play / Pause buttons + time slider)
# ----------------------------------------------------------------------
def animation_controls(times, frame_ms: int, redraw: bool):
    """Return (updatemenus, sliders) for a figure whose frames are named "0".."n-1"."""
    anim = dict(frame=dict(duration=frame_ms, redraw=redraw),
                transition=dict(duration=0), fromcurrent=False, mode="immediate")
    stop = dict(frame=dict(duration=0, redraw=False),
                transition=dict(duration=0), mode="immediate")
    menus = [dict(
        type="buttons", direction="left", showactive=False,
        x=0.0, y=-0.08, xanchor="left", yanchor="top", pad=dict(t=0, r=8),
        buttons=[
            dict(label="▶ Play / Replay", method="animate", args=[None, anim]),
            dict(label="⏸ Pause", method="animate", args=[[None], stop]),
        ],
    )]
    steps = [
        dict(method="animate", label=(f"{t:.1f}s" if k % 10 == 0 else ""),
             args=[[str(k)], dict(mode="immediate", transition=dict(duration=0),
                                  frame=dict(duration=0, redraw=redraw))])
        for k, t in enumerate(times)
    ]
    sliders = [dict(active=len(times) - 1, steps=steps, x=0.30, len=0.70,
                    y=-0.08, yanchor="top", pad=dict(t=0, b=0),
                    currentvalue=dict(visible=False), ticklen=0)]
    return menus, sliders


# ----------------------------------------------------------------------
# 2D animated arm
# ----------------------------------------------------------------------
def build_arm_2d(ex1, ey1, ex2, ey2, times, l1, l2, target, template, frame_ms):
    reach = l1 + l2
    n = len(ex2)
    circle = np.linspace(0, 2 * np.pi, 200)
    bound = reach * 1.12

    traces = [
        go.Scatter(x=reach * np.cos(circle), y=reach * np.sin(circle), mode="lines",
                   line=dict(dash="dot", color="gray"), name="Max reach"),
        go.Scatter(x=[target[0]], y=[target[1]], mode="markers",
                   marker=dict(size=15, symbol="x", color=TARGET), name="Target"),
        go.Scatter(x=ex2, y=ey2, mode="lines", line=dict(color=TRAIL, width=2),
                   name="End-effector path"),
        go.Scatter(x=[0, ex1[-1], ex2[-1]], y=[0, ey1[-1], ey2[-1]], mode="lines+markers",
                   line=dict(color=ARM_BLUE, width=8),
                   marker=dict(size=12, color=ARM_ORANGE), name="Arm"),
    ]
    frames = [
        go.Frame(name=str(k), traces=[2, 3], data=[
            go.Scatter(x=ex2[: k + 1], y=ey2[: k + 1]),
            go.Scatter(x=[0, ex1[k], ex2[k]], y=[0, ey1[k], ey2[k]]),
        ])
        for k in range(n)
    ]
    fig = go.Figure(data=traces, frames=frames)
    menus, sliders = animation_controls(times, frame_ms, redraw=False)
    fig.update_layout(
        template=template, height=620, updatemenus=menus, sliders=sliders,
        margin=dict(l=20, r=20, t=30, b=90),
        xaxis=dict(range=[-bound, bound], title="X (m)", scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-bound, bound], title="Y (m)"),
        legend=dict(orientation="h", y=1.04),
    )
    return fig


# ----------------------------------------------------------------------
# 3D arm
# ----------------------------------------------------------------------
def _cylinder(p0, p1, radius, n=16):
    """Vertices (x, y, z) of an open cylinder between two 3D points."""
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


def _cylinder_faces(n=16):
    i, j, k = [], [], []
    for a in range(n):
        b = (a + 1) % n
        i += [a, b]
        j += [b, n + b]
        k += [n + a, n + a]
    return i, j, k


def _mesh(p0, p1, radius, color, faces):
    x, y, z = _cylinder(p0, p1, radius)
    return go.Mesh3d(x=x, y=y, z=z, i=faces[0], j=faces[1], k=faces[2], color=color,
                     flatshading=True, opacity=1.0, showlegend=False,
                     lighting=dict(ambient=0.55, diffuse=0.85, specular=0.35, roughness=0.5))


def build_arm_3d(ex1, ey1, ex2, ey2, times, l1, l2, target, template, frame_ms):
    reach = l1 + l2
    n = len(ex2)
    r = 0.035 * reach              # link radius
    H = 0.22 * reach               # pedestal height (arm plane)
    Z2 = H + 2.3 * r               # second link sits slightly above the first
    bound = reach * 1.15
    faces = _cylinder_faces()
    circle = np.linspace(0, 2 * np.pi, 200)

    static = [
        # ground plane
        go.Mesh3d(x=[-bound, bound, bound, -bound], y=[-bound, -bound, bound, bound], z=[0, 0, 0, 0],
                  i=[0, 0], j=[1, 2], k=[2, 3], color="gray", opacity=0.18, showlegend=False,
                  hoverinfo="skip"),
        # reach ring on the ground
        go.Scatter3d(x=reach * np.cos(circle), y=reach * np.sin(circle), z=np.zeros_like(circle),
                     mode="lines", line=dict(color="gray", width=3, dash="dot"),
                     name="Max reach", hoverinfo="skip"),
        # base pedestal
        _mesh((0, 0, 0), (0, 0, H + r), 1.7 * r, "#7A7F8C", faces),
        # target marker + vertical guide line
        go.Scatter3d(x=[target[0], target[0]], y=[target[1], target[1]], z=[0, Z2 + r],
                     mode="lines+markers", line=dict(color=TARGET, width=3, dash="dash"),
                     marker=dict(size=[0.1, 7], color=TARGET, symbol="x"), name="Target"),
    ]

    def pose(k):
        p_base = (0, 0, H)
        p_el1 = (ex1[k], ey1[k], H)
        p_el2 = (ex1[k], ey1[k], Z2)
        p_end = (ex2[k], ey2[k], Z2)
        link1 = _mesh(p_base, p_el1, r, ARM_BLUE, faces)
        link2 = _mesh(p_el2, p_end, r, ARM_ORANGE, faces)
        post = _mesh((ex1[k], ey1[k], H - r), (ex1[k], ey1[k], Z2 + r), 1.3 * r, "#B0B6C3", faces)
        joints = go.Scatter3d(x=[0, ex1[k], ex2[k]], y=[0, ey1[k], ey2[k]],
                              z=[H + r, Z2 + r, Z2 + r], mode="markers",
                              marker=dict(size=5, color="white"), name="Joints", showlegend=False)
        trail = go.Scatter3d(x=ex2[: k + 1], y=ey2[: k + 1], z=np.full(k + 1, Z2 + r),
                             mode="lines", line=dict(color=TRAIL, width=6), name="End-effector path")
        return [link1, link2, post, joints, trail]

    dynamic = pose(n - 1)
    dyn_idx = list(range(len(static), len(static) + len(dynamic)))

    frames = []
    for k in range(n):
        p = pose(k)
        data = [
            go.Mesh3d(x=p[0].x, y=p[0].y, z=p[0].z),
            go.Mesh3d(x=p[1].x, y=p[1].y, z=p[1].z),
            go.Mesh3d(x=p[2].x, y=p[2].y, z=p[2].z),
            go.Scatter3d(x=p[3].x, y=p[3].y, z=p[3].z),
            go.Scatter3d(x=p[4].x, y=p[4].y, z=p[4].z),
        ]
        frames.append(go.Frame(name=str(k), traces=dyn_idx, data=data))

    fig = go.Figure(data=static + dynamic, frames=frames)
    menus, sliders = animation_controls(times, frame_ms, redraw=True)
    fig.update_layout(
        template=template, height=640, updatemenus=menus, sliders=sliders,
        margin=dict(l=0, r=0, t=10, b=90), showlegend=False, uirevision="arm3d",
        scene=dict(
            xaxis=dict(range=[-bound, bound], title="X (m)"),
            yaxis=dict(range=[-bound, bound], title="Y (m)"),
            zaxis=dict(range=[0, Z2 + 4 * r], title="Z (m)"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=-1.7, z=1.1)),
        ),
    )
    return fig


# ----------------------------------------------------------------------
# PID response that draws itself
# ----------------------------------------------------------------------
def build_pid_figure(times, a1, a2, target1_deg, target2_deg, template, frame_ms):
    n = len(times)
    traces = [
        go.Scatter(x=times, y=np.full(n, target1_deg), mode="lines", name="θ1 Target",
                   line=dict(color=J1_COLOR, dash="dash")),
        go.Scatter(x=times, y=np.full(n, target2_deg), mode="lines", name="θ2 Target",
                   line=dict(color=J2_COLOR, dash="dash")),
        go.Scatter(x=times, y=a1, mode="lines", name="θ1 Actual", line=dict(color=J1_COLOR, width=3)),
        go.Scatter(x=times, y=a2, mode="lines", name="θ2 Actual", line=dict(color=J2_COLOR, width=3)),
        go.Scatter(x=[times[-1]] * 2, y=[a1[-1], a2[-1]], mode="markers", showlegend=False,
                   marker=dict(size=11, color=[J1_COLOR, J2_COLOR], line=dict(width=2, color="white"))),
    ]
    frames = [
        go.Frame(name=str(k), traces=[2, 3, 4], data=[
            go.Scatter(x=times[: k + 1], y=a1[: k + 1]),
            go.Scatter(x=times[: k + 1], y=a2[: k + 1]),
            go.Scatter(x=[times[k]] * 2, y=[a1[k], a2[k]]),
        ])
        for k in range(n)
    ]
    fig = go.Figure(data=traces, frames=frames)
    menus, sliders = animation_controls(times, frame_ms, redraw=False)
    lo = min(np.min(a1), np.min(a2), target1_deg, target2_deg)
    hi = max(np.max(a1), np.max(a2), target1_deg, target2_deg)
    pad = 0.08 * max(hi - lo, 1.0)
    fig.update_layout(
        template=template, height=520, updatemenus=menus, sliders=sliders,
        margin=dict(l=20, r=20, t=30, b=90),
        xaxis=dict(title="Time (s)", range=[0, times[-1]]),
        yaxis=dict(title="Joint angle (°)", range=[lo - pad, hi + pad]),
        legend=dict(orientation="h", y=1.06),
    )
    return fig