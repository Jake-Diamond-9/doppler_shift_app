import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import BytesIO
import hashlib
import os
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Doppler Shift Calculator for Marching Band",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Performance optimization: Disable some Streamlit features
st.set_option("deprecation.showPyplotGlobalUse", False)


# Authentication
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        # Get password from secrets or environment variable
        correct_password = st.secrets.get(
            "password", os.environ.get("DOPPLER_PASSWORD", "doppler2024")
        )

        if st.session_state["password"] == correct_password:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password.
        else:
            st.session_state["password_correct"] = False

    # First run, show inputs for username + password.
    if "password_correct" not in st.session_state:
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True


# Check authentication
if not check_password():
    st.stop()  # Do not continue if not authenticated.

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #2c3e50;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Main header
st.markdown(
    '<h1 class="main-header">Doppler Shift Calculator</h1>',
    unsafe_allow_html=True,
)

# Introduction
st.markdown(
    """
<div class="info-box">
    <strong>About this app:</strong> Calculate the doppler shift effect for a drill set. 
</div>
""",
    unsafe_allow_html=True,
)


# Convert dot dictionary to position in global coordinate system
def dot2coord(dot):
    # For steps defined on an 8 to 5 grid
    step_size = 5 / 8

    field_width = 160 / 3
    front_hash = 60 / 3
    back_hash = front_hash + 12.5

    if dot["side"] == "side1":
        if dot["io"] == "inside":
            x_coord = dot["yd_io"] + dot["steps_io"] * step_size
        elif dot["io"] == "outside":
            x_coord = dot["yd_io"] - dot["steps_io"] * step_size

    elif dot["side"] == "side2":
        if dot["io"] == "inside":
            x_coord = (100 - dot["yd_io"]) - dot["steps_io"] * step_size
        elif dot["io"] == "outside":
            x_coord = (100 - dot["yd_io"]) + dot["steps_io"] * step_size

    if dot["marker_fb"] == "front sideline":
        if dot["fb"] == "in front":
            ycoord = -dot["steps_fb"] * step_size
        elif dot["fb"] == "behind":
            ycoord = dot["steps_fb"] * step_size

    elif dot["marker_fb"] == "back sideline":
        if dot["fb"] == "in front":
            ycoord = field_width - dot["steps_fb"] * step_size
        elif dot["fb"] == "behind":
            ycoord = field_width + dot["steps_fb"] * step_size

    elif dot["marker_fb"] == "front hash":
        if dot["fb"] == "in front":
            ycoord = front_hash - dot["steps_fb"] * step_size
        elif dot["fb"] == "behind":
            ycoord = front_hash + dot["steps_fb"] * step_size

    elif dot["marker_fb"] == "back hash":
        if dot["fb"] == "in front":
            ycoord = back_hash - dot["steps_fb"] * step_size
        elif dot["fb"] == "behind":
            ycoord = back_hash + dot["steps_fb"] * step_size

    return x_coord, ycoord


# Calculate the velocity along the line between the source and the observer as well as the frequency shift
def doppler(m, n, l, tempo, counts, t_start, f_s):
    r = (
        n - m
    )  # n and m must be numpy arrays of shape (3, 1). l must be the same shape too
    set_dist = np.linalg.norm(r)

    # Check for zero distance to prevent divide by zero
    if set_dist == 0:
        raise ValueError("Start and end positions cannot be the same (distance = 0)")

    step_size = 8 / ((set_dist / counts) / (5 / 8))

    Vs = set_dist / ((60 / tempo) * counts)  # in yds per sec
    t_end = set_dist / Vs + t_start
    num_pts = 100
    t = np.linspace(t_start, t_end, num_pts)
    t = t.reshape(1, num_pts)

    counts_list = t * (tempo / 60)

    Ps = m + r * ((t - t_start) / (t_end - t_start))

    d_so = np.linalg.norm(Ps, axis=0)

    m1 = m[0, 0]
    m2 = m[1, 0]
    m3 = m[2, 0]
    n1 = n[0, 0]
    n2 = n[1, 0]
    n3 = n[2, 0]
    l1 = l[0, 0]
    l2 = l[1, 0]
    l3 = l[2, 0]

    V_so_der = (
        1.0
        / np.sqrt(
            (
                l1
                - m1
                + Vs
                * (m1 - n1)
                * (t - t_start)
                * 1.0
                / np.sqrt((m1 - n1) ** 2 + (m2 - n2) ** 2 + (m3 - n3) ** 2)
            )
            ** 2
            + (
                l2
                - m2
                + Vs
                * (m2 - n2)
                * (t - t_start)
                * 1.0
                / np.sqrt((m1 - n1) ** 2 + (m2 - n2) ** 2 + (m3 - n3) ** 2)
            )
            ** 2
            + (
                l3
                - m3
                + Vs
                * (m3 - n3)
                * (t - t_start)
                * 1.0
                / np.sqrt((m1 - n1) ** 2 + (m2 - n2) ** 2 + (m3 - n3) ** 2)
            )
            ** 2
        )
        * (
            Vs
            * (m1 - n1)
            * (
                l1
                - m1
                + Vs
                * (m1 - n1)
                * (t - t_start)
                * 1.0
                / np.sqrt((m1 - n1) ** 2 + (m2 - n2) ** 2 + (m3 - n3) ** 2)
            )
            * 1.0
            / np.sqrt((m1 - n1) ** 2 + (m2 - n2) ** 2 + (m3 - n3) ** 2)
            * 2.0
            + Vs
            * (m2 - n2)
            * (
                l2
                - m2
                + Vs
                * (m2 - n2)
                * (t - t_start)
                * 1.0
                / np.sqrt((m1 - n1) ** 2 + (m2 - n2) ** 2 + (m3 - n3) ** 2)
            )
            * 1.0
            / np.sqrt((m1 - n1) ** 2 + (m2 - n2) ** 2 + (m3 - n3) ** 2)
            * 2.0
            + Vs
            * (m3 - n3)
            * (
                l3
                - m3
                + Vs
                * (m3 - n3)
                * (t - t_start)
                * 1.0
                / np.sqrt((m1 - n1) ** 2 + (m2 - n2) ** 2 + (m3 - n3) ** 2)
            )
            * 1.0
            / np.sqrt((m1 - n1) ** 2 + (m2 - n2) ** 2 + (m3 - n3) ** 2)
            * 2.0
        )
    ) / 2.0

    c = 375.44  # sound speed in air in units of yd/s
    f_o = f_s * (c / (c + V_so_der))

    cents = (1200 / np.log(2)) * np.log(f_o / f_s)

    return V_so_der, t, counts_list, f_o, cents, step_size


# Sidebar for inputs
st.sidebar.markdown(
    '<h2 class="section-header">Input Parameters</h2>', unsafe_allow_html=True
)

# Musical parameters
st.sidebar.markdown("---")
st.sidebar.markdown("**Musical Parameters**")
tempo = st.sidebar.number_input("Tempo (BPM)", min_value=0.0, value=180.0, step=1.0)
counts = st.sidebar.number_input("Counts", min_value=0.0, value=16.0, step=1.0)
f_s = st.sidebar.number_input(
    "Source Frequency (Hz)", min_value=0.0, value=440.0, step=1.0
)
t_start = 0.0  # Hardcoded to zero

# Observer position
st.sidebar.markdown("---")
st.sidebar.markdown("**Observer Position**")
observer_mode = st.sidebar.radio(
    "Observer Mode", ["Custom", "Lucas Oil Box (approximate)"], key="observer_mode"
)

if observer_mode == "Custom":
    obs_side = st.sidebar.selectbox(
        "Observer Side", ["side1", "side2"], index=0, key="obs_side"
    )
    obs_steps_io = st.sidebar.number_input(
        "Observer Steps In/Out", 0.0, 4.0, 0.0, 0.1, key="obs_steps_io"
    )
    obs_io = st.sidebar.selectbox(
        "Observer In/Out", ["inside", "outside"], index=0, key="obs_io"
    )
    obs_yd = st.sidebar.number_input("Observer Yard Line", 0, 50, 50, 5, key="obs_yd")
    obs_steps_fb = st.sidebar.number_input(
        "Observer Steps Front/Back",
        min_value=0.0,
        value=0.0,
        step=0.1,
        key="obs_steps_fb",
    )
    obs_fb = st.sidebar.selectbox(
        "Observer Front/Back", ["in front", "behind"], index=0, key="obs_fb"
    )
    obs_marker_fb = st.sidebar.selectbox(
        "Observer Marker",
        ["front sideline", "back sideline", "front hash", "back hash"],
        index=0,
        key="obs_marker_fb",
    )
    z_coord_obs = st.sidebar.number_input(
        "Observer Height (yd)", min_value=0.0, value=2.0, step=0.5, key="obs_height"
    )
else:  # Lucas Oil Box
    # Hardcoded Lucas Oil Box observer position
    obs_side = "side1"
    obs_steps_io = 0.0
    obs_io = "inside"
    obs_yd = 50
    obs_steps_fb = 85.0  # 85 steps in front of front sideline
    obs_fb = "in front"
    obs_marker_fb = "front sideline"
    z_coord_obs = 33.0  # 33 yards high

# Start position
st.sidebar.markdown("---")
st.sidebar.markdown("**Performer Start Position**")
start_side = st.sidebar.selectbox(
    "Start Side", ["side1", "side2"], index=0, key="start_side"
)
start_steps_io = st.sidebar.number_input(
    "Start Steps In/Out", 0.0, 4.0, 0.0, 0.1, key="start_steps_io"
)
start_io = st.sidebar.selectbox(
    "Start In/Out", ["inside", "outside"], index=0, key="start_io"
)
start_yd = st.sidebar.number_input("Start Yard Line", 0, 50, 45, 5, key="start_yd")
start_steps_fb = st.sidebar.number_input(
    "Start Steps Front/Back", min_value=0.0, value=0.0, step=0.1, key="start_steps_fb"
)
start_fb = st.sidebar.selectbox(
    "Start Front/Back", ["in front", "behind"], index=0, key="start_fb"
)
start_marker_fb = st.sidebar.selectbox(
    "Start Marker",
    ["front sideline", "back sideline", "front hash", "back hash"],
    index=2,
    key="start_marker_fb",
)
z_coord1 = st.sidebar.number_input(
    "Start Height (yd)", min_value=0.0, value=2.0, step=0.5, key="start_height"
)

# End position
st.sidebar.markdown("---")
st.sidebar.markdown("**Performer End Position**")
end_side = st.sidebar.selectbox("End Side", ["side1", "side2"], index=1, key="end_side")
end_steps_io = st.sidebar.number_input(
    "End Steps In/Out", 0.0, 4.0, 0.0, 0.1, key="end_steps_io"
)
end_io = st.sidebar.selectbox(
    "End In/Out", ["inside", "outside"], index=0, key="end_io"
)
end_yd = st.sidebar.number_input("End Yard Line", 0, 50, 45, 5, key="end_yd")
end_steps_fb = st.sidebar.number_input(
    "End Steps Front/Back", min_value=0.0, value=8.0, step=0.1, key="end_steps_fb"
)
end_fb = st.sidebar.selectbox(
    "End Front/Back", ["in front", "behind"], index=0, key="end_fb"
)
end_marker_fb = st.sidebar.selectbox(
    "End Marker",
    ["front sideline", "back sideline", "front hash", "back hash"],
    index=2,
    key="end_marker_fb",
)
z_coord2 = st.sidebar.number_input(
    "End Height (yd)", min_value=0.0, value=2.0, step=0.5, key="end_height"
)

# Create dot dictionaries
dot_obs = {
    "side": obs_side,
    "steps_io": obs_steps_io,
    "io": obs_io,
    "yd_io": obs_yd,
    "steps_fb": obs_steps_fb,
    "fb": obs_fb,
    "marker_fb": obs_marker_fb,
}

dot1 = {
    "side": start_side,
    "steps_io": start_steps_io,
    "io": start_io,
    "yd_io": start_yd,
    "steps_fb": start_steps_fb,
    "fb": start_fb,
    "marker_fb": start_marker_fb,
}

dot2 = {
    "side": end_side,
    "steps_io": end_steps_io,
    "io": end_io,
    "yd_io": end_yd,
    "steps_fb": end_steps_fb,
    "fb": end_fb,
    "marker_fb": end_marker_fb,
}

# Calculate when button is pressed
if st.sidebar.button("Calculate Doppler Shift", type="primary", key="calc_button"):
    # Important constants, NCAA field with 8 to 5 standard step size
    field_length = 100
    field_width = 160 / 3
    front_hash = 60 / 3
    back_hash = field_width - 60 / 3
    yd_line_dist = 5
    step_size = yd_line_dist / 8

    # Convert dots to global coordinates
    x_coord_obs, y_coord_obs = dot2coord(dot_obs)
    x_coord1, y_coord1 = dot2coord(dot1)
    x_coord2, y_coord2 = dot2coord(dot2)

    l = np.transpose(np.array([[x_coord_obs, y_coord_obs, z_coord_obs]]))
    m = np.transpose(np.array([[x_coord1, y_coord1, z_coord1]]))
    n = np.transpose(np.array([[x_coord2, y_coord2, z_coord2]]))

    # Cache the doppler calculation to avoid recomputing
    @st.cache_data
    def calculate_doppler(m, n, l, tempo, counts, t_start, f_s):
        try:
            return doppler(m, n, l, tempo, counts, t_start, f_s)
        except ValueError as e:
            st.error(f"Calculation error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.stop()

    V_so_der, time, counts_list, f_o, cents, step_size_march = calculate_doppler(
        m, n, l, tempo, counts, t_start, f_s
    )

    # Yard line locations and markers
    yd_markers_locations = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    yd_markers = ["G", 10, 20, 30, 40, 50, 40, 30, 20, 10, "G"]

    # Make arrays for the yard lines, four step lines, and two step lines
    yd_lines = np.arange(0, field_length + yd_line_dist, yd_line_dist)
    four_step_lines_vert = np.arange(
        0, field_length + yd_line_dist / 2, yd_line_dist / 2
    )
    two_step_lines_vert = np.arange(
        0, field_length + yd_line_dist / 4, yd_line_dist / 4
    )
    four_step_lines_horz = np.arange(0, field_width, yd_line_dist / 2)
    two_step_lines_horz = np.arange(0, field_width, yd_line_dist / 4)

    # Create Plotly football field plot
    fig_field_plotly = go.Figure()

    # Add grid lines (two-step lines)
    for x in two_step_lines_vert:
        fig_field_plotly.add_shape(
            type="line",
            x0=x,
            y0=0,
            x1=x,
            y1=field_width,
            line=dict(color="lightgray", width=1),
            layer="below",
        )

    for y in two_step_lines_horz:
        fig_field_plotly.add_shape(
            type="line",
            x0=0,
            y0=y,
            x1=field_length,
            y1=y,
            line=dict(color="lightgray", width=1),
            layer="below",
        )

    # Add four-step lines
    for x in four_step_lines_vert:
        fig_field_plotly.add_shape(
            type="line",
            x0=x,
            y0=0,
            x1=x,
            y1=field_width,
            line=dict(color="gray", width=1),
            layer="below",
        )

    for y in four_step_lines_horz:
        fig_field_plotly.add_shape(
            type="line",
            x0=0,
            y0=y,
            x1=field_length,
            y1=y,
            line=dict(color="gray", width=1),
            layer="below",
        )

    # Add hash marks and sidelines
    fig_field_plotly.add_shape(
        type="line",
        x0=0,
        y0=front_hash,
        x1=field_length,
        y1=front_hash,
        line=dict(color="black", width=2),
    )

    fig_field_plotly.add_shape(
        type="line",
        x0=0,
        y0=back_hash,
        x1=field_length,
        y1=back_hash,
        line=dict(color="black", width=2),
    )

    fig_field_plotly.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=field_length,
        y1=0,
        line=dict(color="black", width=2),
    )

    fig_field_plotly.add_shape(
        type="line",
        x0=0,
        y0=field_width,
        x1=field_length,
        y1=field_width,
        line=dict(color="black", width=2),
    )

    # Add yard lines
    for x in yd_lines:
        fig_field_plotly.add_shape(
            type="line",
            x0=x,
            y0=0,
            x1=x,
            y1=field_width,
            line=dict(color="black", width=2),
        )

    # Add position markers (on top of grid) - moved to end to ensure they appear on top
    # Start marker
    fig_field_plotly.add_trace(
        go.Scatter(
            x=[x_coord1],
            y=[y_coord1],
            mode="markers",
            marker=dict(
                symbol="diamond",
                size=12,
                color="green",
                line=dict(color="black", width=2),
            ),
            name="Start",
            showlegend=True,
            hoverinfo="skip",
            zorder=10000,
        )
    )

    # End marker
    fig_field_plotly.add_trace(
        go.Scatter(
            x=[x_coord2],
            y=[y_coord2],
            mode="markers",
            marker=dict(
                symbol="circle", size=12, color="red", line=dict(color="black", width=2)
            ),
            name="End",
            showlegend=True,
            hoverinfo="skip",
            zorder=10000,
        )
    )

    # Only add observer marker if using custom position
    if observer_mode == "Custom":
        fig_field_plotly.add_trace(
            go.Scatter(
                x=[x_coord_obs],
                y=[y_coord_obs],
                mode="markers",
                marker=dict(
                    symbol="square",
                    size=12,
                    color="blue",
                    line=dict(color="black", width=2),
                ),
                name="Observer",
                showlegend=True,
                hoverinfo="skip",
                zorder=10000,
            )
        )

    # Update layout
    fig_field_plotly.update_layout(
        title=dict(text="Drill Set", font=dict(size=20, color="black")),
        xaxis=dict(
            title="",
            range=[0, field_length],
            tickvals=yd_markers_locations,
            ticktext=yd_markers,
            tickmode="array",
            tickfont=dict(size=16, color="black"),
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=1,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title="",
            range=[0, field_width],
            tickfont=dict(size=12, color="black"),
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=1,
            showticklabels=False,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=500,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(color="black"),
        ),
    )

    # Display the Plotly football field plot
    st.plotly_chart(fig_field_plotly, use_container_width=True)

    # Create Plotly doppler shift plot
    fig_plotly = go.Figure()

    fig_plotly.add_trace(
        go.Scatter(
            x=counts_list[0, :],
            y=cents[0, :],
            mode="lines",
            line=dict(color="blue", width=4),
            name="Doppler Shift",
        )
    )

    fig_plotly.update_layout(
        title=dict(text="Doppler Shift vs Count", font=dict(size=20, color="black")),
        xaxis=dict(
            title=dict(text="Count", font=dict(size=16, color="black")),
            tickfont=dict(size=14, color="black"),
            gridcolor="lightgray",
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(text="Doppler Shift (cents)", font=dict(size=16, color="black")),
            tickfont=dict(size=14, color="black"),
            gridcolor="lightgray",
            showgrid=True,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=500,
    )

    # Display the Plotly plot
    st.plotly_chart(fig_plotly, use_container_width=True)

    # Display results in a nice format
    st.markdown(
        '<h2 class="section-header">Results Summary</h2>', unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Tempo", f"{tempo} BPM")
        st.metric("Counts", counts)

    with col2:
        st.metric("Source Frequency", f"{f_s} Hz")
        st.metric("Step Size", f"{np.round(step_size_march, 1)}")

    with col3:
        st.metric("Mean Doppler Shift", f"{np.round(np.mean(cents[0, :]), 1)} cents")
        st.metric("Std Dev Doppler Shift", f"{np.round(np.std(cents[0, :]), 1)} cents")

    with col4:
        st.metric("Observed Frequency", f"{np.round(np.mean(f_o[0, :]), 1)} Hz")
        st.metric("Observed Std Dev Frequency", f"{np.round(np.std(f_o[0, :]), 1)} Hz")


# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666;'>
    <p>Doppler Shift Calculator for Drill Sets</p>
    <p>Built with Streamlit • NCAA Football Field Standards</p>
    <p style='font-size: 12px; margin-top: 20px;'>© 2025 Jacob M. Diamond. All rights reserved.</p>
</div>
""",
    unsafe_allow_html=True,
)
