# app.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time
import pandas as pd

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from environment import Environment, Order, Building
from ml_predictor.predictor import EnergyTimePredictor
from path_planner import PathPlanner3D
from utils.geometry import calculate_distance_3d

# --- Helper Functions ---
def log_event(message):
    """Adds a timestamped message to the sidebar log."""
    if 'log' not in st.session_state: st.session_state.log = []
    st.session_state.log.insert(0, f"{time.strftime('%H:%M:%S')} - {message}")

def create_box(building: Building):
    """Creates a 3D Plotly mesh for a building."""
    x, y = building.center_xy; dx, dy = building.size_xy[0] / 2, building.size_xy[1] / 2; h = building.height
    x_coords = [x-dx, x+dx, x+dx, x-dx, x-dx, x+dx, x+dx, x-dx]; y_coords = [y-dy, y-dy, y+dy, y+dy, y-dy, y-dy, y+dy, y+dy]; z_coords = [0, 0, 0, 0, h, h, h, h]
    return go.Mesh3d(x=x_coords, y=y_coords, z=z_coords, i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2], j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3], k=[0, 7, 2, 3, 6, 7, 2, 5, 1, 2, 5, 6], color='grey', opacity=0.7, name='Building', hoverinfo='none')

def create_nfz_box(zone):
    """Creates a 3D Plotly mesh for a No-Fly Zone."""
    x_coords = [zone[0], zone[2], zone[2], zone[0], zone[0], zone[2], zone[2], zone[0]]; y_coords = [zone[1], zone[1], zone[3], zone[3], zone[1], zone[1], zone[3], zone[3]]; z_coords = [0, 0, 0, 0, config.MAX_ALTITUDE, config.MAX_ALTITUDE, config.MAX_ALTITUDE, config.MAX_ALTITUDE]
    return go.Mesh3d(x=x_coords, y=y_coords, z=z_coords, color='red', opacity=0.15, name='No-Fly Zone', hoverinfo='name')

# --- Main App Configuration ---
st.set_page_config(layout="wide")
st.title("🚁 Q-DOP: Hierarchical QUBO Path Optimization")
for key, default in [('stage', 'setup'), ('mission_plan', None), ('animation_step', 0), ('log', [])]:
    if key not in st.session_state: st.session_state[key] = default

def setup_stage():
    """Renders the UI for setting up a new mission."""
    st.header("1. Mission Parameters"); log_event("App initialized. Waiting for mission setup.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Departure & Arrival")
        hub_name = st.selectbox("Choose a Departure Hub", list(config.HUBS.keys())); st.session_state.hub_location = config.HUBS[hub_name]
        dest_name = st.selectbox("Choose a Destination", list(config.DESTINATIONS.keys())); st.session_state.destination = config.DESTINATIONS[dest_name]
    with col2:
        st.subheader("Drone & Payload"); st.session_state.num_drones = st.slider("Drones available at Hub", 1, 10, 3, 1)
        st.session_state.payload_kg = st.slider("Payload Weight (kg)", 0.1, config.DRONE_MAX_PAYLOAD_KG, 1.5, 0.1)
    st.subheader("2. Environmental Conditions"); wind_col1, wind_col2 = st.columns(2)
    with wind_col1: st.session_state.wind_speed = st.slider("Wind Speed (m/s)", 0.0, 20.0, 5.0, 0.5)
    with wind_col2: st.session_state.wind_direction = st.slider("Wind Direction (°)", 0, 359, 270, 1)
    st.subheader("3. Optimization Priority"); st.session_state.opt_preference = st.radio("Optimize For:", ["Balanced", "Fastest Delivery", "Fuel Efficient"], horizontal=True, index=0)
    if st.button("🚀 Plan Mission", type="primary", width='stretch'):
        st.session_state.order = Order(id=0, location=st.session_state.destination, payload_kg=st.session_state.payload_kg)
        st.session_state.stage = 'planning'; st.rerun()

def planning_stage():
    """Handles the backend processing for path optimization."""
    log_event("Starting mission planning with Hierarchical QUBO...")
    with st.spinner("Finding optimal waypoint strategy and solving local QUBOs..."):
        env = Environment(wind_speed_mps=st.session_state.wind_speed, wind_direction_deg=st.session_state.wind_direction)
        predictor = EnergyTimePredictor()
        path_planner = PathPlanner3D(env, predictor)
        
        pref = st.session_state.opt_preference
        weights = {'time': 0.5, 'energy': 0.5};
        if pref == "Fastest Delivery": weights = {'time': 0.9, 'energy': 0.1}
        elif pref == "Fuel Efficient": weights = {'time': 0.1, 'energy': 0.9}

        hub_loc, order_loc, payload = st.session_state.hub_location, st.session_state.order.location, st.session_state.payload_kg
        takeoff_start, takeoff_end = hub_loc, (hub_loc[0], hub_loc[1], config.TAKEOFF_ALTITUDE)
        takeoff_path = [takeoff_start, takeoff_end]

        path_to, status_to = path_planner.find_path(takeoff_end, order_loc, payload, weights)
        path_from, status_from = path_planner.find_path(order_loc, takeoff_end, 0, weights)

        if path_to is None or path_from is None:
            st.error(f"Fatal Error: Path planning failed. Status: {status_to or status_from}.");
            st.session_state.mission_plan = None; st.session_state.stage = 'results'; st.rerun(); return
        
        landing_path = [takeoff_end, takeoff_start]
        full_path = takeoff_path + path_to[1:] + path_from[1:] + landing_path[1:]
        
        # Generate Detailed Mission Log and Calculate Final Costs
        mission_log, total_time, total_energy, current_payload = [], 0.0, 0.0, payload
        accel_e = predictor.calculate_inertial_energy(current_payload); total_energy += accel_e
        mission_log.append({"Segment": "Takeoff Accel", "Time (s)": np.nan, "Energy (Wh)": f"{accel_e:.3f}", "Payload (kg)": current_payload, "Altitude (m)": np.nan, "Distance (m)": np.nan, "Notes": "Inertial cost"})
        for i in range(1, len(full_path)):
            p_prev, p1, p2 = (full_path[i-2] if i > 1 else None), full_path[i-1], full_path[i]
            if p1 == order_loc: current_payload = 0
            wind = env.weather.get_wind_at_location(p1[0], p1[1]); t, e = predictor.predict(p1, p2, current_payload, wind, p_prev)
            total_time += t; total_energy += e
            mission_log.append({"Segment": f"Leg {i}", "Time (s)": f"{t:.2f}", "Energy (Wh)": f"{e:.3f}", "Payload (kg)": current_payload, "Altitude (m)": f"{p2[2]:.1f}", "Distance (m)": f"{calculate_distance_3d(p1, p2):.1f}", "Notes": ""})
        decel_e = predictor.calculate_inertial_energy(0); total_energy += decel_e
        mission_log.append({"Segment": "Landing Decel", "Time (s)": np.nan, "Energy (Wh)": f"{decel_e:.3f}", "Payload (kg)": 0, "Altitude (m)": np.nan, "Distance (m)": np.nan, "Notes": "Inertial cost"})

        st.session_state.mission_plan = {'full_path': full_path, 'total_time': total_time, 'total_energy': total_energy, 'env': env, 'mission_log': mission_log, 'solver_status': status_to}
        log_event("Mission plan generated successfully.")
    st.session_state.stage = 'results'; st.session_state.animation_step = 0; st.rerun()

def results_stage():
    """Renders the 3D visualization, stats, and logs for a completed plan."""
    mission_plan = st.session_state.mission_plan
    if not mission_plan:
        st.warning("No mission plan could be generated. Please return to setup.");
        if st.button("⬅️ New Mission", width='stretch'): st.session_state.stage = 'setup'; st.rerun()
        return

    st.header(f"Mission Plan (Optimized for: {st.session_state.opt_preference})")
    status = mission_plan['solver_status']
    if "Optimal" in status: st.success(f"✅ Solver Status: {status}")
    else: st.error(f"⚠️ Solver Status: {status}. The planner failed to produce a valid path.")

    col1, col2 = st.columns([1, 2.5])
    with col1:
        st.subheader("Mission Stats"); st.metric("Total Flight Time", f"{mission_plan['total_time']:.2f} s"); st.metric("Total Energy Consumed", f"{mission_plan['total_energy']:.2f} Wh")
        st.subheader("Animation Controls"); st.session_state.animation_step = st.slider("Simulation Timeline", 0, 100, st.session_state.animation_step)
        if st.button("⬅️ New Mission", width='stretch'): st.session_state.stage = 'setup'; st.rerun()
        with st.expander("Show Detailed Mission Log"): st.dataframe(pd.DataFrame(mission_plan['mission_log']), use_container_width=True)

    plot_placeholder, progress_placeholder = st.empty(), st.empty()
    def draw_scene(animation_step):
        fig = go.Figure(); env = mission_plan['env']; hub_loc, dest_loc = st.session_state.hub_location, st.session_state.destination
        fig.add_trace(go.Scatter3d(x=[hub_loc[0]], y=[hub_loc[1]], z=[hub_loc[2]], mode='markers', marker=dict(size=8, color='cyan', symbol='diamond'), name='Hub'))
        fig.add_trace(go.Scatter3d(x=[dest_loc[0]], y=[dest_loc[1]], z=[dest_loc[2]], mode='markers+text', text=["Dest."], textposition='middle right', marker=dict(size=8, color='lime'), name='Destination'))
        for b in env.buildings: fig.add_trace(create_box(b));
        for nfz in config.NO_FLY_ZONES: fig.add_trace(create_nfz_box(nfz))
        max_time = mission_plan.get('total_time', 0); current_time = (animation_step / 100.0) * max_time if max_time > 0 else 0
        drone_path = mission_plan['full_path']; pos = hub_loc; path_len = len(drone_path)
        if path_len > 1 and max_time > 0:
            time_per_segment = max_time / (path_len - 1); current_segment_idx = min(int(current_time / time_per_segment), path_len - 2)
            segment_progress = (current_time % time_per_segment) / time_per_segment if time_per_segment > 0 else 0
            p_start, p_end = np.array(drone_path[current_segment_idx]), np.array(drone_path[current_segment_idx + 1]); pos = p_start + segment_progress * (p_end - p_start)
        elif path_len == 1: pos = drone_path[0]
        fig.add_trace(go.Scatter3d(x=[pos[0]], y=[pos[1]], z=[pos[2]], mode='markers', marker=dict(size=10, color='red', symbol='circle-open'), name='Live Drone'))
        full_3d_path = np.array(drone_path); fig.add_trace(go.Scatter3d(x=full_3d_path[:,0], y=full_3d_path[:,1], z=full_3d_path[:,2], mode='lines', line=dict(width=4, color='yellow'), name='Optimal Path'))
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), showlegend=True, scene=dict(xaxis_title='Longitude', yaxis_title='Latitude', zaxis_title='Altitude (m)', aspectratio=dict(x=1, y=1, z=0.4), bgcolor='rgb(20, 24, 54)'), legend=dict(font=dict(color='white')))
        plot_placeholder.plotly_chart(fig, use_container_width=True, height=700); progress_placeholder.progress(animation_step / 100.0, text=f"Simulation Time: {current_time:.2f}s / {max_time:.2f}s")
    draw_scene(st.session_state.animation_step)

with st.sidebar:
    st.header("Operations Log"); log_container = st.container(height=400)
    with log_container:
        for entry in st.session_state.log: st.text(entry)

if st.session_state.stage == 'setup': setup_stage()
elif st.session_state.stage == 'planning': planning_stage()
elif st.session_state.stage == 'results': results_stage()