# app.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time
import pandas as pd
import sys, os
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from environment import Environment, Building
from ml_predictor.predictor import EnergyTimePredictor
from path_planner import PathPlanner3D
from utils.geometry import calculate_distance_3d

# --- App State Initialization & Planner Loading ---
@st.cache_resource
def load_planner() -> PathPlanner3D:
    """Cached function to load the planner once and reuse it."""
    log_event("Loading planner and environment grid...")
    initial_weather = Environment.WeatherSystem(scale=150.0, max_speed=10.0)
    env = Environment(weather_system=initial_weather)
    predictor = EnergyTimePredictor()
    planner = PathPlanner3D(env, predictor)
    log_event("Planner loaded successfully.")
    return planner

# Initialize session state keys
for key, default in [('stage', 'setup'), ('mission_plan', None), ('animation_step', 0), ('log', [])]:
    if key not in st.session_state:
        st.session_state[key] = default

def log_event(message: str):
    """Adds a timestamped message to the app's log."""
    st.session_state.log.insert(0, f"{time.strftime('%H:%M:%S')} - {message}")

planner = load_planner()

# --- UI Helper Functions ---
def create_box(building: Building) -> go.Mesh3d:
    x, y = building.center_xy; dx, dy = building.size_xy[0] / 2, building.size_xy[1] / 2; h = building.height
    x_coords = [x-dx, x+dx, x+dx, x-dx, x-dx, x+dx, x+dx, x-dx]; y_coords = [y-dy, y-dy, y+dy, y+dy, y-dy, y-dy, y+dy, y+dy]; z_coords = [0, 0, 0, 0, h, h, h, h]
    return go.Mesh3d(x=x_coords, y=y_coords, z=z_coords, i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2], j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3], k=[0, 7, 2, 3, 6, 7, 2, 5, 1, 2, 5, 6], color='grey', opacity=0.7, name='Building', hoverinfo='none')

def create_nfz_box(zone: list) -> go.Mesh3d:
    x_coords = [zone[0], zone[2], zone[2], zone[0], zone[0], zone[2], zone[2], zone[0]]; y_coords = [zone[1], zone[1], zone[3], zone[3], zone[1], zone[1], zone[3], zone[3]]; z_coords = [0, 0, 0, 0, config.MAX_ALTITUDE, config.MAX_ALTITUDE, config.MAX_ALTITUDE, config.MAX_ALTITUDE]
    return go.Mesh3d(x=x_coords, y=y_coords, z=z_coords, color='red', opacity=0.15, name='No-Fly Zone', hoverinfo='name')

# --- Main App Structure ---
st.set_page_config(layout="wide")
st.title("🚁 Q-DOP: A* Optimization with Intelligent Heuristics")

def setup_stage():
    """UI for the mission setup screen."""
    st.header("1. Mission Parameters")
    if not st.session_state.log: log_event("App initialized. Waiting for mission setup.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Departure & Arrival")
        st.session_state.hub_location = config.HUBS[st.selectbox("Choose a Departure Hub", list(config.HUBS.keys()))]
        st.session_state.destination = config.DESTINATIONS[st.selectbox("Choose a Destination", list(config.DESTINATIONS.keys()))]
    with col2:
        st.subheader("Drone & Payload")
        st.session_state.payload_kg = st.slider("Payload Weight (kg)", 0.1, config.DRONE_MAX_PAYLOAD_KG, 1.5, 0.1)
    
    st.subheader("2. Real-time Environmental Conditions")
    wind_col1, wind_col2 = st.columns(2)
    with wind_col1: st.session_state.max_wind_speed = st.slider("Max Wind Speed (m/s)", 0.0, 25.0, 10.0, 0.5)
    with wind_col2: st.session_state.wind_complexity = st.slider("Wind Pattern Complexity", 10.0, 500.0, 150.0, 10.0)

    st.subheader("3. Optimization Priority")
    st.session_state.optimization_preference = st.radio(
        "Optimize For:", ["Balanced", "Fastest Path", "Most Battery Efficient"],
        horizontal=True, index=0)
    
    if st.button("🚀 Plan Mission", type="primary", use_container_width=True):
        st.session_state.stage = 'planning'; st.rerun()

def planning_stage():
    """Handles the planning logic and spinner."""
    optimization_mode = st.session_state.optimization_preference
    log_event(f"Planning mission with '{optimization_mode}' optimization...")
    
    with st.spinner(f"Optimizing path for '{optimization_mode}'..."):
        if planner is None:
            st.error("Path Planner could not be loaded. Aborting mission."); return

        planner.env.weather.max_speed = st.session_state.max_wind_speed
        planner.env.weather.scale = st.session_state.wind_complexity
        
        mode_map = {"Fastest Path": "time", "Most Battery Efficient": "energy", "Balanced": "balanced"}
        selected_mode = mode_map[optimization_mode]

        hub_loc = st.session_state.hub_location
        order_loc = st.session_state.destination
        payload = st.session_state.payload_kg
        takeoff_end = (hub_loc[0], hub_loc[1], config.TAKEOFF_ALTITUDE)
        
        path_to, status_to = planner.find_path(takeoff_end, order_loc, payload, selected_mode)
        path_from, status_from = planner.find_path(order_loc, takeoff_end, 0, selected_mode)

        if path_to is None or path_from is None:
            st.error(f"Fatal Error: Path planning failed. Status: {status_to or status_from}.");
            st.session_state.mission_plan = None
            st.session_state.stage = 'results'; st.rerun(); return
        
        full_path = [hub_loc] + path_to + path_from[1:] + [hub_loc]
        
        # Final accurate physics calculation is done here, on the final path only.
        mission_log, total_time, total_energy, current_payload = [], 0.0, 0.0, payload
        for i in range(1, len(full_path)):
            p1, p2 = full_path[i-1], full_path[i]
            if np.allclose(p1, order_loc): current_payload = 0
            wind = planner.env.weather.get_wind_at_location(p1[0], p1[1])
            t, e = planner.predictor.predict(p1, p2, current_payload, wind, (full_path[i-2] if i > 1 else None))
            total_time += t; total_energy += e
            mission_log.append({"Segment": f"Leg {i}", "Time (s)": f"{t:.2f}", "Energy (Wh)": f"{e:.3f}", "Payload (kg)": current_payload, "Altitude (m)": f"{p2[2]:.1f}", "Distance (m)": f"{calculate_distance_3d(p1, p2):.1f}"})

        st.session_state.mission_plan = {'full_path': full_path, 'total_time': total_time, 'total_energy': total_energy, 'mission_log': mission_log}
        
        drone_path_np = np.array(st.session_state.mission_plan['full_path'])
        path_distances = np.linalg.norm(np.diff(drone_path_np, axis=0), axis=1)
        st.session_state.mission_plan['cumulative_dist'] = np.insert(np.cumsum(path_distances), 0, 0)
        
        log_event(f"Mission plan found successfully.")
    st.session_state.stage = 'results'; st.session_state.animation_step = 0; st.rerun()

def results_stage():
    """UI for displaying the mission results and animation."""
    mission_plan = st.session_state.mission_plan
    if not mission_plan:
        st.warning("No mission plan could be generated. Return to setup.");
        if st.button("⬅️ New Mission", use_container_width=True): st.session_state.stage = 'setup'; st.rerun()
        return

    st.header(f"Mission Plan (Optimized for: {st.session_state.optimization_preference})")
    
    is_feasible = mission_plan['total_energy'] < config.DRONE_BATTERY_WH
    if is_feasible:
        st.success(f"✅ Path found successfully.")
    else:
        st.error(f"❌ INFEASIBLE: Mission exceeds battery capacity! ({mission_plan['total_energy']:.2f} Wh > {config.DRONE_BATTERY_WH} Wh)")

    col1, col2 = st.columns([1, 2.5])
    with col1:
        st.subheader("Mission Stats")
        st.metric("Total Flight Time", f"{mission_plan['total_time']:.2f} s")
        st.metric("Total Energy Consumed", f"{mission_plan['total_energy']:.2f} Wh", delta=f"{config.DRONE_BATTERY_WH} Wh Capacity", delta_color="inverse")
        st.subheader("Animation Controls")
        st.session_state.animation_step = st.slider("Simulation Timeline", 0, 100, st.session_state.animation_step)
        if st.button("⬅️ New Mission", use_container_width=True): st.session_state.stage = 'setup'; st.rerun()
        with st.expander("Show Detailed Mission Log"): st.dataframe(pd.DataFrame(mission_plan['mission_log']), use_container_width=True)

    with col2:
        fig = go.Figure(); hub_loc, dest_loc = st.session_state.hub_location, st.session_state.destination
        fig.add_traces([
            go.Scatter3d(x=[hub_loc[0]], y=[hub_loc[1]], z=[hub_loc[2]], mode='markers', marker=dict(size=8, color='cyan', symbol='diamond'), name='Hub'),
            go.Scatter3d(x=[dest_loc[0]], y=[dest_loc[1]], z=[dest_loc[2]], mode='markers+text', text=["Dest."], textposition='middle right', marker=dict(size=8, color='lime'), name='Destination')
        ])
        for b in planner.env.buildings: fig.add_trace(create_box(b))
        for nfz in config.NO_FLY_ZONES: fig.add_trace(create_nfz_box(nfz))
        
        drone_path = np.array(mission_plan['full_path'])
        fig.add_trace(go.Scatter3d(x=drone_path[:,0], y=drone_path[:,1], z=drone_path[:,2], mode='lines', line=dict(width=4, color='yellow'), name='Optimal Path'))
        
        cumulative_dist, total_dist = mission_plan['cumulative_dist'], mission_plan['cumulative_dist'][-1]
        target_dist = (st.session_state.animation_step / 100.0) * total_dist
        
        idx = np.searchsorted(cumulative_dist, target_dist) - 1
        pos = drone_path[-1] if st.session_state.animation_step == 100 else drone_path[0]
        if 0 <= idx < len(cumulative_dist) - 1:
            p_start, p_end = drone_path[idx], drone_path[idx+1]
            segment_len = cumulative_dist[idx+1] - cumulative_dist[idx]
            progress = (target_dist - cumulative_dist[idx]) / segment_len if segment_len > 0 else 0
            pos = p_start + progress * (p_end - p_start)

        fig.add_trace(go.Scatter3d(x=[pos[0]], y=[pos[1]], z=[pos[2]], mode='markers', marker=dict(size=10, color='red', symbol='circle-open'), name='Live Drone'))
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene=dict(aspectratio=dict(x=1, y=1, z=0.4), bgcolor='rgb(20, 24, 54)'), legend=dict(font=dict(color='white')))
        st.plotly_chart(fig, use_container_width=True, height=700)

# --- Sidebar ---
with st.sidebar:
    st.header("Operations Log")
    log_container = st.container(height=400)
    with log_container:
        for entry in st.session_state.log: st.text(entry)

# --- Main Execution Logic ---
if planner:
    if st.session_state.stage == 'setup':
        setup_stage()
    elif st.session_state.stage == 'planning':
        planning_stage()
    elif st.session_state.stage == 'results':
        results_stage()