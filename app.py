# app.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time
import pandas as pd
import sys, os
from shapely.geometry import LineString, Polygon

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from environment import Environment, Building, WeatherSystem
from ml_predictor.predictor import EnergyTimePredictor
from path_planner import PathPlanner3D

# --- App State Initialization ---
@st.cache_resource
def load_planner() -> PathPlanner3D:
    log_event("Loading planner and environment...")
    initial_weather = WeatherSystem(scale=150.0, max_speed=10.0)
    env = Environment(weather_system=initial_weather)
    predictor = EnergyTimePredictor()
    planner = PathPlanner3D(env, predictor)
    log_event("Planner loaded successfully.")
    return planner

# Initialize session state keys
defaults = {
    'stage': 'setup', 'log': [], 'mission_running': False,
    'planned_path': None, 'mission_log': [], 'total_time': 0.0,
    'total_energy': 0.0, 'drone_pos': None, 'path_index': 0,
    'initial_payload': 0.0
}
for key, default_val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_val

def log_event(message: str):
    st.session_state.log.insert(0, f"{time.strftime('%H:%M:%S')} - {message}")

def reset_mission_state():
    """Resets all state variables related to a simulation run."""
    for key, default_val in defaults.items():
        if key not in ['stage', 'log']: # Don't reset stage or log on new mission
             st.session_state[key] = default_val
    st.session_state.stage = 'setup'


planner = load_planner()
st.set_page_config(layout="wide")
st.title("🚁 Q-DOP: Dynamic Re-planning with ML Prediction")

# --- UI Helper Functions ---
def create_box(building: Building) -> go.Mesh3d:
    x, y = building.center_xy; dx, dy = building.size_xy[0] / 2, building.size_xy[1] / 2; h = building.height
    x_coords = [x-dx, x+dx, x+dx, x-dx, x-dx, x+dx, x+dx, x-dx]; y_coords = [y-dy, y-dy, y+dy, y+dy, y-dy, y-dy, y+dy, y+dy]; z_coords = [0, 0, 0, 0, h, h, h, h]
    return go.Mesh3d(x=x_coords, y=y_coords, z=z_coords, i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2], j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3], k=[0, 7, 2, 3, 6, 7, 2, 5, 1, 2, 5, 6], color='grey', opacity=0.7, name='Building', hoverinfo='none')

def create_nfz_box(zone: list, color='red', opacity=0.15) -> go.Mesh3d:
    x_coords = [zone[0], zone[2], zone[2], zone[0], zone[0], zone[2], zone[2], zone[0]]; y_coords = [zone[1], zone[1], zone[3], zone[3], zone[1], zone[1], zone[3], zone[3]]; z_coords = [0, 0, 0, 0, config.MAX_ALTITUDE, config.MAX_ALTITUDE, config.MAX_ALTITUDE, config.MAX_ALTITUDE]
    return go.Mesh3d(x=x_coords, y=y_coords, z=z_coords, color=color, opacity=opacity, name='No-Fly Zone', hoverinfo='name')

# --- Replanning Logic ---
def check_replanning_triggers(current_pos, next_pos, planner):
    """Checks if the drone needs to replan its path."""
    # Trigger 1: Path intersects a new dynamic NFZ
    segment = LineString([current_pos[:2], next_pos[:2]])
    for nfz in planner.env.dynamic_nfzs:
        zone_poly = Polygon([(nfz[0], nfz[1]), (nfz[2], nfz[1]), (nfz[2], nfz[3]), (nfz[0], nfz[3])])
        if segment.intersects(zone_poly):
            log_event("🚨 Path invalid! Intersects new NFZ. Replanning...")
            return True
    return False

# --- App Stages ---
def setup_stage():
    """UI for the mission setup screen."""
    st.header("1. Mission Parameters")
    if not st.session_state.log: log_event("App initialized. Waiting for mission setup.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.hub_location = config.HUBS[st.selectbox("Choose Departure Hub", list(config.HUBS.keys()))]
        st.session_state.destination = config.DESTINATIONS[st.selectbox("Choose Destination", list(config.DESTINATIONS.keys()))]
    with col2:
        st.session_state.payload_kg = st.slider("Payload (kg)", 0.1, config.DRONE_MAX_PAYLOAD_KG, 1.5, 0.1)
    
    st.subheader("2. Environmental Conditions")
    wind_col1, wind_col2 = st.columns(2)
    with wind_col1: st.session_state.max_wind_speed = st.slider("Max Wind Speed (m/s)", 0.0, 25.0, 10.0, 0.5)
    with wind_col2: st.session_state.wind_complexity = st.slider("Wind Complexity", 10.0, 500.0, 150.0, 10.0)

    st.subheader("3. Optimization Priority")
    st.session_state.optimization_preference = st.radio("Optimize For:", ["Balanced", "Fastest Path", "Most Battery Efficient"], horizontal=True, index=0)
    if st.session_state.optimization_preference == "Balanced":
        st.session_state.balance_weight = st.slider("Priority", 0.0, 1.0, 0.5, 0.05, format="%.2f", help="0.0=Energy, 1.0=Time")
    
    if st.button("🚀 Plan Initial Mission", type="primary", use_container_width=True):
        st.session_state.stage = 'planning'; st.rerun()

def planning_stage():
    """Handles the initial path planning."""
    optimization_mode = st.session_state.optimization_preference
    log_event(f"Planning initial mission ('{optimization_mode}' optimization)...")
    
    with st.spinner(f"Optimizing initial path... (using HPA* on {os.cpu_count()} cores)"):
        planner.env.weather.max_speed = st.session_state.max_wind_speed
        planner.env.weather.scale = st.session_state.wind_complexity
        
        mode_map = {"Fastest Path": "time", "Most Battery Efficient": "energy", "Balanced": "balanced"}
        selected_mode = mode_map[optimization_mode]
        balance_weight = st.session_state.get('balance_weight', 0.5)

        hub_loc, order_loc = st.session_state.hub_location, st.session_state.destination
        payload = st.session_state.payload_kg
        st.session_state.initial_payload = payload
        
        takeoff_end = (hub_loc[0], hub_loc[1], config.TAKEOFF_ALTITUDE)
        
        path_to, status_to = planner.find_path(takeoff_end, order_loc, payload, selected_mode, balance_weight)
        path_from, status_from = planner.find_path(order_loc, takeoff_end, 0, selected_mode, balance_weight)

        if path_to is None or path_from is None:
            st.error(f"Fatal Error: Path planning failed. Status: {status_to or status_from}.");
            st.button("⬅️ New Mission", on_click=reset_mission_state)
            return
        
        full_path = [hub_loc] + path_to + path_from[1:] + [hub_loc]
        st.session_state.planned_path = full_path
        st.session_state.drone_pos = full_path[0]
        st.session_state.path_index = 0
        log_event("✅ Initial mission plan found successfully.")
    st.session_state.stage = 'simulation'; st.rerun()

def simulation_stage():
    """UI and logic for the live mission simulation."""
    st.header(f"Mission Simulation (Optimized for: {st.session_state.optimization_preference})")
    
    col1, col2 = st.columns([1, 2.5])
    
    with col1:
        st.subheader("Mission Control")
        start_button, stop_button = st.columns(2)
        if start_button.button("▶️ Run Simulation", use_container_width=True, disabled=st.session_state.mission_running):
            st.session_state.mission_running = True
            st.rerun()
        if stop_button.button("⏸️ Pause", use_container_width=True, disabled=not st.session_state.mission_running):
            st.session_state.mission_running = False
            st.rerun()
            
        st.subheader("Mission Stats")
        progress_value = (st.session_state.path_index) / (len(st.session_state.planned_path) -1) if len(st.session_state.planned_path) > 1 else 0
        st.progress(progress_value, text=f"Mission Progress: {progress_value:.0%}")
        
        st.metric("Flight Time", f"{st.session_state.total_time:.2f} s")
        remaining_battery = config.DRONE_BATTERY_WH - st.session_state.total_energy
        st.metric("Battery Remaining", f"{remaining_battery:.2f} Wh", delta=f"{-st.session_state.total_energy:.2f} Wh used")
        
        if st.session_state.path_index >= len(st.session_state.planned_path) - 1 and st.session_state.planned_path:
            st.success("✅ Mission Complete!")
            st.session_state.mission_running = False

        if st.button("⬅️ New Mission", use_container_width=True):
            reset_mission_state()
            st.rerun()

    with col2:
        fig = go.Figure()
        hub_loc, dest_loc = st.session_state.hub_location, st.session_state.destination
        fig.add_traces([
            go.Scatter3d(x=[hub_loc[0]], y=[hub_loc[1]], z=[hub_loc[2]], mode='markers', marker=dict(size=8, color='cyan', symbol='diamond'), name='Hub'),
            go.Scatter3d(x=[dest_loc[0]], y=[dest_loc[1]], z=[dest_loc[2]], mode='markers+text', text=["Dest."], textposition='middle right', marker=dict(size=8, color='lime'), name='Destination')
        ])
        for b in planner.env.buildings: fig.add_trace(create_box(b))
        for nfz in planner.env.static_nfzs: fig.add_trace(create_nfz_box(nfz))
        for dnfz in planner.env.dynamic_nfzs: fig.add_trace(create_nfz_box(dnfz, color='yellow', opacity=0.2))

        path_np = np.array(st.session_state.planned_path)
        fig.add_trace(go.Scatter3d(x=path_np[:,0], y=path_np[:,1], z=path_np[:,2], mode='lines', line=dict(width=4, color='yellow'), name='Planned Path'))
        
        drone_pos = st.session_state.drone_pos
        fig.add_trace(go.Scatter3d(x=[drone_pos[0]], y=[drone_pos[1]], z=[drone_pos[2]], mode='markers', marker=dict(size=10, color='red', symbol='circle-open'), name='Live Drone'))
        
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene=dict(aspectratio=dict(x=1, y=1, z=0.4), bgcolor='rgb(20, 24, 54)'), legend=dict(font=dict(color='white')))
        st.plotly_chart(fig, use_container_width=True, height=700)

# --- Sidebar ---
with st.sidebar:
    st.header("Operations Log")
    log_container = st.container(height=800)
    with log_container:
        for entry in st.session_state.log: st.text(entry)

# --- Main Execution Logic ---
if planner.predictor.models:
    if st.session_state.stage == 'setup':
        setup_stage()
    elif st.session_state.stage == 'planning':
        planning_stage()
    elif st.session_state.stage == 'simulation':
        simulation_stage()

    # --- Simulation Loop ---
    if st.session_state.mission_running:
        path = st.session_state.planned_path
        idx = st.session_state.path_index

        if idx < len(path) - 1:
            p1 = path[idx]
            p2 = path[idx + 1]

            # Update environment
            time_step = 0.5 # Simulated time per loop
            planner.env.update_environment(st.session_state.total_time, time_step)
            
            # Check for replan
            if check_replanning_triggers(p1, p2, planner):
                with st.spinner("Replanning..."):
                    payload = 0 if st.session_state.total_time > st.session_state.mission_log[-1]['cumulative_time']/2 else st.session_state.initial_payload
                    
                    mode_map = {"Fastest Path": "time", "Most Battery Efficient": "energy", "Balanced": "balanced"}
                    selected_mode = mode_map[st.session_state.optimization_preference]
                    balance_weight = st.session_state.get('balance_weight', 0.5)

                    # Create new plan from current location to final hub
                    final_hub = st.session_state.hub_location
                    new_path, status = planner.find_path(p1, final_hub, payload, selected_mode, balance_weight)
                    
                    if new_path:
                        st.session_state.planned_path = path[:idx+1] + new_path[1:]
                        log_event("✅ New path found and integrated.")
                    else:
                        log_event("❌ Replanning failed. Stopping mission.")
                        st.session_state.mission_running = False

            # Recalculate segment with potentially new weather
            current_payload = st.session_state.initial_payload
            if np.allclose(p1, st.session_state.destination): current_payload = 0

            p_prev = path[idx-1] if idx > 0 else None
            wind = planner.env.weather.get_wind_at_location(p1[0], p1[1])
            t, e = planner.predictor.predict(p1, p2, current_payload, wind, p_prev)

            st.session_state.total_time += t
            st.session_state.total_energy += e
            st.session_state.path_index += 1
            st.session_state.drone_pos = p2
            
            time.sleep(0.1) # Pace the animation
            st.rerun()
else:
    st.error("ML Model not found. Please run `data_generator.py` and `train_model.py` first.")