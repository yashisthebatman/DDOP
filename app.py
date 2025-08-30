import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from environment import Environment, Order, Building, WeatherSystem # Updated import
from ml_predictor.predictor import EnergyTimePredictor
from path_planner import PathPlanner3D
from utils.geometry import calculate_distance_3d

# --- App State Initialization ---
for key, default in [('stage', 'setup'), ('mission_plan', None), ('animation_step', 0), ('log', []), ('planner', None)]:
    if key not in st.session_state: st.session_state[key] = default

# --- Helper Functions ---
def log_event(message):
    st.session_state.log.insert(0, f"{time.strftime('%H:%M:%S')} - {message}")

def create_box(building: Building):
    x, y = building.center_xy; dx, dy = building.size_xy[0] / 2, building.size_xy[1] / 2; h = building.height
    x_coords = [x-dx, x+dx, x+dx, x-dx, x-dx, x+dx, x+dx, x-dx]; y_coords = [y-dy, y-dy, y+dy, y+dy, y-dy, y-dy, y+dy, y+dy]; z_coords = [0, 0, 0, 0, h, h, h, h]
    return go.Mesh3d(x=x_coords, y=y_coords, z=z_coords, i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2], j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3], k=[0, 7, 2, 3, 6, 7, 2, 5, 1, 2, 5, 6], color='grey', opacity=0.7, name='Building', hoverinfo='none')

def create_nfz_box(zone):
    x_coords = [zone[0], zone[2], zone[2], zone[0], zone[0], zone[2], zone[2], zone[0]]; y_coords = [zone[1], zone[1], zone[3], zone[3], zone[1], zone[1], zone[3], zone[3]]; z_coords = [0, 0, 0, 0, config.MAX_ALTITUDE, config.MAX_ALTITUDE, config.MAX_ALTITUDE, config.MAX_ALTITUDE]
    return go.Mesh3d(x=x_coords, y=y_coords, z=z_coords, color='red', opacity=0.15, name='No-Fly Zone', hoverinfo='name')

# --- Main App Configuration ---
st.set_page_config(layout="wide")
st.title("🚁 Q-DOP: Real-time Path Optimization with QUBO Heuristics")

# --- UI Stages ---
def setup_stage():
    st.header("1. Mission Parameters"); log_event("App initialized. Waiting for mission setup.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Departure & Arrival")
        st.session_state.hub_location = config.HUBS[st.selectbox("Choose a Departure Hub", list(config.HUBS.keys()))]
        st.session_state.destination = config.DESTINATIONS[st.selectbox("Choose a Destination", list(config.DESTINATIONS.keys()))]
    with col2:
        st.subheader("Drone & Payload"); st.session_state.num_drones = st.slider("Drones available at Hub", 1, 10, 3, 1)
        st.session_state.payload_kg = st.slider("Payload Weight (kg)", 0.1, config.DRONE_MAX_PAYLOAD_KG, 1.5, 0.1)
    
    st.subheader("2. Environmental Conditions")
    wind_col1, wind_col2 = st.columns(2)
    with wind_col1: st.session_state.max_wind_speed = st.slider("Max Wind Speed (m/s)", 0.0, 25.0, 10.0, 0.5, help="Controls the peak speed of wind gusts in the dynamic field.")
    with wind_col2: st.session_state.wind_complexity = st.slider("Wind Pattern Complexity", 10.0, 500.0, 150.0, 10.0, help="Controls the size of wind patterns. Lower values mean larger, broader wind areas.")

    st.subheader("3. Optimization Priority"); st.session_state.opt_preference = st.radio("Optimize For:", ["Balanced", "Fastest Delivery", "Fuel Efficient"], horizontal=True, index=0)
    
    if st.button("🚀 Plan Mission", type="primary", use_container_width=True):
        st.session_state.order = Order(id=0, location=st.session_state.destination, payload_kg=st.session_state.payload_kg)
        st.session_state.stage = 'planning'; st.rerun()

def planning_stage():
    log_event("Starting mission planning...")
    with st.spinner("Initializing path planner and running real-time search..."):
        # Initialize the environment and planner ONCE and store in session state
        if st.session_state.planner is None:
            log_event("First run: Pre-computing QUBO heuristic table. This may take a moment...")
            weather_system = WeatherSystem(scale=st.session_state.wind_complexity, max_speed=st.session_state.max_wind_speed)
            env = Environment(weather_system=weather_system)
            predictor = EnergyTimePredictor()
            st.session_state.planner = PathPlanner3D(env, predictor)
            log_event("Planner initialized successfully.")
        else:
            # On subsequent runs, just update the weather in the existing environment
            log_event("Updating weather conditions for new mission...")
            st.session_state.planner.env.weather.max_speed = st.session_state.max_wind_speed
            st.session_state.planner.env.weather.scale = st.session_state.wind_complexity

        path_planner = st.session_state.planner
        pref = st.session_state.opt_preference
        weights = {'time': 0.5, 'energy': 0.5}
        if pref == "Fastest Delivery": weights = {'time': 0.9, 'energy': 0.1}
        elif pref == "Fuel Efficient": weights = {'time': 0.1, 'energy': 0.9}

        hub_loc, order_loc, payload = st.session_state.hub_location, st.session_state.order.location, st.session_state.payload_kg
        takeoff_end = (hub_loc[0], hub_loc[1], config.TAKEOFF_ALTITUDE)
        
        path_to, status_to = path_planner.find_path(takeoff_end, order_loc, payload, weights)
        path_from, status_from = path_planner.find_path(order_loc, takeoff_end, 0, weights)

        if path_to is None or path_from is None:
            st.error(f"Fatal Error: Path planning failed. Status: {status_to or status_from}.");
            st.session_state.mission_plan = None; st.session_state.stage = 'results'; st.rerun(); return
        
        full_path = [hub_loc] + path_to + path_from[1:] + [hub_loc]
        
        # Generate Detailed Mission Log
        mission_log, total_time, total_energy, current_payload = [], 0.0, 0.0, payload
        for i in range(1, len(full_path)):
            p1, p2 = full_path[i-1], full_path[i]
            if p1 == order_loc: current_payload = 0
            wind = path_planner.env.weather.get_wind_at_location(p1[0], p1[1])
            t, e = path_planner.predictor.predict(p1, p2, current_payload, wind, (full_path[i-2] if i > 1 else None))
            total_time += t; total_energy += e
            mission_log.append({"Segment": f"Leg {i}", "Time (s)": f"{t:.2f}", "Energy (Wh)": f"{e:.3f}", "Payload (kg)": current_payload, "Altitude (m)": f"{p2[2]:.1f}", "Distance (m)": f"{calculate_distance_3d(p1, p2):.1f}"})

        st.session_state.mission_plan = {'full_path': full_path, 'total_time': total_time, 'total_energy': total_energy, 'env': path_planner.env, 'mission_log': mission_log, 'solver_status': status_to}
        log_event("Mission plan generated successfully.")
    st.session_state.stage = 'results'; st.session_state.animation_step = 0; st.rerun()

def results_stage():
    mission_plan = st.session_state.mission_plan
    if not mission_plan:
        st.warning("No mission plan could be generated. Please return to setup.");
        if st.button("⬅️ New Mission", use_container_width=True): st.session_state.stage = 'setup'; st.rerun()
        return

    st.header(f"Mission Plan (Optimized for: {st.session_state.opt_preference})")
    status = mission_plan['solver_status']
    if "Optimal" in status: st.success(f"✅ Solver Status: {status}")
    else: st.warning(f"⚠️ Solver Status: {status}.")

    col1, col2 = st.columns([1, 2.5])
    with col1:
        st.subheader("Mission Stats")
        st.metric("Total Flight Time", f"{mission_plan['total_time']:.2f} s")
        st.metric("Total Energy Consumed", f"{mission_plan['total_energy']:.2f} Wh")
        st.subheader("Animation Controls")
        st.session_state.animation_step = st.slider("Simulation Timeline", 0, 100, st.session_state.animation_step)
        if st.button("⬅️ New Mission", use_container_width=True): st.session_state.stage = 'setup'; st.rerun()
        with st.expander("Show Detailed Mission Log"): st.dataframe(pd.DataFrame(mission_plan['mission_log']), use_container_width=True)

    with col2:
        plot_placeholder = st.empty()
        fig = go.Figure(); env = mission_plan['env']; hub_loc, dest_loc = st.session_state.hub_location, st.session_state.destination
        fig.add_trace(go.Scatter3d(x=[hub_loc[0]], y=[hub_loc[1]], z=[hub_loc[2]], mode='markers', marker=dict(size=8, color='cyan', symbol='diamond'), name='Hub'))
        fig.add_trace(go.Scatter3d(x=[dest_loc[0]], y=[dest_loc[1]], z=[dest_loc[2]], mode='markers+text', text=["Dest."], textposition='middle right', marker=dict(size=8, color='lime'), name='Destination'))
        for b in env.buildings: fig.add_trace(create_box(b));
        for nfz in config.NO_FLY_ZONES: fig.add_trace(create_nfz_box(nfz))
        
        drone_path = mission_plan['full_path']
        full_3d_path = np.array(drone_path); fig.add_trace(go.Scatter3d(x=full_3d_path[:,0], y=full_3d_path[:,1], z=full_3d_path[:,2], mode='lines', line=dict(width=4, color='yellow'), name='Optimal Path'))
        
        # Animate drone position
        path_distances = [0] + [calculate_distance_3d(drone_path[i], drone_path[i+1]) for i in range(len(drone_path)-1)]
        cumulative_dist = np.cumsum(path_distances)
        total_dist = cumulative_dist[-1]
        target_dist = (st.session_state.animation_step / 100.0) * total_dist
        segment_idx = np.searchsorted(cumulative_dist, target_dist, side='right') - 1
        segment_idx = max(0, segment_idx)
        
        pos = drone_path[0]
        if total_dist > 0 and segment_idx < len(drone_path) -1:
            p_start, p_end = np.array(drone_path[segment_idx]), np.array(drone_path[segment_idx+1])
            dist_into_segment = target_dist - cumulative_dist[segment_idx]
            segment_len = path_distances[segment_idx+1]
            progress = dist_into_segment / segment_len if segment_len > 0 else 0
            pos = p_start + progress * (p_end - p_start)
        elif st.session_state.animation_step == 100:
            pos = drone_path[-1]

        fig.add_trace(go.Scatter3d(x=[pos[0]], y=[pos[1]], z=[pos[2]], mode='markers', marker=dict(size=10, color='red', symbol='circle-open'), name='Live Drone'))
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), showlegend=True, scene=dict(xaxis_title='Longitude', yaxis_title='Latitude', zaxis_title='Altitude (m)', aspectratio=dict(x=1, y=1, z=0.4), bgcolor='rgb(20, 24, 54)'), legend=dict(font=dict(color='white')))
        plot_placeholder.plotly_chart(fig, use_container_width=True, height=700)

# --- Sidebar and Stage Routing ---
with st.sidebar:
    st.header("Operations Log"); log_container = st.container(height=400)
    with log_container:
        for entry in st.session_state.log: st.text(entry)

if st.session_state.stage == 'setup': setup_stage()
elif st.session_state.stage == 'planning': planning_stage()
elif st.session_state.stage == 'results': results_stage()