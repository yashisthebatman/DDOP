# app.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time
import pandas as pd
from shapely.geometry import Polygon, Point

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from environment import Environment, Order, Building, WeatherZone, WeatherSystem
from ml_predictor.predictor import EnergyTimePredictor
from optimization.assignment_solver import AssignmentSolver
from path_planner import PathPlanner3D

# --- Helper Functions ---
def log_event(message):
    if 'log' not in st.session_state: st.session_state.log = []
    st.session_state.log.insert(0, f"{time.strftime('%H:%M:%S')} - {message}")

def create_box(building: Building):
    x, y = building.center_xy
    dx, dy = building.size_xy[0] / 2, building.size_xy[1] / 2
    h = building.height
    x_coords = [x-dx, x+dx, x+dx, x-dx, x-dx, x+dx, x+dx, x-dx]
    y_coords = [y-dy, y-dy, y+dy, y+dy, y-dy, y-dy, y+dy, y+dy]
    z_coords = [0, 0, 0, 0, h, h, h, h]
    return go.Mesh3d(x=x_coords, y=y_coords, z=z_coords, i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                     j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3], k=[0, 7, 2, 3, 6, 7, 2, 5, 1, 2, 5, 6],
                     color='grey', opacity=0.7, name='Building', hoverinfo='none',
                     lighting=dict(ambient=0.4, diffuse=1.0, fresnel=0.1, specular=0.5, roughness=0.5),
                     lightposition=dict(x=1000, y=2000, z=3000))

st.set_page_config(layout="wide")
st.title("🚁 Q-DOP: Path Optimization Comparison")

# --- Session State ---
for key, default in [('stage', 'setup'), ('mission_plan', None), ('animation_step', 0),
                     ('is_playing', False), ('log', []), ('orders_df', None), ('weather_df', None)]:
    if key not in st.session_state: st.session_state[key] = default

# --- STAGE 1: SETUP ---
def setup_stage():
    st.header("1. Mission Setup")
    log_event("App initialized. Waiting for mission setup.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Drone Fleet")
        st.session_state.num_drones = st.slider("Number of Drones", 1, 10, 3, 1)

        st.subheader("Weather Conditions")
        if st.session_state.weather_df is None:
            st.session_state.weather_df = pd.DataFrame([
                {"zone_id": 0, "center_lon": -74.00, "center_lat": 40.72, "radius_m": 1500, "wind_speed_mps": 15.0, "wind_direction_deg": 270},
                {"zone_id": 1, "center_lon": -73.98, "center_lat": 40.73, "radius_m": 1000, "wind_speed_mps": 5.0, "wind_direction_deg": 90},
            ])
        st.session_state.weather_df = st.data_editor(st.session_state.weather_df, num_rows="dynamic", use_container_width=True)

    with col2:
        st.subheader("Package Manifest")
        if st.session_state.orders_df is None:
            st.session_state.orders_df = pd.DataFrame([
                {"lat": 40.7128, "lon": -74.0060, "alt_m": 50.0, "payload_kg": 1.5},
                {"lat": 40.7291, "lon": -73.9965, "alt_m": 80.0, "payload_kg": 3.0},
            ])
        st.session_state.orders_df = st.data_editor(st.session_state.orders_df, num_rows="dynamic", use_container_width=True)

    st.header("2. Planning Parameters")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.session_state.path_solver_choice = st.radio("Pathfinding Solver", ["⭐ Google OR-Tools (Classical)", "💡 D-Wave QUBO (Quantum-Inspired)"], horizontal=True)
    with p_col2:
        st.session_state.opt_preference = st.radio("Optimization Priority", ["Fastest Delivery", "Fuel Efficient"], horizontal=True)

    if st.button("🚀 Plan Mission", type="primary", use_container_width=True):
        if st.session_state.orders_df.empty:
            st.error("Cannot plan mission. Please add at least one package to the manifest.")
            return

        orders = [Order(id=i, location=(r["lon"], r["lat"], r["alt_m"]), payload_kg=float(r["payload_kg"]))
                  for i, r in st.session_state.orders_df.iterrows()]

        weather_zones = []
        for i, row in st.session_state.weather_df.iterrows():
            angle_rad = np.deg2rad(270 - row["wind_direction_deg"])
            wind_vec = np.array([np.cos(angle_rad), np.sin(angle_rad), 0]) * row["wind_speed_mps"]
            center_point = (row["center_lon"], row["center_lat"])
            radius_deg = row["radius_m"] / 111320
            poly = Point(center_point).buffer(radius_deg)
            weather_zones.append(WeatherZone(id=i, polygon=poly, wind_vector=wind_vec))
        st.session_state.weather_system = WeatherSystem(weather_zones)
        st.session_state.orders = orders
        st.session_state.stage = 'planning'; st.rerun()

# --- STAGE 2: PLANNING ---
def planning_stage():
    log_event("Starting mission planning...")
    with st.spinner("Executing Mission Plan..."):
        log_event(f"Pathfinding with: {st.session_state.path_solver_choice}")

        env = Environment(st.session_state.num_drones, st.session_state.orders, st.session_state.weather_system)
        predictor = EnergyTimePredictor()
        path_planner = PathPlanner3D(env, predictor)

        solver = AssignmentSolver(env, predictor, path_planner, st.session_state.path_solver_choice)

        weights = {'time': 0.9, 'energy': 0.1} if st.session_state.opt_preference == "Fastest Delivery" else {'time': 0.1, 'energy': 0.9}

        st.session_state.mission_plan = solver.solve(weights)

        log_event("Mission plan generated.")
    st.session_state.stage = 'results'; st.session_state.animation_step = 0; st.session_state.is_playing = False; st.rerun()

# --- STAGE 3: RESULTS ---
def results_stage():
    mission_plan = st.session_state.mission_plan
    if not mission_plan:
        st.error("An error occurred during planning. Returning to setup.")
        st.session_state.stage = 'setup'
        st.rerun()
        return

    st.header(f"Mission Plan (Optimized for {st.session_state.opt_preference})")
    st.subheader(f"Pathfinding Solver: {st.session_state.path_solver_choice}")

    col1, col2 = st.columns([1, 2.5])
    with col1:
        st.subheader("Drone Schedules & Stats")
        if not mission_plan['assignments'] or not any(mission_plan['assignments'].values()):
            st.warning("No solution found or tasks could be assigned.")
            st.metric("Total Time (Makespan)", "0.00 s")
            st.metric("Total Energy", "0.00 Wh")
        else:
            st.metric("Total Time (Makespan)", f"{mission_plan['total_time']:.2f} s")
            st.metric("Total Energy", f"{mission_plan['total_energy']:.2f} Wh")
            for drone_id, tasks in mission_plan['assignments'].items():
                with st.expander(f"**Drone {drone_id} Schedule**"):
                    if not tasks: st.write("Idle")
                    else:
                        for task in tasks: st.info(f"Order {task['order_id']} ({task['start_time']:.1f}s → {task['end_time']:.1f}s)")

        st.subheader("Animation Controls")
        if st.button("▶️ Play / ⏸️ Pause", use_container_width=True): st.session_state.is_playing = not st.session_state.is_playing
        if st.button("🔁 Restart", use_container_width=True):
            st.session_state.animation_step = 0; st.session_state.is_playing = False; st.rerun()
        if st.button("⬅️ New Mission", use_container_width=True):
            st.session_state.stage = 'setup'; st.rerun()

    plot_placeholder = st.empty()
    progress_placeholder = st.empty()

    def draw_scene(animation_step):
        fig = go.Figure()
        env = mission_plan['env']
        hub_loc = config.HUB_LOCATION
        fig.add_trace(go.Scatter3d(x=[hub_loc[0]], y=[hub_loc[1]], z=[hub_loc[2]], mode='markers', marker=dict(size=10, color='blue', symbol='diamond'), name='Hub'))
        if env.orders:
            order_locs = [o.location for o in env.orders]
            fig.add_trace(go.Scatter3d(x=[loc[0] for loc in order_locs], y=[loc[1] for loc in order_locs], z=[loc[2] for loc in order_locs], mode='markers+text', text=[f"O{o.id}" for o in env.orders], textposition='middle right', marker=dict(size=8, color='green'), name='Orders'))
        for b in env.buildings: fig.add_trace(create_box(b))

        max_time = mission_plan.get('total_time', 0)
        current_time = (animation_step / 100) * max_time if max_time > 0 else 0

        live_drones_pos = []
        if 'full_paths' in mission_plan:
            for drone_id, drone_path in mission_plan['full_paths'].items():
                pos = config.HUB_LOCATION
                if drone_path:
                    segment = next((s for s in drone_path if s['start_time'] <= current_time < s['end_time']), None)
                    if segment and segment.get('path'):
                        duration = segment.get('duration', 0)
                        if duration > 0:
                            progress = (current_time - segment['start_time']) / duration
                            path_len = len(segment['path'])
                            idx = min(int(progress * (path_len - 1)), path_len - 1)
                            pos = segment['path'][idx]
                    elif drone_path and current_time >= drone_path[-1]['end_time']:
                        pos = config.HUB_LOCATION
                live_drones_pos.append(pos)

        if live_drones_pos:
            live_drones_pos_arr = np.array(live_drones_pos)
            fig.add_trace(go.Scatter3d(x=live_drones_pos_arr[:, 0], y=live_drones_pos_arr[:, 1], z=live_drones_pos_arr[:, 2], mode='markers', marker=dict(size=10, color='red', symbol='circle-open'), name='Live Drones'))

        colors = ['yellow', 'cyan', 'magenta', 'lime', 'white']
        if 'full_paths' in mission_plan:
            for i, (drone_id, drone_path) in enumerate(mission_plan['full_paths'].items()):
                if not drone_path: continue
                valid_segments = [seg['path'] for seg in drone_path if seg.get('path')]
                if not valid_segments: continue
                full_3d_path = np.concatenate(valid_segments)
                fig.add_trace(go.Scatter3d(x=full_3d_path[:,0], y=full_3d_path[:,1], z=full_3d_path[:,2], mode='lines', line=dict(width=3, color=colors[i % len(colors)]), name=f'Drone {drone_id} Path'))

        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), showlegend=True,
                          scene=dict(xaxis_title='Longitude', yaxis_title='Latitude', zaxis_title='Altitude (m)',
                                     aspectratio=dict(x=1, y=1, z=0.3), bgcolor='rgb(20, 24, 54)'),
                          legend=dict(font=dict(color='white')))

        plot_placeholder.plotly_chart(fig, use_container_width=True, height=700)
        progress_placeholder.progress(animation_step, text=f"Simulation Time: {current_time:.2f}s / {max_time:.2f}s")

    if st.session_state.is_playing:
        if st.session_state.animation_step < 100:
            st.session_state.animation_step += 1
        else:
            st.session_state.is_playing = False
        draw_scene(st.session_state.animation_step)
        time.sleep(0.05)
        st.rerun()
    else:
        draw_scene(st.session_state.animation_step)

# --- App Router ---
with st.sidebar:
    st.header("Operations Log")
    log_container = st.container(height=400)
    with log_container:
        for entry in st.session_state.log:
            st.text(entry)

if st.session_state.stage == 'setup': setup_stage()
elif st.session_state.stage == 'planning': planning_stage()
elif st.session_state.stage == 'results': results_stage()