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
from optimization.assignment_solver import AssignmentSolver
from path_planner import PathPlanner3D

# --- Helper Functions ---
def log_event(message):
    """Appends a timestamped message to the session log."""
    if 'log' not in st.session_state:
        st.session_state.log = []
    log_entry = f"{time.strftime('%H:%M:%S')} - {message}"
    st.session_state.log.insert(0, log_entry)

def create_box(building: Building):
    """Generates vertices for a 3D rectangular building."""
    x, y = building.center_xy
    dx, dy = building.size_xy[0] / 2, building.size_xy[1] / 2
    h = building.height
    
    x_coords = [x-dx, x+dx, x+dx, x-dx, x-dx, x+dx, x+dx, x-dx]
    y_coords = [y-dy, y-dy, y+dy, y+dy, y-dy, y-dy, y+dy, y+dy]
    z_coords = [0, 0, 0, 0, h, h, h, h]
    
    return go.Mesh3d(
        x=x_coords, y=y_coords, z=z_coords,
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 2, 5, 1, 2, 5, 6],
        color='grey', opacity=0.5, name='Building', hoverinfo='none'
    )

st.set_page_config(layout="wide")
st.title("🚁 Q-DOP: 3D Drone Mission Planner")

# --- Session State Initialization ---
for key, default_value in [('stage', 'setup'), ('mission_plan', None), 
                           ('animation_step', 0), ('is_playing', False), ('log', [])]:
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- STAGE 1: SETUP ---
def setup_stage():
    st.header("1. Mission Setup")
    log_event("App initialized. Waiting for mission setup.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Drone Fleet")
        num_drones = st.slider("Number of Drones", 1, 10, config.DEFAULT_NUM_DRONES, 1)
        st.session_state.num_drones = num_drones
    with col2:
        st.subheader("Package Manifest")
        num_orders = st.slider("Number of Packages", 1, 20, 5, 1)
        orders = [Order(id=i, location=(np.random.uniform(config.AREA_BOUNDS[0], config.AREA_BOUNDS[2]), np.random.uniform(config.AREA_BOUNDS[1], config.AREA_BOUNDS[3]), np.random.uniform(50, 150)), payload_kg=round(np.random.uniform(0.5, config.DRONE_MAX_PAYLOAD_KG), 2)) for i in range(num_orders)]
        st.session_state.orders = orders
        df = pd.DataFrame([{'ID': o.id, 'Payload (kg)': o.payload_kg, 'Location': f"{o.location[0]:.4f}, {o.location[1]:.4f}"} for o in orders])
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.header("2. Planning Parameters")
    solver_choice = st.radio("Choose Solver", ["⭐ Google OR-Tools (Classical)", "💡 D-Wave QUBO (Quantum-Inspired)"], horizontal=True)
    opt_preference = st.radio("Choose Priority", ["Fastest Delivery", "Fuel Efficient"], horizontal=True)
    
    if st.button("🚀 Plan Mission", type="primary", use_container_width=True):
        st.session_state.solver_choice, st.session_state.opt_preference = solver_choice, opt_preference
        st.session_state.stage = 'planning'; st.rerun()

# --- STAGE 2: PLANNING (Backend Logic) ---
def planning_stage():
    log_event("Starting mission planning...")
    with st.spinner("Executing Mission Plan..."):
        log_event(f"Solver: {st.session_state.solver_choice} | Priority: {st.session_state.opt_preference}")
        env = Environment(st.session_state.num_drones, st.session_state.orders)
        log_event("Environment with drones, orders, and buildings generated.")
        predictor = EnergyTimePredictor()
        path_planner = PathPlanner3D(env, predictor)
        solver = AssignmentSolver(env, predictor, path_planner)
        weights = {'time': 0.9, 'energy': 0.1} if st.session_state.opt_preference == "Fastest Delivery" else {'time': 0.1, 'energy': 0.9}
        
        if "OR-Tools" in st.session_state.solver_choice:
            st.session_state.mission_plan = solver.solve_with_or_tools(weights)
        else:
            st.session_state.mission_plan = solver.solve_with_qubo(weights)
        
        log_event("Mission plan generated successfully.")
    st.session_state.stage = 'results'; st.session_state.animation_step = 0; st.session_state.is_playing = False; st.rerun()

# --- STAGE 3: RESULTS & VISUALIZATION ---
def results_stage():
    mission_plan = st.session_state.mission_plan
    st.header(f"Mission Plan (Optimized for {st.session_state.opt_preference})")
    st.subheader(f"Solver Used: {st.session_state.solver_choice}")

    col1, col2 = st.columns([1, 2.5])
    with col1:
        st.subheader("Drone Schedules & Stats")
        if not mission_plan or not mission_plan['assignments']:
            st.warning("No solution found or no tasks assigned.")
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

    plot_placeholder = st.empty()
    progress_placeholder = st.empty()

    def draw_scene(animation_step):
        fig = go.Figure()
        env = mission_plan['env']
        
        hub_loc = config.HUB_LOCATION
        fig.add_trace(go.Scatter3d(x=[hub_loc[0]], y=[hub_loc[1]], z=[hub_loc[2]], mode='markers', marker=dict(size=10, color='blue', symbol='diamond'), name='Hub'))
        
        order_locs = [o.location for o in env.orders]
        fig.add_trace(go.Scatter3d(x=[loc[0] for loc in order_locs], y=[loc[1] for loc in order_locs], z=[loc[2] for loc in order_locs], mode='markers+text', text=[f"O{o.id}" for o in env.orders], textposition='middle right', marker=dict(size=8, color='green'), name='Orders'))
        
        for b in env.buildings: fig.add_trace(create_box(b))
        
        max_time = mission_plan['total_time']
        current_time = (animation_step / 100) * max_time if max_time > 0 else 0
        
        live_drones_pos = []
        for drone_id, drone_path in mission_plan['full_paths'].items():
            pos = config.HUB_LOCATION
            if drone_path:
                segment = next((s for s in drone_path if s['start_time'] <= current_time < s['end_time']), None)
                if segment:
                    progress = (current_time - segment['start_time']) / segment['duration']
                    idx = min(int(progress * (len(segment['path']) - 1)), len(segment['path']) - 1)
                    pos = segment['path'][idx]
                elif current_time >= drone_path[-1]['end_time']: pos = config.HUB_LOCATION
            live_drones_pos.append(pos)
        
        # --- THIS IS THE FIX ---
        # Only try to plot live drones if the list is not empty
        if live_drones_pos:
            live_drones_pos = np.array(live_drones_pos)
            fig.add_trace(go.Scatter3d(x=live_drones_pos[:, 0], y=live_drones_pos[:, 1], z=live_drones_pos[:, 2], mode='markers', marker=dict(size=10, color='red', symbol='circle-open'), name='Live Drones'))
        # --- END OF FIX ---

        colors = ['yellow', 'cyan', 'magenta', 'lime', 'white']
        for i, (drone_id, drone_path) in enumerate(mission_plan['full_paths'].items()):
            if not drone_path: continue
            full_3d_path = np.concatenate([seg['path'] for seg in drone_path])
            fig.add_trace(go.Scatter3d(x=full_3d_path[:,0], y=full_3d_path[:,1], z=full_3d_path[:,2], mode='lines', line=dict(width=3, color=colors[i % len(colors)]), name=f'Drone {drone_id} Path'))

        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), showlegend=True,
                          scene=dict(xaxis_title='Longitude', yaxis_title='Latitude', zaxis_title='Altitude (m)',
                                     aspectratio=dict(x=1, y=1, z=0.3), bgcolor='rgb(20, 24, 54)'),
                          legend=dict(font=dict(color='white')))
        
        with plot_placeholder:
            st.plotly_chart(fig, use_container_width=True, height=700)
        with progress_placeholder:
            st.progress(animation_step, text=f"Simulation Time: {current_time:.2f}s / {max_time:.2f}s")
    
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

# --- App Router and Log Display ---
with st.sidebar:
    st.header("Operations Log")
    log_container = st.container(height=300)
    with log_container:
        for entry in st.session_state.log:
            st.text(entry)

if st.session_state.stage == 'setup': setup_stage()
elif st.session_state.stage == 'planning': planning_stage()
elif st.session_state.stage == 'results': results_stage()