import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from environment import Environment, WeatherSystem
from ml_predictor.predictor import EnergyTimePredictor
from path_planner import PathPlanner3D

@st.cache_resource
def load_planner():
    log_event("Loading hybrid planner...")
    env = Environment(WeatherSystem())
    predictor = EnergyTimePredictor()
    planner = PathPlanner3D(env, predictor)
    log_event("Planner loaded.")
    return planner

# Session state defaults
defaults = {
    'stage': 'setup',
    'log': [],
    'mission_running': False,
    'planned_path': None,
    'planned_path_np': None,
    'total_time': 0.0,
    'total_energy': 0.0,
    'drone_pos': None,
    'path_index': 0,
    'initial_payload': 0.0,
    'current_subgoal_index': 0
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def log_event(m):
    st.session_state.log.insert(0, f"{time.strftime('%H:%M:%S')} - {m}")

def reset_mission_state():
    for k, v in defaults.items():
        if k not in ['stage', 'log']:
            st.session_state[k] = v
    st.session_state.stage = 'setup'
    planner.env.dynamic_nfzs = []
    planner.env.event_triggered = False

planner = load_planner()
st.set_page_config(layout="wide")
st.title("🚁 Q-DOP: Hybrid Hierarchical Planner (JPS+D* Lite)")

def create_box(b):
    """Create 3D box visualization for buildings."""
    x, y = b.center_xy
    dx, dy = b.size_xy[0]/2, b.size_xy[1]/2
    h = b.height
    x_c = [x-dx, x+dx, x+dx, x-dx, x-dx, x+dx, x+dx, x-dx]
    y_c = [y-dy, y-dy, y+dy, y+dy, y-dy, y-dy, y+dy, y+dy]
    z_c = [0, 0, 0, 0, h, h, h, h]
    return go.Mesh3d(
        x=x_c, y=y_c, z=z_c,
        i=[7,0,0,0,4,4,6,6,4,0,3,2],
        j=[3,4,1,2,5,6,5,2,0,1,6,3],
        k=[0,7,2,3,6,7,2,5,1,2,5,6],
        color='grey', opacity=0.7, name='Building', hoverinfo='none'
    )

def create_nfz_box(z, c='red', o=0.15):
    """Create 3D box visualization for no-fly zones."""
    x_c = [z[0], z[2], z[2], z[0], z[0], z[2], z[2], z[0]]
    y_c = [z[1], z[1], z[3], z[3], z[1], z[1], z[3], z[3]]
    z_c = [0, 0, 0, 0, config.MAX_ALTITUDE, config.MAX_ALTITUDE, config.MAX_ALTITUDE, config.MAX_ALTITUDE]
    return go.Mesh3d(
        x=x_c, y=y_c, z=z_c,
        color=c, opacity=o, name='No-Fly Zone', hoverinfo='name'
    )

def check_replanning_triggers():
    """Check if replanning is needed due to new obstacles."""
    if planner.env.was_nfz_just_added:
        log_event("🚨 New NFZ detected! Checking path...")
        return True
    return False

def setup_stage():
    """Mission setup interface."""
    st.header("1. Mission Parameters")
    c1, c2 = st.columns(2)
    
    with c1:
        st.session_state.hub_location = config.HUBS[st.selectbox("Hub", list(config.HUBS.keys()))]
        st.session_state.destination = config.DESTINATIONS[st.selectbox("Destination", list(config.DESTINATIONS.keys()))]
    
    with c2:
        st.session_state.payload_kg = st.slider("Payload (kg)", 0.1, config.DRONE_MAX_PAYLOAD_KG, 1.5, 0.1)
    
    st.subheader("2. Optimization Priority")
    st.session_state.optimization_preference = st.radio(
        "Optimize For:", 
        ["Balanced", "Fastest Path", "Most Battery Efficient"], 
        horizontal=True, index=0
    )
    
    if st.session_state.optimization_preference == "Balanced":
        st.session_state.balance_weight = st.slider(
            "Priority", 0.0, 1.0, 0.5, 0.05, format="%.2f", 
            help="0.0=Energy, 1.0=Time"
        )
    
    if st.button("🚀 Plan Mission", type="primary", use_container_width=True):
        st.session_state.stage = 'planning'
        st.rerun()

def planning_stage():
    """Path planning execution."""
    mode = st.session_state.optimization_preference
    log_event(f"Planning initial mission ('{mode}') with JPS...")
    
    with st.spinner("Hierarchical Planner running..."):
        mode_map = {
            "Fastest Path": "time",
            "Most Battery Efficient": "energy", 
            "Balanced": "balanced"
        }
        selected_mode = mode_map[mode]
        balance_weight = st.session_state.get('balance_weight', 0.5)
        
        hub, destination = st.session_state.hub_location, st.session_state.destination
        payload = st.session_state.initial_payload = st.session_state.payload_kg
        takeoff = (hub[0], hub[1], config.TAKEOFF_ALTITUDE)
        
        path, status = planner.find_path(takeoff, destination, payload, selected_mode, balance_weight)
        
        if path is None:
            st.error(f"Path planning failed: {status}")
            st.button("New", on_click=reset_mission_state)
            return
        
        full_path = [hub] + path
        st.session_state.planned_path = full_path
        st.session_state.planned_path_np = np.array(full_path)
        st.session_state.drone_pos = full_path[0]
        st.session_state.path_index = 0
        log_event("✅ Initial mission plan found.")
    
    st.session_state.stage = 'simulation'
    st.rerun()

def simulation_stage():
    """Mission simulation interface."""
    st.header(f"Mission Simulation ({st.session_state.optimization_preference})")
    c1, c2 = st.columns([1, 2.5])
    
    with c1:
        st.subheader("Mission Control")
        b1, b2 = st.columns(2)
        
        if b1.button("▶️ Run", use_container_width=True, disabled=st.session_state.mission_running):
            st.session_state.mission_running = True
            st.rerun()
            
        if b2.button("⏸️ Pause", use_container_width=True, disabled=not st.session_state.mission_running):
            st.session_state.mission_running = False
            st.rerun()
        
        # Progress tracking
        progress = (st.session_state.path_index) / (len(st.session_state.planned_path) - 1) if len(st.session_state.planned_path) > 1 else 0
        st.progress(progress, text=f"Progress: {progress:.0%}")
        
        # Battery status
        remaining_battery = config.DRONE_BATTERY_WH - st.session_state.total_energy
        st.metric("Battery", f"{remaining_battery:.2f} Wh", delta=f"{-st.session_state.total_energy:.2f} Wh used")
        
        # Mission completion check
        if progress >= 1 and st.session_state.planned_path:
            st.success("✅ Mission Complete!")
            st.session_state.mission_running = False
        
        if st.button("⬅️ New Mission", use_container_width=True):
            reset_mission_state()
            st.rerun()
    
    with c2:
        # 3D Visualization
        fig = go.Figure()
        
        hub, dest = st.session_state.hub_location, st.session_state.destination
        
        # Add hubs
        for name, h in config.HUBS.items():
            fig.add_trace(go.Scatter3d(
                x=[h[0]], y=[h[1]], z=[h[2]],
                mode='markers',
                marker=dict(size=8, color='cyan', symbol='diamond'),
                name='Hub'
            ))
        
        # Add destination
        fig.add_trace(go.Scatter3d(
            x=[dest[0]], y=[dest[1]], z=[dest[2]],
            mode='markers+text',
            text=["Dest."],
            marker=dict(size=8, color='lime'),
            name='Destination'
        ))
        
        # Add buildings
        for b in planner.env.buildings:
            fig.add_trace(create_box(b))
        
        # Add no-fly zones
        for nfz in planner.env.static_nfzs:
            fig.add_trace(create_nfz_box(nfz))
        
        for dnfz in planner