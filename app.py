import streamlit as st
import plotly.graph_objects as go
import time
import pandas as pd
import numpy as np

from config import *
from environment import *
from ml_predictor.predictor import *
from utils.coordinate_manager import *
from planners.cbsh_planner import CBSHPlanner # MODIFIED: Using the new CBSH planner
from planners.single_agent_planner import *
from fleet.manager import *
from fleet.cbs_components import *

# --- State Management ---
def initialize_state():
    defaults = {
        'stage': 'setup',
        'log': [],
        'drones': {}, # state of each drone
        'missions': {}, # mission objectives
        'fleet_manager': None,
        'simulation_running': False,
        'simulation_time': 0.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_simulation():
    log = st.session_state.get('log', [])
    for key in list(st.session_state.keys()):
        if key not in ['log', '_global_planner_objects']:
            del st.session_state[key]
    initialize_state()
    st.session_state.log = log
    log_event("Simulation reset. Ready for new fleet planning.")

# --- Helper Functions ---
@st.cache_resource
def load_global_planners():
    log_event("Loading environment and global planners...")
    env = Environment(WeatherSystem())
    predictor = EnergyTimePredictor()
    coord_manager = CoordinateManager()
    # MODIFIED: Instantiate the new CBSHPlanner
    cbsh_planner = CBSHPlanner(env, coord_manager)
    single_agent_planner = SingleAgentPlanner(env, predictor, coord_manager)
    log_event("✅ Planners ready.")
    return {
        "env": env,
        "predictor": predictor,
        "coord_manager": coord_manager,
        "cbs_planner": cbsh_planner, # Key remains the same for compatibility
        "single_agent_planner": single_agent_planner
    }

def log_event(m):
    st.session_state.log.insert(0, f"{time.strftime('%H:%M:%S')} - {m}")

def create_box(b):
    x, y, h = b.center_xy[0], b.center_xy[1], b.height
    dx, dy = b.size_xy[0]/2, b.size_xy[1]/2
    return go.Mesh3d(x=[x-dx,x+dx,x+dx,x-dx,x-dx,x+dx,x+dx,x-dx], y=[y-dy,y-dy,y+dy,y+dy,y-dy,y-dy,y+dy,y+dy], z=[0,0,0,0,h,h,h,h], i=[7,0,0,0,4,4,6,6,4,0,3,2],j=[3,4,1,2,5,6,5,2,0,1,6,3],k=[0,7,2,3,6,7,2,5,1,2,5,6], color='grey', opacity=0.7, name='Building', hoverinfo='none')

def create_nfz_box(z, c='red', o=0.15):
    return go.Mesh3d(x=[z[0],z[2],z[2],z[0],z[0],z[2],z[2],z[0]], y=[z[1],z[1],z[3],z[3],z[1],z[1],z[3],z[3]], z=[0,0,0,0,MAX_ALTITUDE,MAX_ALTITUDE,MAX_ALTITUDE,MAX_ALTITUDE], color=c, opacity=o, name='No-Fly Zone', hoverinfo='name')

# --- UI Stages ---
def setup_stage():
    st.header("1. Fleet Mission Setup")
    if 'num_drones' not in st.session_state: st.session_state.num_drones = 2
    
    st.number_input("Number of Drones", min_value=1, max_value=5, key='num_drones')
    
    cols = st.columns(st.session_state.num_drones)
    hubs_list = list(HUBS.keys())
    dests_list = list(DESTINATIONS.keys())

    for i in range(st.session_state.num_drones):
        with cols[i]:
            st.subheader(f"Drone {i+1}")
            hub_key = f"drone_{i}_hub"
            dest_key = f"drone_{i}_dest"
            payload_key = f"drone_{i}_payload"
            st.selectbox("Start Hub", hubs_list, index=i % len(hubs_list), key=hub_key)
            st.selectbox("Destination", dests_list, index=(i+3) % len(dests_list), key=dest_key)
            st.slider("Payload (kg)", 0.1, DRONE_MAX_PAYLOAD_KG, 1.5, 0.1, key=payload_key)

    st.header("2. Optimization Weights")
    c1, c2, c3 = st.columns(3)
    st.session_state.w_time = c1.slider("Time Weight", 0.0, 1.0, 0.5)
    st.session_state.w_energy = c2.slider("Energy Weight", 0.0, 1.0, 0.2)
    st.session_state.w_risk = c3.slider("Risk Aversion", 0.0, 1.0, 0.1)

    if st.button("🚀 Plan Fleet Mission", type="primary", use_container_width=True):
        planners = st.session_state._global_planner_objects
        st.session_state.fleet_manager = FleetManager(planners['cbs_planner'])
        
        for i in range(st.session_state.num_drones):
            drone_id = f"Drone {i+1}"
            hub_name = st.session_state[f"drone_{i}_hub"]
            dest_name = st.session_state[f"drone_{i}_dest"]
            start_pos = HUBS[hub_name]
            goal_pos = DESTINATIONS[dest_name]
            payload = st.session_state[f"drone_{i}_payload"]
            
            mission = Mission(
                drone_id=drone_id,
                start_pos=start_pos,
                destinations=[goal_pos],
                payload_kg=payload,
                optimization_weights={
                    'w_time': st.session_state.w_time,
                    'w_energy': st.session_state.w_energy,
                    'w_risk': st.session_state.w_risk,
                }
            )
            st.session_state.fleet_manager.add_mission(mission)

            # Initialize drone state
            st.session_state.drones[drone_id] = {
                'pos': start_pos,
                'battery': DRONE_BATTERY_WH,
                'total_time': 0.0,
                'total_energy': 0.0
            }
        st.session_state.stage = 'planning'
        st.rerun()

def planning_stage():
    log_event("Fleet planning initiated...")
    fm = st.session_state.fleet_manager
    with st.spinner("CBSH planner coordinating conflict-free routes for the fleet..."):
        success = fm.execute_planning_cycle()
    
    if success:
        log_event("✅ Fleet plan found!")
        st.session_state.stage = 'simulation'
        st.rerun()
    else:
        st.error("Fleet planning failed. Check logs. Some missions may be impossible.")
        st.button("New Mission", on_click=reset_simulation)

def simulation_stage():
    st.header("Fleet Mission Simulation")
    c1, c2 = st.columns([1, 2.5])
    
    with c1:
        st.subheader("Mission Control")
        b1, b2 = st.columns(2)
        if b1.button("▶️ Run", disabled=st.session_state.simulation_running, use_container_width=True):
            st.session_state.simulation_running = True
            st.rerun()
        if b2.button("⏸️ Pause", disabled=not st.session_state.simulation_running, use_container_width=True):
            st.session_state.simulation_running = False
            st.rerun()

        st.subheader("Fleet Status")
        for drone_id, mission in st.session_state.fleet_manager.missions.items():
            with st.expander(f"{drone_id} ({mission.state})", expanded=True):
                # MODIFIED: Progress is now based on simulation time vs total planned time
                total_planned_time = mission.path[-1][1] if mission.path else 1
                sim_time = st.session_state.simulation_time
                progress = min(sim_time / total_planned_time, 1.0)
                st.progress(progress, text=f"Time: {sim_time:.0f}s / {total_planned_time:.0f}s")
                
                battery_rem = st.session_state.drones[drone_id]['battery']
                st.metric("Battery", f"{battery_rem:.2f} Wh", delta_color="inverse")
        
        if st.button("⬅️ New Mission", use_container_width=True):
            reset_simulation()
            st.rerun()

    with c2:
        fig = go.Figure()
        env = st.session_state._global_planner_objects['env']
        for b in env.buildings: fig.add_trace(create_box(b))
        for nfz in env.static_nfzs: fig.add_trace(create_nfz_box(nfz, c='red'))

        drone_colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (drone_id, mission) in enumerate(st.session_state.fleet_manager.missions.items()):
            color = drone_colors[i % len(drone_colors)]
            # Path for rendering is now in path_world_coords
            path_np = np.array(mission.path_world_coords)
            if path_np.any():
                fig.add_trace(go.Scatter3d(x=path_np[:,0], y=path_np[:,1], z=path_np[:,2], mode='lines', line=dict(color=color, width=4), name=f'{drone_id} Path'))
            
            drone_pos = st.session_state.drones[drone_id]['pos']
            fig.add_trace(go.Scatter3d(x=[drone_pos[0]], y=[drone_pos[1]], z=[drone_pos[2]], mode='markers', marker=dict(size=8, color=color, symbol='cross'), name=drone_id))

        fig.update_layout(margin=dict(l=0,r=0,b=0,t=0), scene=dict(xaxis_title='Lon',yaxis_title='Lat',zaxis_title='Alt (m)',aspectmode='cube'), legend=dict(y=0.99,x=0.01))
        st.plotly_chart(fig, use_container_width=True)

    with c1:
        st.subheader("Event Log")
        st.dataframe(pd.DataFrame(st.session_state.log, columns=["Log Entry"]), height=200, use_container_width=True)


# --- Main Application Logic ---
def main():
    st.set_page_config(layout="wide", page_title="Multi-Agent Drone Planner")
    st.title("🚁 Multi-Agent CBSH Fleet Planner")

    if 'stage' not in st.session_state:
        initialize_state()
    
    st.session_state._global_planner_objects = load_global_planners()

    if st.session_state.stage == 'setup':
        setup_stage()
    elif st.session_state.stage == 'p