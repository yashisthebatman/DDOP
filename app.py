# FILE: app.py

import streamlit as st
import plotly.graph_objects as go
import time
import pandas as pd
import numpy as np
from itertools import cycle
import uuid
import logging

from config import *
import system_state
from environment import Environment, WeatherSystem
from ml_predictor.predictor import EnergyTimePredictor
from utils.coordinate_manager import CoordinateManager
from planners.cbsh_planner import CBSHPlanner
from fleet.manager import FleetManager, Mission
from fleet.cbs_components import Agent
# NEW IMPORTS
from dispatch.vrp_solver import VRPSolver
from dispatch.dispatcher import Dispatcher

# --- Helper & UI Functions ---

def log_event(state, message):
    """Adds a new message to the persistent event log."""
    state['log'].insert(0, f"{time.strftime('%H:%M:%S')} - {message}")

@st.cache_resource
def load_global_planners():
    """
    Initializes and caches the core, heavy objects for planning and simulation.
    This runs only once per session.
    """
    log_event(system_state.load_state(), "Loading environment and global planners...")
    coord_manager = CoordinateManager()
    env = Environment(WeatherSystem(), coord_manager)
    predictor = EnergyTimePredictor()
    cbsh_planner = CBSHPlanner(env, coord_manager)
    fleet_manager = FleetManager(cbsh_planner, predictor)
    # NEW: Initialize dispatcher components
    vrp_solver = VRPSolver(predictor)
    dispatcher = Dispatcher(vrp_solver)
    log_event(system_state.load_state(), "✅ Planners and Dispatcher ready.")
    return {
        "env": env,
        "predictor": predictor,
        "coord_manager": coord_manager,
        "fleet_manager": fleet_manager,
        "dispatcher": dispatcher # NEW
    }

def render_fleet_table(state):
    """Displays a real-time table of the entire drone fleet's status."""
    drones_data = []
    for drone_id, drone in state['drones'].items():
        battery_percent = (drone['battery'] / DRONE_BATTERY_WH) * 100

        # Determine location string
        if drone['status'] in ['IDLE', 'RECHARGING']:
            location_str = drone.get('home_hub', 'Unknown Hub')
        else:
            pos = drone.get('pos', (0, 0, 0))
            location_str = f"({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.1f}m)"

        # Determine current payload
        payload_kg = 0.0
        mission_id = drone.get('mission_id')
        if mission_id and mission_id in state['active_missions']:
            payload_kg = state['active_missions'][mission_id].get('payload_kg', 0.0)

        drones_data.append({
            "ID": drone_id,
            "Status": drone['status'],
            "Location": location_str,
            "Battery": battery_percent,
            "Payload (kg)": f"{payload_kg:.1f}",
            "Mission ID": mission_id or "—"
        })

    if not drones_data:
        st.info("No drones found in system state.")
        return

    df = pd.DataFrame(drones_data)
    st.subheader("🚀 Fleet Status")
    
    # Use st.columns to create a custom, better-looking table
    header_cols = st.columns((2, 2, 4, 3, 2, 3))
    headers = ["Drone ID", "Status", "Location", "Battery (%)", "Payload", "Mission ID"]
    for col, header in zip(header_cols, headers):
        col.markdown(f"**{header}**")

    st.markdown("---")

    for index, row in df.iterrows():
        row_cols = st.columns((2, 2, 4, 3, 2, 3))
        row_cols[0].write(row['ID'])
        row_cols[1].write(row['Status'])
        row_cols[2].write(row['Location'])
        row_cols[3].progress(int(row['Battery']), text=f"{row['Battery']:.1f}%")
        row_cols[4].write(row['Payload (kg)'])
        row_cols[5].code(row['Mission ID'])

def update_simulation(state, fleet_manager):
    """Advances the simulation by one time step and updates drone/mission states."""
    state['simulation_time'] += SIMULATION_TIME_STEP

    # Update recharging drones
    for drone in state['drones'].values():
        if drone['status'] == 'RECHARGING' and state['simulation_time'] >= drone['available_at']:
            drone['status'] = 'IDLE'
            drone['battery'] = DRONE_BATTERY_WH
            log_event(state, f"✅ {drone['id']} has finished recharging and is now IDLE.")

    # Update active missions
    missions_to_complete = []
    for mission_id, mission in list(state['active_missions'].items()):
        drone_id = mission['drone_id']
        drone = state['drones'][drone_id]

        if drone['status'] != 'EN ROUTE': continue
        if mission.get('total_planned_time', 0) <= 0: continue

        progress = (state['simulation_time'] - mission['start_time']) / mission['total_planned_time']
        progress = min(progress, 1.0)

        # Update position based on path progress
        path = mission.get('path_world_coords', [])
        if path:
            path_index = int(progress * (len(path) - 1))
            if path_index < len(path) - 1:
                p1, p2 = np.array(path[path_index]), np.array(path[path_index + 1])
                segment_progress = (progress * (len(path) - 1)) - path_index
                drone['pos'] = tuple(p1 + segment_progress * (p2 - p1))
            else:
                drone['pos'] = path[-1]

        # Update battery
        energy_consumed = progress * mission.get('total_planned_energy', 0)
        drone['battery'] = mission.get('start_battery', DRONE_BATTERY_WH) - energy_consumed

        if progress >= 1.0:
            missions_to_complete.append(mission_id)

    # Process completed missions
    for mission_id in missions_to_complete:
        mission = state['active_missions'][mission_id]
        drone_id = mission['drone_id']
        drone = state['drones'][drone_id]
        
        log_event(state, f"🏁 {drone_id} completed mission {mission_id}.")
        drone['status'] = 'RECHARGING'
        drone['mission_id'] = None
        drone['available_at'] = state['simulation_time'] + DRONE_RECHARGE_TIME_S
        
        for order_id in mission['order_ids']: state['completed_orders'].append(order_id)
        del state['active_missions'][mission_id]

    # NEW: Find and plan missions for drones in 'PLANNING' state
    success, result = fleet_manager.plan_pending_missions(state)
    if not success:
        log_event(state, f"❌ Fleet planning cycle failed: {result.get('error', 'Unknown error')}")

def render_map(state, planners):
    """Renders the 3D Plotly map of the simulation area."""
    fig = go.Figure()
    env = planners['env']

    # Add Hubs and Destinations
    hubs_lon, hubs_lat, hubs_alt = zip(*HUBS.values())
    dests_lon, dests_lat, dests_alt = zip(*[o['pos'] for o in state['pending_orders'].values()]) if state['pending_orders'] else ([], [], [])
    fig.add_trace(go.Scatter3d(x=hubs_lon, y=hubs_lat, z=hubs_alt, mode='markers', marker=dict(size=8, color='green', symbol='diamond'), name='Hubs', text=list(HUBS.keys()), hoverinfo='text'))
    if dests_lon:
        fig.add_trace(go.Scatter3d(x=dests_lon, y=dests_lat, z=dests_alt, mode='markers', marker=dict(size=6, color='purple', symbol='square'), name='Pending Orders', text=list(state['pending_orders'].keys()), hoverinfo='text'))

    # Add Buildings and No-Fly Zones
    for b in env.buildings:
        x, y, h = b.center_xy[0], b.center_xy[1], b.height
        dx, dy = b.size_xy[0] / 2, b.size_xy[1] / 2
        fig.add_trace(go.Mesh3d(x=[x-dx, x+dx, x+dx, x-dx, x-dx, x+dx, x+dx, x-dx], y=[y-dy, y-dy, y+dy, y+dy, y-dy, y-dy, y+dy, y+dy], z=[0,0,0,0,h,h,h,h], i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6], color='grey', opacity=0.7, name='Building'))
    for nfz in env.static_nfzs:
        fig.add_trace(go.Mesh3d(x=[nfz[0],nfz[2],nfz[2],nfz[0]]*2, y=[nfz[1],nfz[1],nfz[3],nfz[3]]*2, z=[0,0,0,0,MAX_ALTITUDE,MAX_ALTITUDE,MAX_ALTITUDE,MAX_ALTITUDE], color='red', opacity=0.15, name='No-Fly Zone'))

    # Add Drone positions and paths
    drone_colors = cycle(['red', 'blue', 'orange', 'magenta', 'cyan'])
    for drone_id, drone in state['drones'].items():
        color = next(drone_colors)
        if drone['status'] == 'EN ROUTE' and drone['mission_id'] in state['active_missions']:
            path = state['active_missions'][drone['mission_id']].get('path_world_coords', [])
            if path:
                path_np = np.array(path)
                fig.add_trace(go.Scatter3d(x=path_np[:,0], y=path_np[:,1], z=path_np[:,2], mode='lines', line=dict(color=color, width=4), name=f'{drone_id} Path'))

        fig.add_trace(go.Scatter3d(x=[drone['pos'][0]], y=[drone['pos'][1]], z=[drone['pos'][2]], mode='markers', marker=dict(size=8, color=color, symbol='cross'), name=drone_id))

    fig.update_layout(margin=dict(l=0,r=0,b=0,t=0), scene=dict(xaxis_title='Lon',yaxis_title='Lat',zaxis_title='Alt (m)',aspectmode='data'), legend=dict(y=0.99,x=0.01))
    st.plotly_chart(fig, use_container_width=True)

# --- Main Application Logic ---
def main():
    st.set_page_config(layout="wide", page_title="Drone Delivery Simulator")
    st.title("🚁 Continuous Drone Delivery Simulator")
    
    # --- Load persistent state and planners ---
    state = system_state.load_state()
    planners = load_global_planners()
    fleet_manager = planners['fleet_manager']
    dispatcher = planners['dispatcher']

    # --- Sidebar for Controls ---
    with st.sidebar:
        st.header("Master Control")
        sim_running = state.get('simulation_running', False)
        
        if st.button("▶️ Run", disabled=sim_running, use_container_width=True):
            state['simulation_running'] = True
            log_event(state, "Simulation started.")
            st.rerun()

        if st.button("⏸️ Pause", disabled=not sim_running, use_container_width=True):
            state['simulation_running'] = False
            log_event(state, "Simulation paused.")
            st.rerun()
        
        st.metric("Simulation Time", f"{state['simulation_time']:.1f}s")
        
        if st.button("⚠️ Reset Simulation State", use_container_width=True, type="secondary"):
            state = system_state.reset_state_file()
            log_event(state, "--- SYSTEM STATE RESET ---")
            st.rerun()

    # --- Main Screen Layout ---
    render_fleet_table(state)
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        # --- NEW: UI for adding orders ---
        st.subheader("📦 Add New Delivery Order")
        with st.form("new_order_form", clear_on_submit=True):
            dest_name = st.selectbox("Destination", list(DESTINATIONS.keys()))
            payload = st.slider("Payload (kg)", 0.1, DRONE_MAX_PAYLOAD_KG, 1.0, 0.1)
            
            submitted = st.form_submit_button("Add to Delivery Queue", type="primary", use_container_width=True)
            if submitted:
                order_id = f"{dest_name.replace(' ', '')}-{uuid.uuid4().hex[:4]}"
                if order_id in state['pending_orders']:
                    st.error("An order for this destination already exists.")
                else:
                    state['pending_orders'][order_id] = {
                        'id': order_id,
                        'pos': DESTINATIONS[dest_name],
                        'payload_kg': payload
                    }
                    log_event(state, f"📥 New order added: {order_id} ({payload}kg).")
                    st.rerun()

        st.subheader("📦 Pending Delivery Orders")
        pending_orders_list = list(state['pending_orders'].values())
        if not pending_orders_list:
            st.info("No pending orders.")
        else:
            # Display pending orders as a simple list
            df_pending = pd.DataFrame(pending_orders_list)
            st.dataframe(df_pending[['id', 'payload_kg']], use_container_width=True, hide_index=True)

        st.subheader("📋 Event Log")
        st.dataframe(pd.DataFrame(state['log'], columns=["Log Entry"]), height=300, use_container_width=True)

    with col2:
        st.subheader("🌐 3D Operations Map")
        render_map(state, planners)

    # --- Simulation Update & Persistence ---
    if state.get('simulation_running', False):
        # NEW: Run dispatcher before updating simulation
        dispatched = dispatcher.dispatch_missions(state)
        if dispatched:
            log_event(state, "🚚 Dispatcher created new missions.")

        update_simulation(state, fleet_manager)

    system_state.save_state(state)
    
    if state.get('simulation_running', False):
        time.sleep(SIMULATION_UI_REFRESH_INTERVAL)
        st.rerun()

if __name__ == "__main__":
    main()