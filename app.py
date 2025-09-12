# FILE: app.py

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import time
import pandas as pd
import numpy as np
from itertools import cycle
import uuid
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from config import *
import system_state
from environment import Environment, WeatherSystem
from ml_predictor.predictor import EnergyTimePredictor
from utils.coordinate_manager import CoordinateManager
from planners.cbsh_planner import CBSHPlanner
from fleet.manager import FleetManager, Mission
from dispatch.vrp_solver import VRPSolver
from dispatch.dispatcher import Dispatcher
# Import the new retrainer module
import training.retrainer as retrainer

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
    state = system_state.load_state()
    log_event(state, "Loading environment and global planners...")
    coord_manager = CoordinateManager()
    env = Environment(WeatherSystem(), coord_manager)
    
    predictor = EnergyTimePredictor()
    predictor.load_model(state.get('active_model_path', MODEL_FILE_PATH))
    
    cbsh_planner = CBSHPlanner(env, coord_manager)
    fleet_manager = FleetManager(cbsh_planner, predictor)
    vrp_solver = VRPSolver(predictor)
    dispatcher = Dispatcher(vrp_solver)
    executor = ThreadPoolExecutor(max_workers=2) # For async planning
    log_event(state, "✅ Planners and Dispatcher ready.")
    return {
        "env": env,
        "predictor": predictor,
        "coord_manager": coord_manager,
        "fleet_manager": fleet_manager,
        "dispatcher": dispatcher,
        "executor": executor
    }

def update_simulation(state, fleet_manager):
    """Advances the simulation by one time step and updates drone/mission states."""
    state['simulation_time'] += SIMULATION_TIME_STEP

    for drone in state['drones'].values():
        if drone['status'] == 'RECHARGING' and state['simulation_time'] >= drone['available_at']:
            drone['status'] = 'IDLE'
            drone['battery'] = DRONE_BATTERY_WH
            log_event(state, f"✅ {drone['id']} has finished recharging and is now IDLE.")

    missions_to_complete = []
    for mission_id, mission in list(state['active_missions'].items()):
        drone_id = mission['drone_id']
        drone = state['drones'][drone_id]
        if drone['status'] != 'EN ROUTE' or mission.get('total_planned_time', 0) <= 0: continue
        
        progress = (state['simulation_time'] - mission['start_time']) / mission['total_planned_time']
        progress = min(progress, 1.0)

        path = mission.get('path_world_coords', [])
        if path:
            path_index = int(progress * (len(path) - 1))
            if path_index < len(path) - 1:
                p1, p2 = np.array(path[path_index]), np.array(path[path_index + 1])
                segment_progress = (progress * (len(path) - 1)) - path_index
                drone['pos'] = tuple(p1 + segment_progress * (p2 - p1))
            else:
                drone['pos'] = path[-1]
        
        energy_consumed = progress * mission.get('total_planned_energy', 0)
        drone['battery'] = mission.get('start_battery', DRONE_BATTERY_WH) - energy_consumed
        if progress >= 1.0: missions_to_complete.append(mission_id)

    for mission_id in missions_to_complete:
        mission = state['active_missions'][mission_id]
        drone_id = mission['drone_id']
        drone = state['drones'][drone_id]
        
        # --- PHASE 4: Create Analytics Log and Real-World Data ---
        actual_duration = state['simulation_time'] - mission['start_time']
        actual_energy = mission['start_battery'] - drone['battery']
        
        log_entry = {
            "mission_id": mission_id,
            "drone_id": drone_id,
            "completion_timestamp": state['simulation_time'],
            "planned_duration_sec": mission['total_planned_time'],
            "actual_duration_sec": actual_duration,
            "planned_energy_wh": mission['total_planned_energy'],
            "actual_energy_wh": actual_energy,
            "number_of_stops": len(mission['destinations']),
            "outcome": "Completed",
        }
        state['completed_missions_log'].append(log_entry)
        
        # Log data for MLOps feedback loop (as a single segment)
        try:
            temp_predictor = EnergyTimePredictor()
            features = temp_predictor._extract_features(
                mission['start_pos'], mission['destinations'][-1], 
                mission['payload_kg'], [0,0,0], None # NOTE: Wind is not captured in sim, using placeholder
            )
            features['actual_time'] = actual_duration
            features['actual_energy'] = actual_energy
            
            feedback_df = pd.DataFrame([features])
            csv_path = 'data/real_world_flight_segments.csv'
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            write_header = not os.path.exists(csv_path)
            feedback_df.to_csv(csv_path, mode='a', header=write_header, index=False)
        except Exception as e:
            log_event(state, f"⚠️ Could not log mission data for ML feedback: {e}")

        log_event(state, f"🏁 {drone_id} completed mission {mission_id}.")
        drone['status'] = 'RECHARGING'
        drone['mission_id'] = None
        drone['available_at'] = state['simulation_time'] + DRONE_RECHARGE_TIME_S
        for order_id in mission['order_ids']:
            if order_id not in state['completed_orders']:
                 state['completed_orders'].append(order_id)
        state['completed_missions'][mission_id] = mission
        del state['active_missions'][mission_id]

def render_map(state, planners):
    fig = go.Figure()
    env = planners['env']
    hubs_lon, hubs_lat, hubs_alt = zip(*HUBS.values())
    dests_lon, dests_lat, dests_alt = zip(*[o['pos'] for o in state['pending_orders'].values()]) if state['pending_orders'] else ([], [], [])
    fig.add_trace(go.Scatter3d(x=hubs_lon, y=hubs_lat, z=hubs_alt, mode='markers', marker=dict(size=8, color='green', symbol='diamond'), name='Hubs', text=list(HUBS.keys()), hoverinfo='text'))
    if dests_lon: fig.add_trace(go.Scatter3d(x=dests_lon, y=dests_lat, z=dests_alt, mode='markers', marker=dict(size=6, color='purple', symbol='square'), name='Pending Orders', text=list(state['pending_orders'].keys()), hoverinfo='text'))
    for b in env.buildings:
        x, y, h, dx, dy = b.center_xy[0], b.center_xy[1], b.height, b.size_xy[0]/2, b.size_xy[1]/2
        fig.add_trace(go.Mesh3d(x=[x-dx,x+dx,x+dx,x-dx]*2, y=[y-dy,y-dy,y+dy,y+dy]*2, z=[0,0,0,0,h,h,h,h], i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6], color='grey', opacity=0.7, name='Building'))
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

def render_operations_page(state, planners):
    """Renders the main simulation and operations view."""
    render_fleet_table(state)
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📦 Add New Delivery Order")
        with st.form("new_order_form", clear_on_submit=True):
            dest_name = st.selectbox("Destination", list(DESTINATIONS.keys()))
            payload = st.slider("Payload (kg)", 0.1, DRONE_MAX_PAYLOAD_KG, 1.0, 0.1)
            if st.form_submit_button("Add to Delivery Queue", type="primary", use_container_width=True):
                order_id = f"{dest_name.replace(' ', '')}-{uuid.uuid4().hex[:4]}"
                state['pending_orders'][order_id] = {'id': order_id, 'pos': DESTINATIONS[dest_name], 'payload_kg': payload}
                log_event(state, f"📥 New order added: {order_id} ({payload}kg)."); st.rerun()
        st.subheader("📦 Pending Delivery Orders")
        if not state['pending_orders']: st.info("No pending orders.")
        else: st.dataframe(pd.DataFrame(list(state['pending_orders'].values()))[['id', 'payload_kg']], use_container_width=True, hide_index=True)
        st.subheader("📋 Event Log")
        st.dataframe(pd.DataFrame(state['log'], columns=["Log Entry"]), height=300, use_container_width=True)

    with col2:
        st.subheader("🌐 3D Operations Map")
        render_map(state, planners)

def render_analytics_page(state):
    """Renders the post-mission analytics dashboard."""
    st.header("📊 Analytics Dashboard")
    
    log_data = state.get('completed_missions_log', [])
    if not log_data:
        st.info("No missions have been completed yet. Run the simulation to generate data.")
        return

    df = pd.DataFrame(log_data)
    
    # --- KPIs ---
    st.subheader("Key Performance Indicators (KPIs)")
    
    # On-Time Rate
    on_time_missions = (df['actual_duration_sec'] <= df['planned_duration_sec']).sum()
    on_time_rate = (on_time_missions / len(df)) * 100 if len(df) > 0 else 0
    
    # Energy Accuracy
    df_valid_energy = df[df['planned_energy_wh'] > 0]
    energy_error_pct = (abs(df_valid_energy['actual_energy_wh'] - df_valid_energy['planned_energy_wh']) / df_valid_energy['planned_energy_wh']).mean() * 100
    if pd.isna(energy_error_pct): energy_error_pct = 0.0

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Missions Flown", len(df))
    kpi2.metric("On-Time Delivery Rate", f"{on_time_rate:.1f}%")
    kpi3.metric("Avg. Energy Prediction Error", f"{energy_error_pct:.1f}%", help="Lower is better. Average of |Actual - Planned| / Planned.")

    st.markdown("---")

    # --- Visualizations ---
    st.subheader("Performance Visualizations")
    
    vis1, vis2 = st.columns(2)
    with vis1:
        # Energy Efficiency Chart
        df['efficiency_wh_per_stop'] = df['actual_energy_wh'] / df['number_of_stops']
        fig_eff = px.line(df, x='completion_timestamp', y='efficiency_wh_per_stop', title="Energy Efficiency Over Time", markers=True)
        fig_eff.update_layout(xaxis_title="Simulation Time (s)", yaxis_title="Energy (Wh) per Stop")
        st.plotly_chart(fig_eff, use_container_width=True)

    with vis2:
        # Missions per Drone
        missions_per_drone = df['drone_id'].value_counts().reset_index()
        missions_per_drone.columns = ['drone_id', 'count']
        fig_drone = px.bar(missions_per_drone, x='drone_id', y='count', title="Missions Completed per Drone")
        fig_drone.update_layout(xaxis_title="Drone ID", yaxis_title="Number of Missions")
        st.plotly_chart(fig_drone, use_container_width=True)
        
    st.markdown("---")
    
    # --- MLOps Section ---
    st.subheader("🧠 Model Management")
    model_path = state.get('active_model_path', "N/A")
    st.info(f"**Active Model:** `{os.path.basename(model_path)}`")
    
    if st.button("Retrain Prediction Model", use_container_width=True, help="Combines historical and new flight data to train an improved model."):
        with st.spinner("Retraining in progress... This may take a moment."):
            success, message = retrainer.retrain_model()
        if success:
            st.success(f"✅ Model retraining complete! {message}")
            # Clear the resource cache to force reloading planners with the new model
            st.cache_resource.clear()
            st.info("Cleared application cache. The new model will be loaded on the next action.")
            time.sleep(2) # Give user time to read the message
            st.rerun()
        else:
            st.error(f"❌ Retraining failed: {message}")

def main():
    st.set_page_config(layout="wide", page_title="Drone Delivery Simulator")
    st.title("🚁 Continuous Drone Delivery Simulator")
    
    state = system_state.load_state()
    planners = load_global_planners()
    if 'planning_future' not in st.session_state: st.session_state.planning_future = None
    
    with st.sidebar:
        st.header("Master Control")
        page = st.radio("Navigation", ["Operations", "Analytics Dashboard"])
        st.markdown("---")
        sim_running = state.get('simulation_running', False)
        if st.button("▶️ Run", disabled=sim_running, use_container_width=True):
            state['simulation_running'] = True; log_event(state, "Simulation started."); st.rerun()
        if st.button("⏸️ Pause", disabled=not sim_running, use_container_width=True):
            state['simulation_running'] = False; log_event(state, "Simulation paused."); st.rerun()
        st.metric("Simulation Time", f"{state['simulation_time']:.1f}s")
        if st.button("⚠️ Reset Simulation State", use_container_width=True, type="secondary"):
            state = system_state.reset_state_file(); log_event(state, "--- SYSTEM STATE RESET ---"); st.rerun()

    if page == "Operations":
        render_operations_page(state, planners)
    elif page == "Analytics Dashboard":
        render_analytics_page(state)

    # --- Simulation Update & Persistence ---
    if state.get('simulation_running', False):
        fleet_manager, dispatcher, executor = planners['fleet_manager'], planners['dispatcher'], planners['executor']
        # --- Asynchronous Fleet Planning Logic ---
        if st.session_state.planning_future and st.session_state.planning_future.done():
            try:
                success, results = st.session_state.planning_future.result()
                if success:
                    log_event(state, "✅ Fleet plan ready. Applying updates.")
                    state['drones'].update(results.get('drone_updates', {}))
                    for mid, updates in results.get('mission_updates', {}).items():
                        if mid in state['active_missions']: state['active_missions'][mid].update(updates)
                    for mid in results.get('successful_mission_ids', []):
                        for oid in state['active_missions'][mid]['order_ids']:
                            if oid in state['pending_orders']: del state['pending_orders'][oid]
                else:
                    log_event(state, f"❌ Fleet planning failed: {results.get('error', 'Unknown')}")
                    state['drones'].update(results.get('drone_updates', {}))
                    for mid in results.get('mission_failures', []):
                        if mid in state['active_missions']: del state['active_missions'][mid]
            except Exception as e:
                log_event(state, f"💥 CRITICAL ERROR in planning thread: {e}")
            finally:
                st.session_state.planning_future = None

        drones_need_planning = any(d['status'] == 'PLANNING' for d in state['drones'].values())
        if drones_need_planning and not st.session_state.planning_future:
            log_event(state, "🤖 Fleet requires planning. Submitting to background worker...")
            st.session_state.planning_future = executor.submit(fleet_manager.plan_pending_missions, state)

        # --- Core Simulation Logic ---
        dispatched = dispatcher.dispatch_missions(state)
        if dispatched: log_event(state, "🚚 Dispatcher created new missions.")
        update_simulation(state, fleet_manager)

    system_state.save_state(state)
    if state.get('simulation_running', False):
        time.sleep(SIMULATION_UI_REFRESH_INTERVAL); st.rerun()

if __name__ == "__main__":
    main()