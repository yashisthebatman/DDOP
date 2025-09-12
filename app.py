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
import training.retrainer as retrainer
import simulation.event_injector as event_injector
import simulation.contingency_planner as contingency_planner
from simulation.contingency_planner import _trigger_emergency_return
from simulation.deconfliction import check_and_resolve_conflicts
from utils.geometry import calculate_distance_3d

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
    # NOTE: We load state here once to initialize planners, but the main app
    # will use the session_state for interactions.
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

def update_simulation(state, planners):
    """Advances the simulation by one time step and updates drone/mission states."""
    state['simulation_time'] += SIMULATION_TIME_STEP
    coord_manager = planners['coord_manager']

    for drone in state['drones'].values():
        if drone['status'] == 'RECHARGING' and state['simulation_time'] >= drone['available_at']:
            drone['status'] = 'IDLE'
            drone['battery'] = DRONE_BATTERY_WH
            log_event(state, f"✅ {drone['id']} has finished recharging and is now IDLE.")

    active_drones = {d['id']: d for d in state['drones'].values() if d['mission_id'] in state['active_missions']}
    if len(active_drones) > 1:
        check_and_resolve_conflicts(active_drones, coord_manager)

    missions_to_complete = []
    for mission_id, mission in list(state['active_missions'].items()):
        if mission.get('is_paused', False): continue

        mission['mission_time_elapsed'] += SIMULATION_TIME_STEP
        drone_id = mission['drone_id']
        drone = state['drones'][drone_id]

        if drone['status'] == 'PERFORMING_DELIVERY':
            if state['simulation_time'] >= drone.get('maneuver_complete_at', float('inf')):
                mission['current_stop_index'] += 1
                drone['status'] = 'EN ROUTE'
                drone.pop('maneuver_complete_at', None)

        elif drone['status'] == 'AVOIDING':
            target_pos_world = drone.get('avoidance_target_pos')
            if not target_pos_world: continue

            current_pos_m = np.array(coord_manager.world_to_meters(drone['pos']))
            target_pos_m = np.array(coord_manager.world_to_meters(target_pos_world))
            vector_m = target_pos_m - current_pos_m
            dist_m = np.linalg.norm(vector_m)
            step_m = DRONE_SPEED_MPS * SIMULATION_TIME_STEP

            if dist_m <= step_m:
                drone['pos'] = target_pos_world
                log_event(state, f"👍 {drone_id} completed avoidance maneuver.")
                drone['status'] = drone.get('original_status_before_avoid', 'EN ROUTE')
                drone.pop('avoidance_target_pos', None)
                drone.pop('original_status_before_avoid', None)
            else:
                new_pos_m = current_pos_m + (vector_m / dist_m) * step_m
                drone['pos'] = coord_manager.meters_to_world(tuple(new_pos_m))

        elif drone['status'] in ['EN ROUTE', 'EMERGENCY_RETURN']:
            mission['flight_time_elapsed'] += SIMULATION_TIME_STEP

            arrived_this_tick = False
            if drone['status'] == 'EN ROUTE':
                num_stops = len(mission.get('stops', []))
                stop_idx = mission.get('current_stop_index', 0)
                if stop_idx < num_stops:
                    target_pos = mission['stops'][stop_idx]['pos']
                    dist_m = calculate_distance_3d(coord_manager.world_to_meters(drone['pos']), coord_manager.world_to_meters(target_pos))
                    if dist_m < 5.0:
                        drone['status'] = 'PERFORMING_DELIVERY'
                        drone['maneuver_complete_at'] = state['simulation_time'] + DELIVERY_MANEUVER_TIME_SEC
                        drone['pos'] = target_pos
                        arrived_this_tick = True
                        log_event(state, f"🎯 {drone_id} arriving at stop {stop_idx+1}. Performing delivery.")
            
            if not arrived_this_tick:
                total_planned_time = mission.get('total_planned_time', 1)
                total_maneuver_time = mission.get('total_maneuver_time', 0)
                flight_time = max(1, total_planned_time - total_maneuver_time)
                flight_progress = min(1.0, mission['flight_time_elapsed'] / flight_time)

                path = mission.get('path_world_coords', [])
                if path:
                    path_index = int(flight_progress * (len(path) - 1))
                    if path_index < len(path) - 1:
                        p1, p2 = np.array(path[path_index]), np.array(path[path_index + 1])
                        segment_progress = (flight_progress * (len(path) - 1)) - path_index
                        drone['pos'] = tuple(p1 + segment_progress * (p2 - p1))
                    else:
                        drone['pos'] = path[-1]
                
                energy_consumed = flight_progress * mission.get('total_planned_energy', 0)
                drone['battery'] = mission.get('start_battery', DRONE_BATTERY_WH) - energy_consumed

        if mission['mission_time_elapsed'] >= mission.get('total_planned_time', float('inf')):
            missions_to_complete.append(mission_id)

    for mission_id in missions_to_complete:
        mission = state['active_missions'][mission_id]
        drone_id = mission['drone_id']
        drone = state['drones'][drone_id]
        
        is_emergency_return = drone['status'] == 'EMERGENCY_RETURN'

        if not is_emergency_return:
            actual_duration = state['simulation_time'] - mission['start_time']
            actual_energy = mission['start_battery'] - drone['battery']
            
            log_entry = {
                "mission_id": mission_id, "drone_id": drone_id,
                "completion_timestamp": state['simulation_time'],
                "planned_duration_sec": mission['total_planned_time'],
                "actual_duration_sec": actual_duration,
                "planned_energy_wh": mission['total_planned_energy'],
                "actual_energy_wh": actual_energy,
                "number_of_stops": len(mission['destinations']), "outcome": "Completed",
            }
            state['completed_missions_log'].append(log_entry)
            
            try:
                temp_predictor = EnergyTimePredictor()
                stops = mission.get('stops', [])
                if stops:
                    features = temp_predictor._extract_features(
                        mission['start_pos'], stops[-1]['pos'], 
                        mission['payload_kg'], [0,0,0], None
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
            for order_id in mission['order_ids']:
                if order_id not in state['completed_orders']:
                     state['completed_orders'].append(order_id)
            state['completed_missions'][mission_id] = mission

        else:
            log_event(state, f"✅ {drone_id} successfully returned to hub after emergency.")
            
        end_hub_id = mission.get('end_hub')
        if end_hub_id and end_hub_id in HUBS:
            drone['home_hub'] = end_hub_id
            drone['pos'] = HUBS[end_hub_id]
            log_event(state, f"🚚 {drone_id} has relocated to new home base: {end_hub_id}.")
        
        drone['status'] = 'RECHARGING'
        drone['mission_id'] = None
        drone['available_at'] = state['simulation_time'] + DRONE_RECHARGE_TIME_S
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
    
    for d_nfz in env.dynamic_nfzs:
        zone = d_nfz['zone']
        fig.add_trace(go.Mesh3d(x=[zone[0],zone[2],zone[2],zone[0]]*2, y=[zone[1],zone[1],zone[3],zone[3]]*2, z=[0,0,0,0,MAX_ALTITUDE,MAX_ALTITUDE,MAX_ALTITUDE,MAX_ALTITUDE], i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6], color='rgba(255,0,0,0.5)', name='Dynamic NFZ'))

    drone_colors = cycle(px.colors.qualitative.Plotly)
    for drone_id, drone in state['drones'].items():
        color = next(drone_colors)
        if drone['status'] in ['EN ROUTE', 'EMERGENCY_RETURN', 'PERFORMING_DELIVERY', 'AVOIDING'] and drone['mission_id'] in state['active_missions']:
            path = state['active_missions'][drone['mission_id']].get('path_world_coords', [])
            if path:
                path_np = np.array(path)
                fig.add_trace(go.Scatter3d(x=path_np[:,0], y=path_np[:,1], z=path_np[:,2], mode='lines', line=dict(color=color, width=4), name=f'{drone_id} Path'))
        fig.add_trace(go.Scatter3d(x=[drone['pos'][0]], y=[drone['pos'][1]], z=[drone['pos'][2]], mode='markers', marker=dict(size=8, color=color, symbol='cross'), name=drone_id))
    
    # --- FIX 1: Set a better initial camera angle to show the 3D perspective ---
    camera = dict(eye=dict(x=-1.5, y=-1.5, z=1))
    fig.update_layout(margin=dict(l=0,r=0,b=0,t=0), scene_camera=camera, scene=dict(xaxis_title='Lon',yaxis_title='Lat',zaxis_title='Alt (m)',aspectmode='data'), legend=dict(y=0.99,x=0.01))
    st.plotly_chart(fig, use_container_width=True)

def render_operations_page(state, planners):
    c1, c2 = st.columns([3, 1])
    with c1:
        st.subheader("Live Fleet Map")
        render_map(state, planners)
    with c2:
        st.subheader("System Log")
        # --- FIX 2: Set a light text color for visibility in dark mode ---
        log_html = "".join([f"<div style='font-size: 13px; font-family: monospace; color: #E0E0E0;'>{msg}</div>" for msg in state['log'][:20]])
        st.components.v1.html(f"<div style='height: 400px; overflow-y: scroll; border: 1px solid #444; padding: 5px;'>{log_html}</div>", height=410)

    st.markdown("---")
    st.subheader("Fleet & Mission Control")
    
    status_cols = st.columns(4)
    status_cols[0].metric("Drones Idle", len([d for d in state['drones'].values() if d['status'] == 'IDLE']))
    status_cols[1].metric("Drones Active", len([d for d in state['drones'].values() if d['status'] not in ['IDLE', 'RECHARGING']]))
    status_cols[2].metric("Pending Orders", len(state['pending_orders']))
    status_cols[3].metric("Completed Orders", len(state['completed_orders']))
    
    for drone_id, drone in sorted(state['drones'].items()):
        with st.expander(f"**{drone_id}** ({drone['status']}) - Battery: {drone['battery']:.1f}Wh - Location: {drone['home_hub']}"):
            if drone['status'] not in ['IDLE', 'RECHARGING']:
                mission = state['active_missions'].get(drone['mission_id'])
                if mission:
                    st.write(f"**Mission:** {mission['mission_id']}")
                    st.write(f"**Orders:** {', '.join(mission['order_ids'])}")
                    st.progress(min(1.0, mission.get('mission_time_elapsed', 0) / mission.get('total_planned_time', 1)))
                    
                    b_cols = st.columns(3)
                    is_paused = mission.get('is_paused', False)
                    if is_paused:
                        if b_cols[0].button("▶️ Resume", key=f"resume_{drone_id}", use_container_width=True):
                            state['active_missions'][drone['mission_id']]['is_paused'] = False
                            log_event(state, f"OPERATOR: Resumed mission for {drone_id}.")
                            st.rerun()
                    else:
                        if b_cols[0].button("⏸️ Pause", key=f"pause_{drone_id}", use_container_width=True):
                            state['active_missions'][drone['mission_id']]['is_paused'] = True
                            log_event(state, f"OPERATOR: Paused mission for {drone_id}.")
                            st.rerun()
                    
                    if b_cols[1].button("❌ Cancel Mission", key=f"cancel_{drone_id}", type="primary", use_container_width=True):
                        _trigger_emergency_return(state, drone_id, "Manual Operator Override", planners)
                        log_event(state, f"OPERATOR: Manually cancelled mission for {drone_id}.")
                        st.rerun()
            else:
                st.write("Awaiting task.")

    st.markdown("---")
    with st.form("add_order_form"):
        st.subheader("Add New Order")
        c1, c2, c3, c4 = st.columns([2,1,1,1])
        dest_name = c1.selectbox("Destination", DESTINATIONS.keys())
        payload = c2.number_input("Payload (kg)", min_value=0.1, max_value=DRONE_MAX_PAYLOAD_KG, value=1.0, step=0.5)
        is_high_priority = c3.checkbox("High Priority", help="High priority orders trigger dispatch immediately.")
        submitted = c4.form_submit_button("Add Order")
        if submitted:
            order_id = f"Order-{uuid.uuid4().hex[:6]}"
            # --- FIX 3 (Part of State Fix): Modify the state IN session_state ---
            st.session_state.system_state['pending_orders'][order_id] = {
                'id': order_id, 'pos': DESTINATIONS[dest_name],
                'payload_kg': payload, 'high_priority': is_high_priority
            }
            log_event(st.session_state.system_state, f"📥 New {'high priority ' if is_high_priority else ''}order added: {order_id} for {dest_name}.")
            st.rerun()

def render_analytics_page(state):
    st.header("📊 Analytics Dashboard")
    log_df = pd.DataFrame(state.get('completed_missions_log', []))

    if log_df.empty:
        st.warning("No missions completed yet. Run the simulation to generate data.")
        return

    c1, c2, c3 = st.columns(3)
    completed_missions = log_df[log_df['outcome'] == 'Completed']
    on_time = (completed_missions['actual_duration_sec'] <= completed_missions['planned_duration_sec']).sum()
    on_time_rate = (on_time / len(completed_missions)) * 100 if not completed_missions.empty else 0
    c1.metric("On-Time Delivery Rate", f"{on_time_rate:.1f}%")

    log_df_valid_planned = completed_missions[completed_missions['planned_energy_wh'] > 0]
    energy_error = (abs(log_df_valid_planned['actual_energy_wh'] - log_df_valid_planned['planned_energy_wh']) / log_df_valid_planned['planned_energy_wh']).mean() * 100
    c2.metric("Energy Prediction Accuracy", f"{100-energy_error:.1f}%" if pd.notna(energy_error) else "N/A")

    c3.metric("Total Missions Flown", len(log_df))

    st.markdown("---")
    st.subheader("Performance Over Time")
    fig = px.line(log_df, x='completion_timestamp', y=['planned_duration_sec', 'actual_duration_sec'], title="Planned vs. Actual Mission Duration", labels={'value': 'Duration (s)', 'completion_timestamp': 'Simulation Time (s)'})
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(layout="wide", page_title="Drone Delivery Simulator")
    st.title("🚁 Continuous Drone Delivery Simulator")
    
    planners = load_global_planners()
    if 'planning_future' not in st.session_state: st.session_state.planning_future = None
    
    # --- FIX 3: Use st.session_state for robust state management ---
    # Load from file only ONCE at the beginning of a user's session.
    if 'system_state' not in st.session_state:
        st.session_state.system_state = system_state.load_state()

    # Use the state from the session for all subsequent operations.
    state = st.session_state.system_state
    
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
            planners['env'].remove_dynamic_obstacles()
            # Reset both the session state and the file on disk.
            st.session_state.system_state = system_state.reset_state_file()
            log_event(st.session_state.system_state, "--- SYSTEM STATE RESET ---"); st.rerun()

    if page == "Operations":
        render_operations_page(state, planners)
    elif page == "Analytics Dashboard":
        render_analytics_page(state)

    # --- Core simulation loop will only run if the toggle is active ---
    if state.get('simulation_running', False):
        fleet_manager, dispatcher, executor, env = planners['fleet_manager'], planners['dispatcher'], planners['executor'], planners['env']
        
        event_injector.inject_random_event(state, env)
        contingency_planner.check_for_contingencies(state, planners)

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

        dispatched = dispatcher.dispatch_missions(state)
        if dispatched: log_event(state, "🚚 Dispatcher created new missions.")
        update_simulation(state, planners)

    # Persist the (potentially modified) session state to disk at the end of every script run.
    system_state.save_state(state)
    if state.get('simulation_running', False):
        time.sleep(SIMULATION_UI_REFRESH_INTERVAL); st.rerun()

if __name__ == "__main__":
    main()