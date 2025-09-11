import streamlit as st
import plotly.graph_objects as go
import time
import pandas as pd
import numpy as np
from itertools import cycle
import warnings

from config import *
from environment import *
from ml_predictor.predictor import *
from utils.coordinate_manager import *
from planners.cbsh_planner import CBSHPlanner
from fleet.manager import *
from fleet.cbs_components import *

# --- State Management ---
def initialize_state():
    """Initializes the session state with default values."""
    if 'stage' not in st.session_state:
        st.session_state.stage = 'setup'
        st.session_state.log = []
        st.session_state.drones = {
            f"Drone {i+1}": {
                'pos': HUBS[list(HUBS.keys())[i % len(HUBS)]],
                'battery': DRONE_BATTERY_WH,
                'payload_kg': DRONE_MAX_PAYLOAD_KG,
                'status': 'IDLE', # IDLE, DISPATCHED, RECHARGING
                'available_at': 0.0
            } for i in range(3)
        }
        st.session_state.fleet_manager = None
        st.session_state.simulation_running = False
        st.session_state.simulation_time = 0.0
        st.session_state.orders = {name: pos for name, pos in DESTINATIONS.items()}

def reset_simulation():
    """Resets the state for a new mission, preserving drone availability."""
    drones_state = st.session_state.drones
    for drone_id, drone in drones_state.items():
        if drone['status'] != 'RECHARGING':
            drone['status'] = 'IDLE'

    log = st.session_state.get('log', [])
    for key in list(st.session_state.keys()):
        if key not in ['log', '_global_planner_objects', 'drones']:
            del st.session_state[key]
    
    initialize_state()
    st.session_state.log = log
    st.session_state.drones = drones_state
    log_event("Simulation reset. Ready for new mission assignment.")

# --- Helper Functions ---
def validate_points(env: Environment):
    """Checks predefined hubs and destinations against the environment's obstacles."""
    logging.info("Validating Hubs and Destinations...")
    points_to_check = {**HUBS, **DESTINATIONS}
    for name, pos in points_to_check.items():
        if env.is_point_obstructed(pos):
            st.warning(f"Configuration Warning: Point '{name}' at {pos} is inside an obstructed area (e.g., a No-Fly Zone).")

@st.cache_resource
def load_global_planners():
    log_event("Loading environment and global planners...")
    coord_manager = CoordinateManager()
    env = Environment(WeatherSystem(), coord_manager)
    validate_points(env)
    
    predictor = EnergyTimePredictor()
    cbsh_planner = CBSHPlanner(env, coord_manager)
    fleet_manager = FleetManager(cbsh_planner, predictor)
    log_event("✅ Planners ready.")
    return {
        "env": env,
        "predictor": predictor,
        "coord_manager": coord_manager,
        "fleet_manager": fleet_manager,
    }

def log_event(m):
    st.session_state.log.insert(0, f"{time.strftime('%H:%M:%S')} - {m}")

def create_box(b):
    x, y, h = b.center_xy[0], b.center_xy[1], b.height
    dx, dy = b.size_xy[0] / 2, b.size_xy[1] / 2
    vertices = [
        [x - dx, y - dy, 0], [x + dx, y - dy, 0], [x + dx, y + dy, 0], [x - dx, y + dy, 0],
        [x - dx, y - dy, h], [x + dx, y - dy, h], [x + dx, y + dy, h], [x - dx, y + dy, h]
    ]
    x_coords, y_coords, z_coords = zip(*vertices)
    i = [0, 0, 0, 0, 4, 4, 1, 1, 2, 2, 3, 3]
    j = [1, 3, 4, 7, 5, 6, 2, 5, 6, 7, 4, 7]
    k = [2, 1, 7, 4, 6, 5, 5, 6, 3, 6, 0, 4]
    return go.Mesh3d(x=x_coords, y=y_coords, z=z_coords, i=i, j=j, k=k,
                     color='grey', opacity=0.7, name='Building', hoverinfo='none')

def create_nfz_box(z, c='red', o=0.15):
    return go.Mesh3d(x=[z[0],z[2],z[2],z[0],z[0],z[2],z[2],z[0]], y=[z[1],z[1],z[3],z[3],z[1],z[1],z[3],z[3]], z=[0,0,0,0,MAX_ALTITUDE,MAX_ALTITUDE,MAX_ALTITUDE,MAX_ALTITUDE], color=c, opacity=o, name='No-Fly Zone', hoverinfo='name')

# --- UI Stages ---
def setup_stage():
    st.header("1. Mission Control Center")
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Available Drones")
        for drone_id, drone in st.session_state.drones.items():
            payload_kg = st.session_state.get(f"{drone_id}_payload", 1.5)
            if drone['status'] == 'RECHARGING' and st.session_state.simulation_time < drone['available_at']:
                st.info(f"**{drone_id}** - RECHARGING (Available in {drone['available_at'] - st.session_state.simulation_time:.0f}s)")
            else:
                if drone['status'] == 'RECHARGING':
                    drone['status'] = 'IDLE'
                    drone['battery'] = DRONE_BATTERY_WH
                
                st.success(f"**{drone_id}** - {drone['status']} | Battery: {drone['battery']:.1f}Wh")
                st.slider("Payload (kg)", 0.1, DRONE_MAX_PAYLOAD_KG, payload_kg, 0.1, key=f"{drone_id}_payload")

    with c2:
        st.subheader("Available Delivery Orders")
        all_orders = list(st.session_state.orders.keys())
        st.session_state.selected_orders = st.multiselect("Select orders to dispatch:", all_orders, default=all_orders[:3])

    if st.button("🚀 Assign & Plan Fleet Mission", type="primary", use_container_width=True):
        planners = st.session_state._global_planner_objects
        fm = planners['fleet_manager']
        fm.missions.clear()

        unassigned_orders = {name: st.session_state.orders[name] for name in st.session_state.selected_orders}
        
        for drone_id, drone in st.session_state.drones.items():
            if drone['status'] == 'IDLE':
                current_pos = drone['pos']
                remaining_payload = st.session_state[f"{drone_id}_payload"]
                drone_missions = []
                
                while remaining_payload > 0 and unassigned_orders:
                    closest_order_name = min(unassigned_orders, key=lambda name: calculate_distance_3d(current_pos, unassigned_orders[name]))
                    
                    mission_leg = Mission(drone_id, current_pos, [unassigned_orders[closest_order_name]], 1.0, {})
                    
                    is_safe, reason = fm.pre_flight_check(mission_leg, drone['battery'])
                    if is_safe:
                        drone_missions.append(unassigned_orders[closest_order_name])
                        current_pos = unassigned_orders[closest_order_name]
                        remaining_payload -= 1.0
                        del unassigned_orders[closest_order_name]
                    else:
                        log_event(f"⚠️ {drone_id} could not take order for {closest_order_name}: {reason}")
                        break
                
                if drone_missions:
                    final_mission = Mission(drone_id, drone['pos'], drone_missions, st.session_state[f"{drone_id}_payload"], {})
                    fm.add_mission(final_mission)
                    log_event(f"Assigned {len(drone_missions)} stops to {drone_id}.")

        if not fm.missions:
            st.error("No valid missions could be assigned. Check drone battery/payload or destination safety.")
        else:
            st.session_state.stage = 'planning'
            st.rerun()

def planning_stage():
    log_event("Fleet planning initiated...")
    fm = st.session_state._global_planner_objects['fleet_manager']
    with st.spinner("CBSH planner coordinating conflict-free routes for the fleet..."):
        success, results = fm.execute_planning_cycle()
    
    if success:
        log_event("✅ Fleet plan found!")
        st.session_state.stage = 'simulation'
        for drone_id in results['planned_missions']:
            st.session_state.drones[drone_id]['status'] = 'DISPATCHED'
        st.rerun()
    else:
        log_event("❌ Fleet planning failed. Some missions were impossible.")
        st.error("Fleet planning failed. Check logs for details.")
        if st.button("New Mission Assignment", use_container_width=True):
            reset_simulation()
            st.rerun()

def simulation_stage():
    fm = st.session_state._global_planner_objects['fleet_manager']
    
    all_drones_finished = True
    for drone_id in fm.missions:
        if st.session_state.drones[drone_id]['status'] == 'DISPATCHED':
            all_drones_finished = False
            mission = fm.missions[drone_id]
            if st.session_state.simulation_time >= mission.total_planned_time:
                st.session_state.drones[drone_id]['status'] = 'RECHARGING'
                st.session_state.drones[drone_id]['available_at'] = st.session_state.simulation_time + DRONE_RECHARGE_TIME_S
                log_event(f"✅ {drone_id} completed its mission and is now recharging.")
    
    if all_drones_finished and st.session_state.simulation_running:
        log_event("🏁 All missions complete.")
        st.session_state.simulation_running = False
    
    st.header("Fleet Mission Simulation")
    c1, c2 = st.columns([1, 2.5])
    
    with c1:
        st.subheader("Mission Control")
        b1, b2 = st.columns(2)
        if b1.button("▶️ Run", disabled=st.session_state.simulation_running or all_drones_finished, use_container_width=True):
            st.session_state.simulation_running = True
            st.rerun()
        if b2.button("⏸️ Pause", disabled=not st.session_state.simulation_running, use_container_width=True):
            st.session_state.simulation_running = False
            st.rerun()

        st.subheader("Fleet Status")
        for drone_id, drone in st.session_state.drones.items():
             with st.expander(f"{drone_id} ({drone['status']})", expanded=True):
                if drone_id in fm.missions and drone['status'] == 'DISPATCHED':
                    mission = fm.missions[drone_id]
                    total_planned_time = mission.total_planned_time if mission.total_planned_time > 0 else 1
                    sim_time = st.session_state.simulation_time
                    progress = min(sim_time / total_planned_time, 1.0)
                    st.progress(progress, text=f"Time: {sim_time:.0f}s / {total_planned_time:.0f}s")
                    st.metric("Battery", f"{drone['battery']:.2f} Wh", delta=f"{-mission.total_planned_energy:.2f} Wh (plan)", delta_color="inverse")
                elif drone['status'] == 'RECHARGING':
                     st.info(f"Recharging... available in {drone['available_at'] - st.session_state.simulation_time:.0f}s")
                else:
                    st.success(f"Idle at Hub | Battery: {drone['battery']:.1f} Wh")

        if st.button("⬅️ New Mission Assignment", use_container_width=True):
            reset_simulation()
            st.rerun()

    with c2:
        fig = go.Figure()
        env = st.session_state._global_planner_objects['env']
        hubs_lon, hubs_lat, hubs_alt = zip(*HUBS.values())
        dests_lon, dests_lat, dests_alt = zip(*DESTINATIONS.values())
        
        fig.add_trace(go.Scatter3d(x=hubs_lon, y=hubs_lat, z=hubs_alt, mode='markers', marker=dict(size=8, color='green', symbol='diamond'), name='Hubs', text=list(HUBS.keys()), hoverinfo='text'))
        fig.add_trace(go.Scatter3d(x=dests_lon, y=dests_lat, z=dests_alt, mode='markers', marker=dict(size=6, color='purple', symbol='square'), name='Destinations', text=list(DESTINATIONS.keys()), hoverinfo='text'))
        for b in env.buildings: fig.add_trace(create_box(b))
        for nfz in env.static_nfzs: fig.add_trace(create_nfz_box(nfz, c='red'))

        drone_colors = cycle(['red', 'blue', 'green', 'orange', 'purple'])
        for drone_id, drone in st.session_state.drones.items():
            color = next(drone_colors)
            if drone_id in fm.missions and drone['status'] == 'DISPATCHED':
                mission = fm.missions[drone_id]
                path_np = np.array(mission.path_world_coords)
                if path_np.any():
                    fig.add_trace(go.Scatter3d(x=path_np[:,0], y=path_np[:,1], z=path_np[:,2], mode='lines', line=dict(color=color, width=4), name=f'{drone_id} Path'))
            
            drone_pos = drone['pos']
            fig.add_trace(go.Scatter3d(x=[drone_pos[0]], y=[drone_pos[1]], z=[drone_pos[2]], mode='markers', marker=dict(size=8, color=color, symbol='cross'), name=drone_id))
        
        fig.update_layout(margin=dict(l=0,r=0,b=0,t=0), scene=dict(xaxis_title='Lon',yaxis_title='Lat',zaxis_title='Alt (m)',aspectmode='data'), legend=dict(y=0.99,x=0.01))
        st.plotly_chart(fig, use_container_width=True)

    with c1:
        st.subheader("Event Log")
        st.dataframe(pd.DataFrame(st.session_state.log, columns=["Log Entry"]), height=200, use_container_width=True)

    if st.session_state.simulation_running:
        st.session_state.simulation_time += SIMULATION_TIME_STEP

        for drone_id, drone in st.session_state.drones.items():
            if drone['status'] != 'DISPATCHED': continue
            
            mission = fm.missions[drone_id]
            if not mission.path_world_coords: continue
            
            total_time = mission.total_planned_time
            progress = min(st.session_state.simulation_time / total_time, 1.0) if total_time > 0 else 1.0
            
            path_len = len(mission.path_world_coords)
            path_index = int(progress * (path_len - 1))
            
            if path_index < path_len - 1:
                p1 = np.array(mission.path_world_coords[path_index])
                p2 = np.array(mission.path_world_coords[path_index + 1])
                segment_progress = (progress * (path_len - 1)) - path_index
                new_pos = p1 + segment_progress * (p2 - p1)
                drone['pos'] = tuple(new_pos)
            else:
                drone['pos'] = mission.path_world_coords[-1]
            
            initial_battery = DRONE_BATTERY_WH
            energy_consumed = progress * mission.total_planned_energy
            drone['battery'] = initial_battery - energy_consumed

        time.sleep(SIMULATION_UI_REFRESH_INTERVAL)
        st.rerun()

# --- Main Application Logic ---
def main():
    st.set_page_config(layout="wide", page_title="Multi-Agent Drone Planner")
    st.title("🚁 Multi-Agent CBSH Fleet Planner")

    initialize_state()
    st.session_state._global_planner_objects = load_global_planners()

    if st.session_state.stage == 'setup':
        setup_stage()
    elif st.session_state.stage == 'planning':
        planning_stage()
    elif st.session_state.stage == 'simulation':
        simulation_stage()

if __name__ == "__main__":
    main()