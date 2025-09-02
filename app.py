# ==============================================================================
# File: app.py
# ==============================================================================
import streamlit as st
import plotly.graph_objects as go
import time
import pandas as pd
# --- (Import all other required classes from your project) ---
# Assuming all other files are in the correct directory structure
from config import *
from utils.geometry import *
from utils.coordinate_manager import *
from utils.heuristics import *
from utils.rrt_star import *
from utils.d_star_lite import *
from environment import *
from ml_predictor.predictor import *
from path_planner import *


def initialize_state():
    defaults = {
        'stage': 'setup', 'log': [], 'mission_running': False, 'planned_path': None,
        'planned_path_np': None, 'total_time': 0.0, 'total_energy': 0.0, 'drone_pos': None,
        'path_index': 0, 'initial_payload': 0.0, 'predicted_time': 0.0, 'predicted_energy': 0.0,
        'destination': None, 'hub_location': None
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

def reset_mission_state():
    log = st.session_state.get('log', [])
    for key in list(st.session_state.keys()):
        if key != 'log':
            del st.session_state[key]
            
    initialize_state()
    st.session_state.log = log
    planner.env.remove_dynamic_obstacles()
    log_event("Mission reset. Ready for new planning.")


@st.cache_resource
def load_planner():
    initialize_state()
    log_event("Loading environment and ML predictor...")
    env = Environment(WeatherSystem())
    predictor = EnergyTimePredictor()
    log_event("Initializing RRT*/D* Lite hybrid planner...")
    planner = PathPlanner3D(env, predictor)
    log_event("✅ Planner ready.")
    return planner

def log_event(m):
    st.session_state.log.insert(0, f"{time.strftime('%H:%M:%S')} - {m}")

def create_box(b):
    x, y, h = b.center_xy[0], b.center_xy[1], b.height
    dx, dy = b.size_xy[0]/2, b.size_xy[1]/2
    return go.Mesh3d(x=[x-dx, x+dx, x+dx, x-dx, x-dx, x+dx, x+dx, x-dx], y=[y-dy, y-dy, y+dy, y+dy, y-dy, y-dy, y+dy, y+dy], z=[0,0,0,0,h,h,h,h], i=[7,0,0,0,4,4,6,6,4,0,3,2],j=[3,4,1,2,5,6,5,2,0,1,6,3],k=[0,7,2,3,6,7,2,5,1,2,5,6], color='grey', opacity=0.7, name='Building', hoverinfo='none')

def create_nfz_box(z, c='red', o=0.15):
    return go.Mesh3d(x=[z[0],z[2],z[2],z[0],z[0],z[2],z[2],z[0]], y=[z[1],z[1],z[3],z[3],z[1],z[1],z[3],z[3]], z=[0,0,0,0,MAX_ALTITUDE,MAX_ALTITUDE,MAX_ALTITUDE,MAX_ALTITUDE], color=c, opacity=o, name='No-Fly Zone', hoverinfo='name')

def check_replanning_triggers():
    if planner.env.was_nfz_just_added:
        log_event("🚨 New NFZ detected! Triggering replan...")
        return True
    return False

def calculate_mission_summary(path, payload):
    if not path or len(path) < 2: return 0, 0
    total_time, total_energy = 0, 0
    for i in range(len(path) - 1):
        p_prev = path[i-1] if i > 0 else None
        # Use a simplified wind vector for summary calculation
        wind_vector = planner.env.weather.get_wind_at_location(path[i][0], path[i][1])
        t, e = planner.predictor.predict(path[i], path[i+1], payload, wind_vector, p_prev)
        total_time += t; total_energy += e
    return total_time, total_energy

def setup_stage():
    st.header("1. Mission Parameters")
    c1, c2 = st.columns(2)
    with c1:
        hub_name = st.selectbox("Select Hub", list(HUBS.keys()))
        st.session_state.hub_location = HUBS[hub_name]
        st.session_state.destination_choice = st.selectbox("Select Destination", list(DESTINATIONS.keys()))
    with c2: st.session_state.payload_kg = st.slider("Payload (kg)", 0.1, DRONE_MAX_PAYLOAD_KG, 1.5, 0.1)
    st.subheader("2. Optimization Priority")
    st.session_state.optimization_preference = st.radio("Optimize For:", ["Balanced", "Fastest Path", "Most Battery Efficient"], horizontal=True, index=0, help="Note: RRT* primarily optimizes for path length. Cost functions are used in predictions.")
    if st.session_state.optimization_preference == "Balanced": st.session_state.balance_weight = st.slider("Priority", 0.0, 1.0, 0.5, 0.05, format="%.2f", help="0.0 = Energy, 1.0 = Time")
    if st.button("🚀 Plan Mission", type="primary", use_container_width=True):
        st.session_state.destination = DESTINATIONS[st.session_state.destination_choice]
        st.session_state.stage = 'planning'; st.rerun()

def planning_stage():
    log_event(f"Planning mission with payload {st.session_state.payload_kg:.2f}kg...")
    with st.spinner("RRT* strategic planner exploring for an optimal route..."):
        hub, dest = st.session_state.hub_location, st.session_state.destination
        payload = st.session_state.payload_kg
        path, status = planner.find_path(
            start_pos=(hub[0], hub[1], TAKEOFF_ALTITUDE), end_pos=dest, 
            payload_kg=payload, mode='time'
        )
        if path is None:
            st.error(f"Path planning failed: {status}"); st.button("New Mission", on_click=reset_mission_state); return
        
        # The path from RRT* starts at the first waypoint *after* takeoff. Prepend the start position.
        full_path = [(hub[0], hub[1], TAKEOFF_ALTITUDE)] + path
        
        pred_time, pred_energy = calculate_mission_summary(full_path, payload)
        st.session_state.update({
            'initial_payload': payload, 'predicted_time': pred_time, 'predicted_energy': pred_energy,
            'planned_path': full_path, 'planned_path_np': np.array(full_path),
            'drone_pos': full_path[0], 'path_index': 0, 'mission_running': False
        })
        log_event(f"✅ Plan found! Est. Time: {pred_time:.1f}s, Est. Energy: {pred_energy:.2f}Wh")
    st.session_state.stage = 'simulation'; st.rerun()

def simulation_stage():
    st.header(f"Mission Simulation ({st.session_state.optimization_preference})")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        st.subheader("Mission Control")
        b1, b2 = st.columns(2)
        if b1.button("▶️ Run", disabled=st.session_state.mission_running, use_container_width=True): st.session_state.mission_running = True; st.rerun()
        if b2.button("⏸️ Pause", disabled=not st.session_state.mission_running, use_container_width=True): st.session_state.mission_running = False; st.rerun()
        st.subheader("Mission Summary (Predicted)")
        st.info(f"**Est. Time:** `{st.session_state.predicted_time:.2f} s`\n\n**Est. Energy:** `{st.session_state.predicted_energy:.2f} Wh`")
        st.subheader("Live Status")
        progress = (st.session_state.path_index / (len(st.session_state.planned_path) - 1)) if st.session_state.planned_path and len(st.session_state.planned_path) > 1 else 0
        st.progress(progress, text=f"Progress: {progress:.0%}")
        st.metric("Mission Time (Elapsed)", f"{st.session_state.total_time:.2f} s")
        battery_remaining = DRONE_BATTERY_WH - st.session_state.total_energy
        st.metric("Battery", f"{battery_remaining:.2f} Wh", delta=f"{-st.session_state.total_energy:.2f} Wh used", delta_color="inverse")
        if progress >= 1 and st.session_state.planned_path: st.success("✅ Mission Complete!"); st.session_state.mission_running = False
        if st.button("⬅️ New Mission", use_container_width=True): reset_mission_state(); st.rerun()
    with c2:
        fig = go.Figure()
        hub, dest = st.session_state.hub_location, st.session_state.destination
        if hub and dest:
            for name, h in HUBS.items(): fig.add_trace(go.Scatter3d(x=[h[0]],y=[h[1]],z=[h[2]],mode='markers',marker=dict(size=8,color='cyan',symbol='diamond'),name=f'Hub: {name}'))
            fig.add_trace(go.Scatter3d(x=[dest[0]],y=[dest[1]],z=[dest[2]],mode='markers+text',text=["Dest."],marker=dict(size=8,color='lime'),name='Destination'))
        for b in planner.env.buildings: fig.add_trace(create_box(b))
        for nfz_data in planner.env.dynamic_nfzs: fig.add_trace(create_nfz_box(nfz_data['zone'], c='purple', o=0.25))
        for nfz in planner.env.static_nfzs: fig.add_trace(create_nfz_box(nfz, c='red'))
        if st.session_state.planned_path_np is not None: fig.add_trace(go.Scatter3d(x=st.session_state.planned_path_np[:,0],y=st.session_state.planned_path_np[:,1],z=st.session_state.planned_path_np[:,2],mode='lines',line=dict(color='blue',width=4),name='Planned Path'))
        if st.session_state.drone_pos: fig.add_trace(go.Scatter3d(x=[st.session_state.drone_pos[0]],y=[st.session_state.drone_pos[1]],z=[st.session_state.drone_pos[2]],mode='markers',marker=dict(size=10,color='red',symbol='cross'),name='Drone'))
        fig.update_layout(margin=dict(l=0,r=0,b=0,t=0), scene=dict(xaxis_title='Lon',yaxis_title='Lat',zaxis_title='Alt (m)',aspectmode='cube'), legend=dict(y=0.99,x=0.01))
        st.plotly_chart(fig, use_container_width=True)
    with c1: st.subheader("Event Log"); st.dataframe(pd.DataFrame(st.session_state.log, columns=["Log Entry"]), height=150, use_container_width=True)

def main():
    """Main application loop and state machine."""
    st.set_page_config(layout="wide", page_title="Hybrid Drone Path Planner")
    st.title("🚁 Hybrid RRT*/D* Lite Drone Mission Planner")

    if 'stage' not in st.session_state:
        initialize_state()
        
    if st.session_state.stage == 'setup': 
        setup_stage()
    elif st.session_state.stage == 'planning': 
        planning_stage()
    elif st.session_state.stage == 'simulation': 
        simulation_stage()

    # ==========================================================================
    # --- START: CORE SIMULATION AND REPLANNING LOOP (MAJOR ADDITION/FIX) ---
    # ==========================================================================
    if 'mission_running' in st.session_state and st.session_state.mission_running:
        # 1. Update environment (weather, dynamic obstacles)
        time_step_sim = 0.5 # A discrete time step for simulation logic
        planner.env.update_environment(st.session_state.total_time, time_step=time_step_sim)

        # 2. Check for replanning triggers
        if check_replanning_triggers():
            st.session_state.mission_running = False # Pause simulation during replan
            with st.spinner("Obstacle detected! Performing hybrid replan..."):
                stale_path_from_drone = st.session_state.planned_path[st.session_state.path_index:]
                
                new_path_from_drone, status = planner.perform_hybrid_replan(
                    current_pos=st.session_state.drone_pos,
                    stale_path=stale_path_from_drone,
                    new_obstacle_bounds=planner.env.dynamic_nfzs[-1]['bounds']
                )

            if new_path_from_drone:
                log_event(f"✅ Replan successful! New path generated. ({status})")
                
                # The path segments *before* the current drone position
                path_traveled = st.session_state.planned_path[:st.session_state.path_index]
                
                # Combine the already traveled path with the new detour path
                new_full_path = path_traveled + new_path_from_drone
                
                st.session_state.planned_path = new_full_path
                st.session_state.planned_path_np = np.array(new_full_path)
                
                # --- BUG FIX: Correctly set the path index ---
                # The new index is the length of the already traveled segment, which points
                # to the drone's current position at the start of the new path segment.
                # Old (buggy) code: st.session_state.path_index = len(path_traveled) - 1
                st.session_state.path_index = len(path_traveled)
                
                # Re-calculate mission summary with the new path
                rem_path = new_full_path[st.session_state.path_index:]
                rem_time, rem_energy = calculate_mission_summary(rem_path, st.session_state.initial_payload)
                st.session_state.predicted_time = st.session_state.total_time + rem_time
                st.session_state.predicted_energy = st.session_state.total_energy + rem_energy
                log_event("Mission summary updated for new route.")

            else:
                log_event(f"❌ Replan FAILED: {status}. Mission paused.")
                st.error(f"Replan failed: {status}")

            planner.env.was_nfz_just_added = False # Reset the trigger
            st.session_state.mission_running = True # Resume simulation
        
        # 3. Advance drone along the path if no replanning occurred
        path = st.session_state.planned_path
        if path and st.session_state.path_index < len(path) - 1:
            p_current = path[st.session_state.path_index]
            p_next = path[st.session_state.path_index + 1]
            p_prev = path[st.session_state.path_index - 1] if st.session_state.path_index > 0 else None
            
            wind = planner.env.weather.get_wind_at_location(p_current[0], p_current[1])
            time_for_leg, energy_for_leg = planner.predictor.predict(
                p_current, p_next, st.session_state.initial_payload, wind, p_prev
            )
            
            # Update mission state
            st.session_state.total_time += time_for_leg
            st.session_state.total_energy += energy_for_leg
            st.session_state.path_index += 1
            st.session_state.drone_pos = path[st.session_state.path_index]
            log_event(f"Moved to waypoint {st.session_state.path_index}/{len(path)-1}. Leg Time: {time_for_leg:.1f}s")
        else:
            # Reached the end of the path
            st.session_state.mission_running = False

        # 4. Control simulation speed and refresh the UI
        time.sleep(0.2) # UI refresh rate
        st.rerun()
    # ==========================================================================
    # --- END: CORE SIMULATION AND REPLANNING LOOP ---
    # ==========================================================================


if __name__ == '__main__':
    # The planner is loaded once and cached by Streamlit
    planner = load_planner()
    main()