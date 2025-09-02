# ... (imports and functions up to `main` are unchanged and correct)
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time
import pandas as pd
from config import *
from environment import Environment, WeatherSystem
from path_planner import PathPlanner3D
from ml_predictor.predictor import EnergyTimePredictor
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
    log = st.session_state.log
    for key in st.session_state.keys(): del st.session_state[key]
    initialize_state()
    st.session_state.log = log
    planner.env.remove_dynamic_obstacles()
    planner.build_abstract_graph()
    log_event("Mission reset. Ready for new planning.")
@st.cache_resource
def load_planner():
    initialize_state()
    log_event("Loading environment and ML predictor...")
    env = Environment(WeatherSystem())
    predictor = EnergyTimePredictor()
    log_event("Initializing hybrid planner...")
    planner = PathPlanner3D(env, predictor)
    log_event("Building abstract graph for strategic planning...")
    planner.build_abstract_graph()
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
        t, e = planner.predictor.predict(path[i], path[i+1], payload, np.array([0,0,0]), p_prev)
        total_time += t; total_energy += e
    return total_time, total_energy
def setup_stage():
    st.header("1. Mission Parameters")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.hub_location = HUBS[st.selectbox("Select Hub", list(HUBS.keys()))]
        st.session_state.destination_choice = st.selectbox("Select Destination", list(DESTINATIONS.keys()))
    with c2: st.session_state.payload_kg = st.slider("Payload (kg)", 0.1, DRONE_MAX_PAYLOAD_KG, 1.5, 0.1)
    st.subheader("2. Optimization Priority")
    st.session_state.optimization_preference = st.radio("Optimize For:", ["Balanced", "Fastest Path", "Most Battery Efficient"], horizontal=True, index=0)
    if st.session_state.optimization_preference == "Balanced": st.session_state.balance_weight = st.slider("Priority", 0.0, 1.0, 0.5, 0.05, format="%.2f", help="0.0 = Energy, 1.0 = Time")
    if st.button("🚀 Plan Mission", type="primary", use_container_width=True):
        st.session_state.destination = DESTINATIONS[st.session_state.destination_choice]
        st.session_state.stage = 'planning'; st.rerun()
def planning_stage():
    mode = st.session_state.optimization_preference
    log_event(f"Planning mission ('{mode}') with payload {st.session_state.payload_kg:.2f}kg...")
    with st.spinner("Hierarchical planner calculating optimal route..."):
        mode_map = {"Fastest Path": "time", "Most Battery Efficient": "energy", "Balanced": "balanced"}
        hub, dest = st.session_state.hub_location, st.session_state.destination
        payload = st.session_state.payload_kg
        path, status = planner.find_path(
            start_pos=(hub[0], hub[1], TAKEOFF_ALTITUDE), end_pos=dest, 
            payload_kg=payload, mode=mode_map[mode], time_weight=st.session_state.get('balance_weight', 0.5)
        )
        if path is None:
            st.error(f"Path planning failed: {status}"); st.button("New Mission", on_click=reset_mission_state); return
        full_path = [hub] + path
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
    if st.session_state.stage == 'setup': setup_stage()
    elif st.session_state.stage == 'planning': planning_stage()
    elif st.session_state.stage == 'simulation': simulation_stage()

    if st.session_state.mission_running:
        planner.env.update_environment(st.session_state.total_time, time_step=0.1)

        if check_replanning_triggers():
            mode = st.session_state.optimization_preference
            mode_map = {"Fastest Path": "time", "Most Battery Efficient": "energy", "Balanced": "balanced"}
            
            with st.spinner("Obstacle detected! Performing hybrid replan..."):
                # Call the new, single, robust replanning method
                new_path, status = planner.perform_hybrid_replan(
                    current_pos=st.session_state.drone_pos,
                    goal_pos=st.session_state.destination,
                    new_obstacle_bounds=planner.env.dynamic_nfzs[-1]['bounds'],
                    payload_kg=st.session_state.initial_payload,
                    mode=mode_map[mode],
                    time_weight=st.session_state.get('balance_weight', 0.5)
                )

            if new_path:
                log_event(f"✅ Replan successful! New path generated. ({status})")
                full_new_path = [st.session_state.drone_pos] + new_path
                st.session_state.update(
                    planned_path=full_new_path,
                    planned_path_np=np.array(full_new_path),
                    path_index=0
                )
            else:
                log_event(f"❌ FATAL: All replanning failed: {status}. Mission halted.")
                st.error(f"MISSION FAILED: No valid path found. Reason: {status}") 
                st.session_state.mission_running = False
            
            planner.env.was_nfz_just_added = False

        path = st.session_state.planned_path
        idx = st.session_state.path_index
        if st.session_state.mission_running and path and idx < len(path) - 1:
            p_prev = path[idx - 1] if idx > 0 else None
            wind_vector = planner.env.weather.get_wind_at_location(path[idx][0], path[idx][1])
            t, e = planner.predictor.predict_energy_time(path[idx], path[idx + 1], st.session_state.initial_payload, wind_vector, p_prev)
            st.session_state.total_time += t
            st.session_state.total_energy += e
            st.session_state.path_index += 1
            st.session_state.drone_pos = path[st.session_state.path_index]

        time.sleep(0.1)
        st.rerun()

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Q-DOP Planner")
    planner = load_planner()
    main()