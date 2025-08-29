# app.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time

# Make sure our project modules are found
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import Environment
from ml_predictor.predictor import EnergyTimePredictor
from optimization.hybrid_solver import HybridSolver
from optimization.qubo_formulator import QuboFormulator
from path_planner import PathPlanner3D

# --- Helper Functions for Plotting ---
def create_cylinder(center_xy, radius, height, num_points=20):
    """Generates vertices for a 3D cylinder (building)."""
    index = np.arange(0, num_points + 1, 1) * (360 / num_points)
    x = center_xy[0] + radius * np.cos(np.radians(index))
    y = center_xy[1] + radius * np.sin(np.radians(index))
    z = np.zeros(len(x))
    
    x = np.append(x, x)
    y = np.append(y, y)
    z = np.append(z, np.full(len(z), height))
    
    return go.Mesh3d(x=x, y=y, z=z, alphahull=0, color='grey', opacity=0.4, name='Building')

# --- Main Application Logic ---
st.set_page_config(layout="wide")
st.title("🚁 Q-DOP: 3D Drone Path Optimization")

if 'solution' not in st.session_state:
    st.session_state.solution = None

def run_optimization():
    """The main logic from our old main.py, adapted for Streamlit."""
    progress_bar = st.progress(0, text="Initializing Environment...")
    env = Environment(seed=42)
    predictor = EnergyTimePredictor()
    solver = HybridSolver()
    path_planner = PathPlanner3D(env)
    time.sleep(1)

    progress_bar.progress(20, text="Building Cost Matrices with ML Predictor...")
    locations_dict = {
        **{f"Depot-{d.id}": d.start_location for d in env.drones},
        **{f"Order-{o.id}": o.location for o in env.orders}
    }
    location_names = list(locations_dict.keys())
    locations = list(locations_dict.values())
    time_matrix, energy_matrix = predictor.build_cost_matrices(locations, env)
    time.sleep(1)

    progress_bar.progress(50, text="Solving VRP with Classical Optimizer (OR-Tools)...")
    or_tools_routes = solver.solve_with_or_tools(time_matrix, energy_matrix, env.drones, env.orders)
    time.sleep(1)

    if not or_tools_routes:
        st.error("No solution found by the optimizer.")
        progress_bar.empty()
        return

    final_paths = {}
    progress_bar.progress(75, text="Generating 3D Obstacle-Avoiding Paths (A*)...")
    for drone_id, route_data in or_tools_routes.items():
        if len(route_data['readable']) > 2: # If the drone is assigned a route
            waypoints = [locations_dict[name] for name in route_data['readable']]
            detailed_path = path_planner.generate_full_trajectory(waypoints)
            if detailed_path: # Only add if path was found
                final_paths[drone_id] = {
                    "waypoints": route_data['readable'],
                    "trajectory": detailed_path
                }
    time.sleep(1)
    progress_bar.progress(100, text="Mission Plan Complete!")
    st.session_state.solution = {"env": env, "paths": final_paths}
    time.sleep(1)
    progress_bar.empty()


# --- UI Layout ---
col1, col2 = st.columns([1, 3])

with col1:
    st.header("Controls")
    if st.button("🚀 Plan Mission", type="primary", use_container_width=True):
        run_optimization()
    
    if st.session_state.solution:
        st.header("Mission Plan")
        paths = st.session_state.solution["paths"]
        if not paths:
            st.warning("No valid paths were generated for any drone.")
        for drone_id, data in paths.items():
            st.subheader(f"Drone {drone_id}")
            st.info(" -> ".join(data['waypoints']))
        
        # Real-time simulation slider
        st.header("Simulation")
        time_step = st.slider("Animate Path", 0, 100, 0, 1)

with col2:
    st.header("3D Mission Visualization")
    fig = go.Figure()
    
    # Set default empty plot
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), showlegend=True,
                    scene=dict(xaxis_title='X (meters)', yaxis_title='Y (meters)', zaxis_title='Altitude (meters)',
                                aspectmode='data'))

    if st.session_state.solution:
        env = st.session_state.solution["env"]
        paths = st.session_state.solution["paths"]
        
        # Plot Drones (Depots)
        fig.add_trace(go.Scatter3d(x=[d.start_location[0] for d in env.drones], 
                                   y=[d.start_location[1] for d in env.drones],
                                   z=[d.start_location[2] for d in env.drones],
                                   mode='markers', marker=dict(size=8, color='blue', symbol='diamond'), name='Depots'))
        # Plot Orders
        fig.add_trace(go.Scatter3d(x=[o.location[0] for o in env.orders], 
                                   y=[o.location[1] for o in env.orders],
                                   z=[o.location[2] for o in env.orders],
                                   mode='markers+text', text=[f"Order {o.id}" for o in env.orders], textposition='top center',
                                   marker=dict(size=8, color='green'), name='Orders'))
        # Plot Buildings
        for b in env.buildings:
            fig.add_trace(create_cylinder(b.center_xy, b.radius, b.height))
            
        # Plot Trajectories
        colors = ['red', 'purple', 'orange', 'cyan']
        for i, (drone_id, data) in enumerate(paths.items()):
            path = np.array(data['trajectory'])
            fig.add_trace(go.Scatter3d(x=path[:,0], y=path[:,1], z=path[:,2],
                                       mode='lines', line=dict(color=colors[i % len(colors)], width=4), 
                                       name=f'Drone {drone_id} Path'))
            
            # "Real-time" drone position
            if path.shape[0] > 1:
                idx = int((time_step / 100) * (path.shape[0] - 1))
                drone_pos = path[idx]
                fig.add_trace(go.Scatter3d(x=[drone_pos[0]], y=[drone_pos[1]], z=[drone_pos[2]],
                                           mode='markers', marker=dict(size=10, color=colors[i % len(colors)]),
                                           name=f'Drone {drone_id} Live'))

    st.plotly_chart(fig, use_container_width=True, height=700)