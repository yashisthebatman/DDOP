# app.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time

# Make sure our project modules are found
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from environment import Environment
from ml_predictor.predictor import EnergyTimePredictor
from optimization.hybrid_solver import HybridSolver
from optimization.qubo_formulator import QuboFormulator
from path_planner import PathPlanner3D

# --- Helper Functions for Plotting ---
def create_cylinder(center_xy, radius, height):
    """Generates vertices for a 3D cylinder (building)."""
    index = np.arange(0, 361, 10)
    x = center_xy[0] + radius * np.cos(np.radians(index))
    y = center_xy[1] + radius * np.sin(np.radians(index))
    z = np.zeros(len(x))
    return go.Mesh3d(x=x, y=y, z=z, i=list(range(len(x)-2)), j=list(range(1,len(x)-1)), k=list(range(2,len(x))),
                     delaunayaxis='z', color='grey', opacity=0.6, name='Building', hoverinfo='none')

def get_optimization_weights(preference):
    if preference == "Fastest Delivery":
        return {'time': 0.9, 'energy': 0.1}
    else: # Fuel Efficient
        return {'time': 0.1, 'energy': 0.9}

# --- Main Application Logic ---
st.set_page_config(layout="wide")
st.title("🚁 Q-DOP: 3D Drone Path Optimization")

if 'solution' not in st.session_state:
    st.session_state.solution = None

# --- UI Sidebar for Controls ---
with st.sidebar:
    st.header("Scenario Controls")
    num_drones = st.slider("Number of Drones (Hubs)", 1, 5, config.DEFAULT_NUM_DRONES, 1)
    num_orders = st.slider("Number of Orders", 2, 20, config.DEFAULT_NUM_ORDERS, 1)
    num_buildings = st.slider("Number of Obstacles", 5, 50, config.DEFAULT_NUM_BUILDINGS, 1)
    
    st.header("Optimization Goal")
    opt_preference = st.radio("Choose Priority", ["Fastest Delivery", "Fuel Efficient"], horizontal=True)
    
    if st.button("🚀 Plan Mission", type="primary", use_container_width=True):
        st.session_state.solution = None # Reset previous solution
        weights = get_optimization_weights(opt_preference)
        
        progress_bar = st.progress(0, text="Initializing Environment...")
        env = Environment(num_drones, num_orders, num_buildings, seed=42)
        predictor = EnergyTimePredictor()
        qubo_formulator = QuboFormulator()
        solver = HybridSolver()
        path_planner = PathPlanner3D(env, predictor, weights)
        time.sleep(1)

        progress_bar.progress(20, text="Building Cost Matrices with ML Predictor...")
        locations_dict = {
            **{f"Depot-{d.id}": d.start_location for d in env.drones},
            **{f"Order-{o.id}": o.location for o in env.orders}
        }
        locations = list(locations_dict.values())
        time_matrix, energy_matrix = predictor.build_cost_matrices(locations, env)
        time.sleep(1)

        solution_data = {"env": env, "locations_dict": locations_dict}

        # --- Solve with Both Solvers ---
        progress_bar.progress(40, text="Solving with Quantum-Inspired QUBO...")
        # CHANGE: Pass the 'weights' dictionary to the QUBO formulator
        qubo, _ = qubo_formulator.build_vrp_qubo(time_matrix, energy_matrix, env.drones, env.orders, weights)
        qubo_sample = solver.solve_qubo(qubo)
        
        num_locs = len(env.drones) + len(env.orders)
        qubo_map = {i: name for i, name in enumerate(locations_dict.keys()) if i < num_locs}
        decoded_qubo_routes = qubo_formulator.decode_solution(qubo_sample, env.drones, env.orders, qubo_map)
        
        progress_bar.progress(60, text="Solving with Classical OR-Tools...")
        or_tools_routes = solver.solve_with_or_tools(time_matrix, energy_matrix, env.drones, env.orders, weights)

        progress_bar.progress(80, text="Generating 3D Obstacle-Avoiding Paths (A*)...")
        # Process and store results for both solvers
        solution_data['qubo_paths'] = path_planner.process_routes(decoded_qubo_routes, locations_dict, is_qubo=True)
        solution_data['ortools_paths'] = path_planner.process_routes(or_tools_routes, locations_dict, is_qubo=False) if or_tools_routes else {}

        st.session_state.solution = solution_data
        progress_bar.progress(100, text="Mission Plan Complete!")
        time.sleep(1)
        progress_bar.empty()
        st.success("Mission plan generated!")

# --- Main Content Area (remains the same) ---
if not st.session_state.solution:
    st.info("Configure your scenario in the sidebar and click 'Plan Mission' to begin.")
else:
    solution = st.session_state.solution
    env = solution["env"]
    
    # --- Tabs for different solver results ---
    tab1, tab2 = st.tabs(["⭐ Classical (OR-Tools) Solution", "💡 Quantum-Inspired (QUBO) Solution"])

    # Create a shared slider for animation
    with st.sidebar:
        st.header("Simulation")
        time_step = st.slider("Animate Path", 0, 100, 0, 1)

    for i, tab in enumerate([tab1, tab2]):
        with tab:
            solver_key = 'ortools_paths' if i == 0 else 'qubo_paths'
            paths = solution.get(solver_key, {})
            
            col1, col2 = st.columns([1, 2.5])
            with col1:
                st.subheader("Mission Plan Details")
                if not paths:
                    st.warning("No valid paths were generated by this solver.")
                total_time, total_energy = 0, 0
                for drone_id, data in paths.items():
                    st.markdown(f"**Drone {drone_id}**")
                    st.info(" -> ".join(data['waypoints']))
                    st.write(f"_Time: {data['time']:.2f}s | Energy: {data['energy']:.2f} Wh_")
                    total_time += data['time']
                    total_energy += data['energy']
                st.markdown("---")
                st.metric("Total Mission Time (Sum)", f"{total_time:.2f} s")
                st.metric("Total Mission Energy", f"{total_energy:.2f} Wh")

            with col2:
                st.subheader("3D Mission Visualization")
                fig = go.Figure()
                
                # Plot static elements
                depot_locs = [d.start_location for d in env.drones]
                order_locs = [o.location for o in env.orders]
                fig.add_trace(go.Scatter3d(x=[loc[0] for loc in depot_locs], y=[loc[1] for loc in depot_locs], z=[loc[2] for loc in depot_locs],
                                           mode='markers', marker=dict(size=10, color='blue', symbol='diamond'), name='Depots'))
                fig.add_trace(go.Scatter3d(x=[loc[0] for loc in order_locs], y=[loc[1] for loc in order_locs], z=[loc[2] for loc in order_locs],
                                           mode='markers+text', text=[f"Order {o.id}" for o in env.orders], textposition='top center',
                                           marker=dict(size=8, color='green'), name='Orders'))
                for b in env.buildings:
                    fig.add_trace(create_cylinder(b.center_xy, b.radius, b.height))
                
                # Plot dynamic paths and live drones
                colors = ['red', 'purple', 'orange', 'cyan']
                for j, (drone_id, data) in enumerate(paths.items()):
                    path = np.array(data['trajectory'])
                    fig.add_trace(go.Scatter3d(x=path[:,0], y=path[:,1], z=path[:,2],
                                               mode='lines', line=dict(color=colors[j % len(colors)], width=4), 
                                               name=f'Drone {drone_id} Path'))
                    
                    if path.shape[0] > 1:
                        idx = int((time_step / 100) * (path.shape[0] - 1))
                        drone_pos = path[idx]
                        fig.add_trace(go.Scatter3d(x=[drone_pos[0]], y=[drone_pos[1]], z=[drone_pos[2]],
                                                   mode='markers', marker=dict(size=12, color=colors[j % len(colors)], symbol='circle'),
                                                   name=f'Drone {drone_id} Live'))

                fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), showlegend=True,
                                  scene=dict(xaxis_title='X (meters)', yaxis_title='Y (meters)', zaxis_title='Altitude (meters)',
                                             aspectmode='data', camera=dict(eye=dict(x=-1.5, y=-1.5, z=1))))
                st.plotly_chart(fig, use_container_width=True, height=700)