# FILE: server.py

import asyncio
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
import json
import numpy as np

from config import *
import system_state
from environment import Environment, WeatherSystem
from ml_predictor.predictor import EnergyTimePredictor
from utils.coordinate_manager import CoordinateManager
from planners.cbsh_planner import CBSHPlanner
from fleet.manager import FleetManager
from dispatch.vrp_solver import VRPSolver
from dispatch.dispatcher import Dispatcher
import simulation.event_injector as event_injector
import simulation.contingency_planner as contingency_planner
from simulation.deconfliction import check_and_resolve_conflicts
from utils.geometry import calculate_distance_3d

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global State & Planners ---
planners = {}
state = {}
connected_clients = set()

# --- Helper function to create building meshes for Plotly ---
def _create_building_mesh_data(building, coord_manager):
    center_x, center_y = building.center_xy
    size_x, size_y = building.size_xy
    height = building.height
    corners_world = [
        (center_x - size_x / 2, center_y - size_y / 2, 0), (center_x + size_x / 2, center_y - size_y / 2, 0),
        (center_x + size_x / 2, center_y + size_y / 2, 0), (center_x - size_x / 2, center_y + size_y / 2, 0),
        (center_x - size_x / 2, center_y - size_y / 2, height), (center_x + size_x / 2, center_y - size_y / 2, height),
        (center_x + size_x / 2, center_y + size_y / 2, height), (center_x - size_x / 2, center_y + size_y / 2, height),
    ]
    corners_m = [coord_manager.world_to_meters(p) for p in corners_world]
    x, y, z = zip(*corners_m)
    return {
        'x': x, 'y': y, 'z': z,
        'i': [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2], 'j': [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3], 'k': [0, 7, 2, 1, 6, 7, 2, 5, 1, 3, 2, 6]
    }

# --- Function to prepare all 3D scene data for the frontend ---
def generate_plotly_data(current_state, coord_manager, env):
    drones = list(current_state.get('drones', {}).values())
    drone_positions_m = [coord_manager.world_to_meters(d['pos']) for d in drones]
    drones_trace = {
        'x': [p[0] for p in drone_positions_m], 'y': [p[1] for p in drone_positions_m], 'z': [p[2] for p in drone_positions_m],
        'text': [f"{d['id']}<br>Status: {d['status']}<br>Battery: {d.get('battery', 0):.1f}Wh" for d in drones],
        'type': 'scatter3d', 'mode': 'markers', 'name': 'Drones',
        'marker': {'size': 5, 'color': 'red'}
    }
    hub_positions_m = [coord_manager.world_to_meters(h) for h in HUBS.values()]
    hubs_trace = {
        'x': [p[0] for p in hub_positions_m], 'y': [p[1] for p in hub_positions_m], 'z': [p[2] for p in hub_positions_m],
        'text': list(HUBS.keys()),
        'type': 'scatter3d', 'mode': 'markers', 'name': 'Hubs',
        'marker': {'size': 8, 'color': 'cyan', 'symbol': 'diamond'}
    }
    paths_x, paths_y, paths_z = [], [], []
    for mission in current_state.get('active_missions', {}).values():
        path = mission.get('path_world_coords', [])
        if path:
            path_m = [coord_manager.world_to_meters(p) for p in path]
            paths_x.extend([p[0] for p in path_m] + [None]); paths_y.extend([p[1] for p in path_m] + [None]); paths_z.extend([p[2] for p in path_m] + [None])
    paths_trace = {
        'x': paths_x, 'y': paths_y, 'z': paths_z,
        'type': 'scatter3d', 'mode': 'lines', 'name': 'Paths',
        # FIX: Make paths highly visible
        'line': {'color': 'magenta', 'width': 4}
    }
    buildings_x, buildings_y, buildings_z, buildings_i, buildings_j, buildings_k = [], [], [], [], [], []
    vertex_offset = 0
    for building in env.buildings:
        mesh = _create_building_mesh_data(building, coord_manager)
        buildings_x.extend(mesh['x']); buildings_y.extend(mesh['y']); buildings_z.extend(mesh['z'])
        buildings_i.extend([i + vertex_offset for i in mesh['i']]); buildings_j.extend([j + vertex_offset for j in mesh['j']]); buildings_k.extend([k + vertex_offset for k in mesh['k']])
        vertex_offset += 8
    buildings_trace = {
        'x': buildings_x, 'y': buildings_y, 'z': buildings_z,
        'i': buildings_i, 'j': buildings_j, 'k': buildings_k,
        'type': 'mesh3d', 'name': 'Buildings', 'color': 'grey', 'opacity': 0.5
    }
    return [drones_trace, hubs_trace, paths_trace, buildings_trace]

# --- (lifespan and update_simulation functions remain unchanged, so they are omitted for brevity) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global state; state.update(system_state.load_state())
    coord_manager = CoordinateManager(); env = Environment(WeatherSystem(), coord_manager)
    predictor = EnergyTimePredictor(); predictor.load_model()
    cbsh_planner = CBSHPlanner(env, coord_manager)
    planners.update({
        'coord_manager': coord_manager, 'env': env, 'predictor': predictor,
        'dispatcher': Dispatcher(VRPSolver(predictor)), 'fleet_manager': FleetManager(cbsh_planner, predictor),
        'executor': ThreadPoolExecutor(max_workers=2)
    })
    for dest_name, pos in DESTINATIONS.items():
        surface_alt = env.get_surface_height((pos[0], pos[1]))
        DESTINATIONS[dest_name] = (pos[0], pos[1], pos[2] if pos[2] > surface_alt else surface_alt)
    simulation_task = asyncio.create_task(simulation_loop())
    logging.info("--- System Initialized and Ready ---")
    yield
    simulation_task.cancel(); planners['executor'].shutdown(wait=False); logging.info("--- System Shutting Down ---")

app = FastAPI(lifespan=lifespan)

def update_simulation(state, planners):
    state['simulation_time'] += SIMULATION_TIME_STEP; coord_manager = planners['coord_manager']
    for drone in state['drones'].values():
        if drone['status'] == 'RECHARGING' and state['simulation_time'] >= drone['available_at']: drone['status'] = 'IDLE'; drone['battery'] = DRONE_BATTERY_WH
    active_drones = {d['id']: d for d in state['drones'].values() if d['status'] in ['EN ROUTE', 'EMERGENCY_RETURN', 'AVOIDING']}
    if len(active_drones) > 1: check_and_resolve_conflicts(active_drones, coord_manager)
    missions_to_complete = []
    for mission_id, mission in list(state['active_missions'].items()):
        if mission.get('is_paused', False): continue
        drone_id = mission['drone_id']; drone = state['drones'][drone_id]
        if drone['status'] == 'AVOIDING':
            target_pos = np.array(drone.get('avoidance_target_pos', drone['pos'])); current_pos = np.array(drone['pos'])
            direction = target_pos - current_pos; distance = np.linalg.norm(direction)
            if distance < DRONE_VERTICAL_SPEED_MPS * SIMULATION_TIME_STEP: drone['pos'] = tuple(target_pos.tolist()); drone['status'] = drone.get('original_status_before_avoid', 'EN ROUTE'); drone.pop('avoidance_target_pos', None); drone.pop('original_status_before_avoid', None)
            else: move_vec = (direction / distance) * DRONE_VERTICAL_SPEED_MPS * SIMULATION_TIME_STEP; drone['pos'] = tuple((current_pos + move_vec).tolist())
            continue
        elif drone['status'] == 'PERFORMING_DELIVERY':
            if state['simulation_time'] >= drone.get('maneuver_complete_at', float('inf')): mission['current_stop_index'] += 1; drone['status'] = 'EN ROUTE'; drone.pop('maneuver_complete_at', None); drone.pop('maneuver_start_pos', None); drone.pop('maneuver_target_pos', None)
            else:
                start_time = drone['maneuver_complete_at'] - DELIVERY_MANEUVER_TIME_SEC; progress = min(1.0, max(0.0, (state['simulation_time'] - start_time) / DELIVERY_MANEUVER_TIME_SEC))
                start_pos = np.array(drone.get('maneuver_start_pos', drone['pos'])); target_pos = np.array(drone.get('maneuver_target_pos', drone['pos']))
                new_z = start_pos[2] + progress * (target_pos[2] - start_pos[2]); drone['pos'] = (float(start_pos[0]), float(start_pos[1]), float(new_z))
            continue
        elif drone['status'] in ['EN ROUTE', 'EMERGENCY_RETURN']:
            if drone['status'] == 'EN ROUTE' and mission.get('current_stop_index', 0) < len(mission.get('stops', [])):
                target_pos = mission['stops'][mission['current_stop_index']]['pos']; dist_to_target_m = calculate_distance_3d(coord_manager.world_to_meters(drone['pos']), coord_manager.world_to_meters(target_pos))
                if dist_to_target_m < 5.0: drone['status'] = 'PERFORMING_DELIVERY'; drone['maneuver_complete_at'] = state['simulation_time'] + DELIVERY_MANEUVER_TIME_SEC; drone['maneuver_start_pos'] = drone['pos']; drone['maneuver_target_pos'] = target_pos; continue
            mission['flight_time_elapsed'] += SIMULATION_TIME_STEP; flight_time = max(1, mission.get('total_planned_time', 1) - mission.get('total_maneuver_time', 0)); progress = min(1.0, mission['flight_time_elapsed'] / flight_time)
            path = mission.get('path_world_coords', [])
            if path and len(path) > 1:
                idx = int(progress * (len(path) - 1))
                if idx < len(path) - 1: p1, p2 = np.array(path[idx]), np.array(path[idx + 1]); seg_prog = (progress * (len(path) - 1)) - idx; new_pos = p1 + seg_prog * (p2 - p1); drone['pos'] = tuple(new_pos.tolist())
                else: drone['pos'] = tuple(path[-1])
            elif path: drone['pos'] = tuple(path[-1])
            drone['battery'] = mission.get('start_battery', DRONE_BATTERY_WH) - (progress * mission.get('total_planned_energy', 0))
        mission['mission_time_elapsed'] += SIMULATION_TIME_STEP
        if mission.get('total_planned_time', 0) > 0 and mission['mission_time_elapsed'] >= mission['total_planned_time']: missions_to_complete.append(mission_id)
    for mission_id in missions_to_complete:
        mission = state['active_missions'][mission_id]; drone_id = mission['drone_id']; drone = state['drones'][drone_id]
        if drone['status'] != 'EMERGENCY_RETURN':
            actual_duration = state['simulation_time'] - mission['start_time']; actual_energy = mission['start_battery'] - drone['battery']
            state['completed_missions_log'].append({"mission_id": mission_id, "drone_id": drone_id, "completion_timestamp": float(state['simulation_time']), "planned_duration_sec": float(mission['total_planned_time']), "actual_duration_sec": float(actual_duration), "planned_energy_wh": float(mission['total_planned_energy']), "actual_energy_wh": float(actual_energy), "number_of_stops": len(mission['destinations']), "outcome": "Completed"})
            [state['completed_orders'].append(oid) for oid in mission['order_ids'] if oid not in state['completed_orders']]; state['completed_missions'][mission_id] = mission
        end_hub_id = mission.get('end_hub')
        if end_hub_id in HUBS: drone['home_hub'] = end_hub_id; hub_pos = HUBS[end_hub_id]; drone['pos'] = (float(hub_pos[0]), float(hub_pos[1]), float(hub_pos[2]))
        drone['status'] = 'RECHARGING'; drone['mission_id'] = None; drone['available_at'] = state['simulation_time'] + DRONE_RECHARGE_TIME_S
        del state['active_missions'][mission_id]

# --- WebSocket Communication ---
async def broadcast_state():
    if connected_clients:
        plotly_data = generate_plotly_data(state, planners['coord_manager'], planners['env'])
        # FIX: Send enriched data payload to the frontend
        full_message = {
            'simulation_state': state,
            'plotly_data': plotly_data,
            'drone_list': list(state.get('drones', {}).values()),
            'pending_orders_list': list(state.get('pending_orders', {}).values()),
            'mission_log': state.get('completed_missions_log', [])
        }
        state_json = json.dumps(full_message, cls=system_state.NumpyJSONEncoder)
        await asyncio.gather(*[client.send_text(state_json) for client in connected_clients])

# --- (WebSocket endpoint, sim loop, and API endpoints are unchanged and omitted for brevity) ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept(); connected_clients.add(websocket); logging.info(f"Client connected. Total clients: {len(connected_clients)}")
    try:
        await broadcast_state()
        while True:
            data = await websocket.receive_json(); command = data.get("type"); payload = data.get("payload")
            if command == "toggle_simulation": state['simulation_running'] = not state.get('simulation_running', False)
            elif command == "reset_simulation":
                state.update(system_state.reset_state_file())
                for dest_name, pos in DESTINATIONS.items(): surface_alt = planners['env'].get_surface_height((pos[0], pos[1])); DESTINATIONS[dest_name] = (pos[0], pos[1], pos[2] if pos[2] > surface_alt else surface_alt)
            elif command == "add_order":
                env = planners['env']; base_pos = DESTINATIONS[payload['dest_name']]; surface_alt = env.get_surface_height((base_pos[0], base_pos[1])); final_pos = (base_pos[0], base_pos[1], surface_alt)
                order_id = f"Order-{uuid.uuid4().hex[:6]}"; state['pending_orders'][order_id] = {'id': order_id, 'pos': final_pos, 'dest_name': payload['dest_name'], 'payload_kg': payload['payload_kg'], 'high_priority': payload['high_priority']}
            elif command == "dispatch_missions": planners['dispatcher'].dispatch_missions(state)
            await broadcast_state()
    except WebSocketDisconnect: connected_clients.remove(websocket); logging.info(f"Client disconnected. Total clients: {len(connected_clients)}")
    except Exception as e: logging.error(f"WebSocket Error: {e}");
async def simulation_loop():
    while True:
        try:
            if state.get('simulation_running', False):
                fleet_manager = planners['fleet_manager']; executor = planners['executor']; planning_future = None
                if planning_future and planning_future.done():
                    try:
                        success, results = planning_future.result()
                        if success:
                            state['drones'].update(results.get('drone_updates', {}));
                            for mid, updates in results.get('mission_updates', {}).items():
                                if mid in state['active_missions']: state['active_missions'][mid].update(updates)
                            for mid in results.get('successful_mission_ids', []):
                                for oid in state['active_missions'][mid]['order_ids']:
                                    if oid in state['pending_orders']: del state['pending_orders'][oid]
                        else: state['drones'].update(results.get('drone_updates', {}))
                    finally: planning_future = None
                drones_need_planning = any(d['status'] == 'PLANNING' for d in state['drones'].values())
                if drones_need_planning and not planning_future: planning_future = executor.submit(fleet_manager.plan_pending_missions, state)
                update_simulation(state, planners); contingency_planner.check_for_contingencies(state, planners); event_injector.inject_random_event(state, planners['env'])
                system_state.save_state(state); await broadcast_state()
            await asyncio.sleep(SIMULATION_UI_REFRESH_INTERVAL)
        except asyncio.CancelledError: logging.info("Simulation loop cancelled."); break
        except Exception as e: logging.error(f"Error in simulation loop: {e}"); await asyncio.sleep(1)
@app.get("/api/destinations")
async def get_destinations(): return JSONResponse(content=DESTINATIONS)
app.mount("/", StaticFiles(directory="web", html=True), name="static")

if __name__ == "__main__": uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)