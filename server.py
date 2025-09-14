# FILE: server.py

import asyncio
import logging
import os
import uuid
import threading
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
from dispatch.dispatcher import Dispatcher, get_hub_by_id
import simulation.event_injector as event_injector
import simulation.contingency_planner as contingency_planner
from simulation.deconfliction import check_and_resolve_conflicts
from utils.geometry import calculate_distance_3d
from demand_analyzer import DemandAnalyzer

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global State & Planners ---
planners = {}
state = {}
connected_clients = set()
state_lock = threading.Lock()
active_planning_futures = []
demand_analyzer_thread = None
shutdown_event = threading.Event()

# --- Helper functions ---
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
    return {'x': x, 'y': y, 'z': z, 'i': [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2], 'j': [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3], 'k': [0, 7, 2, 1, 6, 7, 2, 5, 1, 3, 2, 6]}

def generate_plotly_data(current_state, coord_manager, env):
    drones = list(current_state.get('drones', {}).values())
    drone_positions_m = [coord_manager.world_to_meters(d['pos']) for d in drones]
    drones_trace = {
        'x': [p[0] for p in drone_positions_m], 'y': [p[1] for p in drone_positions_m], 'z': [p[2] for p in drone_positions_m],
        'text': [f"{d['id']}<br>Status: {d['status']}<br>Battery: {d.get('battery', 0):.1f}Wh" for d in drones],
        'type': 'scatter3d', 'mode': 'markers', 'name': 'Drones', 'marker': {'size': 5, 'color': 'red'}
    }
    hub_locations = [h['location'] for h in HUBS]
    hub_names = [h['name'] for h in HUBS]
    hub_positions_m = [coord_manager.world_to_meters(h) for h in hub_locations]
    hubs_trace = {
        'x': [p[0] for p in hub_positions_m], 'y': [p[1] for p in hub_positions_m], 'z': [p[2] for p in hub_positions_m],
        'text': hub_names, 'type': 'scatter3d', 'mode': 'markers', 'name': 'Hubs', 'marker': {'size': 8, 'color': 'cyan', 'symbol': 'diamond'}
    }
    paths_x, paths_y, paths_z = [], [], []
    for mission in current_state.get('active_missions', {}).values():
        path = mission.get('path_world_coords', [])
        if path:
            path_m = [coord_manager.world_to_meters(p) for p in path]
            paths_x.extend([p[0] for p in path_m] + [None]); paths_y.extend([p[1] for p in path_m] + [None]); paths_z.extend([p[2] for p in path_m] + [None])
    paths_trace = {'x': paths_x, 'y': paths_y, 'z': paths_z, 'type': 'scatter3d', 'mode': 'lines', 'name': 'Paths', 'line': {'color': 'magenta', 'width': 4}}
    buildings_x, buildings_y, buildings_z, buildings_i, buildings_j, buildings_k = [], [], [], [], [], []
    vertex_offset = 0
    for building in env.buildings:
        mesh = _create_building_mesh_data(building, coord_manager)
        buildings_x.extend(mesh['x']); buildings_y.extend(mesh['y']); buildings_z.extend(mesh['z'])
        buildings_i.extend([i + vertex_offset for i in mesh['i']]); buildings_j.extend([j + vertex_offset for j in mesh['j']]); buildings_k.extend([k + vertex_offset for k in mesh['k']])
        vertex_offset += 8
    buildings_trace = {'x': buildings_x, 'y': buildings_y, 'z': buildings_z, 'i': buildings_i, 'j': buildings_j, 'k': buildings_k, 'type': 'mesh3d', 'name': 'Buildings', 'color': 'grey', 'opacity': 0.5}
    return [drones_trace, hubs_trace, paths_trace, buildings_trace]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global state, demand_analyzer_thread
    with state_lock: state.update(system_state.load_state())
    coord_manager = CoordinateManager()
    env = Environment(WeatherSystem(), coord_manager)
    predictor = EnergyTimePredictor(); predictor.load_model()
    cbsh_planner = CBSHPlanner(env, coord_manager)
    fleet_manager = FleetManager(cbsh_planner, predictor, state_lock)
    dispatcher = Dispatcher(fleet_manager)
    planners.update({
        'coord_manager': coord_manager, 'env': env, 'predictor': predictor,
        'dispatcher': dispatcher, 'fleet_manager': fleet_manager,
        'executor': ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PLANNERS)
    })
    for dest_name, pos in DESTINATIONS.items():
        surface_alt = env.get_surface_height((pos[0], pos[1]))
        DESTINATIONS[dest_name] = (pos[0], pos[1], pos[2] if pos[2] > surface_alt else surface_alt)
    demand_analyzer = DemandAnalyzer(state, dispatcher, state_lock, shutdown_event)
    demand_analyzer_thread = threading.Thread(target=demand_analyzer.run, daemon=True)
    demand_analyzer_thread.start()
    simulation_task = asyncio.create_task(simulation_loop())
    logging.info("--- System Initialized and Ready ---")
    yield
    shutdown_event.set()
    demand_analyzer_thread.join()
    simulation_task.cancel()
    planners['executor'].shutdown(wait=True)
    logging.info("--- System Shutting Down ---")

app = FastAPI(lifespan=lifespan)

# ... (update_simulation logic is unchanged from the last correct version) ...
def update_simulation(state, planners):
    state['simulation_time'] += SIMULATION_TIME_STEP
    coord_manager = planners['coord_manager']
    for drone in state['drones'].values():
        if drone['status'] == 'RECHARGING' and state['simulation_time'] >= drone['available_at']:
            drone['status'] = 'IDLE'; drone['battery'] = DRONE_BATTERY_WH
    active_drones = {d['id']: d for d in state['drones'].values() if d['status'] not in ['IDLE', 'PLANNING', 'RECHARGING']}
    if len(active_drones) > 1:
        check_and_resolve_conflicts(active_drones, planners)
    missions_to_complete = []
    for mission_id, mission in list(state['active_missions'].items()):
        if mission.get('is_paused', False): continue
        drone_id = mission['drone_id']
        drone = state['drones'][drone_id]
        if 'current_path_target_index' not in mission: mission['current_path_target_index'] = 1
        if drone['status'] == 'PERFORMING_DELIVERY':
            if state['simulation_time'] >= mission.get('maneuver_complete_at', float('inf')):
                mission['current_stop_index'] += 1
                mission['current_path_target_index'] = min(len(mission.get('path_world_coords', [])) -1, mission['current_path_target_index'] + 1)
                drone.pop('maneuver_complete_at', None)
                if mission['current_stop_index'] >= len(mission.get('stops', [])):
                    drone['status'] = 'RETURNING_TO_HUB'
                else:
                    drone['status'] = 'EN ROUTE'
            continue 
        elif drone['status'] == 'AVOIDING':
            target_pos = np.array(drone.get('avoidance_target_pos', drone['pos']))
            current_pos = np.array(drone['pos'])
            direction = target_pos - current_pos
            distance = np.linalg.norm(direction)
            if distance < DRONE_VERTICAL_SPEED_MPS * SIMULATION_TIME_STEP:
                drone['pos'] = tuple(target_pos.tolist())
                drone['status'] = drone.get('original_status_before_avoid', 'EN ROUTE')
                drone.pop('avoidance_target_pos', None); drone.pop('original_status_before_avoid', None)
            else:
                move_vec = (direction / distance) * DRONE_VERTICAL_SPEED_MPS * SIMULATION_TIME_STEP
                drone['pos'] = tuple((current_pos + move_vec).tolist())
            continue
        elif drone['status'] in ['EN ROUTE', 'RETURNING_TO_HUB', 'EMERGENCY_RETURN']:
            path = mission.get('path_world_coords', [])
            if not path or mission['current_path_target_index'] >= len(path): continue
            target_waypoint = np.array(path[mission['current_path_target_index']])
            current_pos_np = np.array(drone['pos'])
            direction = target_waypoint - current_pos_np
            dist_to_wp = np.linalg.norm(direction)
            move_dist = DRONE_SPEED_MPS * SIMULATION_TIME_STEP
            if dist_to_wp < move_dist:
                drone['pos'] = tuple(target_waypoint.tolist())
                mission['current_path_target_index'] += 1
            else:
                drone['pos'] = tuple((current_pos_np + (direction / dist_to_wp) * move_dist).tolist())
            has_deliveries = len(mission.get('stops', [])) > 0
            is_returning = drone['status'] in ['RETURNING_TO_HUB', 'EMERGENCY_RETURN']
            if drone['status'] == 'EN ROUTE' and has_deliveries:
                stop_idx = mission.get('current_stop_index', 0)
                delivery_pos = mission['stops'][stop_idx]['pos']
                dist_to_delivery = calculate_distance_3d(coord_manager.world_to_meters(drone['pos']), coord_manager.world_to_meters(delivery_pos))
                if dist_to_delivery < 5.0:
                    drone['status'] = 'PERFORMING_DELIVERY'
                    mission['maneuver_complete_at'] = state['simulation_time'] + DELIVERY_MANEUVER_TIME_SEC
            elif is_returning or (drone['status'] == 'EN ROUTE' and not has_deliveries):
                hub_pos = mission['destinations'][-1]
                dist_to_hub = calculate_distance_3d(coord_manager.world_to_meters(drone['pos']), coord_manager.world_to_meters(hub_pos))
                if dist_to_hub < 5.0:
                    missions_to_complete.append(mission_id)
                    continue
            mission['flight_time_elapsed'] += SIMULATION_TIME_STEP
            flight_time = max(1, mission.get('total_planned_time', 1))
            progress = min(1.0, mission['flight_time_elapsed'] / flight_time)
            drone['battery'] = mission.get('start_battery', DRONE_BATTERY_WH) - (progress * mission.get('total_planned_energy', 0))
    for mission_id in missions_to_complete:
        mission = state.get('active_missions', {}).get(mission_id)
        if not mission: continue
        drone_id = mission['drone_id']
        drone = state['drones'][drone_id]
        if drone['status'] != 'EMERGENCY_RETURN':
            actual_duration = state['simulation_time'] - mission.get('start_time', 0)
            actual_energy = mission.get('start_battery', DRONE_BATTERY_WH) - drone['battery']
            state['completed_missions_log'].append({"mission_id": mission_id, "drone_id": drone_id, "completion_timestamp": float(state['simulation_time']), "planned_duration_sec": float(mission.get('total_planned_time', 0)), "actual_duration_sec": float(actual_duration), "planned_energy_wh": float(mission.get('total_planned_energy', 0)), "actual_energy_wh": float(actual_energy), "number_of_stops": len(mission.get('stops', [])), "outcome": "Completed"})
            [state['completed_orders'].append(oid) for oid in mission.get('order_ids', []) if oid not in state['completed_orders']]
            state['completed_missions'][mission_id] = mission
        end_hub_id = mission.get('end_hub')
        if end_hub_id:
            end_hub_obj = get_hub_by_id(end_hub_id)
            if end_hub_obj:
                drone['home_hub'] = end_hub_id
                hub_pos = end_hub_obj['location']
                drone['pos'] = (float(hub_pos[0]), float(hub_pos[1]), float(hub_pos[2]))
        drone['status'] = 'RECHARGING'
        drone['mission_id'] = None
        drone['available_at'] = state['simulation_time'] + DRONE_RECHARGE_TIME_S
        if mission_id in state['active_missions']: del state['active_missions'][mission_id]

async def broadcast_state():
    if connected_clients:
        state_snapshot = {}
        with state_lock:
            state_snapshot = json.loads(json.dumps(state, cls=system_state.NumpyJSONEncoder))
        full_message = {'simulation_state': state_snapshot, 'drone_list': list(state_snapshot.get('drones', {}).values())}
        state_json = json.dumps(full_message)
        await asyncio.gather(*[client.send_text(state_json) for client in connected_clients])

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept(); connected_clients.add(websocket)
    try:
        await broadcast_state() # Send initial full state
        while True:
            data = await websocket.receive_json()
            command, payload = data.get("type"), data.get("payload")
            response_message = None
            with state_lock:
                if command == "add_order":
                    dest_pos = DESTINATIONS.get(payload['dest_name'])
                    if dest_pos:
                        order_id = f"Order-{uuid.uuid4().hex[:6]}"
                        order = {'id': order_id, 'pos': dest_pos, **payload}
                        dispatch_status = planners['dispatcher'].dispatch_order(state, order, payload['hub_id'])
                        if dispatch_status == "out_of_range":
                            response_message = {"type": "order_out_of_range", "payload": {"order_id": order_id}}
                        else:
                            state['pending_orders'][order_id] = order
                elif command == "reset_simulation":
                    state.update(system_state.reset_state_file())
            if response_message:
                await websocket.send_text(json.dumps(response_message))
            await broadcast_state()
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
    except Exception as e:
        logging.error(f"WebSocket Error: {e}")

async def simulation_loop():
    global active_planning_futures
    while not shutdown_event.is_set():
        try:
            done_futures = [f for f in active_planning_futures if f.done()]
            for future in done_futures:
                active_planning_futures.remove(future)
                try:
                    success, results = future.result()
                    with state_lock:
                        if success:
                            state['drones'].update(results.get('drone_updates', {}))
                            for mid, updates in results.get('mission_updates', {}).items():
                                if mid in state['active_missions']: state['active_missions'][mid].update(updates)
                            for mid in results.get('successful_mission_ids', []):
                                for oid in state['active_missions'][mid]['order_ids']:
                                    if oid in state['pending_orders']: del state['pending_orders'][oid]
                        else:
                            state['drones'].update(results.get('drone_updates', {}))
                            for mid in results.get('mission_failures', []):
                                if mid in state['active_missions']: del state['active_missions'][mid]
                except Exception as e:
                    logging.error(f"Error processing planning result: {e}")
            fleet_manager = planners['fleet_manager']
            executor = planners['executor']
            while len(active_planning_futures) < MAX_CONCURRENT_PLANNERS and not fleet_manager.planning_queue.empty():
                future = executor.submit(fleet_manager.plan_pending_missions, state)
                active_planning_futures.append(future)
            if state.get('simulation_running', False):
                with state_lock:
                    update_simulation(state, planners)
                    contingency_planner.check_for_contingencies(state, planners)
                    event_injector.inject_random_event(state, planners['env'])
                    system_state.save_state(state)
                await broadcast_state()
            await asyncio.sleep(SIMULATION_UI_REFRESH_INTERVAL)
        except asyncio.CancelledError: 
            break
        except Exception as e:
            logging.error(f"FATAL error in simulation loop: {e}", exc_info=True)
            await asyncio.sleep(1)

@app.get("/api/hubs")
async def get_hubs():
    return JSONResponse(content=HUBS)

@app.get("/api/destinations")
async def get_destinations():
    return JSONResponse(content=DESTINATIONS)

app.mount("/", StaticFiles(directory="web", html=True), name="static")

if __name__ == "__main__": uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)