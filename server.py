# FILE: server.py
import asyncio
import logging
import os
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import asynccontextmanager
from typing import Dict, Any, List

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
from simulation.contingency_planner import check_for_contingencies
from simulation.deconfliction import check_and_resolve_conflicts
from utils.geometry import calculate_distance_3d
from demand_analyzer import DemandAnalyzer

# --- Global State ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
state: Dict[str, Any] = {}
planners: Dict[str, Any] = {}
connected_clients = set()
state_lock = threading.Lock()
active_planning_futures: List[Future] = []
demand_analyzer_thread: threading.Thread | None = None
shutdown_event = threading.Event()

# --- Plotly Data Generation --- (Code Unchanged)
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

# --- Simulation Core Logic ---
def _complete_mission(mission_id: str, state: dict):
    mission = state.get('active_missions', {}).get(mission_id)
    if not mission: return

    drone_id = mission['drone_id']
    drone = state['drones'][drone_id]
    
    if drone['status'] != 'EMERGENCY_RETURN':
        actual_duration = state['simulation_time'] - mission.get('start_time', 0)
        actual_energy = mission.get('start_battery', DRONE_BATTERY_WH) - drone['battery']
        log_entry = { "mission_id": mission_id, "drone_id": drone_id, "completion_timestamp": float(state['simulation_time']), "planned_duration_sec": float(mission.get('total_planned_time', 0)), "actual_duration_sec": float(actual_duration), "planned_energy_wh": float(mission.get('total_planned_energy', 0)), "actual_energy_wh": float(actual_energy), "number_of_stops": len(mission.get('stops', [])), "outcome": "Completed" }
        state['completed_missions_log'].append(log_entry)
        state['completed_missions'][mission_id] = mission
        
    if end_hub_id := mission.get('end_hub'):
        if end_hub_obj := get_hub_by_id(end_hub_id):
            drone['home_hub'] = end_hub_id
            hub_pos = end_hub_obj['location']
            drone['pos'] = (float(hub_pos[0]), float(hub_pos[1]), float(hub_pos[2]))
    
    drone.update({'status': 'RECHARGING', 'mission_id': None, 'available_at': state['simulation_time'] + DRONE_RECHARGE_TIME_S})
    
    if mission_id in state['active_missions']:
        del state['active_missions'][mission_id]

def _handle_delivery_maneuver(drone: dict, mission: dict, state: dict):
    total_mass_kg = DRONE_MASS_KG + mission.get('payload_kg', 0.0)
    hover_power_watts = DRONE_BASE_POWER_WATTS + (total_mass_kg * DRONE_ADDITIONAL_WATTS_PER_KG)
    energy_drain_wh = (hover_power_watts * SIMULATION_TIME_STEP) / 3600.0
    drone['battery'] = max(0, drone['battery'] - energy_drain_wh)

    if state['simulation_time'] >= mission.get('maneuver_complete_at', float('inf')):
        stop_idx = mission.get('current_stop_index', 0)
        if 0 <= stop_idx < len(mission.get('stops', [])):
            order_id = mission['stops'][stop_idx]['id']
            if order_id not in state['completed_orders']:
                state['completed_orders'].append(order_id)
        
        mission['current_stop_index'] += 1
        drone.pop('maneuver_complete_at', None)
        
        if mission['current_stop_index'] >= len(mission.get('stops', [])):
            drone['status'] = 'RETURNING_TO_HUB'
        else:
            drone['status'] = 'EN ROUTE'

def _handle_avoidance_maneuver(drone: dict):
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

def _handle_en_route(drone: dict, mission: dict, state: dict, planners: dict):
    path = mission.get('path_world_coords', [])
    if not path or mission.get('current_path_target_index', 1) >= len(path): return

    p_prev = np.array(drone['pos'])
    target_waypoint = np.array(path[mission['current_path_target_index']])
    direction = target_waypoint - p_prev
    dist_to_wp = np.linalg.norm(direction)
    move_dist = DRONE_SPEED_MPS * SIMULATION_TIME_STEP

    if dist_to_wp < move_dist:
        p_next = tuple(target_waypoint.tolist())
        mission['current_path_target_index'] += 1
    else:
        p_next = tuple((p_prev + (direction / dist_to_wp) * move_dist).tolist())
    
    payload = mission.get('payload_kg', 0.0) if drone['status'] == 'EN ROUTE' else 0.0
    wind = planners['env'].weather.get_wind_at_location(*drone['pos'])
    _, energy_drain_wh = planners['predictor'].fallback_predictor.predict(drone['pos'], p_next, payload, wind, mission.get('last_pos'))
    drone['battery'] = max(0, drone['battery'] - energy_drain_wh)
    
    drone['pos'] = p_next
    mission['last_pos'] = p_prev

    if drone['status'] == 'EN ROUTE':
        stop_idx = mission.get('current_stop_index', 0)
        if 0 <= stop_idx < len(mission.get('stops', [])):
            delivery_pos = mission['stops'][stop_idx]['pos']
            if calculate_distance_3d(drone['pos'], delivery_pos) < 5.0:
                drone['status'] = 'PERFORMING_DELIVERY'
                mission['maneuver_complete_at'] = state['simulation_time'] + DELIVERY_MANEUVER_TIME_SEC
                return

    is_final_leg = drone['status'] in ['RETURNING_TO_HUB', 'EMERGENCY_RETURN'] or \
                   (drone['status'] == 'EN ROUTE' and mission.get('current_stop_index', 0) >= len(mission.get('stops', [])))

    if is_final_leg:
        hub_pos = mission['destinations'][-1]
        if calculate_distance_3d(drone['pos'], hub_pos) < 5.0:
            _complete_mission(mission['mission_id'], state)
            return

def update_simulation(state: dict, planners: dict):
    state['simulation_time'] += SIMULATION_TIME_STEP
    active_drones_list = [d for d in state['drones'].values() if d['status'] not in ['IDLE', 'PLANNING', 'RECHARGING']]
    if len(active_drones_list) > 1:
        check_and_resolve_conflicts({d['id']: d for d in active_drones_list}, planners)
    
    for drone in list(state['drones'].values()):
        if drone.get('is_paused'): continue
        if drone['status'] == 'RECHARGING':
            if state['simulation_time'] >= drone['available_at']:
                drone.update({'status': 'IDLE', 'battery': DRONE_BATTERY_WH})
            continue
        if drone['status'] in ['IDLE', 'PLANNING']:
            continue
        if check_for_contingencies(state, planners, drone):
            continue 
        if mission := state['active_missions'].get(drone.get('mission_id')):
            if drone['status'] == 'PERFORMING_DELIVERY':
                _handle_delivery_maneuver(drone, mission, state)
            elif drone['status'] == 'AVOIDING':
                _handle_avoidance_maneuver(drone)
            elif drone['status'] in ['EN ROUTE', 'RETURNING_TO_HUB', 'EMERGENCY_RETURN']:
                _handle_en_route(drone, mission, state, planners)

# --- FastAPI Application (Unchanged) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global state, demand_analyzer_thread
    with state_lock: state.update(system_state.load_state())
    coord_manager = CoordinateManager()
    env = Environment(WeatherSystem(), coord_manager)
    predictor = EnergyTimePredictor(); predictor.load_model()
    cbsh_planner = CBSHPlanner(env, coord_manager)
    fleet_manager = FleetManager(cbsh_planner, predictor, state_lock)
    dispatcher = Dispatcher(fleet_manager, predictor)
    planners.update({
        'coord_manager': coord_manager, 'env': env, 'predictor': predictor,
        'dispatcher': dispatcher, 'fleet_manager': fleet_manager,
        'executor': ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PLANNERS)
    })
    demand_analyzer = DemandAnalyzer(state, dispatcher, state_lock, shutdown_event)
    demand_analyzer_thread = threading.Thread(target=demand_analyzer.run, daemon=True)
    demand_analyzer_thread.start()
    simulation_task = asyncio.create_task(simulation_loop())
    logging.info("--- System Initialized and Ready ---")
    yield
    shutdown_event.set()
    if demand_analyzer_thread: demand_analyzer_thread.join()
    simulation_task.cancel()
    planners['executor'].shutdown(wait=True)
    logging.info("--- System Shutting Down ---")

app = FastAPI(lifespan=lifespan)

async def broadcast_state():
    if not connected_clients: return
    state_snapshot, plotly_data = {}, None
    with state_lock:
        state_snapshot = json.loads(json.dumps(state, cls=system_state.NumpyJSONEncoder))
        if planners: plotly_data = generate_plotly_data(state, planners['coord_manager'], planners['env'])
    full_message = {'simulation_state': state_snapshot, 'drone_list': list(state_snapshot.get('drones', {}).values()), 'plotly_data': plotly_data}
    state_json = json.dumps(full_message)
    await asyncio.gather(*[client.send_text(state_json) for client in connected_clients], return_exceptions=False)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept(); connected_clients.add(websocket)
    try:
        await broadcast_state() 
        while True:
            data = await websocket.receive_json()
            command, payload = data.get("type"), data.get("payload")
            response_message = None
            with state_lock:
                if command == "add_order":
                    if dest_pos := DESTINATIONS.get(payload['dest_name']):
                        order_id = f"Order-{uuid.uuid4().hex[:6]}"
                        order = {'id': order_id, 'pos': dest_pos, 'dest_name': payload['dest_name'], **payload}
                        if (dispatch_status := planners['dispatcher'].dispatch_order(state, order, payload['hub_id'])) == "out_of_range":
                            response_message = {"type": "order_out_of_range", "payload": {"order_id": order_id}}
                        else: state['pending_orders'][order_id] = order
                elif command == "reset_simulation": state.update(system_state.reset_state_file())
                elif command == "toggle_simulation": state['simulation_running'] = not state.get('simulation_running', False)
            if response_message: await websocket.send_text(json.dumps(response_message))
            await broadcast_state()
    except WebSocketDisconnect: pass
    except Exception as e: logging.error(f"WebSocket Error: {e}")
    finally: connected_clients.remove(websocket)

def _process_planning_results(future: Future, state: dict):
    try:
        success, results = future.result()
        if success:
            state['drones'].update(results.get('drone_updates', {}))
            for mid, updates in results.get('mission_updates', {}).items():
                if mid in state['active_missions']: state['active_missions'][mid].update(updates)
            for mid in results.get('successful_mission_ids', []):
                if mission := state['active_missions'].get(mid):
                    for oid in mission['order_ids']:
                        state['pending_orders'].pop(oid, None)
        else:
            state['drones'].update(results.get('drone_updates', {}))
            for mid in results.get('mission_failures', []):
                state['active_missions'].pop(mid, None)
    except Exception as e:
        logging.error(f"Error processing planning result: {e}")

async def simulation_loop():
    while not shutdown_event.is_set():
        try:
            done_futures = [f for f in active_planning_futures if f.done()]
            for future in done_futures:
                active_planning_futures.remove(future)
                with state_lock: _process_planning_results(future, state)
            
            fleet_manager = planners['fleet_manager']
            while len(active_planning_futures) < MAX_CONCURRENT_PLANNERS and not fleet_manager.planning_queue.empty():
                future = planners['executor'].submit(fleet_manager.plan_pending_missions, state)
                active_planning_futures.append(future)

            if state.get('simulation_running', False):
                with state_lock:
                    update_simulation(state, planners)
                    event_injector.inject_random_event(state, planners['env'])
                    system_state.save_state(state)
                await broadcast_state()
            
            await asyncio.sleep(SIMULATION_UI_REFRESH_INTERVAL)
        except asyncio.CancelledError: break
        except Exception as e:
            logging.error(f"FATAL error in simulation loop: {e}", exc_info=True)
            await asyncio.sleep(1)

@app.get("/api/hubs")
async def get_hubs(): return JSONResponse(content=HUBS)

@app.get("/api/destinations")
async def get_destinations(): return JSONResponse(content=DESTINATIONS)

app.mount("/", StaticFiles(directory="web", html=True), name="static")

if __name__ == "__main__": uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)