# FILE: server.py

import asyncio
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
import json
from dotenv import load_dotenv
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


# --- Load Environment Variables ---
load_dotenv()

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- FastAPI App Initialization ---
app = FastAPI()
planners = {}
state = {}
connected_clients = set()

# --- Core Simulation Logic (Moved from old app.py) ---
def update_simulation(state, planners):
    """Advances the simulation by one time step and updates drone/mission states."""
    state['simulation_time'] += SIMULATION_TIME_STEP
    coord_manager = planners['coord_manager']
    for drone in state['drones'].values():
        if drone['status'] == 'RECHARGING' and state['simulation_time'] >= drone['available_at']:
            drone['status'] = 'IDLE'; drone['battery'] = DRONE_BATTERY_WH
            logging.info(f"✅ {drone['id']} has finished recharging.")
    active_drones = {d['id']: d for d in state['drones'].values() if d['mission_id'] in state['active_missions']}
    if len(active_drones) > 1: check_and_resolve_conflicts(active_drones, coord_manager)
    missions_to_complete = []
    for mission_id, mission in list(state['active_missions'].items()):
        if mission.get('is_paused', False): continue
        mission['mission_time_elapsed'] += SIMULATION_TIME_STEP
        drone_id = mission['drone_id']
        drone = state['drones'][drone_id]
        if drone['status'] == 'PERFORMING_DELIVERY':
            if state['simulation_time'] >= drone.get('maneuver_complete_at', float('inf')):
                mission['current_stop_index'] += 1; drone['status'] = 'EN ROUTE'
                drone.pop('maneuver_complete_at', None); drone.pop('maneuver_start_pos', None); drone.pop('maneuver_target_pos', None)
            else:
                start_time = drone['maneuver_complete_at'] - DELIVERY_MANEUVER_TIME_SEC
                progress = min(1.0, max(0.0, (state['simulation_time'] - start_time) / DELIVERY_MANEUVER_TIME_SEC))
                start_pos = np.array(drone.get('maneuver_start_pos', drone['pos'])); target_pos = np.array(drone.get('maneuver_target_pos', drone['pos']))
                new_z = start_pos[2] + progress * (target_pos[2] - start_pos[2])
                new_pos_np = np.array([start_pos[0], start_pos[1], new_z])
                drone['pos'] = (float(new_pos_np[0]), float(new_pos_np[1]), float(new_pos_np[2]))
        elif drone['status'] in ['EN ROUTE', 'EMERGENCY_RETURN']:
            mission['flight_time_elapsed'] += SIMULATION_TIME_STEP
            arrived = False
            if drone['status'] == 'EN ROUTE' and mission.get('current_stop_index', 0) < len(mission.get('stops', [])):
                target_pos = mission['stops'][mission['current_stop_index']]['pos']
                if calculate_distance_3d(coord_manager.world_to_meters(drone['pos']), coord_manager.world_to_meters(target_pos)) < 5.0:
                    drone['status'] = 'PERFORMING_DELIVERY'; drone['maneuver_complete_at'] = state['simulation_time'] + DELIVERY_MANEUVER_TIME_SEC
                    drone['maneuver_start_pos'] = drone['pos']; drone['maneuver_target_pos'] = target_pos; arrived = True
                    logging.info(f"🎯 {drone_id} arriving at stop {mission['current_stop_index']+1}. Performing delivery.")
            if not arrived:
                flight_time = max(1, mission.get('total_planned_time', 1) - mission.get('total_maneuver_time', 0))
                progress = min(1.0, mission['flight_time_elapsed'] / flight_time)
                path = mission.get('path_world_coords', [])
                if path and len(path) > 1:
                    idx = int(progress * (len(path) - 1))
                    if idx < len(path) - 1:
                        p1, p2 = np.array(path[idx]), np.array(path[idx + 1]); seg_prog = (progress * (len(path) - 1)) - idx
                        new_pos = p1 + seg_prog * (p2 - p1); drone['pos'] = tuple(new_pos.tolist())
                    else: drone['pos'] = tuple(path[-1])
                elif path:
                     drone['pos'] = tuple(path[-1])
                drone['battery'] = mission.get('start_battery', DRONE_BATTERY_WH) - (progress * mission.get('total_planned_energy', 0))
        if mission.get('total_planned_time', 0) > 0 and mission['mission_time_elapsed'] >= mission['total_planned_time']:
            missions_to_complete.append(mission_id)
    for mission_id in missions_to_complete:
        mission = state['active_missions'][mission_id]; drone_id = mission['drone_id']; drone = state['drones'][drone_id]
        if drone['status'] != 'EMERGENCY_RETURN':
            actual_duration = state['simulation_time'] - mission['start_time']; actual_energy = mission['start_battery'] - drone['battery']
            state['completed_missions_log'].append({"mission_id": mission_id, "drone_id": drone_id, "completion_timestamp": float(state['simulation_time']), "planned_duration_sec": float(mission['total_planned_time']), "actual_duration_sec": float(actual_duration), "planned_energy_wh": float(mission['total_planned_energy']), "actual_energy_wh": float(actual_energy), "number_of_stops": len(mission['destinations']), "outcome": "Completed"})
            logging.info(f"🏁 {drone_id} completed mission {mission_id}."); [state['completed_orders'].append(oid) for oid in mission['order_ids'] if oid not in state['completed_orders']]; state['completed_missions'][mission_id] = mission
        else: logging.info(f"✅ {drone_id} successfully returned to hub after emergency.")
        end_hub_id = mission.get('end_hub')
        if end_hub_id in HUBS:
            drone['home_hub'] = end_hub_id; hub_pos = HUBS[end_hub_id]; drone['pos'] = (float(hub_pos[0]), float(hub_pos[1]), float(hub_pos[2]))
            logging.info(f"🚚 {drone_id} has relocated to new home base: {end_hub_id}.")
        drone['status'] = 'RECHARGING'; drone['mission_id'] = None; drone['available_at'] = state['simulation_time'] + DRONE_RECHARGE_TIME_S
        del state['active_missions'][mission_id]

# --- WebSocket Communication ---
async def broadcast_state():
    """Sends the current state to all connected clients."""
    if connected_clients:
        state_json = json.dumps(state, cls=system_state.NumpyJSONEncoder)
        await asyncio.gather(*[client.send_text(state_json) for client in connected_clients])

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles WebSocket connections for real-time updates."""
    await websocket.accept()
    connected_clients.add(websocket)
    logging.info("Client connected. Total clients: %d", len(connected_clients))
    try:
        state_json = json.dumps(state, cls=system_state.NumpyJSONEncoder)
        await websocket.send_text(state_json)
        
        while True:
            data = await websocket.receive_json()
            command = data.get("type")
            payload = data.get("payload")
            
            if command == "toggle_simulation":
                state['simulation_running'] = not state.get('simulation_running', False)
                logging.info(f"Simulation {'started' if state['simulation_running'] else 'paused'}.")
            elif command == "reset_simulation":
                state.update(system_state.reset_state_file())
                for dest_name, pos in DESTINATIONS.items():
                    surface_alt = planners['env'].get_surface_height((pos[0], pos[1]))
                    DESTINATIONS[dest_name] = (pos[0], pos[1], surface_alt)
                logging.info("--- SYSTEM STATE RESET ---")
            elif command == "add_order":
                env = planners['env']
                base_pos = DESTINATIONS[payload['dest_name']]
                surface_alt = env.get_surface_height((base_pos[0], base_pos[1]))
                final_pos = (base_pos[0], base_pos[1], surface_alt)
                order_id = f"Order-{uuid.uuid4().hex[:6]}"
                state['pending_orders'][order_id] = {
                    'id': order_id, 'pos': final_pos, 'dest_name': payload['dest_name'],
                    'payload_kg': payload['payload_kg'], 'high_priority': payload['high_priority']
                }
                logging.info(f"New order added: {order_id} for {payload['dest_name']}.")
            elif command == "dispatch_missions":
                dispatched = planners['dispatcher'].dispatch_missions(state)
                logging.info(f"Dispatch attempted. Success: {dispatched}")
            
            await broadcast_state()

    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        logging.info("Client disconnected. Total clients: %d", len(connected_clients))
    except Exception as e:
        logging.error(f"WebSocket Error: {e}")
        if websocket in connected_clients:
            connected_clients.remove(websocket)

# --- Main Simulation Loop ---
async def simulation_loop():
    """The main, non-blocking simulation loop."""
    fleet_manager = planners['fleet_manager']
    executor = planners['executor']
    planning_future = None

    while True:
        if state.get('simulation_running', False):
            if planning_future and planning_future.done():
                try:
                    success, results = planning_future.result()
                    if success:
                        logging.info("Fleet plan ready. Applying updates.")
                        state['drones'].update(results.get('drone_updates', {}))
                        for mid, updates in results.get('mission_updates', {}).items():
                            if mid in state['active_missions']: state['active_missions'][mid].update(updates)
                        for mid in results.get('successful_mission_ids', []):
                            for oid in state['active_missions'][mid]['order_ids']:
                                if oid in state['pending_orders']: del state['pending_orders'][oid]
                    else:
                        logging.error(f"Fleet planning failed: {results.get('error', 'Unknown')}")
                        state['drones'].update(results.get('drone_updates', {}))
                except Exception as e:
                    logging.error(f"CRITICAL ERROR in planning thread: {e}")
                finally:
                    planning_future = None

            drones_need_planning = any(d['status'] == 'PLANNING' for d in state['drones'].values())
            if drones_need_planning and not planning_future:
                logging.info("Fleet requires planning. Submitting to background worker...")
                planning_future = executor.submit(fleet_manager.plan_pending_missions, state)
            
            update_simulation(state, planners)
            contingency_planner.check_for_contingencies(state, planners)
            event_injector.inject_random_event(state, planners['env'])

            system_state.save_state(state)
            await broadcast_state()
        
        await asyncio.sleep(SIMULATION_UI_REFRESH_INTERVAL)

# --- Application Startup ---
@app.on_event("startup")
async def startup_event():
    """Initializes planners and starts the simulation loop on server start."""
    global state
    state.update(system_state.load_state())
    
    coord_manager = CoordinateManager()
    env = Environment(WeatherSystem(), coord_manager)
    predictor = EnergyTimePredictor()
    predictor.load_model()
    cbsh_planner = CBSHPlanner(env, coord_manager)
    
    planners['coord_manager'] = coord_manager
    planners['env'] = env
    planners['predictor'] = predictor
    planners['dispatcher'] = Dispatcher(VRPSolver(predictor))
    planners['fleet_manager'] = FleetManager(cbsh_planner, predictor)
    planners['executor'] = ThreadPoolExecutor(max_workers=2)

    for dest_name, pos in DESTINATIONS.items():
        surface_alt = env.get_surface_height((pos[0], pos[1]))
        DESTINATIONS[dest_name] = (pos[0], pos[1], surface_alt)
    
    asyncio.create_task(simulation_loop())

# --- API Endpoints for Frontend ---

@app.get("/api/destinations")
async def get_destinations():
    return JSONResponse(content=DESTINATIONS)

@app.get("/api/token")
async def get_token():
    """Securely provides the Cesium Ion token to the frontend."""
    token = os.getenv("CESIUM_ION_TOKEN")
    if not token:
        logging.error("FATAL: CESIUM_ION_TOKEN not found in .env file!")
        return JSONResponse(content={"error": "Server is missing Cesium token."}, status_code=500)
    return JSONResponse(content={"token": token})

# Mount the static frontend files
app.mount("/", StaticFiles(directory="web", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)