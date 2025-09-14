# FILE: dispatch/dispatcher.py
import logging
import uuid
from typing import Dict, Any, List, Optional
import numpy as np

from fleet.manager import Mission, FleetManager
# FIX: Import the new safety margin config
from config import HUBS, RTH_BATTERY_THRESHOLD_FACTOR, DISPATCH_ENERGY_SAFETY_MARGIN
from utils.geometry import calculate_distance_3d
from ml_predictor.predictor import EnergyTimePredictor
from utils.coordinate_manager import CoordinateManager

def get_hub_by_id(hub_id: str) -> Optional[Dict]:
    for hub in HUBS:
        if hub['id'] == hub_id:
            return hub
    return None

class Dispatcher:
    """
    Handles individual order dispatching, including energy prediction,
    "smart" drone selection, and mission creation.
    """
    def __init__(self, fleet_manager: FleetManager, predictor: EnergyTimePredictor):
        self.fleet_manager = fleet_manager
        self.predictor = predictor
        self.coord_manager = CoordinateManager()

    def _get_eligible_drones_at_hub(self, state: Dict[str, Any], hub_id: str) -> List[Dict]:
        """Finds all IDLE drones at a specific hub."""
        return [
            drone for drone in state['drones'].values()
            if drone['status'] == 'IDLE' and drone['home_hub'] == hub_id
        ]

    def _score_drone(self, drone: Dict, proximity_dist_m: float) -> float:
        """Calculates a score for a drone based on multiple factors."""
        battery_weight = 0.5
        health_weight = 0.4
        proximity_weight = 0.1
        norm_battery = drone.get('battery', 0) / 200.0
        norm_health = drone.get('battery_health', 0) / 100.0
        norm_proximity = 1 / (1 + proximity_dist_m)
        score = (
            (norm_battery * battery_weight) +
            (norm_health * health_weight) +
            (norm_proximity * proximity_weight)
        )
        return score

    def dispatch_order(self, state: Dict, order: Dict, hub_id: str) -> str:
        """
        Main dispatch logic for a single order. Finds the best drone and
        enqueues a mission, or returns an out-of-range status.
        """
        start_hub = get_hub_by_id(hub_id)
        if not start_hub:
            return "error_hub_not_found"
        
        _, outbound_energy = self.predictor.predict(start_hub['location'], order['pos'], order['payload_kg'], [0,0,0], None)
        _, return_energy = self.predictor.predict(order['pos'], start_hub['location'], 0, [0,0,0], None)
        
        required_energy = (outbound_energy + return_energy) * RTH_BATTERY_THRESHOLD_FACTOR * DISPATCH_ENERGY_SAFETY_MARGIN

        eligible_drones = self._get_eligible_drones_at_hub(state, hub_id)
        drones_with_enough_battery = [d for d in eligible_drones if d['battery'] > required_energy]

        if not drones_with_enough_battery:
            logging.warning(f"Order {order['id']} out of range for all drones at {hub_id}. Required: {required_energy:.2f}Wh")
            return "out_of_range"
        
        scored_drones = []
        for drone in drones_with_enough_battery:
            drone_pos_m = self.coord_manager.world_to_meters(drone['pos'])
            order_pos_m = self.coord_manager.world_to_meters(order['pos'])
            proximity = calculate_distance_3d(drone_pos_m, order_pos_m)
            score = self._score_drone(drone, proximity)
            scored_drones.append((score, drone))
        
        scored_drones.sort(key=lambda x: (-x[0], x[1]['id']))
        best_drone = scored_drones[0][1]
        best_drone_id = best_drone['id']
        
        mission_id = f"M-{uuid.uuid4().hex[:6]}"
        mission_obj = Mission(
            mission_id=mission_id,
            drone_id=best_drone_id, 
            start_pos=best_drone['pos'],
            destinations=[order['pos'], start_hub['location']],
            payload_kg=order['payload_kg'],
            order_ids=[order['id']]
        )
        mission_obj.stops = [order]
        mission_obj.start_hub = hub_id
        mission_obj.end_hub = hub_id

        self.fleet_manager.add_mission_to_queue(mission_obj)
        logging.info(f"Dispatched order {order['id']} to drone {best_drone_id} from {hub_id}.")
        return "dispatched"

    def create_rebalancing_mission(self, state: Dict, from_hub_id: str, to_hub_id: str):
        """Creates a mission for a drone to fly from one hub to another."""
        eligible_drones = self._get_eligible_drones_at_hub(state, from_hub_id)
        if not eligible_drones:
            logging.warning(f"Rebalancing requested from {from_hub_id}, but no drones are available.")
            return

        drone_to_move = max(eligible_drones, key=lambda d: d.get('charge_cycles', 0))
        from_hub = get_hub_by_id(from_hub_id)
        to_hub = get_hub_by_id(to_hub_id)
        
        mission_id = f"REBALANCE-{uuid.uuid4().hex[:6]}"
        mission_obj = Mission(
            mission_id=mission_id,
            drone_id=drone_to_move['id'],
            start_pos=drone_to_move['pos'],
            destinations=[to_hub['location']],
            payload_kg=0, order_ids=[]
        )
        mission_obj.start_hub = from_hub_id
        mission_obj.end_hub = to_hub_id
        mission_obj.stops = []

        self.fleet_manager.add_mission_to_queue(mission_obj)
        logging.info(f"Created rebalancing mission for {drone_to_move['id']} from {from_hub_id} to {to_hub_id}.")