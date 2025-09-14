# FILE: dispatch/dispatcher.py
import logging
import uuid
from typing import Dict, Any, List, Optional
import numpy as np

from fleet.manager import Mission, FleetManager
from config import HUBS
from utils.geometry import calculate_distance_3d

# Helper function to find a hub by its ID
def get_hub_by_id(hub_id: str) -> Optional[Dict]:
    for hub in HUBS:
        if hub['id'] == hub_id:
            return hub
    return None

class DroneEnergyPredictor:
    """A simple heuristic-based energy predictor."""
    # Rough estimate: Joules per meter, converted to Watt-hours
    # (1 Wh = 3600 Joules). This factor can be replaced by a real model.
    ENERGY_FACTOR_WH_PER_METER = 0.01

    def predict_round_trip_energy(self, start_hub_pos: tuple, dest_pos: tuple) -> float:
        """Estimates the energy for a hub -> destination -> hub round trip."""
        one_way_dist_m = calculate_distance_3d(start_hub_pos, dest_pos)
        round_trip_dist_m = one_way_dist_m * 2
        return round_trip_dist_m * self.ENERGY_FACTOR_WH_PER_METER

class Dispatcher:
    """
    Handles individual order dispatching, including energy prediction,
    "smart" drone selection, and mission creation.
    """
    def __init__(self, fleet_manager: FleetManager):
        self.fleet_manager = fleet_manager
        self.energy_predictor = DroneEnergyPredictor()

    def _get_eligible_drones_at_hub(self, state: Dict[str, Any], hub_id: str) -> List[Dict]:
        """Finds all IDLE drones at a specific hub."""
        return [
            drone for drone in state['drones'].values()
            if drone['status'] == 'IDLE' and drone['home_hub'] == hub_id
        ]

    def _score_drone(self, drone: Dict, proximity_dist_m: float) -> float:
        """Calculates a score for a drone based on multiple factors."""
        # Weights can be tuned
        battery_weight = 0.5
        health_weight = 0.4
        proximity_weight = 0.1

        # Normalize values to be roughly 0-1
        norm_battery = drone.get('battery', 0) / 200.0
        norm_health = drone.get('battery_health', 0) / 100.0
        # Inverse proximity: closer is better. Add 1 to avoid division by zero.
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
        
        # 1. Predict energy for the round trip
        required_energy = self.energy_predictor.predict_round_trip_energy(
            start_hub['location'], order['pos']
        )

        # 2. Find and score eligible drones
        eligible_drones = self._get_eligible_drones_at_hub(state, hub_id)
        drones_with_enough_battery = [d for d in eligible_drones if d['battery'] > required_energy]

        if not drones_with_enough_battery:
            logging.warning(f"Order {order['id']} is out of range for all drones at {hub_id}.")
            return "out_of_range"
        
        # 3. Perform "smart" selection
        scored_drones = []
        for drone in drones_with_enough_battery:
            proximity = calculate_distance_3d(drone['pos'], order['pos'])
            score = self._score_drone(drone, proximity)
            scored_drones.append((score, drone))
        
        # Sort by score descending (highest score is best)
        scored_drones.sort(key=lambda x: x[0], reverse=True)
        best_drone = scored_drones[0][1]
        best_drone_id = best_drone['id']
        
        # 4. Create and enqueue the mission
        mission_id = f"M-{uuid.uuid4().hex[:6]}"
        mission_obj = Mission(
            mission_id=mission_id,
            drone_id=best_drone_id, 
            start_pos=best_drone['pos'],
            destinations=[order['pos'], start_hub['location']], # Delivery -> Return
            payload_kg=order['payload_kg'],
            order_ids=[order['id']]
        )
        mission_obj.stops = [order]
        mission_obj.start_hub = hub_id
        mission_obj.end_hub = hub_id # Simple missions return to the same hub

        self.fleet_manager.add_mission_to_queue(mission_obj)
        logging.info(f"Dispatched order {order['id']} to drone {best_drone_id} from {hub_id}.")
        return "dispatched"

    def create_rebalancing_mission(self, state: Dict, from_hub_id: str, to_hub_id: str):
        """Creates a mission for a drone to fly from one hub to another."""
        eligible_drones = self._get_eligible_drones_at_hub(state, from_hub_id)
        if not eligible_drones:
            logging.warning(f"Rebalancing requested from {from_hub_id}, but no drones are available.")
            return

        # Pick the drone with the most charge cycles (the "most used" one)
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
        mission_obj.stops = [] # No delivery stops

        self.fleet_manager.add_mission_to_queue(mission_obj)
        logging.info(f"Created rebalancing mission for {drone_to_move['id']} from {from_hub_id} to {to_hub_id}.")