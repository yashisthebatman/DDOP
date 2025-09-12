# FILE: dispatch/dispatcher.py
import logging
import uuid
from typing import Dict, Any

from dispatch.vrp_solver import VRPSolver
from fleet.manager import Mission
from config import DRONE_BATTERY_WH

# --- Dispatcher Trigger Conditions ---
MIN_ORDERS_FOR_BATCH = 3

class Dispatcher:
    """Decides when to batch orders and dispatch drones."""

    def __init__(self, vrp_solver: VRPSolver):
        self.vrp_solver = vrp_solver

    def _get_eligible_drones(self, state: Dict[str, Any]) -> list:
        """Finds IDLE drones with sufficient battery for a typical mission."""
        eligible = []
        for drone_id, drone in state['drones'].items():
            # A simple pre-check: require at least 40% battery to be considered for a batch.
            if drone['status'] == 'IDLE' and drone['battery'] > DRONE_BATTERY_WH * 0.4:
                # Add the drone's ID to its own dictionary for easy access in the solver
                drone_with_id = drone.copy()
                drone_with_id['id'] = drone_id
                eligible.append(drone_with_id)
        return eligible

    def dispatch_missions(self, state: Dict[str, Any]) -> bool:
        """
        Main dispatch logic. Checks trigger conditions, runs VRP solver,
        and updates the system state with new missions.
        Returns True if a dispatch occurred, False otherwise.
        """
        pending_orders = list(state['pending_orders'].values())
        
        # --- Trigger Condition Check ---
        if len(pending_orders) < MIN_ORDERS_FOR_BATCH:
            return False # Not enough orders to warrant a batch computation
        
        eligible_drones = self._get_eligible_drones(state)
        if not eligible_drones:
            return False # No drones available

        logging.info("Dispatch trigger conditions met. Running VRP solver...")

        # --- Generate Optimal Tours ---
        tours = self.vrp_solver.generate_tours(eligible_drones, pending_orders)

        if not tours:
            return False

        # --- Update State with New Missions ---
        dispatched_order_ids = set()
        for tour in tours:
            drone_id = tour['drone_id']
            drone = state['drones'][drone_id]
            
            if drone['status'] != 'IDLE':
                logging.warning(f"VRP assigned tour to drone {drone_id}, but its status is now {drone['status']}. Skipping.")
                continue

            order_ids = [stop['id'] for stop in tour['stops']]
            destinations = [stop['pos'] for stop in tour['stops']]
            
            mission_id = f"M-{uuid.uuid4().hex[:6]}"
            
            mission_obj = Mission(
                mission_id=mission_id,
                drone_id=drone_id, 
                start_pos=drone['pos'],
                destinations=destinations, 
                payload_kg=tour['payload'],
                order_ids=order_ids
            )
            
            state['active_missions'][mission_id] = mission_obj.to_dict()

            # Update drone and order states
            drone['status'] = 'PLANNING'
            drone['mission_id'] = mission_id
            
            # MODIFICATION: Do NOT delete orders here. They are deleted in the main
            # app loop only after planning succeeds. This prevents order loss.
            for order_id in order_ids:
                dispatched_order_ids.add(order_id)

        if dispatched_order_ids:
            logging.info(f"Dispatcher created {len(tours)} new missions for orders: {dispatched_order_ids}")
            return True
        
        return False