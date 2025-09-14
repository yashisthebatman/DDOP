# FILE: dispatch/dispatcher.py
import logging
import uuid
from typing import Dict, Any

from dispatch.vrp_solver import VRPSolver
from fleet.manager import Mission, FleetManager
from config import DRONE_BATTERY_WH, HUBS, MIN_ORDERS_TO_DISPATCH

class Dispatcher:
    """Decides when to batch orders and dispatch drones."""

    def __init__(self, vrp_solver: VRPSolver, fleet_manager: FleetManager):
        self.vrp_solver = vrp_solver
        self.fleet_manager = fleet_manager

    def _get_eligible_drones(self, state: Dict[str, Any]) -> list:
        """Finds all IDLE drones with sufficient battery for a typical mission."""
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
        and adds new missions to the FleetManager's planning queue.
        Returns True if a dispatch occurred, False otherwise.
        """
        pending_orders = list(state['pending_orders'].values())
        
        # --- Trigger on batch size OR high priority order ---
        any_high_priority = any(o.get('high_priority', False) for o in pending_orders)
        if len(pending_orders) < MIN_ORDERS_TO_DISPATCH and not any_high_priority:
            return False
        
        eligible_drones = self._get_eligible_drones(state)
        if not eligible_drones:
            logging.info("Dispatch trigger met, but no eligible drones available.")
            return False

        logging.info(f"Dispatch trigger conditions met for {len(pending_orders)} orders. Running VRP solver with {len(eligible_drones)} drones...")

        # --- Generate Optimal Tours ---
        tours = self.vrp_solver.generate_tours(eligible_drones, pending_orders)

        if not tours:
            return False

        # --- Create Mission objects and enqueue them for planning ---
        missions_created = 0
        for tour in tours:
            drone_id = tour['drone_id']
            drone = state['drones'][drone_id]
            
            if drone['status'] != 'IDLE':
                logging.warning(f"VRP assigned tour to drone {drone_id}, but its status is now {drone['status']}. Skipping.")
                continue

            order_ids = [stop['id'] for stop in tour['stops']]
            
            destinations = [stop['pos'] for stop in tour['stops']]
            end_hub_pos = HUBS[tour['end_hub_id']]
            destinations.append(end_hub_pos)
            
            mission_id = f"M-{uuid.uuid4().hex[:6]}"
            
            mission_obj = Mission(
                mission_id=mission_id,
                drone_id=drone_id, 
                start_pos=drone['pos'],
                destinations=destinations, 
                payload_kg=tour['payload'],
                order_ids=order_ids
            )
            # Store full order details and hub info
            mission_obj.stops = tour['stops']
            mission_obj.start_hub = tour['start_hub_id']
            mission_obj.end_hub = tour['end_hub_id']

            # Enqueue mission for the FleetManager to plan.
            # This decouples dispatching from planning.
            self.fleet_manager.add_mission_to_queue(mission_obj)
            missions_created += 1

        if missions_created > 0:
            logging.info(f"Dispatcher created and enqueued {missions_created} new missions.")
            return True
        
        return False