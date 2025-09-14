# FILE: demand_analyzer.py
import logging
import threading
import time
from collections import defaultdict
from typing import Dict, Any

from config import HUBS

class DemandAnalyzer:
    def __init__(self, state: Dict, dispatcher, state_lock: threading.Lock, shutdown_event: threading.Event):
        self.state = state
        self.dispatcher = dispatcher
        self.state_lock = state_lock
        self.shutdown_event = shutdown_event
        self.check_interval_s = 15 * 60  # 15 minutes
        self.hourly_order_counts = defaultdict(int)

    def run(self):
        logging.info("Demand Analyzer thread started.")
        last_hourly_reset = time.time()
        
        while not self.shutdown_event.is_set():
            # Reset hourly counts
            if time.time() - last_hourly_reset > 3600:
                self.hourly_order_counts.clear()
                last_hourly_reset = time.time()
            
            # Perform rebalancing check
            self.check_and_rebalance_fleet()

            # Wait for the next interval or until shutdown is requested
            self.shutdown_event.wait(self.check_interval_s)
        
        logging.info("Demand Analyzer thread shutting down.")

    def check_and_rebalance_fleet(self):
        with self.state_lock:
            # This is a snapshot-in-time analysis
            idle_drones_per_hub = defaultdict(int)
            for drone in self.state['drones'].values():
                if drone['status'] == 'IDLE':
                    idle_drones_per_hub[drone['home_hub']] += 1

        # Find hubs with a surplus and deficit of drones
        hubs_with_surplus = {hub_id: count for hub_id, count in idle_drones_per_hub.items() if count > 3}
        
        # FIX: Iterate over all known HUBS to correctly identify deficits, even for hubs with 0 idle drones.
        hubs_with_deficit = {
            hub['id'] for hub in HUBS 
            if idle_drones_per_hub.get(hub['id'], 0) < 1
        }
        
        # Simple rebalancing logic
        if hubs_with_surplus and hubs_with_deficit:
            for from_hub_id in hubs_with_surplus:
                for to_hub_id in hubs_with_deficit:
                    # In a real system, you'd pick the closest hubs, but for now, any pair works.
                    logging.info(f"Rebalancing triggered: Moving drone from {from_hub_id} to {to_hub_id}")
                    # Dispatcher needs to be thread-safe if it modifies state directly
                    # Our current implementation just adds to a queue, which is safe.
                    self.dispatcher.create_rebalancing_mission(self.state, from_hub_id, to_hub_id)
                    # Only move one drone at a time to prevent over-correction
                    return