# FILE: simulation/event_injector.py
import random
import logging
import numpy as np

from config import DRONE_BATTERY_WH, AREA_BOUNDS

def log_event(state, message):
    """Adds a new message to the persistent event log."""
    import time
    state['log'].insert(0, f"{time.strftime('%H:%M:%S')} - {message}")

def inject_random_event(state, env):
    """
    With a small probability, injects a random failure or event into the simulation.
    """
    # Trigger probability per simulation tick
    if random.random() < 0.005:
        active_drones = [d for d in state['drones'].values() if d['status'] == 'EN ROUTE']
        
        # Only trigger an event if there's a drone to affect
        if not active_drones:
            return

        event_type = random.choice(['BATTERY_FAULT', 'SUDDEN_NFZ'])

        if event_type == 'BATTERY_FAULT':
            drone_to_affect = random.choice(active_drones)
            drone_id = drone_to_affect['id']
            fault_amount = DRONE_BATTERY_WH * 0.25
            state['drones'][drone_id]['battery'] -= fault_amount
            log_event(state, f"⚡️ EVENT: Battery fault on {drone_id}. Lost {fault_amount:.1f}Wh.")

        elif event_type == 'SUDDEN_NFZ':
            lon_min, lat_min, lon_max, lat_max = AREA_BOUNDS
            # Create a reasonably sized NFZ within the bounds
            center_lon = random.uniform(lon_min + 0.005, lon_max - 0.005)
            center_lat = random.uniform(lat_min + 0.005, lat_max - 0.005)
            size = 0.004 # Approx 400m wide
            nfz_bounds = [
                center_lon - size / 2,
                center_lat - size / 2,
                center_lon + size / 2,
                center_lat + size / 2
            ]
            env.add_dynamic_nfz(nfz_bounds)
            log_event(state, f"🚨 EVENT: New temporary No-Fly Zone created near [{center_lon:.3f}, {center_lat:.3f}].")