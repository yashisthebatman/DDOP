# FILE: simulation/event_injector.py
import random
import logging
import numpy as np

from config import AREA_BOUNDS

def log_event(state, message):
    """Adds a new message to the persistent event log."""
    import time
    state['log'].insert(0, f"{time.strftime('%H:%M:%S')} - {message}")

def inject_random_event(state, env):
    """
    With a small probability, injects a random event into the simulation.
    """
    # Trigger probability per simulation tick
    # FIX: Drastically reduced the event probability from 0.005 to 0.0005.
    # This prevents the simulation from being overwhelmed by constant path re-validations.
    if random.random() < 0.0005:
        active_drones = [d for d in state['drones'].values() if d['status'] == 'EN ROUTE']
        
        # Only trigger an event if there's a drone to affect
        if not active_drones:
            return

        # FIX: Removed 'BATTERY_FAULT' event type for more realistic simulation.
        event_type = 'SUDDEN_NFZ'

        if event_type == 'SUDDEN_NFZ':
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