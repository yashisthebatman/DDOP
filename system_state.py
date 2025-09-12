# FILE: system_state.py

import os
import numpy as np
from tinydb import TinyDB
from config import HUBS, DESTINATIONS, DRONE_BATTERY_WH, DRONE_MAX_PAYLOAD_KG, MODEL_FILE_PATH

# --- Constants ---
DB_FILE = 'system_state.json'
# We store the entire application state as a single document in the DB
STATE_DOC_ID = 1

def get_initial_state():
    """
    Defines the default structure and initial values for the system state.
    This is used to create the database file on the first run.
    """
    drones = {
        f"Drone {i+1}": {
            'id': f"Drone {i+1}", # Add id for convenience
            'pos': HUBS[list(HUBS.keys())[i % len(HUBS)]],
            'home_hub': list(HUBS.keys())[i % len(HUBS)],
            'battery': DRONE_BATTERY_WH,
            'max_payload_kg': DRONE_MAX_PAYLOAD_KG,
            'status': 'IDLE',  # IDLE, PLANNING, EN ROUTE, RECHARGING
            'mission_id': None,
            'available_at': 0.0  # Simulation time when the drone becomes available after charging
        } for i in range(3)
    }

    pending_orders = {
        name: {
            'pos': pos,
            # Assign a random, realistic payload to each predefined destination
            'payload_kg': round(np.random.uniform(0.5, DRONE_MAX_PAYLOAD_KG - 0.5), 2),
            'id': name
        } for name, pos in DESTINATIONS.items()
    }

    return {
        'drones': drones,
        'pending_orders': pending_orders,
        'active_missions': {},
        'completed_missions': {},
        'completed_orders': [],
        'simulation_time': 0.0,
        'log': ["System initialized. Welcome to the Drone Delivery Simulator."],
        'simulation_running': False,
        'completed_missions_log': [], # For analytics dashboard
        'active_model_path': MODEL_FILE_PATH # For MLOps feedback loop
    }

def load_state():
    """
    Loads the system state from the TinyDB file.
    If the file or state document doesn't exist, it creates and returns an initial state.
    """
    db = TinyDB(DB_FILE)
    state_doc = db.get(doc_id=STATE_DOC_ID)

    if state_doc:
        # Backwards compatibility: ensure all keys from the initial state are present
        initial_state = get_initial_state()
        for key in initial_state:
            if key not in state_doc:
                state_doc[key] = initial_state[key]
        # Add drone IDs if they are missing from older states
        for drone_id, drone_data in state_doc['drones'].items():
            if 'id' not in drone_data:
                drone_data['id'] = drone_id
        return state_doc
    else:
        # The database is empty, so we create the first state document
        initial_state = get_initial_state()
        db.insert(initial_state)
        # Retrieve it again to ensure it has the doc_id from the DB
        return db.get(doc_id=STATE_DOC_ID)

def save_state(state):
    """
    Saves the entire system state dictionary to the TinyDB file, overwriting the old state.
    """
    db = TinyDB(DB_FILE)
    db.update(state, doc_ids=[STATE_DOC_ID])

def reset_state_file():
    """
    Deletes the existing state file and creates a new one with initial values.
    Useful for resetting the simulation completely.
    """
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    return load_state()