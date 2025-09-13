# FILE: system_state.py

import os
import numpy as np
from tinydb import TinyDB, JSONStorage
import uuid
import json
from config import HUBS, DESTINATIONS, DRONE_BATTERY_WH, DRONE_MAX_PAYLOAD_KG, MODEL_FILE_PATH

# --- Custom JSON Encoder to handle NumPy types ---
class NumpyJSONEncoder(json.JSONEncoder):
    """
    A special JSON encoder that can handle NumPy data types.
    This is the permanent fix to prevent JSONDecodeError corruption.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

# --- Constants ---
DB_FILE = 'system_state.json'
STATE_DOC_ID = 1

def get_initial_state():
    """
    Defines the default structure and initial values for the system state.
    """
    drones = {}
    for i in range(15):
        drone_id = f"Drone {i+1}"
        home_hub_name = list(HUBS.keys())[i % len(HUBS)]
        drones[drone_id] = {
            'id': drone_id,
            'pos': HUBS[home_hub_name],
            'home_hub': home_hub_name,
            'battery': DRONE_BATTERY_WH,
            'max_payload_kg': DRONE_MAX_PAYLOAD_KG,
            'status': 'IDLE',
            'mission_id': None,
            'available_at': 0.0
        }

    # FIX: Start with zero pending orders. The user will add them manually via the UI.
    pending_orders = {}

    return {
        'drones': drones,
        'pending_orders': pending_orders,
        'active_missions': {},
        'completed_missions': {},
        'completed_orders': [],
        'simulation_time': 0.0,
        'log': ["System initialized. Add orders to begin."],
        'simulation_running': False,
        'completed_missions_log': [],
        'active_model_path': MODEL_FILE_PATH
    }

def load_state():
    """
    Loads the system state from the TinyDB file.
    If the file or state document doesn't exist, it creates and returns an initial state.
    """
    db = TinyDB(DB_FILE, storage=JSONStorage, indent=4, cls=NumpyJSONEncoder)
    state_doc = db.get(doc_id=STATE_DOC_ID)

    if state_doc:
        initial_state = get_initial_state()
        # Ensure all keys from the default state exist in the loaded state
        for key in initial_state:
            if key not in state_doc:
                state_doc[key] = initial_state[key]
        for drone_id, drone_data in state_doc['drones'].items():
            if 'id' not in drone_data:
                drone_data['id'] = drone_id
        return state_doc
    else:
        initial_state = get_initial_state()
        db.insert(initial_state)
        return db.get(doc_id=STATE_DOC_ID)

def save_state(state):
    """
    Saves the entire system state dictionary to the TinyDB file, overwriting the old state.
    """
    db = TinyDB(DB_FILE, storage=JSONStorage, indent=4, cls=NumpyJSONEncoder)
    db.update(state, doc_ids=[STATE_DOC_ID])

def reset_state_file():
    """
    Deletes the existing state file and creates a new one with initial values.
    """
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    db = TinyDB(DB_FILE, storage=JSONStorage, indent=4, cls=NumpyJSONEncoder)
    initial_state = get_initial_state()
    db.insert(initial_state)
    return db.get(doc_id=STATE_DOC_ID)