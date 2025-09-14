# FILE: system_state.py

import os
import numpy as np
from tinydb import TinyDB, JSONStorage
import uuid
import json
import random
from config import HUBS, DRONES_PER_HUB, DRONE_BATTERY_WH, DRONE_MAX_PAYLOAD_KG, MODEL_FILE_PATH

# --- Custom JSON Encoder to handle NumPy types ---
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

# --- Constants ---
DB_FILE = 'system_state.json'
STATE_DOC_ID = 1

def get_initial_state():
    """Defines the default structure and initial values for the system state."""
    drones = {}
    drone_counter = 1
    for hub in HUBS:
        for i in range(DRONES_PER_HUB):
            drone_id = f"Drone {drone_counter}"
            drones[drone_id] = {
                'id': drone_id,
                'pos': hub['location'],
                'home_hub': hub['id'], # MODIFIED: Use hub ID
                'battery': DRONE_BATTERY_WH,
                'max_payload_kg': DRONE_MAX_PAYLOAD_KG,
                'status': 'IDLE',
                'mission_id': None,
                'available_at': 0.0,
                # NEW attributes for Phase 2
                'battery_health': round(random.uniform(95.0, 100.0), 1),
                'charge_cycles': random.randint(5, 50)
            }
            drone_counter += 1

    return {
        'drones': drones, 'pending_orders': {}, 'active_missions': {},
        'completed_missions': {}, 'completed_orders': [], 'simulation_time': 0.0,
        'log': ["System initialized. Add orders to begin."], 'simulation_running': False,
        'completed_missions_log': [], 'active_model_path': MODEL_FILE_PATH
    }

def load_state():
    """Loads the system state, ensuring all keys and drone IDs are present."""
    db = TinyDB(DB_FILE, storage=JSONStorage, indent=4, cls=NumpyJSONEncoder)
    state_doc = db.get(doc_id=STATE_DOC_ID)

    if state_doc:
        initial_state = get_initial_state()
        for key in initial_state:
            if key not in state_doc:
                state_doc[key] = initial_state[key]
        for drone_id, drone_data in state_doc.get('drones', {}).items():
            if 'id' not in drone_data:
                drone_data['id'] = drone_id
        return state_doc
    else:
        initial_state = get_initial_state()
        db.insert(initial_state)
        return db.get(doc_id=STATE_DOC_ID)

def save_state(state):
    db = TinyDB(DB_FILE, storage=JSONStorage, indent=4, cls=NumpyJSONEncoder)
    db.update(state, doc_ids=[STATE_DOC_ID])

def reset_state_file():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    db = TinyDB(DB_FILE, storage=JSONStorage, indent=4, cls=NumpyJSONEncoder)
    initial_state = get_initial_state()
    db.insert(initial_state)
    return db.get(doc_id=STATE_DOC_ID)