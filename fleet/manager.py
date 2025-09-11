# FILE: fleet/manager.py
import logging
import pandas as pd
import os
from typing import Dict, List
from fleet.cbs_components import Agent
from planners.cbsh_planner import CBSHPlanner
from ml_predictor.predictor import EnergyTimePredictor
from config import TRAINING_DATA_PATH, RETRAINING_THRESHOLD, MODEL_FILE_PATH
from sklearn.ensemble import RandomForestRegressor
import joblib

class Mission:
    # ... (class is unchanged) ...
    def __init__(self, drone_id, start_pos, destinations, payload_kg, optimization_weights):
        self.drone_id, self.start_pos, self.destinations, self.payload_kg, self.optimization_weights = drone_id, start_pos, destinations, payload_kg, optimization_weights
        self.current_leg_index, self.state, self.path, self.path_world_coords = 0, "PENDING", [], []
        self.total_planned_energy, self.total_planned_time = 0.0, 0.0
    def get_current_leg(self):
        if self.is_complete(): return None, None
        start = self.start_pos if self.current_leg_index == 0 else self.destinations[self.current_leg_index - 1]
        return start, self.destinations[self.current_leg_index]
    def advance_leg(self):
        if not self.is_complete():
            self.current_leg_index += 1
            self.path, self.path_world_coords = [], []
            self.state = "COMPLETED" if self.is_complete() else "PENDING"
    def is_complete(self):
        return self.current_leg_index >= len(self.destinations)

class FleetManager:
    # ... (init and add_mission are unchanged) ...
    def __init__(self, cbs_planner: CBSHPlanner, predictor: EnergyTimePredictor):
        self.missions: Dict[str, Mission] = {}
        self.cbs_planner = cbs_planner
        self.predictor = predictor
        self.fleet_solution = None
        self.data_points_collected = 0
    def add_mission(self, mission: Mission):
        self.missions[mission.drone_id] = mission
        logging.info(f"Added mission for drone {mission.drone_id}")

    def execute_planning_cycle(self):
        active_agents: List[Agent] = [
            Agent(id=m.drone_id, start_pos=m.get_current_leg()[0], goal_pos=m.get_current_leg()[1], config={'payload_kg': m.payload_kg, **m.optimization_weights})
            for m in self.missions.values() if m.state in ["PENDING", "WAITING_FOR_FLEET"] and not m.is_complete()
        ]
        if not active_agents: return False
        
        solution = self.cbs_planner.plan_fleet(active_agents)
        if not solution:
            logging.error("CBSH planning FAILED.")
            for agent in active_agents: self.missions[agent.id].state = "PLANNING_FAILED"
            return False

        logging.info("CBSH planning successful. Updating mission paths.")
        self.fleet_solution, new_data = solution, []
        for agent in active_agents:
            mission = self.missions[agent.id]
            mission.path = solution[agent.id]
            mission.state = "IN_PROGRESS"
            world_path = [p[0] for p in mission.path]
            mission.path_world_coords = self.cbs_planner.smoother.smooth_path(world_path, self.cbs_planner.env)

            # FIX: Add a check for empty paths to prevent crashes.
            if mission.path and len(mission.path) > 1:
                total_energy, p_prev = 0, None
                for i in range(len(world_path) - 1):
                    p1, p2 = world_path[i], world_path[i+1]
                    wind = self.cbs_planner.env.weather.get_wind_at_location(*p1)
                    _, energy_pred = self.predictor.predict(p1, p2, mission.payload_kg, wind, p_prev)
                    total_energy += energy_pred
                    time_true, energy_true = self.predictor.fallback_predictor.predict(p1, p2, mission.payload_kg, wind, p_prev)
                    features = self.predictor._extract_features(p1, p2, mission.payload_kg, wind, p_prev)
                    new_data.append(features + [time_true, energy_true])
                    p_prev = p1
                mission.total_planned_energy = total_energy
                mission.total_planned_time = mission.path[-1][1]
            else: # Handle case of no path or a path with only a start point
                mission.total_planned_energy = 0
                mission.total_planned_time = 0

        if new_data:
            self._log_training_data(new_data)
            self._check_and_trigger_retraining()
        return True
    
    # ... (_log_training_data, _check_and_trigger_retraining, check_if_all_legs_complete are unchanged)
    def _log_training_data(self, new_data: List[List]):
        logging.info(f"Logging {len(new_data)} new data points for future training.")
        self.data_points_collected += len(new_data)
        feature_names = [
            'distance_3d', 'altitude_change', 'horizontal_distance', 'payload_kg', 'wind_speed',
            'wind_alignment', 'turning_angle', 'start_altitude', 'end_altitude',
            'abs_altitude_change', 'actual_time', 'actual_energy'
        ]
        df_new = pd.DataFrame(new_data, columns=feature_names)
        df_new.to_csv(TRAINING_DATA_PATH, mode='a', index=False, header=not os.path.exists(TRAINING_DATA_PATH))

    def _check_and_trigger_retraining(self):
        if self.data_points_collected >= RETRAINING_THRESHOLD:
            logging.info(f"Retraining threshold reached. Starting model retraining...")
            try:
                df = pd.read_csv(TRAINING_DATA_PATH)
                if len(df) < 2:
                    logging.warning("Not enough data to retrain. Skipping.")
                    return
                features = [col for col in df.columns if col not in ['actual_time', 'actual_energy']]
                X, y_time, y_energy = df[features], df['actual_time'], df['actual_energy']
                params = {"n_estimators": 100, "max_depth": 10, "random_state": 42, "n_jobs": -1}
                time_model = RandomForestRegressor(**params).fit(X, y_time)
                energy_model = RandomForestRegressor(**params).fit(X, y_energy)
                joblib.dump({"time_model": time_model, "energy_model": energy_model}, MODEL_FILE_PATH)
                logging.info("✅ Model retraining complete. New model is now active.")
                self.predictor.load_model()
                self.data_points_collected = 0
            except Exception as e:
                logging.error(f"❌ Retraining failed: {e}")

    def check_if_all_legs_complete(self) -> bool:
        if not self.fleet_solution: return False
        return all(self.missions[drone_id].state == "WAITING_FOR_FLEET" for drone_id in self.fleet_solution.keys())