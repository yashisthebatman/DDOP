# FILE: fleet/manager.py
import logging
from typing import Dict, List, Tuple
from fleet.cbs_components import Agent
from planners.cbsh_planner import CBSHPlanner
from ml_predictor.predictor import EnergyTimePredictor
from config import RTH_BATTERY_THRESHOLD_FACTOR
from utils.geometry import calculate_distance_3d

class Mission:
    def __init__(self, drone_id, start_pos, destinations, payload_kg, optimization_weights):
        self.drone_id, self.start_pos, self.destinations, self.payload_kg, self.optimization_weights = drone_id, start_pos, destinations, payload_kg, optimization_weights
        self.path, self.path_world_coords = [], []
        self.total_planned_energy, self.total_planned_time = 0.0, 0.0
        self.state = "PENDING"
        self.current_leg_index = 0

    def is_complete(self):
        return self.current_leg_index >= len(self.destinations)

    def advance_leg(self):
        if not self.is_complete():
            self.current_leg_index += 1
            self.state = "COMPLETED" if self.is_complete() else "PENDING"

class FleetManager:
    def __init__(self, cbs_planner: CBSHPlanner, predictor: EnergyTimePredictor):
        self.missions: Dict[str, Mission] = {}
        self.cbs_planner = cbs_planner
        self.predictor = predictor

    def add_mission(self, mission: Mission):
        self.missions[mission.drone_id] = mission
        logging.info(f"Added mission for drone {mission.drone_id}")
    
    def pre_flight_check(self, mission: Mission, current_battery_wh: float) -> Tuple[bool, str]:
        start_pos, end_pos = mission.start_pos, mission.destinations[0]
        _, energy_to_dest = self.predictor.fallback_predictor.predict(start_pos, end_pos, mission.payload_kg, [0,0,0])
        _, energy_to_home = self.predictor.fallback_predictor.predict(end_pos, mission.start_pos, 0, [0,0,0])
        required_energy = (energy_to_dest + energy_to_home) * RTH_BATTERY_THRESHOLD_FACTOR
        if required_energy > current_battery_wh:
            return False, f"Insufficient battery. Required: {required_energy:.2f}Wh, Available: {current_battery_wh:.2f}Wh"
        return True, "OK"

    def execute_planning_cycle(self) -> Tuple[bool, Dict]:
        active_agents: List[Agent] = []
        for m in self.missions.values():
            config = {'payload_kg': m.payload_kg, **m.optimization_weights}
            agent = Agent(id=m.drone_id, start_pos=m.start_pos, goal_pos=m.destinations[-1], config=config)
            active_agents.append(agent)

        if not active_agents:
            return False, {"error": "No active agents for planning."}
        
        solution = self.cbs_planner.plan_fleet(active_agents)
        if not solution:
            logging.error("CBSH planning FAILED for the fleet.")
            return False, {"error": "CBS could not find a solution for the fleet."}

        logging.info("CBSH planning successful. Updating mission paths.")
        planned_missions = []
        for agent_id, path in solution.items():
            mission = self.missions[agent_id]
            mission.path = path
            world_path = [p[0] for p in mission.path]
            mission.path_world_coords = self.cbs_planner.smoother.smooth_path(world_path, self.cbs_planner.env)
            if mission.path and len(mission.path) > 1:
                total_energy, p_prev = 0, None
                for i in range(len(world_path) - 1):
                    p1, p2 = world_path[i], world_path[i+1]
                    wind = self.cbs_planner.env.weather.get_wind_at_location(*p1)
                    _, energy_pred = self.predictor.predict(p1, p2, mission.payload_kg, wind, p_prev)
                    total_energy += energy_pred
                    p_prev = p1
                mission.total_planned_energy = total_energy
                mission.total_planned_time = mission.path[-1][1]
                planned_missions.append(agent_id)
            else:
                mission.total_planned_energy = mission.total_planned_time = 0

        if not planned_missions:
             return False, {"error": "Planning succeeded but no valid paths were generated."}
        return True, {"planned_missions": planned_missions}