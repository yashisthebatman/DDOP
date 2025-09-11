# FILE: fleet/manager.py
import logging
from typing import Dict, List
from fleet.cbs_components import Agent
from planners.cbsh_planner import CBSHPlanner  # MODIFIED IMPORT

class Mission:
    """Represents a multi-stop mission for a single drone."""
    def __init__(self, drone_id, start_pos, destinations, payload_kg, optimization_weights):
        self.drone_id = drone_id
        self.start_pos = start_pos
        self.destinations = destinations
        self.payload_kg = payload_kg
        self.optimization_weights = optimization_weights
        self.current_leg_index = 0
        self.state = "PENDING" # PENDING, IN_PROGRESS, WAITING_FOR_FLEET, COMPLETED, RTH
        self.path = [] # The full TIMED path for the current leg [((x,y,z), t), ...]
        self.path_world_coords = [] # Smoothed world coordinates for rendering/simulation
        self.path_progress_index = 0
        
    def get_current_leg(self):
        if self.is_complete():
            return None, None
        
        start = self.start_pos if self.current_leg_index == 0 else self.destinations[self.current_leg_index - 1]
        end = self.destinations[self.current_leg_index]
        return start, end

    def advance_leg(self):
        if not self.is_complete():
            self.current_leg_index += 1
            self.path = []
            self.path_world_coords = []
            self.path_progress_index = 0
            if self.is_complete():
                self.state = "COMPLETED"
            else:
                 self.state = "PENDING" # Ready for next planning cycle

    def is_complete(self):
        return self.current_leg_index >= len(self.destinations)

class FleetManager:
    """Manages the state and planning for a fleet of drones and their missions."""
    def __init__(self, cbs_planner: CBSHPlanner): # MODIFIED TYPE HINT
        self.missions: Dict[str, Mission] = {}
        self.cbs_planner = cbs_planner
        self.fleet_solution = None

    def add_mission(self, mission: Mission):
        self.missions[mission.drone_id] = mission
        logging.info(f"Added mission for drone {mission.drone_id}")

    def execute_planning_cycle(self):
        """
        Gathers active mission legs, plans them with CBSH, and updates mission paths.
        """
        active_agents: List[Agent] = []
        
        for mission in self.missions.values():
            if mission.state in ["PENDING", "WAITING_FOR_FLEET"] and not mission.is_complete():
                start_world, goal_world = mission.get_current_leg()
                
                # The new planner takes world coordinates directly
                agent_config = {
                    'payload_kg': mission.payload_kg,
                    **mission.optimization_weights
                }
                # The Agent for CBSH now uses world coordinates for start/goal
                active_agents.append(Agent(id=mission.drone_id, start_pos=start_world, goal_pos=goal_world, config=agent_config))

        if not active_agents:
            logging.info("Planning cycle skipped: No active missions requiring planning.")
            return False

        logging.info(f"Starting CBSH planning for {len(active_agents)} agents.")
        solution = self.cbs_planner.plan_fleet(active_agents)

        if solution:
            logging.info("CBSH planning successful. Updating mission paths.")
            self.fleet_solution = solution
            for agent in active_agents:
                mission = self.missions[agent.id]
                mission.path = solution[agent.id] # This is now a timed path
                mission.state = "IN_PROGRESS"
                mission.path_progress_index = 0
                
                # Extract just the world coordinates for simulation/rendering
                world_path = [p[0] for p in mission.path]
                
                # Post-process with smoothing (optional, RRT* paths are already quite smooth)
                smoothed_path = self.cbs_planner.smoother.smooth_path(world_path, self.cbs_planner.env)
                mission.path_world_coords = smoothed_path
            return True
        else:
            logging.error("CBSH planning FAILED. Missions cannot proceed.")
            for agent in active_agents:
                 self.missions[agent.id].state = "PLANNING_FAILED"
            return False

    def check_if_all_legs_complete(self) -> bool:
        """Checks if all drones in the current planning cycle have finished their leg."""
        if not self.fleet_solution: # No active plan
            return False
            
        active_drones = self.fleet_solution.keys()
        for drone_id in active_drones:
            if self.missions[drone_id].state != "WAITING_FOR_FLEET":
                return False
        return True