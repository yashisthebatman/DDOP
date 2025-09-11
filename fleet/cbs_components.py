# FILE: fleet/cbs_components.py
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

# Type alias for clarity
GridPosition = Tuple[int, int, int]
WorldPosition = Tuple[float, float, float] # This is now the primary position type
AgentID = Any

@dataclass
class Agent:
    """Represents a single agent (drone) for the CBS planner."""
    id: AgentID
    # MODIFIED: The planner now operates in continuous world space.
    start_pos: WorldPosition
    goal_pos: WorldPosition
    config: Dict[str, Any]  # payload_kg, optimization weights, etc.

@dataclass(frozen=True, eq=True)
class Constraint:
    """A space-time constraint for an agent."""
    agent_id: AgentID
    position: GridPosition # Constraints are applied to the discrete grid for simplicity
    timestamp: int

@dataclass(frozen=True, eq=True)
class Conflict:
    """Represents a collision between two agents."""
    agent1_id: AgentID
    agent2_id: AgentID
    # FIX: A conflict now stores both agent positions to allow for more intelligent
    # constraint generation, which is necessary to prevent infinite loops when
    # dealing with proximity (continuous space) conflicts.
    agent1_pos: GridPosition
    agent2_pos: GridPosition
    timestamp: int

    @property
    def is_vertex_conflict(self) -> bool:
        """Returns true if both agents are in the same grid cell."""
        return self.agent1_pos == self.agent2_pos

@dataclass
class CTNode:
    """A node in the Constraint Tree for the high-level CBS search."""
    constraints: set[Constraint] = field(default_factory=set)
    solution: Dict[AgentID, List[Tuple[WorldPosition, int]]] = field(default_factory=dict)
    cost: float = 0.0

    def __lt__(self, other: 'CTNode') -> bool:
        # For priority queue comparison
        return self.cost < other.cost