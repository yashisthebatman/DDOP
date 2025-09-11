from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

# Type alias for clarity
GridPosition = Tuple[int, int, int]
WorldPosition = Tuple[float, float, float]
AgentID = Any

@dataclass
class Agent:
    """Represents a single agent (drone) for the CBS planner."""
    id: AgentID
    start_pos: GridPosition
    goal_pos: GridPosition
    config: Dict[str, Any]  # payload_kg, optimization weights, etc.

@dataclass(frozen=True, eq=True)
class Constraint:
    """A space-time constraint for an agent."""
    agent_id: AgentID
    position: GridPosition
    timestamp: int

@dataclass(frozen=True, eq=True)
class Conflict:
    """Represents a collision between two agents."""
    agent1_id: AgentID
    agent2_id: AgentID
    position: GridPosition
    timestamp: int

@dataclass
class CTNode:
    """A node in the Constraint Tree for the high-level CBS search."""
    constraints: set[Constraint] = field(default_factory=set)
    solution: Dict[AgentID, List[Tuple[GridPosition, int]]] = field(default_factory=dict)
    cost: float = 0.0

    def __lt__(self, other: 'CTNode') -> bool:
        # For priority queue comparison
        return self.cost < other.cost