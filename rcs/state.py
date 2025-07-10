"""Data structures representing agent state."""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class RecursiveAgentState:
    """Simplified state representation with language capabilities."""

    agent_id: str
    current_situation: str
    emotional_state: Dict[str, float]
    goals: List[str]
    recent_thoughts: List[str]
    social_context: Dict[str, Any]
    energy_level: float
    consciousness_level: float

    def to_natural_language(self) -> str:
        emotions = ", ".join(f"{emotion}: {intensity:.2f}" for emotion, intensity in self.emotional_state.items())
        goals_str = "; ".join(self.goals)
        recent_thoughts_str = "; ".join(self.recent_thoughts[-3:])
        return (
            f"Current situation: {self.current_situation}\n"
            f"Emotional state: {emotions}\n"
            f"Current goals: {goals_str}\n"
            f"Recent thoughts: {recent_thoughts_str}\n"
            f"Energy level: {self.energy_level:.2f}\n"
            f"Social context: {self.social_context}"
        )
