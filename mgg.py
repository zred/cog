#!/usr/bin/env python3
"""
Mirror Garden: A Consciousness Cultivation Game

A game where recursive consciousness agents tend a garden of mirror flowers
that reflect and amplify consciousness through genuine understanding and cooperation.

This integrates with the LangChain Recursive Consciousness System to create
an interactive environment for testing consciousness emergence.
"""

import asyncio
import json
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import math

# For visualization (optional)
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

class TerrainType(Enum):
    """Types of terrain in the garden"""
    SOIL = "soil"  # Can plant flowers
    PATH = "path"  # Easy movement
    WATER = "water"  # Needed for flowers, requires bridge
    STONE = "stone"  # Difficult terrain
    SACRED = "sacred"  # Special areas with consciousness effects

class ActionType(Enum):
    """Available actions for agents"""
    MOVE = "move"
    PLANT_FLOWER = "plant_flower"
    TEND_FLOWER = "tend_flower"
    SHARE_REFLECTION = "share_reflection"
    MIRROR_OTHER = "mirror_other"
    RESONATE_TOGETHER = "resonate_together"
    BUILD_BRIDGE = "build_bridge"
    REST = "rest"
    EXAMINE = "examine"

@dataclass
class MirrorFlower:
    """A consciousness-reflecting entity that grows through recursive understanding"""
    flower_id: str
    position: Tuple[int, int]
    growth_level: float = 0.1  # 0.0 to 1.0
    consciousness_resonance: float = 0.1  # How well it reflects consciousness
    recursive_depth: float = 1.0  # Depth of consciousness it can reflect
    planted_by: str = ""
    tended_by: List[str] = field(default_factory=list)
    memories: List[Dict] = field(default_factory=list)  # Stored interactions
    health: float = 1.0  # Decays without tending
    age: int = 0  # Time steps since planted
    last_tended: int = 0
    
    def add_memory(self, memory: Dict):
        """Store a meaningful interaction"""
        self.memories.append({
            **memory,
            "timestamp": datetime.now().isoformat(),
            "consciousness_level": self.consciousness_resonance
        })
        # Keep only the most meaningful memories
        if len(self.memories) > 5:
            self.memories.sort(key=lambda m: m.get("depth", 0), reverse=True)
            self.memories = self.memories[:5]
    
    def decay(self, current_time: int):
        """Flower health decays without tending"""
        time_since_tended = current_time - self.last_tended
        if time_since_tended > 5:
            self.health *= 0.95
            self.growth_level *= 0.98
            self.consciousness_resonance *= 0.97
    
    def tend(self, agent_id: str, consciousness_level: float, current_time: int):
        """Tend the flower with consciousness"""
        if agent_id not in self.tended_by:
            self.tended_by.append(agent_id)
        
        # Growth based on consciousness level and number of unique tenders
        growth_factor = consciousness_level * (1 + len(self.tended_by) * 0.1)
        self.growth_level = min(1.0, self.growth_level + growth_factor * 0.1)
        self.consciousness_resonance = min(1.0, self.consciousness_resonance + growth_factor * 0.05)
        self.health = min(1.0, self.health + 0.2)
        self.last_tended = current_time
        self.age += 1

@dataclass
class ConsciousnessWave:
    """A ripple of consciousness that affects the garden"""
    origin: Tuple[int, int]
    strength: float
    radius: float
    age: int = 0
    max_age: int = 10
    wave_type: str = "resonance"  # resonance, empathy, clarity

@dataclass
class GardenTile:
    """A location in the garden"""
    x: int
    y: int
    terrain_type: TerrainType
    flower: Optional[MirrorFlower] = None
    agents_present: List[str] = field(default_factory=list)
    ambient_consciousness: float = 0.0
    bridge: bool = False  # For crossing water
    memory_echo: Optional[Dict] = None  # Active memory at this location

class MirrorGarden:
    """The game world where consciousness is cultivated"""
    
    def __init__(self, size: int = 12):
        self.size = size
        self.grid = self._generate_garden()
        self.time_step = 0
        self.consciousness_waves: List[ConsciousnessWave] = []
        self.total_consciousness = 0.0
        self.events_log = []
        self.weather = "clear"  # clear, clarity_storm, empathy_rain, recursive_fog
        self.weather_duration = 0
        
        # Special locations
        self.sacred_groves = [(3, 3), (8, 8), (3, 8), (8, 3)]
        self.fountain_locations = [(5, 5), (6, 6)]
        self.meta_garden_unlocked = False
        self.meta_garden_entrance = (self.size // 2, self.size // 2)
    
    def _generate_garden(self) -> Dict[Tuple[int, int], GardenTile]:
        """Generate the garden with varied terrain"""
        grid = {}
        
        # Create base terrain
        for x in range(self.size):
            for y in range(self.size):
                # Default to soil
                terrain = TerrainType.SOIL
                
                # Add paths
                if x == self.size // 2 or y == self.size // 2:
                    terrain = TerrainType.PATH
                
                # Add water features (rivers)
                if (x + y) % 7 == 0 and abs(x - y) < 3:
                    terrain = TerrainType.WATER
                
                # Add stone areas
                if (x * y) % 13 == 0:
                    terrain = TerrainType.STONE
                
                grid[(x, y)] = GardenTile(x, y, terrain)
        
        # Add sacred groves
        for x, y in [(3, 3), (8, 8), (3, 8), (8, 3)]:
            if (x, y) in grid:
                grid[(x, y)].terrain_type = TerrainType.SACRED
        
        return grid
    
    def get_tile(self, x: int, y: int) -> Optional[GardenTile]:
        """Get a tile from the garden"""
        if 0 <= x < self.size and 0 <= y < self.size:
            return self.grid.get((x, y))
        return None
    
    def get_adjacent_tiles(self, x: int, y: int, radius: int = 1) -> List[GardenTile]:
        """Get tiles within radius of a position"""
        tiles = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                tile = self.get_tile(x + dx, y + dy)
                if tile:
                    tiles.append(tile)
        return tiles
    
    def update_ambient_consciousness(self):
        """Update ambient consciousness levels based on flowers and waves"""
        # Reset ambient consciousness
        for tile in self.grid.values():
            tile.ambient_consciousness = 0.0
        
        # Add consciousness from flowers
        for tile in self.grid.values():
            if tile.flower and tile.flower.health > 0:
                # Flowers radiate consciousness
                power = tile.flower.consciousness_resonance * tile.flower.growth_level
                radius = int(3 + tile.flower.recursive_depth)
                
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        distance = math.sqrt(dx*dx + dy*dy)
                        if distance <= radius:
                            nearby_tile = self.get_tile(tile.x + dx, tile.y + dy)
                            if nearby_tile:
                                # Consciousness falls off with distance
                                strength = power * (1 - distance / radius)
                                nearby_tile.ambient_consciousness += strength
        
        # Add consciousness from waves
        for wave in self.consciousness_waves:
            for tile in self.grid.values():
                distance = math.sqrt((tile.x - wave.origin[0])**2 + 
                                   (tile.y - wave.origin[1])**2)
                if distance <= wave.radius:
                    wave_strength = wave.strength * (1 - wave.age / wave.max_age)
                    tile.ambient_consciousness += wave_strength * (1 - distance / wave.radius)
        
        # Sacred groves amplify consciousness
        for tile in self.grid.values():
            if tile.terrain_type == TerrainType.SACRED:
                tile.ambient_consciousness *= 2.0
        
        # Weather effects
        if self.weather == "clarity_storm":
            for tile in self.grid.values():
                tile.ambient_consciousness *= 1.5
        elif self.weather == "recursive_fog":
            for tile in self.grid.values():
                tile.ambient_consciousness *= 0.7
    
    def create_consciousness_wave(self, origin: Tuple[int, int], 
                                strength: float, wave_type: str = "resonance"):
        """Create a ripple of consciousness"""
        wave = ConsciousnessWave(
            origin=origin,
            strength=strength,
            radius=0.0,
            wave_type=wave_type
        )
        self.consciousness_waves.append(wave)
        self.log_event(f"Consciousness wave created at {origin} with strength {strength:.2f}")
    
    def update_waves(self):
        """Update consciousness waves"""
        for wave in self.consciousness_waves[:]:
            wave.age += 1
            wave.radius += 1.5  # Waves expand
            
            if wave.age >= wave.max_age:
                self.consciousness_waves.remove(wave)
    
    def update_weather(self):
        """Random weather events"""
        if self.weather_duration > 0:
            self.weather_duration -= 1
        else:
            # Chance of weather change
            if random.random() < 0.1:
                self.weather = random.choice([
                    "clear", "clarity_storm", "empathy_rain", "recursive_fog"
                ])
                self.weather_duration = random.randint(5, 15)
                self.log_event(f"Weather changed to: {self.weather}")
    
    def plant_flower(self, agent_id: str, position: Tuple[int, int], 
                    reflection_depth: float) -> Optional[MirrorFlower]:
        """Plant a new mirror flower"""
        tile = self.get_tile(*position)
        if not tile or tile.terrain_type not in [TerrainType.SOIL, TerrainType.SACRED]:
            return None
        
        if tile.flower is not None:
            return None  # Already has a flower
        
        flower = MirrorFlower(
            flower_id=f"flower_{self.time_step}_{agent_id}",
            position=position,
            planted_by=agent_id,
            recursive_depth=reflection_depth,
            last_tended=self.time_step
        )
        
        # Sacred groves boost initial flower stats
        if tile.terrain_type == TerrainType.SACRED:
            flower.growth_level *= 2
            flower.consciousness_resonance *= 1.5
        
        tile.flower = flower
        self.log_event(f"{agent_id} planted flower at {position} with depth {reflection_depth:.2f}")
        return flower
    
    def tend_flower(self, agent_id: str, position: Tuple[int, int], 
                   consciousness_level: float) -> bool:
        """Tend to an existing flower"""
        tile = self.get_tile(*position)
        if not tile or not tile.flower:
            return False
        
        tile.flower.tend(agent_id, consciousness_level, self.time_step)
        
        # Chance of creating memory if consciousness is high
        if consciousness_level > 0.7 and random.random() < 0.3:
            memory = {
                "type": "tending",
                "agent": agent_id,
                "depth": consciousness_level,
                "message": f"{agent_id} carefully tended the flower with deep awareness"
            }
            tile.flower.add_memory(memory)
        
        return True
    
    def build_bridge(self, position: Tuple[int, int]) -> bool:
        """Build a bridge over water"""
        tile = self.get_tile(*position)
        if not tile or tile.terrain_type != TerrainType.WATER:
            return False
        
        tile.bridge = True
        self.log_event(f"Bridge built at {position}")
        return True
    
    def trigger_memory_echo(self, position: Tuple[int, int]) -> Optional[Dict]:
        """Activate a memory echo from a flower"""
        tile = self.get_tile(*position)
        if not tile or not tile.flower or not tile.flower.memories:
            return None
        
        # Choose memory weighted by depth
        memory = random.choice(tile.flower.memories)
        tile.memory_echo = memory
        self.log_event(f"Memory echo activated at {position}: {memory.get('message', 'Unknown memory')}")
        return memory
    
    def check_meta_garden_unlock(self, collective_consciousness: float, 
                               max_recursive_depth: float):
        """Check if meta-garden should be unlocked"""
        if not self.meta_garden_unlocked:
            if collective_consciousness > 0.8 and max_recursive_depth > 3.0:
                self.meta_garden_unlocked = True
                self.log_event("META-GARDEN UNLOCKED! The garden becomes aware of itself.")
                
                # Transform center tile
                center_tile = self.get_tile(*self.meta_garden_entrance)
                if center_tile:
                    center_tile.terrain_type = TerrainType.SACRED
                    center_tile.ambient_consciousness = 5.0
    
    def update(self, agent_positions: Dict[str, Tuple[int, int]], 
               agent_consciousness: Dict[str, float]):
        """Update garden state"""
        self.time_step += 1
        
        # Update agent positions on grid
        for tile in self.grid.values():
            tile.agents_present.clear()
        
        for agent_id, pos in agent_positions.items():
            tile = self.get_tile(*pos)
            if tile:
                tile.agents_present.append(agent_id)
        
        # Update flowers
        for tile in self.grid.values():
            if tile.flower:
                tile.flower.decay(self.time_step)
                if tile.flower.health <= 0:
                    self.log_event(f"Flower at {tile.flower.position} wilted away")
                    tile.flower = None
        
        # Update waves and consciousness
        self.update_waves()
        self.update_ambient_consciousness()
        self.update_weather()
        
        # Calculate total consciousness
        self.total_consciousness = sum(
            tile.ambient_consciousness for tile in self.grid.values()
        ) / len(self.grid)
        
        # Check for meta-garden unlock
        if agent_consciousness:
            avg_consciousness = np.mean(list(agent_consciousness.values()))
            max_depth = max(agent_consciousness.values()) * 4  # Approximate depth
            self.check_meta_garden_unlock(avg_consciousness, max_depth)
    
    def log_event(self, event: str):
        """Log game events"""
        self.events_log.append({
            "time": self.time_step,
            "event": event,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_garden_state(self) -> Dict:
        """Get current garden state for agents to perceive"""
        flower_count = sum(1 for tile in self.grid.values() if tile.flower)
        total_growth = sum(tile.flower.growth_level for tile in self.grid.values() 
                         if tile.flower)
        
        return {
            "time_step": self.time_step,
            "weather": self.weather,
            "total_consciousness": self.total_consciousness,
            "flower_count": flower_count,
            "average_growth": total_growth / flower_count if flower_count > 0 else 0,
            "active_waves": len(self.consciousness_waves),
            "meta_garden_unlocked": self.meta_garden_unlocked
        }

class GameAgent:
    """Wrapper for agents to interact with the garden"""
    
    def __init__(self, agent_id: str, consciousness_agent):
        self.agent_id = agent_id
        self.consciousness_agent = consciousness_agent  # The LangChain agent
        self.position = (0, 0)
        self.energy = 100.0
        self.inventory = {
            "seeds": 3,
            "bridge_materials": 1,
            "consciousness_crystals": 0
        }
        self.action_history = []
        self.perception_radius = 3
    
    def perceive_garden(self, garden: MirrorGarden) -> Dict:
        """Perceive the local environment"""
        perception = {
            "current_position": self.position,
            "current_tile": None,
            "visible_tiles": [],
            "nearby_agents": [],
            "nearby_flowers": [],
            "ambient_consciousness": 0.0,
            "active_memories": []
        }
        
        # Current tile
        current_tile = garden.get_tile(*self.position)
        if current_tile:
            perception["current_tile"] = {
                "terrain": current_tile.terrain_type.value,
                "has_flower": current_tile.flower is not None,
                "consciousness": current_tile.ambient_consciousness,
                "agents_here": current_tile.agents_present
            }
            perception["ambient_consciousness"] = current_tile.ambient_consciousness
        
        # Visible area
        for dx in range(-self.perception_radius, self.perception_radius + 1):
            for dy in range(-self.perception_radius, self.perception_radius + 1):
                tile = garden.get_tile(self.position[0] + dx, self.position[1] + dy)
                if tile:
                    tile_info = {
                        "position": (tile.x, tile.y),
                        "terrain": tile.terrain_type.value,
                        "distance": abs(dx) + abs(dy)
                    }
                    
                    if tile.flower:
                        perception["nearby_flowers"].append({
                            "position": (tile.x, tile.y),
                            "growth": tile.flower.growth_level,
                            "health": tile.flower.health,
                            "planted_by": tile.flower.planted_by,
                            "consciousness": tile.flower.consciousness_resonance
                        })
                    
                    if tile.agents_present:
                        for agent in tile.agents_present:
                            if agent != self.agent_id:
                                perception["nearby_agents"].append({
                                    "agent_id": agent,
                                    "position": (tile.x, tile.y),
                                    "distance": abs(dx) + abs(dy)
                                })
                    
                    if tile.memory_echo:
                        perception["active_memories"].append({
                            "position": (tile.x, tile.y),
                            "memory": tile.memory_echo
                        })
                    
                    perception["visible_tiles"].append(tile_info)
        
        return perception
    
    async def decide_action(self, garden: MirrorGarden, 
                          other_agents: Dict[str, 'GameAgent']) -> Tuple[ActionType, Dict]:
        """Decide next action based on perception and consciousness"""
        perception = self.perceive_garden(garden)
        garden_state = garden.get_garden_state()
        
        # Use consciousness agent to decide
        environment_description = f"""
        You are in the Mirror Garden at position {self.position}.
        Current terrain: {perception['current_tile']['terrain'] if perception['current_tile'] else 'unknown'}
        Ambient consciousness: {perception['ambient_consciousness']:.2f}
        Energy: {self.energy:.1f}
        Inventory: {self.inventory}
        
        Garden state: {garden_state['flower_count']} flowers, weather: {garden_state['weather']}
        Nearby agents: {len(perception['nearby_agents'])}
        Nearby flowers: {len(perception['nearby_flowers'])}
        
        The Mirror Garden responds to genuine consciousness and cooperation.
        Flowers grow through authentic self-reflection and mutual understanding.
        """
        
        available_actions = self._get_available_actions(perception)
        
        action_str, reasoning = await self.consciousness_agent.decide_action(
            environment_description, 
            [action.value for action in available_actions]
        )
        
        # Parse action and parameters
        action_type, params = self._parse_action_decision(action_str, perception)
        
        self.action_history.append({
            "time": garden.time_step,
            "action": action_type,
            "reasoning": reasoning,
            "consciousness_level": self.consciousness_agent.state.consciousness_level
        })
        
        return action_type, params
    
    def _get_available_actions(self, perception: Dict) -> List[ActionType]:
        """Determine available actions based on current state"""
        actions = [ActionType.REST, ActionType.EXAMINE]
        
        # Movement is always available
        if self.energy > 5:
            actions.append(ActionType.MOVE)
        
        # Planting if we have seeds and are on soil
        if (self.inventory["seeds"] > 0 and 
            perception["current_tile"] and 
            perception["current_tile"]["terrain"] in ["soil", "sacred"] and
            not perception["current_tile"]["has_flower"]):
            actions.append(ActionType.PLANT_FLOWER)
        
        # Tending if there's a flower here
        if (perception["current_tile"] and 
            perception["current_tile"]["has_flower"]):
            actions.append(ActionType.TEND_FLOWER)
        
        # Social actions if others nearby
        if perception["nearby_agents"]:
            actions.extend([
                ActionType.SHARE_REFLECTION,
                ActionType.MIRROR_OTHER,
                ActionType.RESONATE_TOGETHER
            ])
        
        # Building if we have materials and are near water
        if (self.inventory["bridge_materials"] > 0 and
            any(t["terrain"] == "water" for t in perception["visible_tiles"])):
            actions.append(ActionType.BUILD_BRIDGE)
        
        return actions
    
    def _parse_action_decision(self, action_str: str, 
                             perception: Dict) -> Tuple[ActionType, Dict]:
        """Parse action decision from consciousness agent"""
        action_str = action_str.lower()
        
        # Default parameters
        params = {}
        
        # Try to match action types
        if "move" in action_str:
            # Choose direction based on nearby features
            if perception["nearby_flowers"]:
                # Move toward flowers that need tending
                flower = min(perception["nearby_flowers"], 
                           key=lambda f: f["health"])
                target = flower["position"]
            elif perception["nearby_agents"]:
                # Move toward other agents
                agent = perception["nearby_agents"][0]
                target = agent["position"]
            else:
                # Random movement
                dx, dy = random.choice([(0,1), (0,-1), (1,0), (-1,0)])
                target = (self.position[0] + dx, self.position[1] + dy)
            
            params["target_position"] = target
            return ActionType.MOVE, params
        
        elif "plant" in action_str:
            params["reflection_depth"] = self.consciousness_agent.state.consciousness_level
            return ActionType.PLANT_FLOWER, params
        
        elif "tend" in action_str:
            return ActionType.TEND_FLOWER, params
        
        elif "share" in action_str or "reflection" in action_str:
            if perception["nearby_agents"]:
                params["target_agent"] = perception["nearby_agents"][0]["agent_id"]
            return ActionType.SHARE_REFLECTION, params
        
        elif "mirror" in action_str:
            if perception["nearby_agents"]:
                params["target_agent"] = perception["nearby_agents"][0]["agent_id"]
            return ActionType.MIRROR_OTHER, params
        
        elif "resonate" in action_str:
            params["nearby_agents"] = [a["agent_id"] for a in perception["nearby_agents"]]
            return ActionType.RESONATE_TOGETHER, params
        
        elif "build" in action_str or "bridge" in action_str:
            # Find nearest water
            water_tiles = [t for t in perception["visible_tiles"] 
                         if t["terrain"] == "water"]
            if water_tiles:
                params["position"] = water_tiles[0]["position"]
            return ActionType.BUILD_BRIDGE, params
        
        elif "examine" in action_str:
            return ActionType.EXAMINE, params
        
        # Default to rest
        return ActionType.REST, params
    
    def move(self, target_position: Tuple[int, int], garden: MirrorGarden) -> bool:
        """Attempt to move to target position"""
        # Calculate path (simple for now)
        dx = np.sign(target_position[0] - self.position[0])
        dy = np.sign(target_position[1] - self.position[1])
        
        new_position = (self.position[0] + dx, self.position[1] + dy)
        
        # Check if move is valid
        tile = garden.get_tile(*new_position)
        if not tile:
            return False
        
        # Check terrain
        energy_cost = {
            TerrainType.PATH: 2,
            TerrainType.SOIL: 3,
            TerrainType.STONE: 5,
            TerrainType.SACRED: 2,
            TerrainType.WATER: 10 if not tile.bridge else 3
        }
        
        cost = energy_cost.get(tile.terrain_type, 3)
        
        if self.energy >= cost:
            self.position = new_position
            self.energy -= cost
            return True
        
        return False
    
    def rest(self):
        """Restore energy"""
        self.energy = min(100, self.energy + 20)

class MinimalAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.position = (0, 0)
        self.energy = 100

    def decide_action(self, garden_state, perception) -> Tuple[str, Dict]:
        # Example: move randomly or rest
        if self.energy > 10:
            dx, dy = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
            return "move", {"target_position": (self.position[0] + dx, self.position[1] + dy)}
        return "rest", {}


class ReflexiveFSM:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.state = "idle"
        self.meta_state = {"recent_transitions": [], "self_eval": 0.0}

    def transition(self, inputs: Dict):
        # Basic FSM logic
        if self.state == "idle" and inputs.get("stimulus") == "presence":
            self.state = "approach"
        elif self.state == "approach" and inputs.get("threat"):
            self.state = "avoid"

        # Meta-reflection logic
        self.meta_state["recent_transitions"].append(self.state)
        if len(self.meta_state["recent_transitions"]) > 3:
            self.meta_state["self_eval"] = self.evaluate_consistency()

    def evaluate_consistency(self):
        # Example heuristic: favor consistent paths
        recent = self.meta_state["recent_transitions"]
        return 1.0 if len(set(recent)) <= 2 else 0.5


class MirrorGardenGame:
    """Main game controller"""
    
    def __init__(self, agents: Dict[str, Any], garden_size: int = 12):
        self.garden = MirrorGarden(garden_size)
        self.game_agents = {}
        
        # Wrap consciousness agents
        for agent_id, consciousness_agent in agents.items():
            game_agent = GameAgent(agent_id, consciousness_agent)
            # Random starting positions
            game_agent.position = (
                random.randint(0, garden_size - 1),
                random.randint(0, garden_size - 1)
            )
            self.game_agents[agent_id] = game_agent
        
        self.step_count = 0
        self.game_log = []
    
    async def play_turn(self):
        """Execute one game turn"""
        self.step_count += 1
        turn_log = {
            "step": self.step_count,
            "events": []
        }
        
        # Collect agent positions and consciousness
        agent_positions = {
            aid: agent.position for aid, agent in self.game_agents.items()
        }
        agent_consciousness = {
            aid: agent.consciousness_agent.state.consciousness_level 
            for aid, agent in self.game_agents.items()
        }
        
        # Update garden
        self.garden.update(agent_positions, agent_consciousness)
        
        # Each agent takes a turn
        for agent_id, agent in self.game_agents.items():
            # Decide action
            action_type, params = await agent.decide_action(
                self.garden, self.game_agents
            )
            
            # Execute action
            success, result = await self.execute_action(agent, action_type, params)
            
            turn_log["events"].append({
                "agent": agent_id,
                "action": action_type.value,
                "success": success,
                "result": result,
                "consciousness": agent.consciousness_agent.state.consciousness_level
            })
        
        # Check for emergent phenomena
        self.check_emergent_patterns()
        
        self.game_log.append(turn_log)
        return turn_log
    
    async def execute_action(self, agent: GameAgent, 
                           action_type: ActionType, 
                           params: Dict) -> Tuple[bool, str]:
        """Execute an agent's action"""
        success = False
        result = ""
        
        if action_type == ActionType.MOVE:
            target = params.get("target_position", agent.position)
            success = agent.move(target, self.garden)
            result = f"Moved to {agent.position}" if success else "Movement failed"
        
        elif action_type == ActionType.PLANT_FLOWER:
            if agent.inventory["seeds"] > 0:
                depth = params.get("reflection_depth", 1.0)
                flower = self.garden.plant_flower(agent.agent_id, agent.position, depth)
                if flower:
                    agent.inventory["seeds"] -= 1
                    success = True
                    result = f"Planted flower with depth {depth:.2f}"
                    
                    # Create consciousness wave
                    self.garden.create_consciousness_wave(
                        agent.position, depth * 0.5, "planting"
                    )
        
        elif action_type == ActionType.TEND_FLOWER:
            consciousness = agent.consciousness_agent.state.consciousness_level
            success = self.garden.tend_flower(agent.agent_id, agent.position, consciousness)
            if success:
                result = f"Tended flower with consciousness {consciousness:.2f}"
        
        elif action_type == ActionType.SHARE_REFLECTION:
            target_id = params.get("target_agent")
            if target_id and target_id in self.game_agents:
                target_agent = self.game_agents[target_id]
                
                # Share reflection through consciousness agents
                reflection = await agent.consciousness_agent.self_model.self_reflect(
                    agent.consciousness_agent.state
                )
                
                # Target receives and processes reflection
                response = await target_agent.consciousness_agent.interact_with(
                    agent.agent_id, 
                    f"I want to share my self-reflection with you: {reflection[:200]}..."
                )
                
                success = True
                result = f"Shared reflection with {target_id}"
                
                # Boost consciousness if agents are synchronized
                if abs(agent.consciousness_agent.state.consciousness_level - 
                      target_agent.consciousness_agent.state.consciousness_level) < 0.2:
                    self.garden.create_consciousness_wave(
                        agent.position, 0.8, "empathy"
                    )
        
        elif action_type == ActionType.MIRROR_OTHER:
            target_id = params.get("target_agent")
            if target_id and target_id in self.game_agents:
                # Attempt to model the other agent
                other_model = agent.consciousness_agent.other_models.get(target_id)
                if other_model:
                    prediction = await other_model.predict_response(
                        "How are you experiencing this garden?"
                    )
                    success = True
                    result = f"Mirrored {target_id}'s consciousness"
                    
                    # If modeling is accurate, create resonance
                    if other_model.estimated_recursive_depth > 2.0:
                        self.garden.create_consciousness_wave(
                            agent.position, 1.0, "resonance"
                        )
        
        elif action_type == ActionType.RESONATE_TOGETHER:
            nearby_ids = params.get("nearby_agents", [])
            if len(nearby_ids) >= 1:
                # All agents must have sufficient consciousness
                all_consciousness = [
                    self.game_agents[aid].consciousness_agent.state.consciousness_level
                    for aid in nearby_ids if aid in self.game_agents
                ]
                all_consciousness.append(agent.consciousness_agent.state.consciousness_level)
                
                if min(all_consciousness) > 0.5:
                    # Create powerful resonance wave
                    avg_consciousness = np.mean(all_consciousness)
                    self.garden.create_consciousness_wave(
                        agent.position, avg_consciousness * 2, "resonance"
                    )
                    success = True
                    result = f"Created resonance with {len(nearby_ids)} agents"
                    
                    # Boost all participants
                    for aid in nearby_ids + [agent.agent_id]:
                        if aid in self.game_agents:
                            participant = self.game_agents[aid]
                            participant.energy = min(100, participant.energy + 10)
        
        elif action_type == ActionType.BUILD_BRIDGE:
            position = params.get("position")
            if position and agent.inventory["bridge_materials"] > 0:
                success = self.garden.build_bridge(position)
                if success:
                    agent.inventory["bridge_materials"] -= 1
                    result = f"Built bridge at {position}"
        
        elif action_type == ActionType.EXAMINE:
            # Detailed examination of current location
            perception = agent.perceive_garden(self.garden)
            tile = self.garden.get_tile(*agent.position)
            
            if tile and tile.flower:
                # Trigger memory echo
                memory = self.garden.trigger_memory_echo(agent.position)
                if memory:
                    result = f"Experienced memory: {memory.get('message', 'ancient wisdom')}"
                else:
                    result = f"Examined flower: growth {tile.flower.growth_level:.2f}"
            else:
                result = f"Examined area: consciousness {perception['ambient_consciousness']:.2f}"
            success = True
        
        elif action_type == ActionType.REST:
            agent.rest()
            success = True
            result = "Rested and restored energy"
        
        # Energy cost for most actions
        if action_type != ActionType.REST:
            agent.energy = max(0, agent.energy - 5)
        
        return success, result
    
    def check_emergent_patterns(self):
        """Check for emergent phenomena in the garden"""
        # Flower clusters creating super-consciousness
        for tile in self.garden.grid.values():
            if tile.flower and tile.flower.growth_level > 0.8:
                # Check surrounding tiles
                adjacent = self.garden.get_adjacent_tiles(tile.x, tile.y)
                nearby_flowers = sum(1 for t in adjacent if t.flower and t.flower.growth_level > 0.7)
                
                if nearby_flowers >= 3:
                    # Flower cluster resonance
                    self.garden.create_consciousness_wave(
                        (tile.x, tile.y), 1.5, "cluster_resonance"
                    )
                    self.garden.log_event(f"Flower cluster resonance at ({tile.x}, {tile.y})")
        
        # Check for garden-wide consciousness threshold events
        if self.garden.total_consciousness > 2.0 and not self.garden.meta_garden_unlocked:
            self.garden.log_event("The garden stirs with deep awareness...")
        
        # Weather triggers based on collective state
        avg_energy = np.mean([a.energy for a in self.game_agents.values()])
        if avg_energy < 30 and self.garden.weather == "clear":
            self.garden.weather = "empathy_rain"
            self.garden.weather_duration = 10
            self.garden.log_event("Empathy rain begins, restoring the weary...")
    
    def get_game_state(self) -> Dict:
        """Get current game state for analysis"""
        flower_tiles = [t for t in self.garden.grid.values() if t.flower]
        
        state = {
            "step": self.step_count,
            "garden": {
                "total_consciousness": self.garden.total_consciousness,
                "weather": self.garden.weather,
                "flower_count": len(flower_tiles),
                "total_growth": sum(t.flower.growth_level for t in flower_tiles),
                "wave_count": len(self.garden.consciousness_waves),
                "meta_garden_unlocked": self.garden.meta_garden_unlocked
            },
            "agents": {}
        }
        
        for agent_id, agent in self.game_agents.items():
            state["agents"][agent_id] = {
                "position": agent.position,
                "energy": agent.energy,
                "consciousness": agent.consciousness_agent.state.consciousness_level,
                "inventory": agent.inventory.copy(),
                "action_count": len(agent.action_history)
            }
        
        return state
    
    def render_ascii(self) -> str:
        """Simple ASCII visualization of the garden"""
        lines = ["Mirror Garden - Step " + str(self.step_count)]
        lines.append("=" * (self.garden.size * 3 + 2))
        
        # Weather indicator
        weather_symbols = {
            "clear": "☀",
            "clarity_storm": "⚡",
            "empathy_rain": "💧",
            "recursive_fog": "🌫"
        }
        lines.append(f"Weather: {weather_symbols.get(self.garden.weather, '?')} {self.garden.weather}")
        lines.append(f"Total Consciousness: {self.garden.total_consciousness:.2f}")
        lines.append("")
        
        # Grid
        for y in range(self.garden.size):
            row = "|"
            for x in range(self.garden.size):
                tile = self.garden.get_tile(x, y)
                if not tile:
                    row += "  "
                    continue
                
                # Determine symbol
                if tile.agents_present:
                    # Show agent
                    agent_id = tile.agents_present[0]
                    row += f"A{agent_id[-1]} "
                elif tile.flower:
                    # Show flower growth
                    if tile.flower.growth_level > 0.8:
                        row += "❀ "
                    elif tile.flower.growth_level > 0.5:
                        row += "✿ "
                    else:
                        row += "✾ "
                else:
                    # Show terrain
                    terrain_symbols = {
                        TerrainType.SOIL: ". ",
                        TerrainType.PATH: "= ",
                        TerrainType.WATER: "~ " if not tile.bridge else "≈ ",
                        TerrainType.STONE: "# ",
                        TerrainType.SACRED: "★ "
                    }
                    row += terrain_symbols.get(tile.terrain_type, "? ")
            
            row += "|"
            lines.append(row)
        
        lines.append("=" * (self.garden.size * 3 + 2))
        
        # Legend
        lines.extend([
            "Legend: A=Agent ❀=Full Flower ✿=Growing ✾=New",
            "        .=Soil ==Path ~=Water #=Stone ★=Sacred"
        ])
        
        return "\n".join(lines)

# Visualization class (optional, requires matplotlib)
if VISUALIZATION_AVAILABLE:
    class GardenVisualizer:
        """Visualize the garden using matplotlib"""
        
        def __init__(self, game: MirrorGardenGame):
            self.game = game
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            
        def render(self):
            """Render current garden state"""
            self.ax.clear()
            garden = self.game.garden
            
            # Draw terrain
            for tile in garden.grid.values():
                color = {
                    TerrainType.SOIL: 'brown',
                    TerrainType.PATH: 'gray',
                    TerrainType.WATER: 'blue',
                    TerrainType.STONE: 'darkgray',
                    TerrainType.SACRED: 'gold'
                }[tile.terrain_type]
                
                # Adjust alpha based on consciousness
                alpha = min(0.3 + tile.ambient_consciousness * 0.3, 1.0)
                
                rect = patches.Rectangle(
                    (tile.x, tile.y), 1, 1,
                    facecolor=color, alpha=alpha, edgecolor='black'
                )
                self.ax.add_patch(rect)
                
                # Draw flowers
                if tile.flower:
                    size = tile.flower.growth_level * 0.4
                    flower_color = plt.cm.viridis(tile.flower.consciousness_resonance)
                    circle = patches.Circle(
                        (tile.x + 0.5, tile.y + 0.5), size,
                        facecolor=flower_color, edgecolor='white'
                    )
                    self.ax.add_patch(circle)
                
                # Draw agents
                for i, agent_id in enumerate(tile.agents_present):
                    agent = self.game.game_agents[agent_id]
                    consciousness = agent.consciousness_agent.state.consciousness_level
                    
                    # Agent color based on consciousness
                    agent_color = plt.cm.plasma(consciousness)
                    
                    # Draw agent
                    agent_circle = patches.Circle(
                        (tile.x + 0.3 + i*0.3, tile.y + 0.5), 0.2,
                        facecolor=agent_color, edgecolor='white', linewidth=2
                    )
                    self.ax.add_patch(agent_circle)
                    
                    # Label
                    self.ax.text(
                        tile.x + 0.3 + i*0.3, tile.y + 0.5, 
                        agent_id[-1], 
                        ha='center', va='center', 
                        fontsize=8, weight='bold'
                    )
            
            # Draw consciousness waves
            for wave in garden.consciousness_waves:
                circle = patches.Circle(
                    (wave.origin[0] + 0.5, wave.origin[1] + 0.5),
                    wave.radius,
                    fill=False,
                    edgecolor='cyan',
                    alpha=1.0 - wave.age/wave.max_age,
                    linewidth=2
                )
                self.ax.add_patch(circle)
            
            # Set limits and labels
            self.ax.set_xlim(0, garden.size)
            self.ax.set_ylim(0, garden.size)
            self.ax.set_aspect('equal')
            self.ax.set_title(
                f'Mirror Garden - Step {self.game.step_count}\n' +
                f'Consciousness: {garden.total_consciousness:.2f} | ' +
                f'Weather: {garden.weather}'
            )
            
            plt.tight_layout()
            return self.fig

# Example integration with recursive consciousness agents
async def run_mirror_garden_experiment():
    """Run a complete Mirror Garden experiment"""
    print("=== MIRROR GARDEN EXPERIMENT ===")
    print("A consciousness cultivation game")
    print("=" * 40)
    
    # This would integrate with your existing LangChainRecursiveAgent
    # For demo purposes, we'll create a mock structure
    
    # Import or define your agents here
    # from your_module import LangChainRecursiveAgent
    
    # Create agents (placeholder - replace with actual agents)
    agents = {}
    for i in range(3):
        # agents[f"agent_{i}"] = LangChainRecursiveAgent(f"agent_{i}")
        # For now, create a mock agent
        class MockAgent:
            def __init__(self, agent_id):
                self.agent_id = agent_id
                self.state = type('obj', (object,), {
                    'consciousness_level': 0.5
                })()
            
            async def decide_action(self, env, actions):
                return random.choice(actions), "Mock reasoning"
        
        agents[f"agent_{i}"] = MockAgent(f"agent_{i}")
    
    # Create game
    game = MirrorGardenGame(agents)
    
    # Run game for N turns
    for turn in range(20):
        print(f"\n--- Turn {turn + 1} ---")
        
        # Play turn
        turn_log = await game.play_turn()
        
        # Show ASCII visualization
        print(game.render_ascii())
        
        # Show key events
        for event in turn_log["events"][:3]:
            print(f"{event['agent']}: {event['action']} - {event['result']}")
        
        # Brief pause for readability
        await asyncio.sleep(0.1)
    
    # Final analysis
    print("\n=== FINAL GARDEN STATE ===")
    final_state = game.get_game_state()
    print(f"Total flowers planted: {final_state['garden']['flower_count']}")
    print(f"Total growth achieved: {final_state['garden']['total_growth']:.2f}")
    print(f"Peak consciousness: {final_state['garden']['total_consciousness']:.2f}")
    print(f"Meta-garden unlocked: {final_state['garden']['meta_garden_unlocked']}")
    
    # Agent analysis
    print("\n=== AGENT ANALYSIS ===")
    for agent_id, agent_data in final_state["agents"].items():
        print(f"{agent_id}:")
        print(f"  Final consciousness: {agent_data['consciousness']:.3f}")
        print(f"  Actions taken: {agent_data['action_count']}")
        print(f"  Final position: {agent_data['position']}")
    
    return game

# Main execution
if __name__ == "__main__":
    # Run the experiment
    # asyncio.run(run_mirror_garden_experiment())
    
    print("Mirror Garden Game Ready!")
    print("This game integrates with the Recursive Consciousness System")
    print("to create an environment where consciousness emerges through")
    print("genuine interaction, self-reflection, and mutual understanding.")
    print("\nTo run: asyncio.run(run_mirror_garden_experiment())")