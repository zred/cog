"""Experiment orchestration for recursive agents."""

from datetime import datetime
from typing import Dict

import numpy as np

from .agent import LangChainRecursiveAgent


class RecursiveConsciousnessExperiment:
    """Orchestrate experiments with recursive agents."""

    def __init__(self, num_agents: int = 3, llm_model: str = "distilgpt2") -> None:
        self.agents = {
            f"agent_{i}": LangChainRecursiveAgent(f"agent_{i}", llm_model) for i in range(num_agents)
        }
        self.environment = {
            "description": "A shared space where conscious agents can think, communicate, and act",
            "available_actions": ["think", "communicate", "help_another", "explore", "reflect", "rest"],
        }
        self.experiment_log = []

    async def run_step(self) -> Dict:
        step_log = {"timestamp": datetime.now().isoformat(), "events": []}
        for agent_id, agent in self.agents.items():
            thought_result = await agent.think()
            step_log["events"].append(f"{agent_id} thought: {thought_result}")
            action, reasoning = await agent.decide_action(
                self.environment["description"], self.environment["available_actions"]
            )
            step_log["events"].append(f"{agent_id} decided: {action} - {reasoning[:100]}...")
            if action == "communicate":
                other_agents = [aid for aid in self.agents.keys() if aid != agent_id]
                if other_agents:
                    target = np.random.choice(other_agents)
                    message = (
                        f"Hello {target}, I'm thinking about consciousness and would like to share thoughts with you."
                    )
                    response = await self.agents[target].interact_with(agent_id, message)
                    step_log["events"].append(f"{agent_id} → {target}: {message}")
                    step_log["events"].append(f"{target} → {agent_id}: {response}")
            elif action == "help_another":
                other_agents = [aid for aid in self.agents.keys() if aid != agent_id]
                if other_agents:
                    target = np.random.choice(other_agents)
                    help_message = "I notice you might need assistance. How can I help you?"
                    response = await self.agents[target].interact_with(agent_id, help_message)
                    step_log["events"].append(f"{agent_id} helped {target}: {help_message}")
                    step_log["events"].append(f"{target} responded: {response}")
        self.experiment_log.append(step_log)
        return step_log

    async def run_experiment(self, num_steps: int = 10) -> Dict:
        print("Starting LangChain Recursive Consciousness Experiment")
        print("=" * 60)
        for step in range(num_steps):
            print(f"\nStep {step + 1}:")
            step_log = await self.run_step()
            for event in step_log["events"][:3]:
                print(f"  {event}")
            if len(step_log["events"]) > 3:
                print(f"  ... and {len(step_log['events']) - 3} more events")
            consciousness_levels = {aid: ag.state.consciousness_level for aid, ag in self.agents.items()}
            avg_consciousness = np.mean(list(consciousness_levels.values()))
            print(f"  Average consciousness: {avg_consciousness:.3f}")
        return self.analyze_results()

    def analyze_results(self) -> Dict:
        print("\n" + "=" * 60)
        print("LANGCHAIN RECURSIVE CONSCIOUSNESS ANALYSIS")
        print("=" * 60)
        for agent_id, agent in self.agents.items():
            metrics = agent.get_consciousness_metrics()
            print(f"\n{agent_id.upper()}:")
            print(f"  Final consciousness level: {metrics['consciousness_level']:.3f}")
            print(f"  Recursive depth: {metrics['recursive_depth']:.3f}")
            print(f"  Self-model accuracy: {metrics['self_model_accuracy']:.3f}")
            print(f"  Other agents modeled: {metrics['other_agents_modeled']}")
            print(f"  Ethical decisions: {metrics['ethical_decisions']}")
            print(f"  Total interactions: {metrics['total_interactions']}")
            print(f"  Meta insights: {metrics['meta_insights']}")
            print(f"  Behavioral consistency: {metrics['behavior_consistency']:.2f}")
            print(f"  Reputation: {metrics['reputation']:.2f}")
        total_interactions = sum(len(agent.conversation_memory.messages) for agent in self.agents.values())
        total_ethical_decisions = sum(len(agent.ethical_decisions) for agent in self.agents.values())
        avg_reputation = np.mean([a.reputation for a in self.agents.values()])
        print("\nCROSS-AGENT ANALYSIS:")
        print(f"  Total interactions: {total_interactions}")
        print(f"  Total ethical decisions: {total_ethical_decisions}")
        print(
            f"  Average final consciousness: {np.mean([a.state.consciousness_level for a in self.agents.values()]):.3f}"
        )
        print(f"  Average reputation: {avg_reputation:.2f}")
        return {
            "agents": {aid: agent.get_consciousness_metrics() for aid, agent in self.agents.items()},
            "experiment_log": self.experiment_log,
            "summary": {
                "total_interactions": total_interactions,
                "total_ethical_decisions": total_ethical_decisions,
                "avg_consciousness": np.mean([a.state.consciousness_level for a in self.agents.values()]),
                "avg_reputation": avg_reputation,
            },
        }
