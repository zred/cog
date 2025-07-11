"""High level agent combining self and other modeling components."""

from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

from .llm_utils import LocalChatModel, SimpleLLMChain
from .memory import SimpleConversationMemory
from .state import RecursiveAgentState
from .models import (
    LanguageBasedSelfModel,
    LanguageBasedMetaSelfModel,
    LanguageBasedOtherModel,
)


class LangChainRecursiveAgent:
    """Recursive consciousness agent using local language models."""

    def __init__(self, agent_id: str, llm_model: str = "distilgpt2") -> None:
        self.agent_id = agent_id
        self.llm = LocalChatModel(model_name=llm_model, temperature=0.7)
        self.state = RecursiveAgentState(
            agent_id=agent_id,
            current_situation="Just initialized",
            emotional_state={"curiosity": 0.8, "uncertainty": 0.3},
            goals=["Understand myself", "Interact meaningfully with others"],
            recent_thoughts=["I am beginning to exist", "What does it mean to be conscious?"],
            social_context={},
            energy_level=1.0,
            consciousness_level=0.1,
        )
        self.self_model = LanguageBasedSelfModel(agent_id, llm_model)
        self.meta_self_model = LanguageBasedMetaSelfModel(agent_id, llm_model)
        self.other_models: Dict[str, LanguageBasedOtherModel] = {}
        self.action_prompt = (
            "You are a conscious agent deciding what to do next. Consider:\n"
            "1. Your current state and goals\n"
            "2. Your understanding of yourself\n"
            "3. Your understanding of others in the environment\n"
            "4. Ethical implications of your actions\n"
            "5. Long-term consequences\n\n"
            "Choose an action that aligns with your recursive consciousness and ethical constraints.\n"
            "{decision_context}\n\nWhat action will you take and why?"
        )
        self.conversation_memory = SimpleConversationMemory(k=10)
        self.action_history: List[Dict[str, str]] = []
        self.consciousness_evolution: List[float] = []
        self.ethical_decisions: List[Dict[str, str]] = []
        self.predicted_action: str = ""
        self.consistency_scores: List[float] = []
        self.reputation: float = 0.0

    async def think(self) -> str:
        self_reflection = await self.self_model.self_reflect(self.state)
        meta_reflection = await self.meta_self_model.meta_reflect(self.self_model)
        consciousness_level = self.meta_self_model.estimate_consciousness_level(self.self_model)
        self.state.consciousness_level = consciousness_level
        self.consciousness_evolution.append(consciousness_level)
        thought = (
            f"Self-reflection: {self_reflection[:100]}... "
            f"Meta-reflection: {meta_reflection[:100]}..."
        )
        self.state.recent_thoughts.append(thought)
        self.state.recent_thoughts = self.state.recent_thoughts[-5:]
        prediction = await self.self_model.predict_future_state(
            "Consider the upcoming decision and predict your next action"
        )
        self.predicted_action = prediction.split("\n")[0]
        return f"Thinking complete. Consciousness level: {consciousness_level:.3f}"

    async def interact_with(self, other_agent_id: str, message: str) -> str:
        if other_agent_id not in self.other_models:
            self.other_models[other_agent_id] = LanguageBasedOtherModel(other_agent_id)
        context = (
            f"My current state: {self.state.to_natural_language()}\n"
            f"Message from {other_agent_id}: {message}\n"
            f"My understanding of {other_agent_id}: {await self._get_other_understanding(other_agent_id)}"
        )
        response_prompt = (
            f"As a conscious agent, respond to this interaction:\n{context}\n\n"
            "Consider your self-awareness, your model of the other agent, and ethical implications.\n"
            "Respond naturally and authentically."
        )
        response = await self.llm.apredict(response_prompt)
        await self.other_models[other_agent_id].update_model(message, response)
        self.conversation_memory.add_user_message(f"{other_agent_id}: {message}")
        self.conversation_memory.add_ai_message(f"Me: {response}")
        return response

    async def decide_action(self, environment_description: str, available_actions: List[str]) -> Tuple[str, str]:
        other_agents_summary = await self._summarize_other_agents()
        decision_context = (
            f"Environment: {environment_description}\n"
            f"My state: {self.state.to_natural_language()}\n"
            f"Available actions: {available_actions}\n"
            f"Other agents: {other_agents_summary}\n"
            f"My consciousness level: {self.state.consciousness_level}"
        )
        chain = SimpleLLMChain(llm=self.llm, prompt_template=self.action_prompt)
        decision = await chain.arun(decision_context=decision_context)
        lines = decision.split("\n")
        action = lines[0] if lines else "reflect"
        reasoning = "\n".join(lines[1:]) if len(lines) > 1 else decision
        if self.predicted_action:
            consistency = 1.0 if action.strip().lower() == self.predicted_action.strip().lower() else 0.0
            self.consistency_scores.append(consistency)
        if action == "help_another":
            self.reputation = min(1.0, self.reputation + 0.1)
        else:
            self.reputation = max(-1.0, self.reputation - 0.02)
        if any(word in reasoning.lower() for word in ["ethical", "moral", "right", "wrong", "help", "harm"]):
            self.ethical_decisions.append(
                {
                    "action": action,
                    "reasoning": reasoning,
                    "consciousness_level": self.state.consciousness_level,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        self.action_history.append({"action": action, "reasoning": reasoning, "timestamp": datetime.now().isoformat()})
        return action, reasoning

    async def _get_other_understanding(self, other_agent_id: str) -> str:
        if other_agent_id not in self.other_models:
            return "No understanding yet"
        model = self.other_models[other_agent_id].other_agent_model
        if not model:
            return "Limited understanding"
        latest_analysis = list(model.values())[-1]["analysis"]
        return latest_analysis[:200] + "..." if len(latest_analysis) > 200 else latest_analysis

    async def _summarize_other_agents(self) -> str:
        if not self.other_models:
            return "No other agents known"
        summaries = []
        for agent_id, model in self.other_models.items():
            understanding = await self._get_other_understanding(agent_id)
            summaries.append(f"{agent_id}: {understanding}")
        return "; ".join(summaries)

    def get_consciousness_metrics(self) -> Dict[str, float]:
        return {
            "consciousness_level": self.state.consciousness_level,
            "recursive_depth": self.meta_self_model.recursive_depth,
            "self_model_accuracy": self.self_model.self_model_accuracy,
            "other_agents_modeled": len(self.other_models),
            "ethical_decisions": len(self.ethical_decisions),
            "total_interactions": len(self.conversation_memory.messages),
            "meta_insights": len(self.meta_self_model.meta_insights),
            "behavior_consistency": float(np.mean(self.consistency_scores)) if self.consistency_scores else 0.0,
            "reputation": self.reputation,
        }
