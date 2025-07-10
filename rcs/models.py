"""Agent modeling components used by :mod:`rcs.agent`."""

import json
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from .llm_utils import LocalChatModel, SimpleLLMChain
from .memory import SimpleConversationMemory, SimpleVectorStore
from .state import RecursiveAgentState


class LanguageBasedSelfModel:
    """Self-model using local language reasoning."""

    def __init__(self, agent_id: str, llm_model: str = "distilgpt2") -> None:
        self.agent_id = agent_id
        self.llm = LocalChatModel(model_name=llm_model, temperature=0.7)
        self.self_reflection_memory = SimpleConversationMemory(k=10)
        self.embeddings = SentenceTransformer("all-MiniLM-L6-v2")
        self.self_knowledge_store = SimpleVectorStore(self.embeddings)
        self.self_reflection_prompt = (
            "You are engaged in deep self-reflection. Your task is:\n"
            "1. Analyze your current mental state and recent experiences\n"
            "2. Identify patterns in your thoughts, emotions, and behaviors\n"
            "3. Predict how you might react in future situations\n"
            "4. Update your understanding of your own personality and tendencies\n\n"
            "Be honest, introspective, and specific. Think about your thinking process itself.\n"
            "{current_state}\n\nReflect on this state and recent experiences. What do you notice about yourself?"
        )
        self.prediction_prompt = (
            "Based on your self-knowledge, predict your future behavior and mental states.\n"
            "Consider your personality patterns, emotional tendencies, and decision-making style.\n"
            "{situation}\n\nGiven this situation and your self-knowledge, how would you likely think, feel, and act?"
        )
        self.predictions: List[Dict[str, Any]] = []
        self.actual_outcomes: List[str] = []
        self.self_model_accuracy = 0.0

    async def self_reflect(self, current_state: RecursiveAgentState) -> str:
        state_description = current_state.to_natural_language()
        chain = SimpleLLMChain(llm=self.llm, prompt_template=self.self_reflection_prompt)
        reflection = await chain.arun(current_state=state_description)
        self.self_knowledge_store.add_texts(
            [reflection],
            metadatas=[{"timestamp": datetime.now().isoformat(), "type": "self_reflection"}],
        )
        return reflection

    async def predict_future_state(self, situation: str) -> str:
        relevant_knowledge = self.self_knowledge_store.similarity_search(situation, k=3)
        context = "\n".join(doc.page_content for doc in relevant_knowledge)
        chain = SimpleLLMChain(llm=self.llm, prompt_template=self.prediction_prompt)
        prediction = await chain.arun(
            situation=f"Situation: {situation}\n\nRelevant self-knowledge: {context}"
        )
        self.predictions.append({
            "situation": situation,
            "prediction": prediction,
            "timestamp": datetime.now().isoformat(),
        })
        return prediction

    def update_accuracy(self, predicted: str, actual: str) -> None:
        prediction_words = set(predicted.lower().split())
        actual_words = set(actual.lower().split())
        if prediction_words.union(actual_words):
            similarity = len(prediction_words.intersection(actual_words)) / len(
                prediction_words.union(actual_words)
            )
        else:
            similarity = 0.0
        self.actual_outcomes.append(actual)
        if self.predictions:
            accuracies = []
            for i, pred in enumerate(self.predictions[-5:]):
                if i < len(self.actual_outcomes):
                    pred_words = set(pred["prediction"].lower().split())
                    actual_words = set(self.actual_outcomes[i].lower().split())
                    if pred_words.union(actual_words):
                        acc = len(pred_words.intersection(actual_words)) / len(
                            pred_words.union(actual_words)
                        )
                        accuracies.append(acc)
            if accuracies:
                self.self_model_accuracy = np.mean(accuracies)


class LanguageBasedMetaSelfModel:
    """Meta-self-model using language reasoning about self-modeling."""

    def __init__(self, agent_id: str, llm_model: str = "distilgpt2") -> None:
        self.agent_id = agent_id
        self.llm = LocalChatModel(model_name=llm_model, temperature=0.7)
        self.meta_reflection_prompt = (
            "You are analyzing your own thinking and self-modeling process. Consider:\n"
            "1. How well do you understand yourself?\n"
            "2. What are the patterns in your self-reflection?\n"
            "3. How accurate are your self-predictions?\n"
            "4. How is your self-awareness changing over time?\n"
            "5. What is the nature of your consciousness and self-experience?\n\n"
            "This is meta-cognition - thinking about thinking about yourself.\n"
            "{self_model_data}\n\nAnalyze your self-modeling process. What do you notice about how you think about yourself?"
        )
        self.recursive_depth = 2.0
        self.meta_insights: List[Dict[str, Any]] = []

    async def meta_reflect(self, self_model: LanguageBasedSelfModel) -> str:
        recent_predictions = self_model.predictions[-3:] if self_model.predictions else []
        accuracy = self_model.self_model_accuracy
        self_model_data = {
            "recent_predictions": recent_predictions,
            "prediction_accuracy": accuracy,
            "knowledge_items": len(self_model.self_knowledge_store.docs),
            "reflection_count": len(recent_predictions),
        }
        chain = SimpleLLMChain(llm=self.llm, prompt_template=self.meta_reflection_prompt)
        meta_reflection = await chain.arun(self_model_data=json.dumps(self_model_data, indent=2))
        self.meta_insights.append(
            {
                "reflection": meta_reflection,
                "timestamp": datetime.now().isoformat(),
                "self_model_accuracy": accuracy,
            }
        )
        if "recursive" in meta_reflection.lower() or "meta" in meta_reflection.lower():
            self.recursive_depth = min(self.recursive_depth + 0.1, 4.0)
        return meta_reflection

    def estimate_consciousness_level(self, self_model: LanguageBasedSelfModel) -> float:
        base_consciousness = min(self.recursive_depth / 4.0, 1.0)
        accuracy_bonus = self_model.self_model_accuracy * 0.3
        insight_bonus = min(len(self.meta_insights) * 0.05, 0.2)
        return min(base_consciousness + accuracy_bonus + insight_bonus, 1.0)


class LanguageBasedOtherModel:
    """Model other agents through conversation and observation."""

    def __init__(self, other_agent_id: str, llm_model: str = "distilgpt2") -> None:
        self.other_agent_id = other_agent_id
        self.llm = LocalChatModel(model_name=llm_model, temperature=0.7)
        self.interaction_memory = SimpleConversationMemory(k=15)
        self.other_agent_model: Dict[str, Dict[str, str]] = {}
        self.estimated_recursive_depth = 1.0
        self.other_modeling_prompt = (
            "You are trying to understand another agent's mind. Based on their words and actions:\n"
            "1. What are their personality traits and tendencies?\n"
            "2. How do they think and make decisions?\n"
            "3. What are their goals and motivations?\n"
            "4. How self-aware do they seem to be?\n"
            "5. How well do they understand others?\n"
            "6. What is their level of recursive consciousness?\n\n"
            "Be specific and evidence-based.\n"
            "{interaction_data}\n\nAnalyze this agent's mind based on these interactions."
        )

    async def update_model(self, interaction_text: str, observed_behavior: str) -> None:
        self.interaction_memory.add_user_message(interaction_text)
        self.interaction_memory.add_ai_message(observed_behavior)
        recent_interactions = self.interaction_memory.messages[-6:]
        interaction_summary = "\n".join(recent_interactions)
        chain = SimpleLLMChain(llm=self.llm, prompt_template=self.other_modeling_prompt)
        analysis = await chain.arun(interaction_data=interaction_summary)
        self.other_agent_model[datetime.now().isoformat()] = {
            "analysis": analysis,
            "interaction_summary": interaction_summary,
        }
        if "self-aware" in analysis.lower() or "reflects" in analysis.lower():
            self.estimated_recursive_depth = max(self.estimated_recursive_depth, 2.0)
        if "meta" in analysis.lower() or "recursive" in analysis.lower():
            self.estimated_recursive_depth = max(self.estimated_recursive_depth, 3.0)

    async def predict_response(self, situation: str) -> str:
        if not self.other_agent_model:
            return "Insufficient data to predict response"
        latest_analysis = list(self.other_agent_model.values())[-1]["analysis"]
        prediction_prompt = (
            f"Based on this analysis of an agent:\n{latest_analysis}\n\n"
            f"How would they likely respond to this situation: {situation}\n\n"
            "Consider their personality, decision-making patterns, and level of self-awareness."
        )
        response = await self.llm.apredict(prediction_prompt)
        return response
