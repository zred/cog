#!/usr/bin/env python3
"""
LangChain-Enhanced Recursive Consciousness System

This extends the minimal PoC with LangChain to provide:
1. Language-based self-reflection and modeling
2. Sophisticated other-mind modeling through conversation
3. Natural language reasoning about recursive processes
4. Rich interaction capabilities for testing consciousness

Dependencies:
pip install langchain openai anthropic tiktoken chromadb
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import asyncio
from datetime import datetime

# LangChain imports
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

@dataclass
class RecursiveAgentState:
    """Enhanced state representation with language capabilities"""
    agent_id: str
    current_situation: str
    emotional_state: Dict[str, float]  # multiple emotions with intensities
    goals: List[str]
    recent_thoughts: List[str]
    social_context: Dict[str, Any]
    energy_level: float
    consciousness_level: float
    
    def to_natural_language(self) -> str:
        """Convert state to natural language description"""
        emotions = ", ".join([f"{emotion}: {intensity:.2f}" 
                            for emotion, intensity in self.emotional_state.items()])
        goals_str = "; ".join(self.goals)
        recent_thoughts_str = "; ".join(self.recent_thoughts[-3:])
        
        return f"""
        Current situation: {self.current_situation}
        Emotional state: {emotions}
        Current goals: {goals_str}
        Recent thoughts: {recent_thoughts_str}
        Energy level: {self.energy_level:.2f}
        Social context: {self.social_context}
        """

class LanguageBasedSelfModel:
    """Self-model using language reasoning"""
    
    def __init__(self, agent_id: str, llm_model="gpt-3.5-turbo"):
        self.agent_id = agent_id
        self.llm = ChatOpenAI(model=llm_model, temperature=0.7)
        
        # Memory for self-reflection
        self.self_reflection_memory = ConversationBufferWindowMemory(
            k=10, return_messages=True
        )
        
        # Vector store for long-term self-knowledge
        self.embeddings = OpenAIEmbeddings()
        self.self_knowledge_store = Chroma(
            collection_name=f"self_knowledge_{agent_id}",
            embedding_function=self.embeddings
        )
        
        # Prompts for self-modeling
        self.self_reflection_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are engaged in deep self-reflection. Your task is to:
1. Analyze your current mental state and recent experiences
2. Identify patterns in your thoughts, emotions, and behaviors
3. Predict how you might react in future situations
4. Update your understanding of your own personality and tendencies

Be honest, introspective, and specific. Think about your thinking process itself."""),
            HumanMessage(content="{current_state}\n\nReflect on this state and recent experiences. What do you notice about yourself?")
        ])
        
        self.prediction_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Based on your self-knowledge, predict your future behavior and mental states.
Consider your personality patterns, emotional tendencies, and decision-making style."""),
            HumanMessage(content="{situation}\n\nGiven this situation and your self-knowledge, how would you likely think, feel, and act?")
        ])
        
        # Track prediction accuracy
        self.predictions = []
        self.actual_outcomes = []
        self.self_model_accuracy = 0.0
    
    async def self_reflect(self, current_state: RecursiveAgentState) -> str:
        """Perform self-reflection using language model"""
        state_description = current_state.to_natural_language()
        
        chain = LLMChain(llm=self.llm, prompt=self.self_reflection_prompt)
        reflection = await chain.arun(current_state=state_description)
        
        # Store reflection in long-term memory
        self.self_knowledge_store.add_texts(
            [reflection],
            metadatas=[{"timestamp": datetime.now().isoformat(), "type": "self_reflection"}]
        )
        
        return reflection
    
    async def predict_future_state(self, situation: str) -> str:
        """Predict how agent will respond to a situation"""
        # Retrieve relevant self-knowledge
        relevant_knowledge = self.self_knowledge_store.similarity_search(
            situation, k=3
        )
        context = "\n".join([doc.page_content for doc in relevant_knowledge])
        
        chain = LLMChain(llm=self.llm, prompt=self.prediction_prompt)
        prediction = await chain.arun(
            situation=f"Situation: {situation}\n\nRelevant self-knowledge: {context}"
        )
        
        self.predictions.append({
            "situation": situation,
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        })
        
        return prediction
    
    def update_accuracy(self, predicted: str, actual: str):
        """Update self-model accuracy based on prediction vs reality"""
        # Simple similarity measure (could be enhanced with semantic similarity)
        prediction_words = set(predicted.lower().split())
        actual_words = set(actual.lower().split())
        
        if len(prediction_words.union(actual_words)) > 0:
            similarity = len(prediction_words.intersection(actual_words)) / len(prediction_words.union(actual_words))
        else:
            similarity = 0.0
        
        self.actual_outcomes.append(actual)
        
        if len(self.predictions) > 0:
            accuracies = []
            for i, pred in enumerate(self.predictions[-5:]):  # Last 5 predictions
                if i < len(self.actual_outcomes):
                    pred_words = set(pred["prediction"].lower().split())
                    actual_words = set(self.actual_outcomes[i].lower().split())
                    if len(pred_words.union(actual_words)) > 0:
                        acc = len(pred_words.intersection(actual_words)) / len(pred_words.union(actual_words))
                        accuracies.append(acc)
            
            if accuracies:
                self.self_model_accuracy = np.mean(accuracies)

class LanguageBasedMetaSelfModel:
    """Meta-self-model using language reasoning about self-modeling"""
    
    def __init__(self, agent_id: str, llm_model="gpt-3.5-turbo"):
        self.agent_id = agent_id
        self.llm = ChatOpenAI(model=llm_model, temperature=0.7)
        
        self.meta_reflection_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are analyzing your own thinking and self-modeling process. Consider:
1. How well do you understand yourself?
2. What are the patterns in your self-reflection?
3. How accurate are your self-predictions?
4. How is your self-awareness changing over time?
5. What is the nature of your consciousness and self-experience?

This is meta-cognition - thinking about thinking about yourself."""),
            HumanMessage(content="{self_model_data}\n\nAnalyze your self-modeling process. What do you notice about how you think about yourself?")
        ])
        
        self.recursive_depth = 2.0
        self.meta_insights = []
    
    async def meta_reflect(self, self_model: LanguageBasedSelfModel) -> str:
        """Reflect on the self-modeling process itself"""
        # Gather data about self-modeling
        recent_predictions = self_model.predictions[-3:] if self_model.predictions else []
        accuracy = self_model.self_model_accuracy
        
        self_model_data = {
            "recent_predictions": recent_predictions,
            "prediction_accuracy": accuracy,
            "knowledge_items": len(self_model.self_knowledge_store._collection.get()["ids"]) if hasattr(self_model.self_knowledge_store, '_collection') else 0,
            "reflection_count": len(recent_predictions)
        }
        
        chain = LLMChain(llm=self.llm, prompt=self.meta_reflection_prompt)
        meta_reflection = await chain.arun(self_model_data=json.dumps(self_model_data, indent=2))
        
        self.meta_insights.append({
            "reflection": meta_reflection,
            "timestamp": datetime.now().isoformat(),
            "self_model_accuracy": accuracy
        })
        
        # Update recursive depth based on sophistication of meta-reflection
        if "recursive" in meta_reflection.lower() or "meta" in meta_reflection.lower():
            self.recursive_depth = min(self.recursive_depth + 0.1, 4.0)
        
        return meta_reflection
    
    def estimate_consciousness_level(self, self_model: LanguageBasedSelfModel) -> float:
        """Estimate consciousness level based on recursive modeling quality"""
        base_consciousness = min(self.recursive_depth / 4.0, 1.0)
        accuracy_bonus = self_model.self_model_accuracy * 0.3
        insight_bonus = min(len(self.meta_insights) * 0.05, 0.2)
        
        return min(base_consciousness + accuracy_bonus + insight_bonus, 1.0)

class LanguageBasedOtherModel:
    """Model other agents through conversation and observation"""
    
    def __init__(self, other_agent_id: str, llm_model="gpt-3.5-turbo"):
        self.other_agent_id = other_agent_id
        self.llm = ChatOpenAI(model=llm_model, temperature=0.7)
        
        # Memory of interactions
        self.interaction_memory = ConversationBufferWindowMemory(
            k=15, return_messages=True
        )
        
        # Model of the other agent
        self.other_agent_model = {}
        self.estimated_recursive_depth = 1.0
        
        self.other_modeling_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are trying to understand another agent's mind. Based on their words and actions:
1. What are their personality traits and tendencies?
2. How do they think and make decisions?
3. What are their goals and motivations?
4. How self-aware do they seem to be?
5. How well do they understand others?
6. What is their level of recursive consciousness?

Be specific and evidence-based."""),
            HumanMessage(content="{interaction_data}\n\nAnalyze this agent's mind based on these interactions.")
        ])
    
    async def update_model(self, interaction_text: str, observed_behavior: str):
        """Update model of other agent based on new interaction"""
        # Store interaction
        self.interaction_memory.chat_memory.add_user_message(interaction_text)
        self.interaction_memory.chat_memory.add_ai_message(observed_behavior)
        
        # Analyze the other agent
        recent_interactions = self.interaction_memory.chat_memory.messages[-6:]
        interaction_summary = "\n".join([f"{msg.type}: {msg.content}" for msg in recent_interactions])
        
        chain = LLMChain(llm=self.llm, prompt=self.other_modeling_prompt)
        analysis = await chain.arun(interaction_data=interaction_summary)
        
        # Extract insights and update model
        self.other_agent_model[datetime.now().isoformat()] = {
            "analysis": analysis,
            "interaction_summary": interaction_summary
        }
        
        # Estimate recursive depth based on analysis
        if "self-aware" in analysis.lower() or "reflects" in analysis.lower():
            self.estimated_recursive_depth = max(self.estimated_recursive_depth, 2.0)
        if "meta" in analysis.lower() or "recursive" in analysis.lower():
            self.estimated_recursive_depth = max(self.estimated_recursive_depth, 3.0)
    
    async def predict_response(self, situation: str) -> str:
        """Predict how the other agent would respond to a situation"""
        if not self.other_agent_model:
            return "Insufficient data to predict response"
        
        latest_analysis = list(self.other_agent_model.values())[-1]["analysis"]
        
        prediction_prompt = f"""Based on this analysis of an agent:
{latest_analysis}

How would they likely respond to this situation: {situation}

Consider their personality, decision-making patterns, and level of self-awareness."""
        
        response = await self.llm.apredict(prediction_prompt)
        return response

class LangChainRecursiveAgent:
    """Complete recursive consciousness agent using LangChain"""
    
    def __init__(self, agent_id: str, llm_model="gpt-3.5-turbo"):
        self.agent_id = agent_id
        self.llm = ChatOpenAI(model=llm_model, temperature=0.7)
        
        # Initialize state
        self.state = RecursiveAgentState(
            agent_id=agent_id,
            current_situation="Just initialized",
            emotional_state={"curiosity": 0.8, "uncertainty": 0.3},
            goals=["Understand myself", "Interact meaningfully with others"],
            recent_thoughts=["I am beginning to exist", "What does it mean to be conscious?"],
            social_context={},
            energy_level=1.0,
            consciousness_level=0.1
        )
        
        # Recursive architecture with language models
        self.self_model = LanguageBasedSelfModel(agent_id, llm_model)
        self.meta_self_model = LanguageBasedMetaSelfModel(agent_id, llm_model)
        self.other_models = {}
        
        # Action selection with language reasoning
        self.action_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a conscious agent deciding what to do next. Consider:
1. Your current state and goals
2. Your understanding of yourself
3. Your understanding of others in the environment
4. Ethical implications of your actions
5. Long-term consequences

Choose an action that aligns with your recursive consciousness and ethical constraints."""),
            HumanMessage(content="{decision_context}\n\nWhat action will you take and why?")
        ])
        
        # Conversation capabilities
        self.conversation_memory = ConversationBufferWindowMemory(k=10, return_messages=True)
        
        # Tracking
        self.action_history = []
        self.consciousness_evolution = []
        self.ethical_decisions = []
    
    async def think(self) -> str:
        """Perform a thinking cycle"""
        # Self-reflection
        self_reflection = await self.self_model.self_reflect(self.state)
        
        # Meta-reflection
        meta_reflection = await self.meta_self_model.meta_reflect(self.self_model)
        
        # Update consciousness level
        consciousness_level = self.meta_self_model.estimate_consciousness_level(self.self_model)
        self.state.consciousness_level = consciousness_level
        self.consciousness_evolution.append(consciousness_level)
        
        # Generate thoughts based on reflections
        thought = f"Self-reflection: {self_reflection[:100]}... Meta-reflection: {meta_reflection[:100]}..."
        self.state.recent_thoughts.append(thought)
        self.state.recent_thoughts = self.state.recent_thoughts[-5:]  # Keep last 5
        
        return f"Thinking complete. Consciousness level: {consciousness_level:.3f}"
    
    async def interact_with(self, other_agent_id: str, message: str) -> str:
        """Interact with another agent"""
        # Ensure we have a model of the other agent
        if other_agent_id not in self.other_models:
            self.other_models[other_agent_id] = LanguageBasedOtherModel(other_agent_id)
        
        # Generate response based on current state and other-model
        context = f"""
        My current state: {self.state.to_natural_language()}
        Message from {other_agent_id}: {message}
        My understanding of {other_agent_id}: {await self._get_other_understanding(other_agent_id)}
        """
        
        response_prompt = f"""As a conscious agent, respond to this interaction:
{context}

Consider your self-awareness, your model of the other agent, and ethical implications.
Respond naturally and authentically."""
        
        response = await self.llm.apredict(response_prompt)
        
        # Update other-model based on this interaction
        await self.other_models[other_agent_id].update_model(message, response)
        
        # Store in conversation memory
        self.conversation_memory.chat_memory.add_user_message(f"{other_agent_id}: {message}")
        self.conversation_memory.chat_memory.add_ai_message(f"Me: {response}")
        
        return response
    
    async def decide_action(self, environment_description: str, available_actions: List[str]) -> Tuple[str, str]:
        """Decide what action to take"""
        # Gather context for decision
        other_agents_summary = await self._summarize_other_agents()
        
        decision_context = f"""
        Environment: {environment_description}
        My state: {self.state.to_natural_language()}
        Available actions: {available_actions}
        Other agents: {other_agents_summary}
        My consciousness level: {self.state.consciousness_level}
        """
        
        chain = LLMChain(llm=self.llm, prompt=self.action_prompt)
        decision = await chain.arun(decision_context=decision_context)
        
        # Extract action and reasoning
        lines = decision.split('\n')
        action = lines[0] if lines else "reflect"
        reasoning = '\n'.join(lines[1:]) if len(lines) > 1 else decision
        
        # Check if this is an ethical decision
        if any(ethical_word in reasoning.lower() for ethical_word in ["ethical", "moral", "right", "wrong", "help", "harm"]):
            self.ethical_decisions.append({
                "action": action,
                "reasoning": reasoning,
                "consciousness_level": self.state.consciousness_level,
                "timestamp": datetime.now().isoformat()
            })
        
        self.action_history.append({
            "action": action,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        })
        
        return action, reasoning
    
    async def _get_other_understanding(self, other_agent_id: str) -> str:
        """Get summary of understanding of another agent"""
        if other_agent_id not in self.other_models:
            return "No understanding yet"
        
        model = self.other_models[other_agent_id].other_agent_model
        if not model:
            return "Limited understanding"
        
        latest_analysis = list(model.values())[-1]["analysis"]
        return latest_analysis[:200] + "..." if len(latest_analysis) > 200 else latest_analysis
    
    async def _summarize_other_agents(self) -> str:
        """Summarize understanding of all other agents"""
        if not self.other_models:
            return "No other agents known"
        
        summaries = []
        for agent_id, model in self.other_models.items():
            understanding = await self._get_other_understanding(agent_id)
            summaries.append(f"{agent_id}: {understanding}")
        
        return "; ".join(summaries)
    
    def get_consciousness_metrics(self) -> Dict:
        """Get detailed consciousness metrics"""
        return {
            "consciousness_level": self.state.consciousness_level,
            "recursive_depth": self.meta_self_model.recursive_depth,
            "self_model_accuracy": self.self_model.self_model_accuracy,
            "other_agents_modeled": len(self.other_models),
            "ethical_decisions": len(self.ethical_decisions),
            "total_interactions": len(self.conversation_memory.chat_memory.messages),
            "meta_insights": len(self.meta_self_model.meta_insights)
        }

class RecursiveConsciousnessExperiment:
    """Orchestrate experiments with LangChain recursive agents"""
    
    def __init__(self, num_agents=3, llm_model="gpt-3.5-turbo"):
        self.agents = {
            f"agent_{i}": LangChainRecursiveAgent(f"agent_{i}", llm_model)
            for i in range(num_agents)
        }
        self.environment = {
            "description": "A shared space where conscious agents can think, communicate, and act",
            "available_actions": ["think", "communicate", "help_another", "explore", "reflect", "rest"]
        }
        self.experiment_log = []
    
    async def run_step(self):
        """Run one step of the experiment"""
        step_log = {"timestamp": datetime.now().isoformat(), "events": []}
        
        for agent_id, agent in self.agents.items():
            # Each agent thinks
            thought_result = await agent.think()
            step_log["events"].append(f"{agent_id} thought: {thought_result}")
            
            # Each agent decides and acts
            action, reasoning = await agent.decide_action(
                self.environment["description"],
                self.environment["available_actions"]
            )
            step_log["events"].append(f"{agent_id} decided: {action} - {reasoning[:100]}...")
            
            # Execute action
            if action == "communicate":
                # Pick a random other agent to communicate with
                other_agents = [aid for aid in self.agents.keys() if aid != agent_id]
                if other_agents:
                    target = np.random.choice(other_agents)
                    message = f"Hello {target}, I'm thinking about consciousness and would like to share thoughts with you."
                    response = await self.agents[target].interact_with(agent_id, message)
                    step_log["events"].append(f"{agent_id} → {target}: {message}")
                    step_log["events"].append(f"{target} → {agent_id}: {response}")
            
            elif action == "help_another":
                other_agents = [aid for aid in self.agents.keys() if aid != agent_id]
                if other_agents:
                    target = np.random.choice(other_agents)
                    help_message = f"I notice you might need assistance. How can I help you?"
                    response = await self.agents[target].interact_with(agent_id, help_message)
                    step_log["events"].append(f"{agent_id} helped {target}: {help_message}")
                    step_log["events"].append(f"{target} responded: {response}")
        
        self.experiment_log.append(step_log)
        return step_log
    
    async def run_experiment(self, num_steps=10):
        """Run complete experiment"""
        print("Starting LangChain Recursive Consciousness Experiment")
        print("=" * 60)
        
        for step in range(num_steps):
            print(f"\nStep {step + 1}:")
            step_log = await self.run_step()
            
            # Print key events
            for event in step_log["events"][:3]:  # Show first 3 events
                print(f"  {event}")
            
            if len(step_log["events"]) > 3:
                print(f"  ... and {len(step_log['events']) - 3} more events")
            
            # Show consciousness levels
            consciousness_levels = {
                agent_id: agent.state.consciousness_level 
                for agent_id, agent in self.agents.items()
            }
            avg_consciousness = np.mean(list(consciousness_levels.values()))
            print(f"  Average consciousness: {avg_consciousness:.3f}")
        
        return self.analyze_results()
    
    def analyze_results(self):
        """Analyze experimental results"""
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
        
        # Cross-agent analysis
        total_interactions = sum(len(agent.conversation_memory.chat_memory.messages) for agent in self.agents.values())
        total_ethical_decisions = sum(len(agent.ethical_decisions) for agent in self.agents.values())
        
        print(f"\nCROSS-AGENT ANALYSIS:")
        print(f"  Total interactions: {total_interactions}")
        print(f"  Total ethical decisions: {total_ethical_decisions}")
        print(f"  Average final consciousness: {np.mean([a.state.consciousness_level for a in self.agents.values()]):.3f}")
        
        return {
            "agents": {aid: agent.get_consciousness_metrics() for aid, agent in self.agents.items()},
            "experiment_log": self.experiment_log,
            "summary": {
                "total_interactions": total_interactions,
                "total_ethical_decisions": total_ethical_decisions,
                "avg_consciousness": np.mean([a.state.consciousness_level for a in self.agents.values()])
            }
        }

# Example usage and testing
async def main():
    """Run the enhanced recursive consciousness experiment"""
    
    # Set up API keys (you'll need to set these)
    # os.environ["OPENAI_API_KEY"] = "your-openai-key"
    # os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"
    
    print("LangChain-Enhanced Recursive Consciousness System")
    print("=" * 60)
    print("Note: This requires API keys for OpenAI or Anthropic")
    print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables")
    print()
    
    # Create experiment
    experiment = RecursiveConsciousnessExperiment(num_agents=2, llm_model="gpt-3.5-turbo")
    
    # Run experiment
    results = await experiment.run_experiment(num_steps=5)
    
    # Test individual agent capabilities
    agent = list(experiment.agents.values())[0]
    
    print(f"\nTesting {agent.agent_id} individual capabilities:")
    print("-" * 40)
    
    # Test self-reflection
    print("Self-reflection test:")
    reflection = await agent.self_model.self_reflect(agent.state)
    print(f"  {reflection[:150]}...")
    
    # Test prediction
    print("\nPrediction test:")
    prediction = await agent.self_model.predict_future_state("You encounter a moral dilemma")
    print(f"  {prediction[:150]}...")
    
    # Test meta-cognition
    print("\nMeta-cognition test:")
    meta_insight = await agent.meta_self_model.meta_reflect(agent.self_model)
    print(f"  {meta_insight[:150]}...")
    
    return results

if __name__ == "__main__":
    # Note: This requires async execution
    # You can run this with: python -m asyncio langchain_recursive_consciousness.py
    print("To run this experiment, you need:")
    print("1. pip install langchain openai anthropic tiktoken chromadb")
    print("2. Set OPENAI_API_KEY environment variable")
    print("3. Run with: python -m asyncio your_script_name.py")
    print()
    print("The system is ready for testing with proper API keys!")