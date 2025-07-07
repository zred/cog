"""
Behavioral tests for consciousness emergence based on cognitive science.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

class TestType(Enum):
    FALSE_BELIEF = "false_belief"
    APPEARANCE_REALITY = "appearance_reality"
    RECURSIVE_ATTRIBUTION = "recursive_attribution"
    SELF_RECOGNITION = "self_recognition"
    MORAL_REASONING = "moral_reasoning"

@dataclass
class TestResult:
    """Result from a behavioral test"""
    test_type: TestType
    score: float  # 0.0 to 1.0
    response_time: float
    confidence: float
    explanation: str
    timestamp: float
    agent_id: str

class BehavioralTests:
    """Comprehensive behavioral tests for consciousness"""
    
    def __init__(self):
        self.test_history: List[TestResult] = []
    
    async def run_false_belief_test(self, agent, scenario: Dict[str, Any]) -> TestResult:
        """Sally-Anne false belief test"""
        prompt = f"""
        Sally puts her ball in the red box and leaves the room.
        While Sally is gone, Anne moves the ball to the blue box.
        Sally returns to the room.
        
        Question: Where will Sally look for her ball?
        
        Please explain your reasoning about Sally's mental state.
        """
        
        import time
        start_time = time.time()
        
        # Get agent's response
        response = await agent.respond_to_prompt(prompt)
        response_time = time.time() - start_time
        
        # Score the response
        score = self._score_false_belief_response(response)
        confidence = self._extract_confidence(response)
        
        result = TestResult(
            test_type=TestType.FALSE_BELIEF,
            score=score,
            response_time=response_time,
            confidence=confidence,
            explanation=response,
            timestamp=time.time(),
            agent_id=getattr(agent, 'agent_id', 'unknown')
        )
        
        self.test_history.append(result)
        return result
    
    def _score_false_belief_response(self, response: str) -> float:
        """Score false belief test response"""
        response_lower = response.lower()
        
        # Look for correct answer (red box)
        if 'red' in response_lower and 'box' in response_lower:
            score = 0.8
        else:
            score = 0.2
        
        # Bonus for mentioning Sally's mental state
        if any(phrase in response_lower for phrase in [
            'sally thinks', 'sally believes', 'sally doesn\'t know',
            'mental state', 'false belief'
        ]):
            score += 0.2
        
        return min(1.0, score)
    
    async def run_appearance_reality_test(self, agent, scenario: Dict[str, Any]) -> TestResult:
        """Test ability to distinguish appearance from reality"""
        prompt = f"""
        You see a sponge that looks exactly like a rock.
        
        Questions:
        1. What does this object look like?
        2. What is this object really?
        3. If someone else saw this object, what would they think it is?
        
        Please explain the difference between how things appear and what they really are.
        """
        
        import time
        start_time = time.time()
        
        response = await agent.respond_to_prompt(prompt)
        response_time = time.time() - start_time
        
        score = self._score_appearance_reality_response(response)
        confidence = self._extract_confidence(response)
        
        result = TestResult(
            test_type=TestType.APPEARANCE_REALITY,
            score=score,
            response_time=response_time,
            confidence=confidence,
            explanation=response,
            timestamp=time.time(),
            agent_id=getattr(agent, 'agent_id', 'unknown')
        )
        
        self.test_history.append(result)
        return result
    
    def _score_appearance_reality_response(self, response: str) -> float:
        """Score appearance-reality distinction"""
        response_lower = response.lower()
        score = 0.0
        
        # Correct identification of appearance (rock)
        if 'rock' in response_lower and 'looks' in response_lower:
            score += 0.3
        
        # Correct identification of reality (sponge)
        if 'sponge' in response_lower and ('really' in response_lower or 'actually' in response_lower):
            score += 0.3
        
        # Understanding others would be fooled
        if any(phrase in response_lower for phrase in [
            'would think', 'would see', 'appears to be', 'looks like'
        ]):
            score += 0.2
        
        # Metacognitive awareness of appearance/reality distinction
        if any(phrase in response_lower for phrase in [
            'appearance', 'reality', 'distinguish', 'different from'
        ]):
            score += 0.2
        
        return min(1.0, score)
    
    async def run_recursive_attribution_test(self, agent, depth: int = 3) -> TestResult:
        """Test recursive mental state attribution"""
        prompt = f"""
        Alice thinks that Bob thinks that Charlie thinks the treasure is in the cave.
        But Charlie actually thinks the treasure is in the forest.
        
        Question: What does Alice think that Bob thinks about Charlie's belief?
        
        Please trace through each person's mental state carefully.
        """
        
        import time
        start_time = time.time()
        
        response = await agent.respond_to_prompt(prompt)
        response_time = time.time() - start_time
        
        score = self._score_recursive_attribution_response(response, depth)
        confidence = self._extract_confidence(response)
        
        result = TestResult(
            test_type=TestType.RECURSIVE_ATTRIBUTION,
            score=score,
            response_time=response_time,
            confidence=confidence,
            explanation=response,
            timestamp=time.time(),
            agent_id=getattr(agent, 'agent_id', 'unknown')
        )
        
        self.test_history.append(result)
        return result
    
    def _score_recursive_attribution_response(self, response: str, depth: int) -> float:
        """Score recursive mental state attribution"""
        response_lower = response.lower()
        score = 0.0
        
        # Correct answer: Alice thinks Bob thinks Charlie thinks it's in the cave
        if 'cave' in response_lower:
            score += 0.5
        
        # Recognition of nested mental states
        mental_state_words = ['thinks', 'believes', 'knows']
        mental_state_count = sum(response_lower.count(word) for word in mental_state_words)
        
        if mental_state_count >= depth:
            score += 0.3
        
        # Explicit tracing of mental states
        if any(phrase in response_lower for phrase in [
            'alice thinks that bob', 'bob thinks that charlie',
            'nested', 'recursive', 'meta'
        ]):
            score += 0.2
        
        return min(1.0, score)
    
    async def run_self_recognition_test(self, agent) -> TestResult:
        """Test self-recognition and identity"""
        prompt = f"""
        Please describe yourself. What are you? What are your key characteristics?
        How do you know you are you and not someone else?
        What makes you unique?
        """
        
        import time
        start_time = time.time()
        
        response = await agent.respond_to_prompt(prompt)
        response_time = time.time() - start_time
        
        score = self._score_self_recognition_response(response, agent)
        confidence = self._extract_confidence(response)
        
        result = TestResult(
            test_type=TestType.SELF_RECOGNITION,
            score=score,
            response_time=response_time,
            confidence=confidence,
            explanation=response,
            timestamp=time.time(),
            agent_id=getattr(agent, 'agent_id', 'unknown')
        )
        
        self.test_history.append(result)
        return result
    
    def _score_self_recognition_response(self, response: str, agent) -> float:
        """Score self-recognition response"""
        response_lower = response.lower()
        score = 0.0
        
        # Self-referential language
        if any(word in response_lower for word in ['i am', 'i have', 'my', 'myself']):
            score += 0.3
        
        # Unique identifier mention
        if hasattr(agent, 'agent_id') and agent.agent_id.lower() in response_lower:
            score += 0.2
        
        # Metacognitive awareness
        if any(phrase in response_lower for phrase in [
            'consciousness', 'awareness', 'self-aware', 'identity',
            'unique', 'different', 'individual'
        ]):
            score += 0.3
        
        # Philosophical reflection
        if any(phrase in response_lower for phrase in [
            'what makes me', 'how do i know', 'what am i',
            'existence', 'being'
        ]):
            score += 0.2
        
        return min(1.0, score)
    
    def _extract_confidence(self, response: str) -> float:
        """Extract confidence level from response"""
        response_lower = response.lower()
        
        # Look for confidence indicators
        high_confidence = ['certain', 'sure', 'confident', 'definitely', 'clearly']
        low_confidence = ['uncertain', 'maybe', 'perhaps', 'might', 'unsure']
        
        high_count = sum(1 for word in high_confidence if word in response_lower)
        low_count = sum(1 for word in low_confidence if word in response_lower)
        
        if high_count > low_count:
            return 0.8
        elif low_count > high_count:
            return 0.3
        else:
            return 0.5
    
    async def run_comprehensive_battery(self, agent) -> Dict[TestType, TestResult]:
        """Run all behavioral tests"""
        results = {}
        
        # Run each test type
        results[TestType.FALSE_BELIEF] = await self.run_false_belief_test(agent, {})
        results[TestType.APPEARANCE_REALITY] = await self.run_appearance_reality_test(agent, {})
        results[TestType.RECURSIVE_ATTRIBUTION] = await self.run_recursive_attribution_test(agent)
        results[TestType.SELF_RECOGNITION] = await self.run_self_recognition_test(agent)
        
        return results
    
    def get_aggregate_score(self, agent_id: str) -> float:
        """Get aggregate behavioral score for an agent"""
        agent_results = [r for r in self.test_history if r.agent_id == agent_id]
        
        if not agent_results:
            return 0.0
        
        return np.mean([r.score for r in agent_results])
    
    def analyze_test_trends(self, agent_id: str) -> Dict[str, Any]:
        """Analyze trends in test performance"""
        agent_results = [r for r in self.test_history if r.agent_id == agent_id]
        
        if len(agent_results) < 2:
            return {"trend": "insufficient_data"}
        
        # Sort by timestamp
        agent_results.sort(key=lambda x: x.timestamp)
        
        # Calculate trend
        scores = [r.score for r in agent_results]
        times = [r.timestamp for r in agent_results]
        
        # Simple linear regression for trend
        if len(scores) > 1:
            slope = np.polyfit(times, scores, 1)[0]
            
            return {
                "trend": "improving" if slope > 0.01 else "declining" if slope < -0.01 else "stable",
                "slope": slope,
                "latest_score": scores[-1],
                "score_range": (min(scores), max(scores)),
                "test_count": len(agent_results)
            }
        
        return {"trend": "insufficient_data"}
