"""Enhanced dialogue manager with context tracking, repair strategies, and coherence checking.

Features:
- Conversation context tracking with semantic understanding
- Repair strategies for handling misunderstandings or contradictions
- Coherence checking for response quality
- Turn management and conversation flow
- Intent tracking and dialogue state management
- Error recovery strategies
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from logging import getLogger

logger = getLogger(__name__)


class DialogueState(Enum):
    """States in a dialogue flow."""
    GREETING = "greeting"
    QUERY = "query"
    CLARIFICATION = "clarification"
    RESPONSE = "response"
    REPAIR = "repair"
    CLOSING = "closing"


class IntentType(Enum):
    """Types of user intents."""
    QUESTION = "question"
    STATEMENT = "statement"
    CORRECTION = "correction"
    CLARIFICATION = "clarification"
    ACKNOWLEDGMENT = "acknowledgment"
    COMMAND = "command"


@dataclass
class DialogueTurn:
    """Represents one turn in a conversation."""
    role: str  # "user" or "assistant"
    text: str
    timestamp: float = field(default_factory=time.time)
    intent: Optional[IntentType] = None
    entities: List[str] = field(default_factory=list)
    sentiment: float = 0.0  # -1 (negative) to 1 (positive)
    coherence_score: float = 0.5
    context_references: List[int] = field(default_factory=list)  # Indices of referenced turns
    follow_up: Optional[str] = None


@dataclass
class DialogueContext:
    """Semantic context of a conversation."""
    topics: List[str] = field(default_factory=list)
    mentioned_entities: List[str] = field(default_factory=list)
    unresolved_questions: List[str] = field(default_factory=list)
    corrections_made: List[Dict[str, Any]] = field(default_factory=list)
    state: DialogueState = DialogueState.GREETING
    turn_count: int = 0
    coherence_history: List[float] = field(default_factory=list)


class RepairStrategy:
    """Handles dialogue repair when misunderstandings occur."""
    
    def __init__(self):
        self.confidence_threshold = 0.6
    
    def should_repair(self, coherence_score: float, response_confidence: float) -> bool:
        """Determine if repair is needed."""
        return coherence_score < 0.5 or response_confidence < self.confidence_threshold
    
    def suggest_clarification(self, last_user_turn: str, topics: List[str]) -> Optional[str]:
        """Suggest clarification based on last user input."""
        clarification_prompts = [
            f"Could you clarify what you mean by '{last_user_turn.split()[0]}'?",
            f"Are you asking about {topics[0] if topics else 'this'}?",
            "Can you provide more details?",
            "I'm not sure I understood. Could you rephrase?",
        ]
        
        # Select based on context
        if len(topics) > 0:
            return f"Are you asking about {topics[0]}?"
        return clarification_prompts[0]
    
    def suggest_correction(self, conflicting_claims: List[Dict[str, Any]]) -> Optional[str]:
        """Suggest how to correct conflicting information."""
        if len(conflicting_claims) < 2:
            return None
        
        claim1 = conflicting_claims[0].get("text", "")
        claim2 = conflicting_claims[1].get("text", "")
        
        return f"I notice conflicting statements: '{claim1}' vs '{claim2}'. Which is correct?"


class DialogueManager:
    """Enhanced dialogue manager with context tracking and repair strategies."""
    
    def __init__(self, max_turns: int = 16, max_history_size: int = 32):
        self.max_turns = max_turns
        self.max_history_size = max_history_size
        self.turns: List[DialogueTurn] = []
        self.context = DialogueContext()
        self.repair_strategy = RepairStrategy()
        self.turn_index: Dict[int, DialogueTurn] = {}  # For quick access
        
    def add_user(
        self,
        text: str,
        intent: Optional[IntentType] = None,
        entities: Optional[List[str]] = None,
        sentiment: float = 0.0,
    ) -> None:
        """Add user turn with optional metadata."""
        turn = DialogueTurn(
            role="user",
            text=text,
            intent=intent or self._infer_intent(text),
            entities=entities or [],
            sentiment=sentiment,
            context_references=self._find_context_references(text),
        )
        
        self._add_turn(turn)
        self.context.turn_count += 1
        
        # Update context based on intent
        if turn.intent == IntentType.QUESTION:
            self.context.unresolved_questions.append(text)
        elif turn.intent == IntentType.CORRECTION:
            self.context.corrections_made.append({
                "text": text,
                "turn_index": len(self.turns) - 1,
                "timestamp": turn.timestamp,
            })
        
        # Update topics
        self._update_topics(text)
        self._update_entities(entities or [])
        
        logger.debug(f"User: {text[:50]}... (intent: {turn.intent.value if turn.intent else 'unknown'})")
    
    def add_assistant(
        self,
        text: str,
        coherence_score: float = 0.7,
        response_confidence: float = 0.7,
    ) -> None:
        """Add assistant turn with quality metrics."""
        turn = DialogueTurn(
            role="assistant",
            text=text,
            coherence_score=coherence_score,
            context_references=self._find_context_references(text),
        )
        
        self._add_turn(turn)
        self.context.coherence_history.append(coherence_score)
        
        # Check if repair is needed
        if self.repair_strategy.should_repair(coherence_score, response_confidence):
            logger.warning(f"Low quality response detected (coherence: {coherence_score})")
            turn.follow_up = self.repair_strategy.suggest_clarification(
                self._get_last_user_text(),
                self.context.topics,
            )
        
        logger.debug(f"Assistant: {text[:50]}... (coherence: {coherence_score:.2f})")
    
    def _add_turn(self, turn: DialogueTurn) -> None:
        """Add turn and manage history size."""
        idx = len(self.turns)
        self.turns.append(turn)
        self.turn_index[idx] = turn
        self._trim()
    
    def _trim(self) -> None:
        """Keep dialogue history within bounds."""
        if len(self.turns) > self.max_history_size:
            removed_count = len(self.turns) - self.max_history_size
            self.turns = self.turns[removed_count:]
            # MEMORY LEAK FIX: Rebuild indices efficiently without complex comprehension
            if self.turn_index:
                new_index = {}
                for i, turn in enumerate(self.turns):
                    new_index[i] = turn
                self.turn_index = new_index
            else:
                self.turn_index = {i: turn for i, turn in enumerate(self.turns)}
    
    def _infer_intent(self, text: str) -> IntentType:
        """Infer user intent from text."""
        text_lower = text.lower().strip()
        
        # Question detection
        if "?" in text or any(q in text_lower for q in ["what", "why", "how", "when", "where", "who"]):
            return IntentType.QUESTION
        
        # Correction detection
        if any(c in text_lower for c in ["actually", "correction", "no that", "i meant", "i was wrong"]):
            return IntentType.CORRECTION
        
        # Clarification request
        if any(c in text_lower for c in ["clarify", "explain", "what do you mean", "could you"]):
            return IntentType.CLARIFICATION
        
        # Acknowledgment
        if any(a in text_lower for a in ["ok", "yes", "no", "sure", "got it", "understood"]):
            return IntentType.ACKNOWLEDGMENT
        
        # Command
        if text_lower.startswith(("help", "show", "tell", "do", "make")):
            return IntentType.COMMAND
        
        # Default to statement
        return IntentType.STATEMENT
    
    def _update_topics(self, text: str) -> None:
        """Extract and update conversation topics."""
        # Simple extraction: capitalize nouns (basic heuristic)
        words = text.split()
        for word in words:
            if len(word) > 3 and word[0].isupper():
                if word not in self.context.topics:
                    self.context.topics.append(word)
                    if len(self.context.topics) > 10:
                        self.context.topics = self.context.topics[-10:]
    
    def _update_entities(self, entities: List[str]) -> None:
        """Update mentioned entities."""
        for entity in entities:
            if entity not in self.context.mentioned_entities:
                self.context.mentioned_entities.append(entity)
                if len(self.context.mentioned_entities) > 20:
                    self.context.mentioned_entities = self.context.mentioned_entities[-20:]
    
    def _find_context_references(self, text: str) -> List[int]:
        """Find which previous turns are referenced in this text."""
        references = []
        
        # Simple: look for pronouns and repeated concepts
        for i, turn in enumerate(self.turns[-5:]):  # Check last 5 turns
            turn_words = set(turn.text.lower().split())
            text_words = set(text.lower().split())
            
            # If significant word overlap, mark as referenced
            if len(turn_words & text_words) / max(len(turn_words), 1) > 0.3:
                references.append(len(self.turns) - 5 + i)
        
        return references
    
    def _get_last_user_text(self) -> str:
        """Get last user utterance."""
        for turn in reversed(self.turns):
            if turn.role == "user":
                return turn.text
        return ""
    
    def get_formatted(self) -> str:
        """Get formatted dialogue history."""
        parts = []
        for i, turn in enumerate(self.turns):
            prefix = "User" if turn.role == "user" else "Assistant"
            parts.append(f"{prefix}: {turn.text}")
            
            # Add metadata if available
            if turn.role == "assistant" and turn.follow_up:
                parts.append(f"  [Note: {turn.follow_up}]")
        
        return "\n".join(parts)
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of dialogue context."""
        return {
            "state": self.context.state.value,
            "turn_count": self.context.turn_count,
            "topics": self.context.topics[-5:],  # Last 5 topics
            "unresolved_questions": self.context.unresolved_questions[-3:],  # Last 3
            "mentioned_entities": self.context.mentioned_entities[-5:],
            "avg_coherence": sum(self.context.coherence_history[-5:]) / max(1, len(self.context.coherence_history[-5:])),
            "corrections_count": len(self.context.corrections_made),
        }
    
    def should_repair(self) -> bool:
        """Determine if dialogue repair is needed."""
        if not self.context.coherence_history:
            return False
        
        recent_coherence = sum(self.context.coherence_history[-3:]) / 3
        return recent_coherence < 0.6
    
    def suggest_repair(self) -> Optional[str]:
        """Suggest a repair for dialogue flow."""
        last_user = self._get_last_user_text()
        
        if self.context.unresolved_questions:
            return f"Should we continue discussing: '{self.context.unresolved_questions[-1]}'?"
        
        if self.context.topics:
            return f"Let's clarify about {self.context.topics[-1]}."
        
        return self.repair_strategy.suggest_clarification(last_user, self.context.topics)
    
    def get_recent_context(self, num_turns: int = 4) -> str:
        """Get recent dialogue context for LLM input."""
        recent = self.turns[-num_turns * 2:] if len(self.turns) >= num_turns * 2 else self.turns
        
        parts = []
        for turn in recent:
            prefix = "User" if turn.role == "user" else "Assistant"
            parts.append(f"{prefix}: {turn.text}")
        
        return "\n".join(parts)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dialogue statistics."""
        user_turns = [t for t in self.turns if t.role == "user"]
        assistant_turns = [t for t in self.turns if t.role == "assistant"]
        
        return {
            "total_turns": len(self.turns),
            "user_turns": len(user_turns),
            "assistant_turns": len(assistant_turns),
            "avg_user_length": sum(len(t.text) for t in user_turns) / max(1, len(user_turns)),
            "avg_assistant_length": sum(len(t.text) for t in assistant_turns) / max(1, len(assistant_turns)),
            "avg_coherence": sum(t.coherence_score for t in assistant_turns) / max(1, len(assistant_turns)),
            "topics": self.context.topics,
            "unresolved_count": len(self.context.unresolved_questions),
        }


__all__ = ["DialogueManager", "DialogueTurn", "DialogueContext", "DialogueState", "IntentType", "RepairStrategy"]

