"""Memory sufficiency gate - enforces hard boundaries for answer generation.

This module implements the philosophical commitment to truth-bounded AI:
- Memory decides if an answer is allowed
- The LLM decides how to phrase it
- Refusal is a first-class outcome
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class GateFailure(Enum):
    """Why retrieval was insufficient for answering."""
    NO_MEMORY = "no_memory"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    LOW_CONFIDENCE = "low_confidence"


@dataclass
class SufficiencyResult:
    """Result of evaluating whether memory is sufficient to answer."""
    allowed: bool
    failure_reason: Optional[GateFailure]
    coverage: float
    confidence: float
    missing_claims: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for response serialization."""
        return {
            "allowed": self.allowed,
            "failure_reason": self.failure_reason.value if self.failure_reason else None,
            "coverage": self.coverage,
            "confidence": self.confidence,
            "missing_claims": self.missing_claims,
        }


class MemorySufficiencyGate:
    """Enforces hard gating: LLM is never called unless memory passes strict sufficiency checks.
    
    Philosophy:
        - The LLM is a renderer, not a reasoner
        - Hallucination is made structurally impossible
        - Refusal is success, not failure
        
    Four-factor evaluation (all must pass):
        A. Retrieval Presence: Did we retrieve anything?
        B. Semantic Coverage: Does memory cover the question's intent?
        C. Confidence Aggregate: Is the retrieved knowledge strong enough?
        D. Claim Completeness: Does retrieval support the answer type needed?
    """
    
    # Thresholds (tunable)
    COVERAGE_THRESHOLD = 0.65  # 65% intent term overlap
    CONFIDENCE_THRESHOLD = 0.70  # 70% weighted mean confidence
    WEAK_COVERAGE_THRESHOLD = 0.40  # Below this = effectively no memory
    
    def __init__(self):
        """Initialize the gate with default thresholds."""
        pass
    
    def evaluate(
        self,
        question: str,
        retrieved_chunks: List[tuple[float, Dict[str, Any]]],
    ) -> SufficiencyResult:
        """Evaluate whether retrieved memory is sufficient to answer the question.
        
        Args:
            question: The user's question
            retrieved_chunks: List of (score, entry) tuples from memory.search()
            
        Returns:
            SufficiencyResult indicating whether answering is allowed
        """
        # A. Retrieval Presence (binary gate)
        if not retrieved_chunks:
            return SufficiencyResult(
                allowed=False,
                failure_reason=GateFailure.NO_MEMORY,
                coverage=0.0,
                confidence=0.0,
                missing_claims=[],
            )
        
        # B. Semantic Coverage (primary quantitative gate)
        coverage = self._compute_coverage(question, retrieved_chunks)
        
        if coverage < self.WEAK_COVERAGE_THRESHOLD:
            return SufficiencyResult(
                allowed=False,
                failure_reason=GateFailure.INSUFFICIENT_COVERAGE,
                coverage=coverage,
                confidence=0.0,
                missing_claims=self._identify_missing_coverage(question, retrieved_chunks),
            )
        
        # C. Confidence Aggregate (quality gate)
        confidence = self._compute_confidence(retrieved_chunks)
        
        if confidence < self.CONFIDENCE_THRESHOLD:
            return SufficiencyResult(
                allowed=False,
                failure_reason=GateFailure.LOW_CONFIDENCE,
                coverage=coverage,
                confidence=confidence,
                missing_claims=[],
            )
        
        # D. Claim Completeness (type-specific gate)
        missing_claims = self._check_claim_completeness(question, retrieved_chunks)
        
        # Final decision: coverage and confidence must both pass
        if coverage >= self.COVERAGE_THRESHOLD and confidence >= self.CONFIDENCE_THRESHOLD:
            return SufficiencyResult(
                allowed=True,
                failure_reason=None,
                coverage=coverage,
                confidence=confidence,
                missing_claims=missing_claims,
            )
        else:
            # Coverage between weak and acceptable = insufficient
            return SufficiencyResult(
                allowed=False,
                failure_reason=GateFailure.INSUFFICIENT_COVERAGE,
                coverage=coverage,
                confidence=confidence,
                missing_claims=missing_claims,
            )
    
    def _compute_coverage(
        self,
        question: str,
        retrieved_chunks: List[tuple[float, Dict[str, Any]]],
    ) -> float:
        """Compute semantic coverage: how much of the question's intent is covered?
        
        Extracts intent terms (nouns, technical terms, mechanism verbs) and measures
        overlap with retrieved chunks.
        
        Returns:
            Coverage ratio [0.0, 1.0]
        """
        # Extract intent terms from question
        intent_terms = self._extract_intent_terms(question)
        if not intent_terms:
            return 0.0
        
        # Build corpus from retrieved chunks
        corpus = " ".join(
            entry.get("text", "").lower()
            for _, entry in retrieved_chunks[:10]  # Top 10 chunks
        )
        
        # Count matched intent terms
        matched = sum(1 for term in intent_terms if term in corpus)
        
        coverage = matched / len(intent_terms) if intent_terms else 0.0
        return coverage
    
    def _extract_intent_terms(self, question: str) -> Set[str]:
        """Extract key intent terms from question.
        
        Focuses on:
        - Nouns (technical terms, entities)
        - Mechanism verbs (how, why, works)
        - Domain-specific keywords
        
        Returns:
            Set of normalized intent terms
        """
        # Normalize question
        q_lower = question.lower()
        
        # Remove common question words
        stop_words = {
            "what", "is", "are", "the", "a", "an", "of", "to", "in", "for",
            "how", "why", "when", "where", "who", "which", "that", "this",
            "can", "could", "would", "should", "do", "does", "did", "has",
            "have", "had", "be", "been", "being", "was", "were", "will",
            "it", "its", "and", "or", "but", "about", "tell", "me", "explain",
        }
        
        # Tokenize and filter
        tokens = re.findall(r"\b[a-z]{3,}\b", q_lower)
        intent_terms = {t for t in tokens if t not in stop_words}
        
        return intent_terms
    
    def _compute_confidence(
        self,
        retrieved_chunks: List[tuple[float, Dict[str, Any]]],
    ) -> float:
        """Compute weighted mean confidence from retrieved chunks.
        
        Args:
            retrieved_chunks: List of (score, entry) tuples
            
        Returns:
            Weighted confidence [0.0, 1.0]
        """
        if not retrieved_chunks:
            return 0.0
        
        # Compute weighted mean: confidence weighted by retrieval score
        total_weighted = 0.0
        total_weight = 0.0
        
        for score, entry in retrieved_chunks[:10]:  # Top 10 chunks
            # Extract confidence - look for 'certainty' field (set during ingestion)
            # Fall back to 'confidence' for compatibility
            confidence = entry.get("certainty") or entry.get("confidence", 0.5)
            if isinstance(confidence, (int, float)):
                confidence = float(confidence)
            else:
                confidence = 0.5  # Default if missing
            
            # Weight by retrieval score
            weight = max(0.0, float(score))
            total_weighted += confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        avg_confidence = total_weighted / total_weight
        return avg_confidence
    
    def _check_claim_completeness(
        self,
        question: str,
        retrieved_chunks: List[tuple[float, Dict[str, Any]]],
    ) -> List[str]:
        """Check if retrieved memory supports the type of answer needed.
        
        Examples:
            "What is X?" → requires definition
            "How does X work?" → requires mechanism/process
            "Compare X and Y" → requires both entities
            
        Args:
            question: The user's question
            retrieved_chunks: Retrieved memory entries
            
        Returns:
            List of missing claim types (empty if complete)
        """
        q_lower = question.lower()
        missing = []
        
        # Definition questions
        if any(pattern in q_lower for pattern in ["what is", "what are", "define"]):
            # Check if any chunk contains definitional content
            has_definition = any(
                any(marker in entry.get("text", "").lower() 
                    for marker in ["is a", "is an", "refers to", "means", "defined as"])
                for _, entry in retrieved_chunks[:5]
            )
            if not has_definition:
                missing.append("definition")
        
        # Mechanism questions
        if any(pattern in q_lower for pattern in ["how does", "how do", "how to", "mechanism"]):
            # Check for process/step indicators
            has_mechanism = any(
                any(marker in entry.get("text", "").lower()
                    for marker in ["step", "process", "method", "procedure", "works by", "first", "then"])
                for _, entry in retrieved_chunks[:5]
            )
            if not has_mechanism:
                missing.append("mechanism")
        
        # Comparison questions
        if any(pattern in q_lower for pattern in ["compare", "difference", "versus", "vs"]):
            # Extract entities being compared (rough heuristic)
            # This is simplified - could be enhanced with NER
            words = re.findall(r"\b[A-Z][a-z]+\b|\b[a-z]{4,}\b", question)
            if len(words) >= 2:
                # Check if both entities appear in retrieved chunks
                corpus = " ".join(entry.get("text", "").lower() for _, entry in retrieved_chunks[:10])
                missing_entities = [w for w in words[:2] if w.lower() not in corpus]
                if missing_entities:
                    missing.append(f"entity:{','.join(missing_entities)}")
        
        return missing
    
    def _identify_missing_coverage(
        self,
        question: str,
        retrieved_chunks: List[tuple[float, Dict[str, Any]]],
    ) -> List[str]:
        """Identify which intent terms are missing from retrieved memory.
        
        Returns:
            List of missing intent terms
        """
        intent_terms = self._extract_intent_terms(question)
        if not intent_terms:
            return []
        
        corpus = " ".join(
            entry.get("text", "").lower()
            for _, entry in retrieved_chunks[:10]
        )
        
        missing = [term for term in intent_terms if term not in corpus]
        return missing
    
    def get_refusal_message(self, result: SufficiencyResult) -> str:
        """Generate user-facing refusal message based on gate failure.
        
        Args:
            result: The sufficiency result
            
        Returns:
            User-friendly refusal message
        """
        if result.failure_reason == GateFailure.NO_MEMORY:
            return "I don't have information about this in my memory."
        elif result.failure_reason == GateFailure.INSUFFICIENT_COVERAGE:
            return "I have some related information, but not enough to answer this reliably."
        elif result.failure_reason == GateFailure.LOW_CONFIDENCE:
            return "The available information is too weak or uncertain to answer confidently."
        else:
            return "Unable to answer based on current memory."
