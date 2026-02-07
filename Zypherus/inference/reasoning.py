"""Enhanced reasoning engine with symbolic rules and confidence propagation.

Provides structured reasoning capabilities including:
- Multi-tier confidence classification (known, probable, uncertain)
- Symbolic rule support (transitivity, negation, quantifiers)
- Confidence propagation through inference chains
- Domain-specific reasoning rules
- Contradiction detection and resolution
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import re
from logging import getLogger

logger = getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence tier classification."""
    CERTAIN = 0.90
    HIGH = 0.75
    PROBABLE = 0.60
    MODERATE = 0.45
    LOW = 0.35
    VERY_LOW = 0.20
    UNKNOWN = 0.0


@dataclass
class SymbolicRule:
    """Represents a logical rule for reasoning."""
    name: str
    pattern: str  # Regex or predicate pattern
    conclusion_template: str  # Template with {subject}, {predicate}, {object}
    confidence_boost: float = 0.1  # How much to boost confidence
    condition: Optional[Callable[[Dict], bool]] = None  # Optional condition check
    
    def apply(self, claim: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply rule to claim, returning enhanced claim or None if not applicable."""
        if self.condition and not self.condition(claim):
            return None
        
        raw = claim.get("raw", "")
        if not re.search(self.pattern, raw, re.IGNORECASE):
            return None
        
        # Build conclusion from template
        subj = claim.get("subject", "")
        pred = claim.get("predicate", "")
        obj = claim.get("object", "")
        
        try:
            conclusion = self.conclusion_template.format(
                subject=subj, predicate=pred, object=obj
            )
        except (KeyError, ValueError):
            conclusion = raw
        
        # Create inference chain record
        inference_chain = claim.get("inference_chain", [])
        inference_chain.append({
            "rule": self.name,
            "confidence_boost": self.confidence_boost,
        })
        
        # Create enhanced claim
        new_confidence = min(1.0, claim.get("confidence", 0.5) + self.confidence_boost)
        
        return {
            **claim,
            "conclusion": conclusion,
            "confidence": new_confidence,
            "inference_chain": inference_chain,
            "derived_by_rule": self.name,
        }


@dataclass
class InferenceChain:
    """Represents a chain of reasoning steps."""
    claims: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0
    steps: List[str] = field(default_factory=list)
    contradicted_by: List[str] = field(default_factory=list)
    
    def propagate_confidence(self) -> float:
        """Propagate confidence through chain (conjunction of steps)."""
        if not self.claims:
            return 1.0
        return min(c.get("confidence", 1.0) for c in self.claims)


class ReasoningEngine:
    """Enhanced reasoning engine with symbolic rules and confidence propagation."""
    
    def __init__(self, claim_store: Any):
        self.claim_store = claim_store
        self.symbolic_rules = self._init_symbolic_rules()
        self.reasoning_history: List[Dict] = []
        
    def _init_symbolic_rules(self) -> List[SymbolicRule]:
        """Initialize default symbolic rules."""
        return [
            # Transitivity rules
            SymbolicRule(
                name="transitive_relation",
                pattern=r"(\w+)\s+(is|are|related to)\s+(\w+)",
                conclusion_template="{subject} {predicate} {object}",
                confidence_boost=0.05,
            ),
            # Negation handling
            SymbolicRule(
                name="negation_inference",
                pattern=r"(not|no|never|cannot)",
                conclusion_template="{subject} NOT {predicate}",
                confidence_boost=0.0,  # No boost for negation (maintain confidence)
                condition=lambda c: c.get("confidence", 0) < 0.8,  # Only for uncertain claims
            ),
            # Quantifier rules
            SymbolicRule(
                name="universal_quantifier",
                pattern=r"(all|every|each)\s+(\w+)",
                conclusion_template="All {object} {predicate}",
                confidence_boost=0.1,
            ),
            # Temporal rules
            SymbolicRule(
                name="temporal_reasoning",
                pattern=r"(before|after|during|while)\s+(\w+)",
                conclusion_template="{subject} {predicate} {object}",
                confidence_boost=0.05,
            ),
        ]
    
    def register_rule(self, rule: SymbolicRule) -> None:
        """Register a custom reasoning rule."""
        self.symbolic_rules.append(rule)
        logger.debug(f"Registered rule: {rule.name}")
    
    def apply_symbolic_rules(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply symbolic rules to enhance claims."""
        enhanced = []
        for claim in claims:
            best_enhancement = claim  # Start with original
            for rule in self.symbolic_rules:
                enhanced_claim = rule.apply(claim)
                if enhanced_claim and enhanced_claim.get("confidence", 0) > best_enhancement.get("confidence", 0):
                    best_enhancement = enhanced_claim
            enhanced.append(best_enhancement)
        return enhanced
    
    def propagate_confidence(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Propagate confidence based on relationships between claims."""
        result = []
        for i, claim in enumerate(claims):
            conf = claim.get("confidence", 0.5)
            
            # Boost confidence if supporting claims exist
            supporting = self._find_supporting_claims(claim, claims)
            if supporting:
                support_conf = sum(c.get("confidence", 0.5) for c in supporting) / len(supporting)
                conf = min(1.0, (conf + support_conf) / 2 + 0.05)
            
            # Reduce confidence if contradictions exist
            contradicting = self._find_contradictions(claim, claims)
            if contradicting:
                max_contra_conf = max(c.get("confidence", 0.5) for c in contradicting)
                if max_contra_conf > conf:
                    conf = max(0.1, conf - 0.2)
            
            result.append({
                **claim,
                "confidence": conf,
                "supporting_claims": len(supporting),
                "contradicting_claims": len(contradicting),
            })
        
        return result
    
    def _find_supporting_claims(self, claim: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find claims that support the given claim."""
        subject = claim.get("subject", "").lower()
        obj = claim.get("object", "").lower()
        
        supporting = []
        for other in candidates:
            if other is claim:
                continue
            other_subj = other.get("subject", "").lower()
            other_obj = other.get("object", "").lower()
            
            # Same subject-object pair
            if subject and obj and subject in other_subj and obj in other_obj:
                supporting.append(other)
            # Semantic similarity check
            elif self._semantic_overlap(subject, other_subj) and self._semantic_overlap(obj, other_obj):
                supporting.append(other)
        
        return supporting
    
    def _find_contradictions(self, claim: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find claims that contradict the given claim."""
        contradicting = []
        subject = claim.get("subject", "").lower()
        predicate = claim.get("predicate", "").lower()
        obj = claim.get("object", "").lower()
        
        for other in candidates:
            if other is claim:
                continue
            
            # Explicit contradiction marking
            if other.get("contradicting"):
                other_subject = other.get("subject", "").lower()
                if subject and subject in other_subject:
                    contradicting.append(other)
                    continue
            
            # Negation contradiction
            other_raw = (other.get("raw") or "").lower()
            if "not" in other_raw or "no " in other_raw or "never" in other_raw:
                if subject in other_raw and obj in other_raw:
                    contradicting.append(other)
        
        return contradicting
    
    def _semantic_overlap(self, text1: str, text2: str, threshold: float = 0.5) -> bool:
        """Check semantic overlap between two text snippets."""
        if not text1 or not text2:
            return False
        
        # Simple word overlap for now
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1 & words2) / max(len(words1), len(words2))
        return overlap >= threshold
    
    def classify_claim(self, claim: Dict[str, Any]) -> str:
        """Classify claim by confidence level."""
        conf = claim.get("confidence", 0.0)
        
        if conf >= ConfidenceLevel.CERTAIN.value:
            return "certain"
        elif conf >= ConfidenceLevel.HIGH.value:
            return "known"
        elif conf >= ConfidenceLevel.PROBABLE.value:
            return "probable"
        elif conf >= ConfidenceLevel.MODERATE.value:
            return "moderate"
        elif conf >= ConfidenceLevel.LOW.value:
            return "uncertain"
        else:
            return "unknown"
    
    def reason(self, question: str, claims: List[Dict[str, Any]], concepts: List[str]) -> Dict[str, Any]:
        """Enhanced reasoning with symbolic rules and confidence propagation."""
        
        # Apply symbolic rules to enhance claims
        claims = self.apply_symbolic_rules(claims)
        
        # Propagate confidence through relationships
        claims = self.propagate_confidence(claims)
        
        # Classify claims by confidence
        known = []
        probable = []
        uncertain = []
        contradictions = []
        
        for c in claims:
            classification = self.classify_claim(c)
            
            if classification == "known" or classification == "certain":
                known.append(c)
            elif classification == "probable" or classification == "moderate":
                probable.append(c)
            elif classification == "uncertain" or classification == "unknown":
                uncertain.append(c)
            
            if c.get("contradicting") or c.get("contradicting_claims", 0) > 0:
                contradictions.append(c)
        
        # Sort by confidence
        known.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        probable.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        uncertain.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        
        # Extract conclusion from most confident claim
        conclusion = ""
        conclusion_confidence = 0.0
        
        if known:
            top = known[0]
            raw = (top.get("raw") or "").strip()
            if raw and len(raw.split()) >= 4:
                conclusion = raw
            else:
                conclusion = " ".join([p for p in [top.get("subject"), top.get("predicate"), top.get("object")] if p]).strip()
            conclusion_confidence = top.get("confidence", 0.0)
        elif probable:
            top = probable[0]
            raw = (top.get("raw") or "").strip()
            if raw and len(raw.split()) >= 4:
                conclusion = raw
            else:
                conclusion = " ".join([p for p in [top.get("subject"), top.get("predicate"), top.get("object")] if p]).strip()
            conclusion_confidence = top.get("confidence", 0.0)
        
        result = {
            "known": known,
            "probable": probable,
            "uncertain": uncertain,
            "contradictions": contradictions,
            "conclusion": conclusion,
            "conclusion_confidence": conclusion_confidence,
            "reasoning_depth": len(claims),
            "contradiction_detected": len(contradictions) > 0,
            "inference_chains": self._build_inference_chains(known + probable),
        }
        
        # Record reasoning history
        self.reasoning_history.append({
            "question": question,
            "result": result,
        })
        
        return result
    
    def _build_inference_chains(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build inference chains from claims with derivation rules."""
        chains = []
        for claim in claims:
            if "inference_chain" in claim:
                chains.append({
                    "conclusion": claim.get("conclusion", claim.get("raw", "")),
                    "confidence": claim.get("confidence", 0.0),
                    "steps": claim.get("inference_chain", []),
                    "derived_by_rule": claim.get("derived_by_rule"),
                })
        return chains


__all__ = ["ReasoningEngine", "SymbolicRule", "ConfidenceLevel", "InferenceChain"]
