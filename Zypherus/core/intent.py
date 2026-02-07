"""Phase 3: Retrieval Intent Classification

Classify question intent to enable smart routing:
- Definition/comparison queries need strong definitions
- How-to queries need process knowledge
- Factual queries need facts and evidence
- Exploratory queries can tolerate uncertainty

Enables:
- Definition-first retrieval for "What is X?"
- Process-first retrieval for "How do I X?"
- Evidence-first for "Is X true?"
"""

import re
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("ACE")


class IntentClassifier:
	"""Classify question intent for routing and confidence gating."""
	
	# Intent patterns
	DEFINITION_PATTERNS = [
		r"what\s+(?:is|are|does)\s+",
		r"define\s+",
		r"what\s+do\s+you\s+mean\s+by\s+",
		r"meaning\s+of\s+",
		r"explain\s+(?:what|how)\s+",
		r"describe\s+",
	]
	
	COMPARISON_PATTERNS = [
		r"(?:what|what\'s)\s+(?:the\s+)?(?:difference|differences|similarities?|similarities)\s+",
		r"compare\s+",
		r"vs\.?|versus|vs\b",
		r"(?:similar|different|same|between)\s+",
		r"difference\s+between\s+",
	]
	
	PROCESS_PATTERNS = [
		r"how\s+(?:to|do|does|can|should|would)\s+",
		r"how\s+(?:do|does)\s+",
		r"steps\s+to\s+",
		r"process\s+(?:of|for)\s+",
		r"way\s+to\s+",
	]
	
	CAUSAL_PATTERNS = [
		r"why\s+",
		r"what\s+causes?",
		r"caused\s+by",
		r"because\s+",
		r"reason\s+(?:for|why)",
	]
	
	EXISTENCE_PATTERNS = [
		r"(?:is|are|has|have|does)\s+",
		r"exist(?:s)?",
		r"true\s+(?:or|that)",
		r"(?:is|are)\s+.*\s+(?:real|true|possible|correct)",
	]
	
	EXPLORATORY_PATTERNS = [
		r"tell\s+me\s+(?:about|of)",
		r"what\s+(?:about|of)\s+",
		r"info(?:rmation)?\s+(?:about|on)",
		r"background",
		r"overview",
		r"introduction\s+to",
	]
	
	def __init__(self):
		"""Initialize classifier with compiled patterns."""
		self.patterns = {
			"definition": [re.compile(p, re.IGNORECASE) for p in self.DEFINITION_PATTERNS],
			"comparison": [re.compile(p, re.IGNORECASE) for p in self.COMPARISON_PATTERNS],
			"process": [re.compile(p, re.IGNORECASE) for p in self.PROCESS_PATTERNS],
			"causal": [re.compile(p, re.IGNORECASE) for p in self.CAUSAL_PATTERNS],
			"existence": [re.compile(p, re.IGNORECASE) for p in self.EXISTENCE_PATTERNS],
			"exploratory": [re.compile(p, re.IGNORECASE) for p in self.EXPLORATORY_PATTERNS],
		}
	
	def classify(self, question: str) -> Dict[str, any]:
		"""Classify question intent.
		
		Returns:
			Dict with:
			- primary_intent: Main intent type
			- secondary_intent: Secondary intent if present
			- confidence: 0.0-1.0 confidence in classification
			- requires_definition: Whether strong definition needed
			- requires_process: Whether process knowledge needed
			- requires_evidence: Whether factual evidence needed
			- can_tolerate_uncertainty: Whether exploratory answers ok
			- min_confidence: Minimum confidence threshold for answer
		"""
		question_lower = question.lower().strip()
		scores = {}
		
		# Score each intent type
		for intent_type, patterns in self.patterns.items():
			matches = sum(1 for p in patterns if p.search(question_lower))
			scores[intent_type] = matches
		
		if not any(scores.values()):
			# No strong match - default to exploratory
			scores["exploratory"] = 1
		
		# Get sorted intents by score
		sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
		primary_intent = sorted_intents[0][0]
		primary_score = sorted_intents[0][1]
		secondary_intent = sorted_intents[1][0] if len(sorted_intents) > 1 and sorted_intents[1][1] > 0 else None
		
		# Normalize confidence (higher score = higher confidence, max 5 patterns)
		confidence = min(1.0, primary_score / 2.0)  # Max out at 2 matches = 1.0 confidence
		
		# Set routing requirements based on intent
		routing = self._get_routing_requirements(primary_intent, question_lower)
		
		return {
			"primary_intent": primary_intent,
			"secondary_intent": secondary_intent,
			"confidence": confidence,
			"requires_definition": routing["requires_definition"],
			"requires_process": routing["requires_process"],
			"requires_evidence": routing["requires_evidence"],
			"can_tolerate_uncertainty": routing["can_tolerate_uncertainty"],
			"min_confidence": routing["min_confidence"],
		}
	
	def _get_routing_requirements(self, intent: str, question_lower: str) -> Dict[str, any]:
		"""Get retrieval and confidence requirements based on intent."""
		requirements = {
			"definition": {
				"requires_definition": True,
				"requires_process": False,
				"requires_evidence": False,
				"can_tolerate_uncertainty": False,
				"min_confidence": 0.70,
			},
			"comparison": {
				"requires_definition": True,
				"requires_process": False,
				"requires_evidence": True,
				"can_tolerate_uncertainty": False,
				"min_confidence": 0.65,
			},
			"process": {
				"requires_definition": False,
				"requires_process": True,
				"requires_evidence": True,
				"can_tolerate_uncertainty": False,
				"min_confidence": 0.60,
			},
			"causal": {
				"requires_definition": False,
				"requires_process": False,
				"requires_evidence": True,
				"can_tolerate_uncertainty": False,
				"min_confidence": 0.55,
			},
			"existence": {
				"requires_definition": False,
				"requires_process": False,
				"requires_evidence": True,
				"can_tolerate_uncertainty": False,
				"min_confidence": 0.60,
			},
			"exploratory": {
				"requires_definition": False,
				"requires_process": False,
				"requires_evidence": False,
				"can_tolerate_uncertainty": True,
				"min_confidence": 0.30,
			},
		}
		
		return requirements.get(intent, requirements["exploratory"])


class KnownUnknownsRegistry:
	"""Track what ACE knows vs doesn't know for honest abstention."""
	
	def __init__(self):
		"""Initialize registry."""
		self.known_concepts: Dict[str, float] = {}  # concept -> max confidence
		self.unknown_concepts: set = set()  # Concepts we've explicitly failed on
		self.failed_queries: List[str] = []  # Failed question patterns
		self.knowledge_gaps: List[Dict] = []  # Identified gaps
	
	def record_success(self, concept: str, confidence: float):
		"""Record that ACE knows this concept."""
		self.known_concepts[concept] = max(self.known_concepts.get(concept, 0.0), confidence)
		if concept in self.unknown_concepts:
			self.unknown_concepts.remove(concept)
	
	def record_failure(self, question: str, concept: str = None):
		"""Record that ACE failed on a question."""
		self.failed_queries.append(question)
		if concept:
			self.unknown_concepts.add(concept)
	
	def record_gap(self, gap_description: str, related_concept: str = None):
		"""Record an identified knowledge gap."""
		self.knowledge_gaps.append({
			"description": gap_description,
			"related_concept": related_concept,
			"timestamp": None,
		})
	
	def is_known(self, concept: str, min_confidence: float = 0.5) -> bool:
		"""Check if ACE knows this concept."""
		if concept in self.unknown_concepts:
			return False
		return self.known_concepts.get(concept, 0.0) >= min_confidence
	
	def get_confidence(self, concept: str) -> Optional[float]:
		"""Get ACE's confidence in knowing this concept."""
		if concept in self.unknown_concepts:
			return 0.0
		return self.known_concepts.get(concept, None)
	
	def get_unknown_concepts(self) -> List[str]:
		"""Get list of unknown concepts."""
		return list(self.unknown_concepts)
	
	def get_failures(self, limit: int = 10) -> List[str]:
		"""Get recent failures."""
		return self.failed_queries[-limit:]
	
	def get_gaps(self, limit: int = 10) -> List[Dict]:
		"""Get identified knowledge gaps."""
		return self.knowledge_gaps[-limit:]
	
	def get_summary(self) -> Dict[str, any]:
		"""Get registry summary."""
		return {
			"known_count": len(self.known_concepts),
			"unknown_count": len(self.unknown_concepts),
			"failed_queries_count": len(self.failed_queries),
			"gaps_count": len(self.knowledge_gaps),
			"known_concepts": list(self.known_concepts.keys()),
			"unknown_concepts": list(self.unknown_concepts),
		}
