"""Phase 2: Knowledge Graph & Bidirectional Memory Links

Enables semantic relationships between concepts:
- Contains relationships: "Water contains Hydrogen"
- Causal relationships: "Gravity causes objects to fall"
- Implication relationships: "Programmer implies writes code"
- Type relationships: "Python is a programming language"

Enables queries like:
- "What contains Hydrogen?" (reverse lookup)
- "What causes disease?" (multi-hop reasoning)
- "What are types of programming languages?" (category queries)
"""

import json
import logging
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger("ZYPHERUS")


class RelationshipStore:
	"""Stores bidirectional semantic relationships between concepts."""
	
	# Relationship types
	RELATIONSHIP_TYPES = {
		"contains": "X contains Y (part-of relationship)",
		"part_of": "X is part of Y (inverse of contains)",
		"causes": "X causes Y (causal relationship)",
		"caused_by": "X is caused by Y (inverse of causes)",
		"is_type_of": "X is a type of Y (taxonomy)",
		"has_type": "X has type Y (inverse of is_type_of)",
		"implies": "X implies Y (logical implication)",
		"implied_by": "X is implied by Y (inverse of implies)",
		"similar_to": "X is similar to Y (similarity)",
		"opposite_of": "X is opposite of Y (antonym)",
		"related_to": "X is related to Y (general relationship)",
	}
	
	def __init__(self):
		# Forward relationships: concept1 -> [(rel_type, concept2, strength)]
		self.forward: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
		# Reverse relationships: concept2 -> [(rel_type, concept1, strength)]
		self.reverse: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
		# Relationship metadata
		self.metadata: Dict[str, Dict] = {}
	
	def add_relationship(
		self, 
		concept1: str, 
		rel_type: str, 
		concept2: str, 
		strength: float = 0.95,
		source: str = "inferred",
	):
		"""Add a bidirectional relationship between concepts.
		
		Args:
			concept1: Source concept
			rel_type: Type of relationship (from RELATIONSHIP_TYPES)
			concept2: Target concept
			strength: Confidence in relationship (0.0-1.0)
			source: Where this relationship came from
		"""
		if rel_type not in self.RELATIONSHIP_TYPES:
			logger.warning(f"Unknown relationship type: {rel_type}")
			return False
		
		concept1 = str(concept1).strip().lower()
		concept2 = str(concept2).strip().lower()
		strength = float(max(0.0, min(1.0, strength)))
		
		if not concept1 or not concept2 or concept1 == concept2:
			return False
		
		# Determine inverse relationship type
		inverse_types = {
			"contains": "part_of",
			"part_of": "contains",
			"causes": "caused_by",
			"caused_by": "causes",
			"is_type_of": "has_type",
			"has_type": "is_type_of",
			"implies": "implied_by",
			"implied_by": "implies",
			"similar_to": "similar_to",  # Symmetric
			"opposite_of": "opposite_of",  # Symmetric
			"related_to": "related_to",  # Symmetric
		}
		
		inverse_rel_type = inverse_types.get(rel_type, "related_to")
		
		# Add forward relationship
		rel_key = f"{concept1}#{rel_type}#{concept2}"
		self.forward[concept1].append((rel_type, concept2, strength))
		
		# Add reverse relationship
		if inverse_rel_type != rel_type:  # Avoid duplicates for symmetric relations
			self.reverse[concept2].append((inverse_rel_type, concept1, strength))
		
		# Store metadata
		self.metadata[rel_key] = {
			"concept1": concept1,
			"rel_type": rel_type,
			"concept2": concept2,
			"strength": strength,
			"source": source,
		}
		
		return True
	
	def get_relationships(self, concept: str, rel_type: Optional[str] = None) -> List[Tuple[str, str, float]]:
		"""Get outgoing relationships from concept.
		
		Args:
			concept: Concept to query
			rel_type: Optional filter by relationship type
		
		Returns:
			List of (rel_type, target_concept, strength)
		"""
		concept = str(concept).strip().lower()
		rels = self.forward.get(concept, [])
		
		if rel_type:
			rels = [(r, c, s) for r, c, s in rels if r == rel_type]
		
		# Sort by strength descending
		return sorted(rels, key=lambda x: x[2], reverse=True)
	
	def get_reverse_relationships(self, concept: str, rel_type: Optional[str] = None) -> List[Tuple[str, str, float]]:
		"""Get incoming relationships to concept.
		
		Args:
			concept: Concept to query
			rel_type: Optional filter by relationship type
		
		Returns:
			List of (rel_type, source_concept, strength)
		"""
		concept = str(concept).strip().lower()
		rels = self.reverse.get(concept, [])
		
		if rel_type:
			rels = [(r, c, s) for r, c, s in rels if r == rel_type]
		
		return sorted(rels, key=lambda x: x[2], reverse=True)
	
	def find_path(self, start: str, end: str, max_depth: int = 3) -> Optional[List[str]]:
		"""Find shortest path between two concepts.
		
		Args:
			start: Starting concept
			end: Target concept
			max_depth: Maximum path length
		
		Returns:
			Path as list of concepts, or None if no path exists
		"""
		start = str(start).strip().lower()
		end = str(end).strip().lower()
		
		if start == end:
			return [start]
		
		visited = set()
		queue = [(start, [start])]
		
		while queue:
			current, path = queue.pop(0)
			
			if len(path) > max_depth:
				continue
			
			if current in visited:
				continue
			visited.add(current)
			
			# Check all relationships from current
			for rel_type, target, strength in self.forward.get(current, []):
				if target == end:
					return path + [target]
				if target not in visited:
					queue.append((target, path + [target]))
			
			# Also check reverse relationships
			for rel_type, source, strength in self.reverse.get(current, []):
				if source == end:
					return path + [source]
				if source not in visited:
					queue.append((source, path + [source]))
		
		return None
	
	def get_all_related(self, concept: str, depth: int = 1) -> Set[str]:
		"""Get all concepts related to given concept within depth.
		
		Args:
			concept: Starting concept
			depth: Depth of search
		
		Returns:
			Set of related concepts
		"""
		concept = str(concept).strip().lower()
		related = set()
		visited = set()
		queue = [(concept, 0)]
		
		while queue:
			current, d = queue.pop(0)
			
			if current in visited or d > depth:
				continue
			visited.add(current)
			
			# Forward relationships
			for rel_type, target, strength in self.forward.get(current, []):
				if target not in visited:
					related.add(target)
					queue.append((target, d + 1))
			
			# Reverse relationships
			for rel_type, source, strength in self.reverse.get(current, []):
				if source not in visited:
					related.add(source)
					queue.append((source, d + 1))
		
		related.discard(concept)  # Don't include the concept itself
		return related
	
	def save(self, path: str = "data/knowledge/relationships.json"):
		"""Save relationships to file."""
		data = {
			"forward": {k: list(v) for k, v in self.forward.items()},
			"reverse": {k: list(v) for k, v in self.reverse.items()},
			"metadata": self.metadata,
		}
		with open(path, "w", encoding="utf-8") as f:
			json.dump(data, f, ensure_ascii=False, indent=2)
	
	def load(self, path: str = "data/knowledge/relationships.json"):
		"""Load relationships from file."""
		try:
			with open(path, "r", encoding="utf-8") as f:
				data = json.load(f)
			
			self.forward = defaultdict(list)
			self.reverse = defaultdict(list)
			self.metadata = {}
			
			for k, v in data.get("forward", {}).items():
				self.forward[k] = [tuple(item) for item in v]
			
			for k, v in data.get("reverse", {}).items():
				self.reverse[k] = [tuple(item) for item in v]
			
			self.metadata = data.get("metadata", {})
		except Exception as e:
			logger.exception(f"Failed to load relationships: {e}")


__all__ = ["RelationshipStore"]
