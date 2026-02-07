"""Relationship Extraction for Phase 2 Knowledge Graph

Automatically extracts relationships from ingested text:
- Part-whole: "Water contains hydrogen"
- Type hierarchy: "Python is a programming language"
- Causality: "Gravity causes objects to fall"
- Implication: "If X then Y"
"""

import re
from typing import List, Tuple
import logging

logger = logging.getLogger("ZYPHERUS")


class RelationshipExtractor:
	"""Extract semantic relationships from text."""
	
	# Patterns for different relationship types
	CONTAINS_PATTERNS = [
		r"(\w+)\s+(?:contains?|is made of|consists of|includes?|composed of)\s+(\w+)",
		r"(\w+)\s+(?:has|possess)\s+(?:a |an )?(\w+)",
	]
	
	CAUSE_PATTERNS = [
		r"(\w+)\s+(?:causes?|leads to|results in|produces?)\s+(\w+)",
		r"(\w+)\s+(?:is caused by|results from)\s+(\w+)",
	]
	
	TYPE_PATTERNS = [
		r"(\w+)\s+(?:is|are)\s+(?:a |an )?(?:type of|kind of|form of|example of)\s+(\w+)",
		r"(\w+)\s+(?:is|are)\s+(?:a |an )(\w+)",  # Simple "X is a Y"
	]
	
	IMPLIES_PATTERNS = [
		r"(?:if|when)\s+(\w+)(?:\s+then)?\s+(\w+)",
		r"(\w+)\s+(?:implies?|means?|suggests?)\s+(\w+)",
	]
	
	RELATED_PATTERNS = [
		r"(\w+)\s+and\s+(\w+)\s+(?:are related|interact|work together)",
		r"(\w+)\s+(?:is related to|is associated with|connects to)\s+(\w+)",
	]
	
	@staticmethod
	def extract_concepts_and_relationships(text: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
		"""Extract concepts and relationships from text.
		
		Returns:
			(concepts_list, relationships_list)
			where relationships_list = [(concept1, rel_type, concept2), ...]
		"""
		concepts = set()
		relationships = []
		
		text_lower = text.lower()
		
		# Extract contains relationships
		for pattern in RelationshipExtractor.CONTAINS_PATTERNS:
			matches = re.finditer(pattern, text_lower, re.IGNORECASE)
			for match in matches:
				c1, c2 = match.groups()
				concepts.add(c1)
				concepts.add(c2)
				relationships.append((c1, "contains", c2))
		
		# Extract cause relationships
		for pattern in RelationshipExtractor.CAUSE_PATTERNS:
			matches = re.finditer(pattern, text_lower, re.IGNORECASE)
			for match in matches:
				groups = match.groups()
				if len(groups) == 2:
					c1, c2 = groups
					concepts.add(c1)
					concepts.add(c2)
					
					# Check if it's a "caused by" pattern
					if "caused by" in match.group(0).lower() or "results from" in match.group(0).lower():
						relationships.append((c1, "caused_by", c2))
					else:
						relationships.append((c1, "causes", c2))
		
		# Extract type relationships
		for pattern in RelationshipExtractor.TYPE_PATTERNS:
			matches = re.finditer(pattern, text_lower, re.IGNORECASE)
			for match in matches:
				groups = match.groups()
				if len(groups) == 2:
					c1, c2 = groups
					concepts.add(c1)
					concepts.add(c2)
					relationships.append((c1, "is_type_of", c2))
		
		# Extract implication relationships
		for pattern in RelationshipExtractor.IMPLIES_PATTERNS:
			matches = re.finditer(pattern, text_lower, re.IGNORECASE)
			for match in matches:
				groups = match.groups()
				if len(groups) == 2:
					c1, c2 = groups
					concepts.add(c1)
					concepts.add(c2)
					relationships.append((c1, "implies", c2))
		
		# Extract general relationships
		for pattern in RelationshipExtractor.RELATED_PATTERNS:
			matches = re.finditer(pattern, text_lower, re.IGNORECASE)
			for match in matches:
				c1, c2 = match.groups()
				concepts.add(c1)
				concepts.add(c2)
				relationships.append((c1, "related_to", c2))
		
		return list(concepts), relationships


__all__ = ["RelationshipExtractor"]
