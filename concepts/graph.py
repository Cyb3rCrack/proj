"""Lightweight concept graph.

Tracks concept nodes and co-occurrence edges.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List


class ConceptGraph:
	def __init__(self):
		self.nodes: Dict[str, Dict[str, Any]] = {}
		self.edges: Dict[tuple, Dict[str, Any]] = {}

	def upsert_node(self, concept: str):
		if concept not in self.nodes:
			self.nodes[concept] = {"created": time.time(), "meta": {}}

	def upsert_edge(self, subject: str, relation: str, object_: str):
		key = (subject, relation, object_)
		self.edges[key] = {"updated": time.time()}

	def observe_cooccurrence(self, concepts: List[str], relation: str = "co_occurs"):
		concepts = [c for c in (concepts or []) if isinstance(c, str) and c.strip()]
		if len(concepts) > 25:
			concepts = concepts[:25]
		for i in range(len(concepts)):
			for j in range(i + 1, len(concepts)):
				self.upsert_edge(concepts[i], relation, concepts[j])

	def get_related_concepts(self, concept: str) -> List[str]:
		res = set()
		for (s, _r, o) in self.edges.keys():
			if s == concept:
				res.add(o)
			if o == concept:
				res.add(s)
		return list(res)

	def propagate_confidence(self):
		return


__all__ = ["ConceptGraph"]
