"""Structured reasoning/triage.

This is intentionally lightweight: it ranks claims by confidence and surfaces
contradictions. It does not call an LLM.
"""

from __future__ import annotations

from typing import Any, Dict, List


class ReasoningEngine:
	def __init__(self, claim_store: Any):
		self.claim_store = claim_store

	def reason(self, question: str, claims: List[Dict[str, Any]], concepts: List[str]) -> Dict[str, Any]:
		known = []
		uncertain = []
		contradictions = []
		HIGH = 0.75
		LOW = 0.35
		for c in claims:
			if c.get("confidence", 0.0) >= HIGH:
				known.append(c)
			elif c.get("confidence", 0.0) <= LOW:
				uncertain.append(c)
		for c in claims:
			if c.get("contradicting"):
				contradictions.append(c)

		conclusion = None
		if known:
			top = max(known, key=lambda x: x.get("confidence", 0.0))
			raw = (top.get("raw") or "").strip()
			if raw and len(raw.split()) >= 4:
				conclusion = raw
			else:
				conclusion = " ".join([p for p in [top.get("subject"), top.get("predicate"), top.get("object")] if p]).strip()
		else:
			conclusion = ""

		return {
			"known": known,
			"uncertain": uncertain,
			"contradictions": contradictions,
			"conclusion": conclusion,
		}


__all__ = ["ReasoningEngine"]
