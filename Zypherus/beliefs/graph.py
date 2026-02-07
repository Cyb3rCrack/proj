"""BeliefGraph (claim dependency graph + controlled inference)."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger("ACE")


class BeliefGraph:
	"""Tracks dependencies between claims.

	v1 focuses on causal-chain support:
	- If A causes B and B causes C, infer A may_cause C (low-confidence derived claim)
	"""

	ALLOWED_RELATIONS = {"supports", "implies", "causes"}

	def __init__(self):
		self.supports: Dict[str, Dict[str, Dict[str, Any]]] = {}  # src -> dst -> meta
		self.dependents: Dict[str, Set[str]] = {}  # dst -> {src}

	def add_edge(
		self,
		src: str,
		dst: str,
		*,
		relation: str = "supports",
		weight: float = 1.0,
		meta: Optional[Dict[str, Any]] = None,
	):
		src = str(src or "").strip()
		dst = str(dst or "").strip()
		if not src or not dst or src == dst:
			return

		rel = str(relation or "supports").strip().lower()
		if rel not in self.ALLOWED_RELATIONS:
			rel = "supports"

		try:
			w = float(weight)
		except Exception:
			w = 1.0
		w = max(0.0, min(1.0, w))

		self.supports.setdefault(src, {})
		self.supports[src][dst] = {
			"relation": rel,
			"weight": w,
			"ts": time.time(),
			**(meta or {}),
		}
		self.dependents.setdefault(dst, set()).add(src)

	def upstream(self, node: str, *, max_depth: int = 2) -> Set[str]:
		node = str(node or "").strip()
		if not node:
			return set()
		max_depth = int(max(1, max_depth))

		seen: Set[str] = set()
		frontier: Set[str] = {node}
		for _ in range(max_depth):
			nxt: Set[str] = set()
			for n in frontier:
				for src in self.dependents.get(n, set()):
					if src not in seen:
						seen.add(src)
						nxt.add(src)
			frontier = nxt
			if not frontier:
				break
		return seen

	def infer_causal_chains(self, claims: Dict[str, Dict[str, Any]], *, max_new: int = 25) -> List[Dict[str, Any]]:
		"""Infer new low-confidence causal claims from existing cause edges."""
		if not isinstance(claims, dict) or not claims:
			return []

		edges: List[tuple] = []  # (a, b, cid)
		for cid, rec in claims.items():
			try:
				if (rec.get("predicate") or "").strip().lower() != "causes":
					continue
				if int(rec.get("polarity", 1) or 1) < 0:
					continue
				a = (rec.get("subject") or "").strip()
				b = (rec.get("object") or "").strip()
				if not a or not b:
					continue
				edges.append((a, b, str(cid)))
			except Exception:
				continue

		if not edges:
			return []

		by_src: Dict[str, List[tuple]] = {}
		by_mid: Dict[str, List[tuple]] = {}
		for a, b, cid in edges:
			by_src.setdefault(a, []).append((b, cid))
			by_mid.setdefault(b, []).append((a, cid))

		derived: List[Dict[str, Any]] = []
		seen_key: Set[str] = set()

		for mid, incoming in by_mid.items():
			outgoing = by_src.get(mid, [])
			if not outgoing:
				continue
			for a, cid1 in incoming:
				for c, cid2 in outgoing:
					if not a or not c or a == c:
						continue
					key = f"{a}||{c}"
					if key in seen_key:
						continue
					seen_key.add(key)

					derived.append(
						{
							"subject": a,
							"predicate": "causes",
							"predicate_token": "may cause",
							"predicate_tense": None,
							"object": c,
							"polarity": 1,
							"modality": "uncertain",
							"raw": f"{a} may cause {c}",
							"_derived_from": [cid1, cid2],
						}
					)
					if len(derived) >= int(max_new):
						return derived

		return derived

	def save(self, path: str = "belief_graph.json"):
		try:
			payload = {"supports": self.supports}
			with open(path, "w", encoding="utf-8") as f:
				json.dump(payload, f, ensure_ascii=False, indent=2)
		except Exception:
			return

	def load(self, path: str = "belief_graph.json"):
		try:
			with open(path, "r", encoding="utf-8") as f:
				payload = json.load(f) or {}
		except Exception:
			return

		supports = payload.get("supports", {})
		if not isinstance(supports, dict):
			return

		self.supports = {}
		self.dependents = {}
		for src, dsts in supports.items():
			if not isinstance(dsts, dict):
				continue
			for dst, meta in dsts.items():
				if not isinstance(meta, dict):
					meta = {}
				self.add_edge(str(src), str(dst), relation=meta.get("relation", "supports"), weight=meta.get("weight", 1.0), meta=meta)


__all__ = ["BeliefGraph"]
