"""Evidence memory index (RAG) with optional FAISS acceleration.

This module replaces the old `from aimaker import KnowledgeIndex` wrapper.
The index treats inner product as cosine similarity by contract: embeddings are
expected to be L2-normalized at creation/load time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ACE")

try:
	import faiss  # type: ignore

	_faiss_available = True
except Exception:
	faiss = None  # type: ignore
	_faiss_available = False


class KnowledgeIndex:
	"""Stores evidence entries and retrieves relevant ones by similarity.

	Public API matches legacy behavior used across the codebase:
	- `add(embedding, text, source, mtype="evidence") -> entry_id | None`
	- `add_entities(entry_id, entities)`
	- `save(path)` / `load(path)`
	- `search(query_embedding, top_k=10, min_score=0.0, query_text="") -> List[(score, entry)]`
	"""

	def __init__(
		self,
		*,
		use_faiss: bool = True,
		max_total_entries: int = 5000,
		max_entries_per_source: int = 800,
		weights: Optional[Dict[str, float]] = None,
	):
		self.entries: List[Dict[str, Any]] = []
		self.sources: set = set()  # (source, text_hash)
		self.entity_index: Dict[str, List[str]] = {}

		self.max_total_entries = int(max_total_entries)
		self.max_entries_per_source = int(max_entries_per_source)

		self.weights = {
			"sim": 0.75,
			"lex": 0.15,
			"recency": 0.10,
		}
		if isinstance(weights, dict):
			for k, v in weights.items():
				try:
					self.weights[str(k)] = float(v)
				except Exception:
					continue

		self.last_retrieval_diagnostics: List[Dict[str, Any]] = []

		self._use_faiss = bool(use_faiss and _faiss_available)
		self._faiss_index = None
		self._dim: Optional[int] = None

	def _root_source(self, source: str) -> str:
		s = (source or "").strip()
		return s.split("#", 1)[0] if s else ""

	def _rebuild_faiss(self):
		if not (self._use_faiss and _faiss_available):
			self._faiss_index = None
			self._dim = None
			return
		if not self.entries:
			self._faiss_index = None
			self._dim = None
			return
		try:
			self._dim = int(len(self.entries[0]["embedding"]))
			self._faiss_index = faiss.IndexFlatIP(self._dim)  # type: ignore[attr-defined]
			mats = np.vstack([e["embedding"] for e in self.entries]).astype(np.float32)
			self._faiss_index.add(mats)
		except Exception:
			logger.exception("Failed to rebuild FAISS index; disabling FAISS")
			self._use_faiss = False
			self._faiss_index = None
			self._dim = None

	def _evict_if_needed(self):
		if self.max_total_entries <= 0 and self.max_entries_per_source <= 0:
			return
		if not self.entries:
			return

		removed_ids = set()
		removed_dedup = set()

		if self.max_entries_per_source > 0:
			by_root: Dict[str, List[Dict[str, Any]]] = {}
			for e in self.entries:
				by_root.setdefault(self._root_source(e.get("source", "")), []).append(e)
			for _root, lst in by_root.items():
				if len(lst) <= self.max_entries_per_source:
					continue
				lst_sorted = sorted(lst, key=lambda x: float(x.get("ts", 0.0) or 0.0))
				evict_count = len(lst_sorted) - self.max_entries_per_source
				evict = [e for e in lst_sorted if e.get("type") == "evidence"][:evict_count]
				if len(evict) < evict_count:
					extra = [e for e in lst_sorted if e not in evict][: (evict_count - len(evict))]
					evict.extend(extra)
				for e in evict:
					removed_ids.add(e.get("id"))
					if e.get("_dedup_key"):
						removed_dedup.add(e.get("_dedup_key"))

		if self.max_total_entries > 0 and len(self.entries) - len(removed_ids) > self.max_total_entries:
			remaining = [e for e in self.entries if e.get("id") not in removed_ids]
			remaining_sorted = sorted(remaining, key=lambda x: float(x.get("ts", 0.0) or 0.0))
			evict_count = len(remaining_sorted) - self.max_total_entries
			evict = [e for e in remaining_sorted if e.get("type") == "evidence"][:evict_count]
			if len(evict) < evict_count:
				extra = [e for e in remaining_sorted if e not in evict][: (evict_count - len(evict))]
				evict.extend(extra)
			for e in evict:
				removed_ids.add(e.get("id"))
				if e.get("_dedup_key"):
					removed_dedup.add(e.get("_dedup_key"))

		if not removed_ids:
			return

		self.entries = [e for e in self.entries if e.get("id") not in removed_ids]

		try:
			for k in removed_dedup:
				if k in self.sources:
					self.sources.remove(k)
		except Exception:
			pass

		try:
			for ent, ids in list((self.entity_index or {}).items()):
				if not ids:
					continue
				new_ids = [i for i in ids if i not in removed_ids]
				if new_ids:
					self.entity_index[ent] = new_ids
				else:
					self.entity_index.pop(ent, None)
		except Exception:
			pass

		try:
			if self._use_faiss:
				self._rebuild_faiss()
		except Exception:
			pass

	def add(self, embedding, text, source, mtype: str = "evidence"):
		try:
			text_hash = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
			key = (source, text_hash)
		except Exception:
			key = (source, str(hash(text)))
			text_hash = hashlib.sha256(str(key).encode("utf-8")).hexdigest()

		if key in self.sources:
			return None
		self.sources.add(key)

		entry_id = f"{source}#{text_hash}"
		vec = np.array(embedding, dtype=np.float32)

		entry = {
			"id": entry_id,
			"embedding": vec,
			"text": text,
			"source": source,
			"type": mtype,
			"ts": time.time(),
			"_dedup_key": key,
		}
		self.entries.append(entry)

		if self._use_faiss:
			try:
				if self._dim is None:
					self._dim = int(len(vec))
					self._faiss_index = faiss.IndexFlatIP(self._dim)  # type: ignore[attr-defined]
				self._faiss_index.add(np.array([vec], dtype=np.float32))
			except Exception:
				self._use_faiss = False

		try:
			self._evict_if_needed()
		except Exception:
			pass

		return entry_id

	def add_entities(self, entry_id: str, entities: List[str]):
		for ent in entities or []:
			ent = (ent or "").strip()
			if not ent:
				continue
			norm = re.sub(r"[^a-z0-9 ]+", "", ent.lower()).strip()
			if not norm:
				continue
			self.entity_index.setdefault(norm, [])
			if entry_id not in self.entity_index[norm]:
				self.entity_index[norm].append(entry_id)

	def save(self, path: str = "memory.json"):
		out = []
		for e in self.entries:
			out.append(
				{
					"id": e.get("id"),
					"embedding": e["embedding"].tolist(),
					"text": e.get("text", ""),
					"source": e.get("source", "unknown"),
					"type": e.get("type", "evidence"),
					"ts": e.get("ts", time.time()),
				}
			)
		payload = {"entries": out, "entity_index": self.entity_index}
		with open(path, "w", encoding="utf-8") as f:
			json.dump(payload, f, ensure_ascii=False, indent=2)

		try:
			if self._use_faiss and self._faiss_index is not None:
				faiss_path = os.path.splitext(path)[0] + ".faiss"
				faiss.write_index(self._faiss_index, faiss_path)  # type: ignore[attr-defined]
		except Exception:
			logger.exception("Failed to write FAISS index to disk")

	def load(self, path: str = "memory.json"):
		try:
			with open(path, "r", encoding="utf-8") as f:
				payload = json.load(f) or {}
				items = payload.get("entries", []) or []
				saved_entity_index = payload.get("entity_index", {}) or {}
		except Exception:
			return

		self.entries = []
		self.sources = set()

		for it in items:
			try:
				emb = np.array(it.get("embedding") or [], dtype=np.float32)
				try:
					n = float(np.linalg.norm(emb))
					if n > 0 and not np.isnan(n):
						emb = emb / n
				except Exception:
					pass

				eid = it.get("id")
				if not eid:
					text_hash = hashlib.sha256((it.get("text", "") or "").encode("utf-8")).hexdigest()
					eid = f"{it.get('source', 'unknown')}#{text_hash}"

				text_hash = hashlib.sha256((it.get("text", "") or "").encode("utf-8")).hexdigest()
				entry = {
					"id": eid,
					"embedding": emb,
					"text": it.get("text", ""),
					"source": it.get("source", "unknown"),
					"type": it.get("type", "evidence"),
					"ts": it.get("ts", time.time()),
					"_dedup_key": (it.get("source", "unknown"), text_hash),
				}
				self.entries.append(entry)
				self.sources.add((entry["source"], text_hash))
			except Exception:
				continue

		try:
			if self._use_faiss and len(self.entries) > 0:
				self._rebuild_faiss()
		except Exception:
			pass

		try:
			converted: Dict[str, List[str]] = {}
			for ent, lst in (saved_entity_index or {}).items():
				if not lst:
					continue
				if isinstance(lst, list) and lst and isinstance(lst[0], int):
					ids = []
					for idx in lst:
						if 0 <= idx < len(self.entries):
							ids.append(self.entries[idx]["id"])
					if ids:
						converted[str(ent)] = ids
				else:
					converted[str(ent)] = [str(i) for i in (lst or [])]
			self.entity_index = converted
		except Exception:
			self.entity_index = {}

	def search(
		self,
		query_embedding,
		top_k: int = 10,
		min_score: float = 0.0,
		query_text: str = "",
	) -> List[Tuple[float, Dict[str, Any]]]:
		if query_embedding is None:
			return []

		scores: List[Tuple[float, Dict[str, Any]]] = []
		self.last_retrieval_diagnostics = []

		try:
			qvec = np.array(query_embedding, dtype=np.float32)

			if self._use_faiss and self._faiss_index is not None and len(self.entries) > 0:
				try:
					D, I = self._faiss_index.search(
						np.array([qvec], dtype=np.float32),
						min(top_k, len(self.entries)),
					)
					for score_val, idx in zip(D[0].tolist(), I[0].tolist()):
						if idx < 0 or idx >= len(self.entries):
							continue
						e = self.entries[idx]
						sim = float(score_val)
						sim = max(-1.0, min(1.0, sim))

						lex = 0.0
						if query_text:
							q_tokens = set(re.findall(r"\w+", query_text.lower()))
							t_tokens = set(re.findall(r"\w+", (e.get("text", "") or "").lower()))
							if q_tokens:
								lex = len(q_tokens & t_tokens) / float(len(q_tokens))

						age = max(0.0, time.time() - float(e.get("ts", 0) or 0.0))
						recency = max(0.0, 1.0 - (age / (3600.0 * 24.0 * 30.0)))
						tw = {"summary": 1.15, "fact": 1.2, "evidence": 1.0}.get(e.get("type", "evidence"), 1.0)
						base = (self.weights["sim"] * sim) + (self.weights["lex"] * lex) + (self.weights["recency"] * recency)
						score = float(base) * float(tw)
						score = max(0.0, float(score))
						if score >= min_score:
							scores.append((score, e))
							self.last_retrieval_diagnostics.append(
								{
									"id": e.get("id"),
									"sim": sim,
									"lex": lex,
									"recency": recency,
									"type_weight": tw,
									"final_score": score,
								}
							)

					scores.sort(key=lambda x: x[0], reverse=True)
					return scores[:top_k]
				except Exception:
					logger.exception("FAISS search failed, falling back to linear search")
					self._use_faiss = False

			for e in self.entries:
				try:
					emb = e.get("embedding")
					if emb is None:
						continue

					sim = float(np.dot(qvec, emb))
					sim = max(-1.0, min(1.0, sim))

					lex = 0.0
					if query_text:
						q_tokens = set(re.findall(r"\w+", query_text.lower()))
						t_tokens = set(re.findall(r"\w+", (e.get("text", "") or "").lower()))
						if q_tokens:
							lex = len(q_tokens & t_tokens) / float(len(q_tokens))

					age = max(0.0, time.time() - float(e.get("ts", 0) or 0.0))
					recency = max(0.0, 1.0 - (age / (3600.0 * 24.0 * 30.0)))
					tw = {"summary": 1.15, "fact": 1.2, "evidence": 1.0}.get(e.get("type", "evidence"), 1.0)
					base = (self.weights["sim"] * sim) + (self.weights["lex"] * lex) + (self.weights["recency"] * recency)
					score = float(base) * float(tw)
					score = max(0.0, float(score))
					if score >= min_score:
						scores.append((score, e))
						self.last_retrieval_diagnostics.append(
							{
								"id": e.get("id"),
								"sim": sim,
								"lex": lex,
								"recency": recency,
								"type_weight": tw,
								"final_score": score,
							}
						)
				except Exception:
					continue

			scores.sort(key=lambda x: x[0], reverse=True)
			self.last_retrieval_diagnostics.sort(key=lambda d: d["final_score"], reverse=True)
			return scores[:top_k]
		except Exception:
			logger.exception("KnowledgeIndex.search failed")
			return []


__all__ = ["KnowledgeIndex"]
