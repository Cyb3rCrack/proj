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

logger = logging.getLogger("ZYPHERUS")

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
		self.entity_index: Dict[str, set] = {}  # MEMORY LEAK FIX: Use sets instead of lists for O(1) lookup

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

	def add(self, embedding, text, source, mtype: str = "evidence", *, generated_by: str = "user", certainty: float = 0.95, is_assumption: bool = False):
		# CRITICAL: Hard block on 'user_input' as source (prevents 80% of memory bugs)
		if source == "user_input" or (source and source.startswith("user_input#")):
			logger.warning(f"[MEMORY] BLOCKED: Attempted to store entry with source='{source}'. This is forbidden to prevent memory contamination.")
			return None
		
		# VALIDATION: Block only UNMISTAKABLE assistant chatter (avoid false positives)
		# These patterns ONLY block if they occur at the start and are clearly conversational
		import re
		text_lower = (text or "").lower().strip()
		
		# STRICT: Only block the most obvious assistant chatter patterns
		# (educate headers like "What is X?" are legitimate knowledge)
		unmistakable_chatter = [
			r'^i\'?ll be happy to help',
			r'^i\'?m happy to help',
			r'^i\'?m here to help',
			r'^please provide the',
			r'^i cannot answer',
			r'^i don\'?t have access',
			r'^as an ai',
			r'^as a language model',
		]
		
		# Check if text starts with unmistakable chatter
		for pattern in unmistakable_chatter:
			if re.match(pattern, text_lower, re.IGNORECASE):
				logger.warning(f"[MEMORY] BLOCKED: Content appears to be assistant chatter or user question, not knowledge.")
				return None
		
		# VALIDATION: Minimum quality thresholds
		if len(text.split()) < 5:
			logger.warning(f"[MEMORY] BLOCKED: Content too short (< 5 words).")
			return None
		
		# DEDUPLICATION: Check if this is already known (avoid storing restatements)
		# Compare embedding similarity to existing entries
		embedding_vec = np.array(embedding, dtype=np.float32)
		
		if len(self.entries) > 0:
			# Calculate similarity to all existing entries
			existing_vecs = np.vstack([e["embedding"] for e in self.entries]).astype(np.float32)
			similarities = np.dot(existing_vecs, embedding_vec)  # Cosine similarity (embeddings are normalized)
			max_similarity = np.max(similarities)
			
			# If nearly identical to existing content (> 0.95), skip it
			# This allows different explanations/perspectives while avoiding exact duplicates
			if max_similarity > 0.95:
				logger.debug(f"[MEMORY] SKIPPED: Content nearly identical to existing knowledge (similarity: {max_similarity:.3f}).")
				return None
		
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

		# Clamp certainty to [0.0, 1.0]
		certainty = float(max(0.0, min(1.0, certainty)))
		
		# Validate generated_by
		generated_by = str(generated_by) if generated_by in ("user", "ace", "system") else "user"

		entry = {
			"id": entry_id,
			"embedding": vec,
			"text": text,
			"source": source,
			"type": mtype,
			"ts": time.time(),
			"_dedup_key": key,
			"generated_by": generated_by,  # PHASE 1: Self-output quarantine
			"certainty": certainty,  # PHASE 1: Speculation tagging
			"is_assumption": bool(is_assumption),  # PHASE 1: Assumption memory
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

		return entry_id

	def remove_entries(self, entry_ids: List[str]) -> int:
		"""Remove entries by id and rebuild indexes.

		Returns number of entries removed.
		"""
		if not entry_ids:
			return 0

		ids = set(entry_ids)
		removed = [e for e in self.entries if e.get("id") in ids]
		if not removed:
			return 0

		self.entries = [e for e in self.entries if e.get("id") not in ids]

		try:
			for e in removed:
				key = e.get("_dedup_key")
				if key in self.sources:
					self.sources.remove(key)
		except Exception:
			pass

		try:
			for ent, id_set in list((self.entity_index or {}).items()):
				if not id_set:
					continue
				new_ids = [i for i in id_set if i not in ids]
				if new_ids:
					self.entity_index[ent] = new_ids
				else:
					self.entity_index.pop(ent, None)
		except Exception:
			pass

		try:
			self._rebuild_faiss()
		except Exception:
			pass

		return len(removed)

	def remove_by_source_prefix(self, prefix: str) -> int:
		"""Remove entries whose source starts with prefix.

		Returns number of entries removed.
		"""
		prefix = (prefix or "").strip()
		if not prefix:
			return 0
		entry_ids = [e.get("id") for e in self.entries if str(e.get("source", "")).startswith(prefix)]
		entry_ids = [i for i in entry_ids if i]
		return self.remove_entries(entry_ids)

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
			# MEMORY LEAK FIX: Use sets for O(1) membership checking
			if norm not in self.entity_index:
				self.entity_index[norm] = set()
			self.entity_index[norm].add(entry_id)

	def save(self, path: str = "data/memory/memory.json"):
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
					"generated_by": e.get("generated_by", "user"),  # PHASE 1
					"certainty": float(e.get("certainty", 0.95)),  # PHASE 1
					"is_assumption": bool(e.get("is_assumption", False)),  # PHASE 1
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

	def load(self, path: str = "data/memory/memory.json"):
		try:
			with open(path, "r", encoding="utf-8") as f:
				payload = json.load(f) or {}
				items = payload.get("entries", []) or []
				saved_entity_index = payload.get("entity_index", {}) or {}
		except Exception:
			return

		self.entries = []
		self.sources = set()
		contaminated_count = 0

		for it in items:
			try:
				text = it.get("text", "")
				source = it.get("source", "unknown")
				
				# VALIDATE: Block user_input sources
				if source == "user_input" or source.startswith("user_input#"):
					contaminated_count += 1
					continue
				
				# VALIDATE: Block assistant chatter and user questions
				text_lower = text.lower()
				if any(phrase in text_lower for phrase in [
					"i'll be happy to", "please provide", "i'm happy to",
					"what went wrong", "can you give me", "tell me about",
					"i don't have", "i cannot", "as an ai",
					"sorry, i", "i apologize", "thank you for",
					"it seems like there is no",
				]):
					contaminated_count += 1
					continue
				
				# VALIDATE: Block too-short content
				if len(text.split()) < 5:
					contaminated_count += 1
					continue
				
				emb = np.array(it.get("embedding") or [], dtype=np.float32)
				try:
					n = float(np.linalg.norm(emb))
					if n > 0 and not np.isnan(n):
						emb = emb / n
				except Exception:
					pass

				eid = it.get("id")
				if not eid:
					text_hash = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
					eid = f"{source}#{text_hash}"

				text_hash = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
				entry = {
					"id": eid,
					"embedding": emb,
					"text": text,
					"source": source,
					"type": it.get("type", "evidence"),
					"ts": it.get("ts", time.time()),
					"_dedup_key": (source, text_hash),
					"generated_by": it.get("generated_by", "user"),  # PHASE 1: Backward compatible
					"certainty": float(it.get("certainty", 0.95)),  # PHASE 1: Backward compatible
					"is_assumption": bool(it.get("is_assumption", False)),  # PHASE 1: Backward compatible
				}
				self.entries.append(entry)
				self.sources.add((entry["source"], text_hash))
			except Exception:
				continue
		
		if contaminated_count > 0:
			logger.warning(f"[MEMORY] Filtered out {contaminated_count} contaminated entries on load")

		try:
			if self._use_faiss and len(self.entries) > 0:
				self._rebuild_faiss()
		except Exception:
			pass

		try:
			# MEMORY LEAK FIX: Convert loaded lists to sets for O(1) lookup
			converted: Dict[str, set] = {}
			for ent, lst in (saved_entity_index or {}).items():
				if not lst:
					continue
				if isinstance(lst, list) and lst and isinstance(lst[0], int):
					ids = set()
					for idx in lst:
						if 0 <= idx < len(self.entries):
							ids.add(self.entries[idx]["id"])
					if ids:
						converted[str(ent)] = ids
				else:
					converted[str(ent)] = set(str(i) for i in (lst or []))
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
						# Priority: definition > fact > summary > evidence
						tw = {"definition": 1.5, "fact": 1.2, "summary": 1.1, "evidence": 1.0}.get(e.get("type", "evidence"), 1.0)
						# PHASE 1: Apply certainty weight from entry
						certainty_weight = float(e.get("certainty", 0.95))
						base = (self.weights["sim"] * sim) + (self.weights["lex"] * lex) + (self.weights["recency"] * recency)
						score = float(base) * float(tw) * float(certainty_weight)
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
									"certainty": certainty_weight,
									"generated_by": e.get("generated_by", "user"),
									"is_assumption": e.get("is_assumption", False),
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
					# Priority: definition > fact > summary > evidence
					tw = {"definition": 1.5, "fact": 1.2, "summary": 1.1, "evidence": 1.0}.get(e.get("type", "evidence"), 1.0)
					# PHASE 1: Apply certainty weight from entry
					certainty_weight = float(e.get("certainty", 0.95))
					base = (self.weights["sim"] * sim) + (self.weights["lex"] * lex) + (self.weights["recency"] * recency)
					score = float(base) * float(tw) * float(certainty_weight)
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
								"certainty": certainty_weight,
								"generated_by": e.get("generated_by", "user"),
								"is_assumption": e.get("is_assumption", False),
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
