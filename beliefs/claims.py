"""ClaimStore (structured belief system + contradiction tracking)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List

from ace.beliefs.graph import BeliefGraph
from ace.extraction.parsing import parse_claim
from ace.llm.nli import NLIContradictionChecker


def infer_domain(text: str) -> str:
	"""Very small heuristic domain tagger: physics | biology | general."""
	t = (text or "").lower()
	if not t:
		return "general"

	physics_weights = {
		"mass": 2.0,
		"weight": 2.0,
		"gravity": 1.5,
		"gravitational": 1.5,
		"force": 1.0,
		"newton": 1.0,
		"acceleration": 1.0,
		"energy": 1.0,
		"work": 1.0,
		"power": 1.0,
		"heat": 2.0,
		"temperature": 2.0,
		"kelvin": 1.5,
		"celsius": 1.0,
		"fahrenheit": 1.0,
		"pressure": 1.0,
		"density": 1.0,
		"volume": 1.0,
		"momentum": 1.0,
		"velocity": 1.0,
	}
	biology_weights = {
		"cell": 1.5,
		"cells": 1.5,
		"dna": 2.0,
		"rna": 2.0,
		"gene": 1.5,
		"genes": 1.5,
		"protein": 1.5,
		"proteins": 1.5,
		"enzyme": 1.0,
		"bacteria": 1.0,
		"virus": 1.0,
		"viruses": 1.0,
		"species": 1.0,
		"evolution": 1.0,
		"mitosis": 1.0,
		"meiosis": 1.0,
		"photosynthesis": 1.0,
		"chromosome": 1.0,
		"chromosomes": 1.0,
	}

	tokens = re.findall(r"\w+", t)
	phys = sum(float(physics_weights.get(tok, 0.0)) for tok in tokens)
	bio = sum(float(biology_weights.get(tok, 0.0)) for tok in tokens)

	if phys > bio and phys >= 1.0:
		return "physics"
	if bio > phys and bio >= 1.0:
		return "biology"
	return "general"


class ClaimStore:
	def __init__(self):
		self.claims: Dict[str, Dict[str, Any]] = {}
		self.nli = NLIContradictionChecker()
		self.graph = BeliefGraph()
		self.revisions: Dict[str, List[Dict[str, Any]]] = {}

		self.decay_half_life_days = 45.0
		self.inference_weight = 0.60
		self.max_inferred_per_cycle = 25

		self.subject_index: Dict[str, set] = {}
		self.revisions_archive: List[Dict[str, Any]] = []

	def _subject_key(self, subject: str) -> str:
		return self._norm_key(subject or "")

	def _index_claim(self, cid: str, rec: Dict[str, Any]):
		sk = self._subject_key(rec.get("subject") or "")
		if not sk:
			return
		self.subject_index.setdefault(sk, set()).add(cid)

	def migrate(self, payload: dict) -> dict:
		if not isinstance(payload, dict):
			return {"schema_version": 1, "saved_at": time.time(), "claims": {}}

		version = int(payload.get("schema_version", 0) or 0)

		if version == 0:
			if "claims" not in payload:
				if any(isinstance(v, dict) for v in payload.values()):
					payload = {"schema_version": 0, "saved_at": time.time(), "claims": payload}
				else:
					payload = {"schema_version": 0, "saved_at": time.time(), "claims": {}}

			for rec in (payload.get("claims", {}) or {}).values():
				if not isinstance(rec, dict):
					continue
				rec.setdefault("modality", "fact")
				rec.setdefault("polarity", 1)
				rec.setdefault("confidence", 0.0)
				rec.setdefault("created", time.time())
				rec.setdefault("last_updated", rec.get("created", time.time()))
				rec.setdefault("_derived", False)

			payload["schema_version"] = 1

		payload.setdefault("saved_at", time.time())
		payload.setdefault("claims", {})
		return payload

	def save(self, path: str = "claims.json"):
		payload = {"schema_version": 1, "saved_at": time.time(), "claims": {}}

		for cid, rec in (self.claims or {}).items():
			payload["claims"][cid] = {
				"id": rec.get("id", cid),
				"subject": rec.get("subject"),
				"predicate": rec.get("predicate"),
				"predicate_token": rec.get("predicate_token"),
				"predicate_tense": rec.get("predicate_tense"),
				"object": rec.get("object"),
				"polarity": rec.get("polarity"),
				"modality": rec.get("modality"),
				"raw": rec.get("raw"),
				"domain": rec.get("domain", "general"),
				"confidence": rec.get("confidence"),
				"created": rec.get("created"),
				"last_updated": rec.get("last_updated"),
				"_derived": bool(rec.get("_derived", False)),
				"supporting_evidence": list(rec.get("supporting_evidence", set()) or []),
				"contradicting_evidence": list(rec.get("contradicting_evidence", set()) or []),
				"supporting_claims": list(rec.get("supporting_claims", set()) or []),
				"contradicting_claims": list(rec.get("contradicting_claims", set()) or []),
			}

		try:
			payload["revisions"] = self.revisions or {}
		except Exception:
			payload["revisions"] = {}

		try:
			with open(path, "w", encoding="utf-8") as f:
				json.dump(payload, f, ensure_ascii=False, indent=2)
		except Exception:
			return

	def load(self, path: str = "claims.json"):
		if not os.path.exists(path):
			return
		try:
			with open(path, "r", encoding="utf-8") as f:
				payload = json.load(f) or {}
		except Exception:
			return

		payload = self.migrate(payload)
		claims_raw = payload.get("claims", {}) or {}
		self.claims = {}
		self.subject_index = {}

		for cid, rec in (claims_raw or {}).items():
			if not isinstance(rec, dict):
				continue
			self.claims[cid] = {
				**rec,
				"supporting_evidence": set(rec.get("supporting_evidence", []) or []),
				"contradicting_evidence": set(rec.get("contradicting_evidence", []) or []),
				"contradicting_claims": set(rec.get("contradicting_claims", []) or []),
				"supporting_claims": set(rec.get("supporting_claims", []) or []),
			}

			self.claims[cid].setdefault("id", cid)
			self.claims[cid].setdefault("modality", "fact")
			self.claims[cid].setdefault("polarity", 1)
			self.claims[cid].setdefault("domain", self.claims[cid].get("domain") or "general")
			self.claims[cid].setdefault("created", time.time())
			self.claims[cid].setdefault("last_updated", self.claims[cid].get("created", time.time()))

			try:
				self._index_claim(cid, self.claims[cid])
			except Exception:
				pass

		try:
			self.revisions = payload.get("revisions", {}) or {}
		except Exception:
			self.revisions = {}

		try:
			self._rebuild_graph_from_claims()
		except Exception:
			pass

		for _cid, rec in (self.claims or {}).items():
			try:
				rec["confidence"] = float(self.recompute_confidence(rec))
			except Exception:
				rec["confidence"] = 0.0

	def _rebuild_graph_from_claims(self):
		self.graph = BeliefGraph()
		for cid, rec in (self.claims or {}).items():
			for src in (rec.get("supporting_claims") or []):
				if not src or src == cid:
					continue
				self.graph.add_edge(str(src), str(cid), relation="implies", weight=self.inference_weight)

	def _domain(self, claim_or_rec: Dict[str, Any]) -> str:
		dom = (claim_or_rec or {}).get("domain")
		dom = (dom or "").strip().lower()
		return dom if dom in {"physics", "biology", "general"} else "general"

	def _claim_text(self, claim_or_rec: Dict[str, Any]) -> str:
		raw = (claim_or_rec.get("raw") or "").strip()
		if raw:
			return raw
		subj = (claim_or_rec.get("subject") or "").strip()
		pred = (claim_or_rec.get("predicate") or "").strip()
		obj = (claim_or_rec.get("object") or "").strip()
		parts = [p for p in [subj, pred, obj] if p]
		return " ".join(parts).strip()

	def _norm_key(self, text: str) -> str:
		return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())).strip()

	def _claim_id(self, claim: Dict[str, Any]) -> str:
		subj = (claim.get("subject") or "").strip().lower()
		pred = (claim.get("predicate") or "").strip().lower()
		obj = (claim.get("object") or "").strip().lower()
		pol = str(int(claim.get("polarity", 1)))
		mod = (claim.get("modality") or "fact").strip().lower()
		tense = (claim.get("predicate_tense") or "").strip().lower()
		raw_norm = self._norm_key(claim.get("raw") or "")
		raw_hash = hashlib.sha256(raw_norm.encode("utf-8")).hexdigest()[:16] if raw_norm else ""
		key = f"{subj}|{pred}|{obj}|{pol}|{mod}|{tense}|{raw_hash}"
		return hashlib.sha256(key.encode("utf-8")).hexdigest()

	def observe_claim(self, claim_input: Any, evidence_id: str):
		if isinstance(claim_input, str):
			claim = parse_claim(claim_input)
		else:
			claim = claim_input or {}

		cid = self._claim_id(claim)
		rec = self.claims.get(cid)
		if rec is None:
			rec = {
				"id": cid,
				"subject": claim.get("subject"),
				"predicate": claim.get("predicate"),
				"predicate_token": claim.get("predicate_token"),
				"predicate_tense": claim.get("predicate_tense"),
				"object": claim.get("object"),
				"polarity": int(claim.get("polarity", 1)),
				"modality": claim.get("modality", "fact"),
				"raw": claim.get("raw", ""),
				"domain": claim.get("domain") or infer_domain(claim.get("raw", "") or ""),
				"supporting_evidence": set(),
				"contradicting_evidence": set(),
				"contradicting_claims": set(),
				"supporting_claims": set(),
				"created": time.time(),
				"last_updated": time.time(),
				"confidence": 0.0,
			}
			self.claims[cid] = rec
			try:
				self._index_claim(cid, rec)
			except Exception:
				pass

		rec["last_updated"] = time.time()
		rec["supporting_evidence"].add(evidence_id)

		try:
			this_text = self._claim_text(rec)
			this_subj = self._norm_key(rec.get("subject") or "")
			this_obj = self._norm_key(rec.get("object") or "")
			this_dom = self._domain(rec)

			candidate_ids = list(self.subject_index.get(this_subj, set())) if this_subj else []
			if len(candidate_ids) > 2000:
				candidate_ids = candidate_ids[-2000:]

			for other_id in candidate_ids:
				if other_id == cid:
					continue
				other = self.claims.get(other_id)
				if not other:
					continue

				other_dom = self._domain(other)
				if this_dom != other_dom:
					continue

				other_subj = self._norm_key(other.get("subject") or "")
				other_obj = self._norm_key(other.get("object") or "")

				likely_related = False
				if this_subj and other_subj and this_subj == other_subj:
					if this_obj and other_obj:
						likely_related = True
				if not likely_related:
					a = set(self._norm_key(this_text).split())
					b = set(self._norm_key(self._claim_text(other)).split())
					if a and b and (len(a & b) / float(max(1, min(len(a), len(b))))) >= 0.50:
						likely_related = True
				if not likely_related:
					continue

				other_text = self._claim_text(other)
				if this_text and other_text and self.nli._ensure_loaded():
					if self.nli.is_contradiction(this_text, other_text):
						rec["contradicting_claims"].add(other_id)
						other.setdefault("contradicting_claims", set()).add(cid)
						self._log_revision(cid, "contradiction_detected", {"other_claim": other_id, "via": "nli"})
						self._log_revision(other_id, "contradiction_detected", {"other_claim": cid, "via": "nli"})
				else:
					if this_subj and other_subj and this_obj and other_obj and this_subj == other_subj and this_obj == other_obj:
						if int(other.get("polarity", 1)) != int(rec.get("polarity", 1)):
							rec["contradicting_claims"].add(other_id)
							other.setdefault("contradicting_claims", set()).add(cid)
							self._log_revision(cid, "contradiction_detected", {"other_claim": other_id, "via": "heuristic"})
							self._log_revision(other_id, "contradiction_detected", {"other_claim": cid, "via": "heuristic"})
		except Exception:
			pass

		rec["confidence"] = self.recompute_confidence(rec)

	def observe_derived_claim(self, claim: Dict[str, Any], supporting_claim_ids: List[str]):
		if not isinstance(claim, dict):
			return

		inherited_domain = (claim.get("domain") or "").strip().lower() if isinstance(claim.get("domain"), str) else ""
		if not inherited_domain:
			try:
				upstream_domains = [self._domain(self.claims.get(u, {})) for u in (supporting_claim_ids or []) if u in self.claims]
				upstream_domains = [d for d in upstream_domains if d]
				if upstream_domains and len(set(upstream_domains)) == 1:
					inherited_domain = upstream_domains[0]
			except Exception:
				inherited_domain = ""
		if not inherited_domain:
			inherited_domain = infer_domain(claim.get("raw", "") or "")
		claim["domain"] = inherited_domain

		cid = self._claim_id(claim)
		rec = self.claims.get(cid)
		if rec is None:
			rec = {
				"id": cid,
				"subject": claim.get("subject"),
				"predicate": claim.get("predicate"),
				"object": claim.get("object"),
				"polarity": int(claim.get("polarity", 1)),
				"modality": claim.get("modality", "uncertain"),
				"raw": claim.get("raw", ""),
				"domain": claim.get("domain") or inherited_domain,
				"supporting_evidence": set(),
				"contradicting_evidence": set(),
				"contradicting_claims": set(),
				"supporting_claims": set(supporting_claim_ids or []),
				"created": time.time(),
				"last_updated": time.time(),
				"confidence": 0.0,
				"_derived": True,
			}
			self.claims[cid] = rec
			self._log_revision(cid, "derived_claim_added", {"supporting_claims": list(rec.get("supporting_claims", []))})
		else:
			rec.setdefault("supporting_claims", set()).update(set(supporting_claim_ids or []))
			rec["last_updated"] = time.time()

		rec["confidence"] = self.recompute_confidence(rec)

	def contradict_claim(self, claim_input: Any, evidence_id: str):
		if isinstance(claim_input, str):
			claim = parse_claim(claim_input)
		else:
			claim = claim_input or {}
		cid = self._claim_id(claim)
		rec = self.claims.get(cid)
		if rec is None:
			rec = {
				"id": cid,
				"subject": claim.get("subject"),
				"predicate": claim.get("predicate"),
				"object": claim.get("object"),
				"polarity": int(claim.get("polarity", -1)),
				"modality": claim.get("modality", "fact"),
				"raw": claim.get("raw", ""),
				"supporting_evidence": set(),
				"contradicting_evidence": set([evidence_id]),
				"contradicting_claims": set(),
				"supporting_claims": set(),
				"created": time.time(),
				"last_updated": time.time(),
				"confidence": 0.0,
			}
			self.claims[cid] = rec
		else:
			rec.setdefault("contradicting_evidence", set()).add(evidence_id)
			rec["last_updated"] = time.time()

		rec["confidence"] = self.recompute_confidence(rec)
		self._log_revision(cid, "contradicting_evidence_added", {"evidence_id": evidence_id})

	def _log_revision(self, claim_id: str, event: str, details: Dict[str, Any] = None):
		if not claim_id:
			return
		self.revisions.setdefault(claim_id, []).append({"ts": time.time(), "event": str(event), "details": details or {}})

	def recompute_confidence(self, claim_rec: Dict[str, Any]) -> float:
		s = len(claim_rec.get("supporting_evidence", []))
		c = len(claim_rec.get("contradicting_evidence", []))
		cc = len(claim_rec.get("contradicting_claims", []))
		k = 1.0
		denom = float(s + c + cc + k)
		base = 0.0 if denom <= 0 else float(s) / denom

		inherited = 0.0
		try:
			supporting_claims = list(claim_rec.get("supporting_claims", []) or [])
			if supporting_claims:
				upstream_confs = [float(self.claims.get(u, {}).get("confidence", 0.0)) for u in supporting_claims if u in self.claims]
				if upstream_confs:
					inherited = float(min(upstream_confs)) * float(self.inference_weight)
		except Exception:
			inherited = 0.0

		penalty = 1.0
		if (c + cc) > 0:
			penalty = 1.0 / (1.0 + 0.75 * float(c + cc))

		decay = 1.0
		try:
			hl_days = max(1.0, float(self.decay_half_life_days))
			last_updated = float(claim_rec.get("last_updated", claim_rec.get("created", time.time())))
			age_days = max(0.0, (time.time() - last_updated) / (3600.0 * 24.0))
			decay = float(0.5 ** (age_days / hl_days))
		except Exception:
			decay = 1.0

		conf = max(base, inherited) * penalty * decay
		return max(0.0, min(1.0, float(conf)))

	def update_beliefs(self):
		try:
			derived = self.graph.infer_causal_chains(self.claims, max_new=self.max_inferred_per_cycle)
			for d in derived:
				supporting = list(d.get("_derived_from") or [])
				self.observe_derived_claim(d, supporting)
				for src in supporting:
					self.graph.add_edge(src, self._claim_id(d), relation="implies", weight=self.inference_weight)
		except Exception:
			pass

		try:
			for cid, rec in self.claims.items():
				before = float(rec.get("confidence", 0.0))
				after = float(self.recompute_confidence(rec))
				rec["confidence"] = after
				if abs(after - before) >= 0.10:
					self._log_revision(cid, "confidence_updated", {"before": before, "after": after})
		except Exception:
			pass

		try:
			for cid, rec in self.claims.items():
				upstream = self.graph.upstream(cid)
				if not upstream:
					continue
				this_dom = self._domain(rec)
				upstream_confs = []
				for u in upstream:
					if u not in self.claims:
						continue
					if this_dom != self._domain(self.claims.get(u, {})):
						continue
					upstream_confs.append(float(self.claims[u].get("confidence", 0.0) or 0.0))
				if not upstream_confs:
					continue

				inherited = max(0.0, min(1.0, float(min(upstream_confs))))
				before = float(rec.get("confidence", 0.0) or 0.0)
				after = max(0.0, min(1.0, before * inherited))
				if abs(after - before) >= 0.05:
					rec["confidence"] = after
					self._log_revision(
						cid,
						"dependency_propagation",
						{"before": before, "after": after, "inherited": inherited},
					)
		except Exception:
			pass

	def retrieve_relevant(self, evidence_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		out = []
		evidence_ids = set(e.get("id") for e in evidence_list if e.get("id"))
		for cid, rec in self.claims.items():
			if evidence_ids & rec.get("supporting_evidence", set()):
				contradicting = set(rec.get("contradicting_claims", set())) | set(rec.get("contradicting_evidence", set()))
				try:
					age_days = (time.time() - float(rec.get("created", time.time()))) / 86400.0
				except Exception:
					age_days = None
				out.append(
					{
						"id": cid,
						"subject": rec.get("subject"),
						"predicate": rec.get("predicate"),
						"object": rec.get("object"),
						"polarity": rec.get("polarity"),
						"modality": rec.get("modality"),
						"raw": rec.get("raw"),
						"domain": rec.get("domain", "general"),
						"age_days": age_days,
						"confidence": rec.get("confidence", 0.0),
						"supporting": list(rec.get("supporting_evidence", [])),
						"contradicting": list(contradicting),
						"contradicting_claims": list(rec.get("contradicting_claims", [])),
						"contradicting_evidence": list(rec.get("contradicting_evidence", [])),
					}
				)
		return out


__all__ = ["ClaimStore"]
