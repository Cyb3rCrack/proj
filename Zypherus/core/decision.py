"""Centralized decision logic for answer/clarify/refuse."""

from __future__ import annotations

import os
import logging
from typing import Any, Dict


logger = logging.getLogger("ACE")


class DecisionPolicy:
	"""Centralized decision logic for answer/clarify/refuse."""

	def __init__(self, min_confidence: float = 0.55):
		try:
			self.min_confidence = float(os.getenv("ACE_MIN_CONF", str(min_confidence)))
		except Exception:
			self.min_confidence = min_confidence

		def _fenv(name: str, default: float) -> float:
			try:
				return float(os.getenv(name, str(default)))
			except Exception:
				return float(default)

		self.w_verify = _fenv("ACE_W_VERIFY", 0.50)
		self.w_retrieval = _fenv("ACE_W_RETRIEVAL", 0.30)
		self.w_claim = _fenv("ACE_W_CLAIM", 0.20)

		self.strong_contradiction_threshold = 0.60

	def decide(
		self,
		retrieval_metrics: Dict[str, Any],
		verification: Dict[str, Any],
		symbolic_conflicts: Dict[str, Any],
		question_info: Dict[str, Any] | None = None,
	) -> Dict[str, Any]:
		raw_conf = verification.get("confidence", None)
		if isinstance(raw_conf, (int, float)):
			verification_conf = float(raw_conf)
		else:
			verification_conf = 0.5

		# Apply grounding bonus if synthesis followed structured format
		grounding_bonus = float(verification.get("grounding_bonus", 0.0) or 0.0)
		verification_conf = max(0.0, min(1.0, verification_conf + grounding_bonus))

		unsupported = verification.get("unsupported_claims", []) or []

		retrieval_strength = float(retrieval_metrics.get("retrieval_strength", 0.0) or 0.0)
		retrieval_strength = max(0.0, min(1.0, retrieval_strength))

		claim_strength = float(symbolic_conflicts.get("claim_strength", 0.0) or 0.0)
		claim_strength = max(0.0, min(1.0, claim_strength))

		blended = (self.w_verify * verification_conf) + (self.w_retrieval * retrieval_strength) + (self.w_claim * claim_strength)
		blended = max(0.0, min(1.0, float(blended)))

		if unsupported:
			blended = max(0.0, min(1.0, float(blended) * 0.60))

		contradictions = symbolic_conflicts.get("contradictions") or []
		strong_conflicts = [
			c
			for c in contradictions
			if float(c.get("confidence", 0.0) or 0.0) >= self.strong_contradiction_threshold
		]
		if strong_conflicts:
			return {
				"action": "dispute",
				"reason": "strong_contradictions_detected",
				"confidence": blended,
				"strong_conflicts": strong_conflicts,
			}

		min_conf = float(self.min_confidence)
		try:
			if (question_info or {}).get("requires_high_confidence"):
				min_conf = max(min_conf, 0.65)
		except Exception:
			pass

		if blended < min_conf:
			decision = {"action": "clarify", "reason": "low_blended_confidence", "confidence": blended}
			try:
				logger.info(
					f"[Decision] intent={(question_info or {}).get('intent')} blended={blended:.3f} "
					f"verify={verification_conf:.3f} retrieval={retrieval_strength:.3f} claim={claim_strength:.3f} -> clarify"
				)
			except Exception:
				pass
			return decision

		decision = {"action": "accept", "confidence": blended}
		try:
			logger.info(
				f"[Decision] intent={(question_info or {}).get('intent')} blended={blended:.3f} "
				f"verify={verification_conf:.3f} retrieval={retrieval_strength:.3f} claim={claim_strength:.3f} -> accept"
			)
		except Exception:
			pass
		return decision


__all__ = ["DecisionPolicy"]

