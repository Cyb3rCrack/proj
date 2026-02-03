"""Question interpretation and intent classification."""

from __future__ import annotations

from typing import Any, Dict


class QuestionInterpreter:
	"""Lightweight intent classifier to make answer behavior intent-aware."""

	def interpret(self, question: str) -> Dict[str, Any]:
		q = (question or "").strip().lower()
		intent = "factual"

		if not q:
			intent = "factual"
		elif "difference between" in q or " vs " in q or "versus" in q or q.startswith("compare "):
			intent = "compare"
		elif q.endswith("?") and q.startswith(
			(
				"is ",
				"are ",
				"does ",
				"do ",
				"did ",
				"can ",
				"could ",
				"should ",
				"would ",
				"will ",
				"has ",
				"have ",
			)
		):
			intent = "verify"
		elif q.startswith(("what is", "what are", "define ")):
			intent = "define"
		elif q.startswith(("why", "how")):
			intent = "causal"
		elif q.startswith(("if ", "suppose ", "imagine ", "what would happen")):
			intent = "hypothetical"

		requires_high_confidence = intent in {"define", "compare"}

		return {
			"intent": intent,
			"requires_high_confidence": requires_high_confidence,
		}


__all__ = ["QuestionInterpreter"]

