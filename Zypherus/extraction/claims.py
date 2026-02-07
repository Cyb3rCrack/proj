"""Atomic claim extraction.

Heuristic splitter: declarative-only, with a minimal S/V/O shape.
"""

from __future__ import annotations

import re
from typing import List


def extract_atomic_claims(text: str) -> List[str]:
	parts = re.split(r"(?<=[.!?])\s+", text or "")
	claims: List[str] = []

	verb_patterns = [
		r"\bis\b", r"\bare\b", r"\bwas\b", r"\bwere\b",
		r"\bhas\b", r"\bhave\b", r"\bhad\b",
		r"\bcauses\b", r"\bcause\b", r"\bleads\b", r"\blead\b",
		r"\bresults\b", r"\bresult\b",
		r"\bincludes\b", r"\binclude\b",
		r"\bconsists\b", r"\bconsist\b",
		r"\bcontains\b", r"\bcontain\b",
		r"\bmeans\b", r"\bmean\b",
		r"\brefers\b", r"\brefer\b",
		r"\bdefines\b", r"\bdefine\b",
	]
	verb_re = re.compile("|".join(verb_patterns), flags=re.IGNORECASE)

	for p in parts:
		s = (p or "").strip()
		if not s:
			continue

		# skip questions
		if s.endswith("?"):
			continue

		# skip very short
		if len(s.split()) < 5:
			continue

		# require a verb connector
		if not verb_re.search(s):
			continue

		# avoid modal-only leading stubs
		if re.match(r"^(might|may|could|should|would|can)\b\s*$", s, flags=re.IGNORECASE):
			continue

		# minimal left/right shape around a connector-like token
		if not re.search(r"\b\w+\b\s+(is|are|was|were|has|have|had|means|refers|includes|contains|causes|leads)\b\s+\b\w+\b", s, flags=re.IGNORECASE):
			continue

		claims.append(s)

	return claims


__all__ = ["extract_atomic_claims"]

