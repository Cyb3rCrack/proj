"""Concept extraction.

Prefers spaCy NER when available; otherwise returns empty to avoid noisy regex.
"""

from __future__ import annotations

from typing import List

from ace.utils.text import get_nlp


def extract_concepts(text: str) -> List[str]:
	if not text:
		return []
	nlp = get_nlp()
	if nlp is None:
		return []
	try:
		doc = nlp(text)
		ents = [ent.text for ent in doc.ents]
		uniq = []
		for e in ents:
			if e not in uniq:
				uniq.append(e)
		return uniq
	except Exception:
		return []


__all__ = ["extract_concepts"]

