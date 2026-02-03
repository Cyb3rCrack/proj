"""Text chunking."""

from __future__ import annotations

import re
from typing import List, Iterable

from ace.utils.text import get_nlp


def chunk_text(text: str, chunk_size: int = 500) -> Iterable[str]:
	"""Chunk by sentences first, then pack into ~chunk_size words.

	This preserves local coherence better than naive word slicing.
	"""
	if not text:
		return

	chunk_size = int(max(50, chunk_size))
	sentences: List[str] = []

	nlp = get_nlp()
	if nlp is not None:
		try:
			doc = nlp(text)
			sentences = [s.text.strip() for s in doc.sents if s.text and s.text.strip()]
		except Exception:
			sentences = []

	if not sentences:
		# Fallback sentence splitter
		sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

	buf: List[str] = []
	buf_words = 0

	def flush():
		nonlocal buf, buf_words
		if buf:
			yield " ".join(buf).strip()
		buf = []
		buf_words = 0

	for sent in sentences:
		words = sent.split()
		if not words:
			continue

		# If a single sentence is enormous, split it.
		if len(words) > chunk_size:
			for part_i in range(0, len(words), chunk_size):
				part = " ".join(words[part_i:part_i + chunk_size]).strip()
				if part:
					for c in flush():
						yield c
					yield part
			continue

		if buf_words + len(words) > chunk_size:
			for c in flush():
				yield c

		buf.append(sent)
		buf_words += len(words)

	for c in flush():
		yield c


__all__ = ["chunk_text"]

