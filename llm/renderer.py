"""LLM renderer (Ollama)."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import requests


class LLMRenderer:
	def __init__(self, model: str = "llama3.2:3b"):
		self.model = model
		self.base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

	def _extract_json_object(self, text: str) -> Optional[dict]:
		if not text:
			return None
		s = text.strip()

		if s.startswith("{"):
			try:
				obj = json.loads(s)
				return obj if isinstance(obj, dict) else None
			except Exception:
				pass

		m = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
		if m:
			try:
				obj = json.loads(m.group(1))
				return obj if isinstance(obj, dict) else None
			except Exception:
				pass

		dec = json.JSONDecoder()
		for i, ch in enumerate(s):
			if ch != '{':
				continue
			try:
				obj, _end = dec.raw_decode(s[i:])
				return obj if isinstance(obj, dict) else None
			except Exception:
				continue
		return None

	def answer(
		self,
		question: str,
		context_chunks: List[Any],
		convo_history: str = "",
		reasoning: Any = None,
		style_hint: str = "",
	) -> str:
		context = "\n\n".join(
			f"[Source: {e.get('source','unknown')}] {e.get('text','')[:1000]}" for _, e in (context_chunks or [])
		)
		context = context[:12000]

		reasoning_text = ""
		if reasoning:
			if isinstance(reasoning, dict):
				known = "\n".join(f"- {k}" for k in (reasoning.get("known") or []))
				uncertain = "\n".join(f"- {u}" for u in (reasoning.get("uncertain") or []))
				concl = reasoning.get("conclusion", "")
				reasoning_text = f"Reasoning Draft:\nKnown:\n{known}\n\nUncertain:\n{uncertain}\n\nConclusion:\n{concl}\n\n"
			else:
				reasoning_text = f"Reasoning Draft:\n{str(reasoning)}\n\n"

		prompt = (
			"Answer the question using a complete, explicit sentence.\n"
			"Do NOT answer with a fragment, a single noun, or an entity-only reply unless asked to.\n"
			"If the question asks for a relation (e.g., 'What orbits the Sun?'), include the verb in your answer.\n"
			"Prefer 1-2 sentences. If uncertain, qualify clearly.\n\n"
		)
		if style_hint:
			prompt += f"STYLE REQUIREMENT (must follow): {style_hint}\n\n"
		prompt += (
			f"Conversation:\n{convo_history}\n\n"
			f"Context:\n{context}\n\n"
			f"{reasoning_text}"
			f"Question: {question}\n\nAnswer:"
		)

		try:
			response = requests.post(
				f"{self.base_url}/api/generate",
				json={"model": self.model, "prompt": prompt, "stream": False, "max_tokens": 512},
				timeout=60,
			)
			response.raise_for_status()
			return response.json().get("response", "").strip()
		except Exception:
			return self._fallback_answer(question, context_chunks)

	def verify_answer(self, answer: str, context_chunks) -> Dict[str, Any]:
		context = "\n\n".join(f"[Source: {e['source']}] {e['text'][:1000]}" for _, e in context_chunks)
		prompt = f"""
You are a fact-check assistant. Given the CONTEXT and an ANSWER, identify which claims are directly supported by the context.

Context:
{context}

Answer:
{answer}

Return ONLY valid JSON with keys: confidence (0-1), unsupported_claims (list of short strings).
If all claims are supported, unsupported_claims must be an empty list.
"""
		try:
			r = requests.post(
				f"{self.base_url}/api/generate",
				json={"model": self.model, "prompt": prompt, "stream": False, "max_tokens": 300},
				timeout=60,
			)
			r.raise_for_status()
			txt = r.json().get("response", "")
			obj = self._extract_json_object(txt)
			if isinstance(obj, dict):
				return obj

			conf = 0.0
			m_conf = re.search(r"([0-9]*\.?[0-9]+)", txt)
			if m_conf:
				try:
					val = float(m_conf.group(1))
					if 0.0 <= val <= 1.0:
						conf = val
				except Exception:
					pass

			unsupported = []
			if "unsupported" in txt.lower() or "not supported" in txt.lower():
				unsupported.append("model indicates unsupported claims")
			return {"confidence": conf or 0.5, "unsupported_claims": unsupported}
		except Exception:
			return {"confidence": None, "unsupported_claims": [], "status": "unavailable"}

	def _fallback_answer(self, question: str, context_chunks):
		if not context_chunks:
			return f"(Ollama offline) I don't have context to answer: {question}"
		try:
			first_entry = (
				context_chunks[0][1]
				if isinstance(context_chunks[0], (list, tuple)) and len(context_chunks[0]) >= 2
				else context_chunks[0]
			)
			context_text = (first_entry.get("text", "") or "")[:200] if isinstance(first_entry, dict) else str(first_entry)[:200]
		except Exception:
			context_text = ""

		q_lower = (question or "").lower()
		if any(word in q_lower for word in ["what", "tell", "explain", "describe"]):
			return f"Based on available context: {context_text}... (To get full answers, ensure Ollama is running at {self.base_url})"
		if any(word in q_lower for word in ["how", "can", "do"]):
			return f"I found relevant information: {context_text}... (Start Ollama for detailed answers)"
		return f"Context found: {context_text}... (Ollama needed for full response)"


__all__ = ["LLMRenderer"]

