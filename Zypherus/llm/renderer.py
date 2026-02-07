"""LLM renderer (Ollama)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("ZYPHERUS.LLM")


@dataclass
class CacheEntry:
	value: str
	expires_at: float
	created_at: float


@dataclass
class LLMetrics:
	requests: int = 0
	cache_hits: int = 0
	cache_misses: int = 0
	timeouts: int = 0
	connection_errors: int = 0
	other_errors: int = 0
	total_latency_s: float = 0.0
	last_latency_s: float = 0.0
	last_error: str = ""


class CircuitBreaker:
	def __init__(self, failure_threshold: int, cooldown_s: float):
		self.failure_threshold = max(1, failure_threshold)
		self.cooldown_s = max(0.0, cooldown_s)
		self._failures = 0
		self._open_until = 0.0

	def is_open(self) -> bool:
		return time.time() < self._open_until

	def record_success(self) -> None:
		self._failures = 0
		self._open_until = 0.0

	def record_failure(self) -> None:
		self._failures += 1
		if self._failures >= self.failure_threshold:
			self._open_until = time.time() + self.cooldown_s
			self._failures = 0

	def status(self) -> Dict[str, Any]:
		return {
			"open": self.is_open(),
			"open_until": self._open_until,
			"failure_threshold": self.failure_threshold,
			"cooldown_s": self.cooldown_s,
		}


class RateLimiter:
	def __init__(self, min_interval_s: float):
		self.min_interval_s = max(0.0, min_interval_s)
		self._lock = threading.Lock()
		self._last_call = 0.0

	def wait(self) -> None:
		if self.min_interval_s <= 0.0:
			return
		with self._lock:
			now = time.monotonic()
			elapsed = now - self._last_call
			if elapsed < self.min_interval_s:
				time.sleep(self.min_interval_s - elapsed)
			self._last_call = time.monotonic()


class LRUCache:
	def __init__(self, max_items: int, ttl_s: float):
		self.max_items = max(1, max_items)
		self.ttl_s = max(0.0, ttl_s)
		self._lock = threading.Lock()
		self._data: OrderedDict[str, CacheEntry] = OrderedDict()

	def get(self, key: str) -> Optional[str]:
		with self._lock:
			entry = self._data.get(key)
			if not entry:
				return None
			if entry.expires_at > 0 and time.time() > entry.expires_at:
				del self._data[key]
				return None
			self._data.move_to_end(key)
			return entry.value

	def set(self, key: str, value: str) -> None:
		with self._lock:
			expires_at = time.time() + self.ttl_s if self.ttl_s > 0 else 0.0
			self._data[key] = CacheEntry(value=value, expires_at=expires_at, created_at=time.time())
			self._data.move_to_end(key)
			while len(self._data) > self.max_items:
				self._data.popitem(last=False)


class LLMRenderer:
	def __init__(self, model: Optional[str] = None):
		self.model = model or os.getenv("ACE_MODEL", "llama3.2:3b")
		self.base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
		self.fallback_models = [m.strip() for m in os.getenv("ACE_MODEL_FALLBACKS", "").split(",") if m.strip()]
		self.active_model = self.model

		# All timeouts configurable via env vars
		self.connect_timeout_s = float(os.getenv("ACE_OLLAMA_CONNECT_TIMEOUT_S", "5"))
		self.request_timeout_s = float(os.getenv("ACE_OLLAMA_TIMEOUT_S", "20"))
		self.total_budget_s = float(os.getenv("ACE_OLLAMA_BUDGET_S", "45"))

		# All character limits configurable
		self.max_context_chars = int(os.getenv("ZYPHERUS_MAX_CONTEXT_CHARS", "12000"))
		self.max_chunk_chars = int(os.getenv("ZYPHERUS_MAX_CHUNK_CHARS", "1000"))
		self.max_fallback_chars = int(os.getenv("ZYPHERUS_MAX_FALLBACK_CHARS", "200"))

		# Token limits for different operations
		self.tokens_answer = int(os.getenv("ACE_TOKENS_ANSWER", "512"))
		self.tokens_verify = int(os.getenv("ACE_TOKENS_VERIFY", "300"))
		self.tokens_default = int(os.getenv("ACE_TOKENS_DEFAULT", "256"))

		# Prompts configurable via env vars with defaults
		self.prompt_answer_instructions = os.getenv(
			"ACE_PROMPT_ANSWER",
			"Answer the question using a complete, explicit sentence.\n"
			"Do NOT answer with a fragment, a single noun, or an entity-only reply unless asked to.\n"
			"If the question asks for a relation (e.g., 'What orbits the Sun?'), include the verb in your answer.\n"
			"For code requests, include a fenced code block (```lang) and a brief explanation.\n"
			"Prefer 1-2 sentences. If uncertain, qualify clearly.\n\n"
		)

		self.prompt_verify_template = os.getenv(
			"ACE_PROMPT_VERIFY",
			"You are a fact-check assistant. Given the CONTEXT and an ANSWER, identify which claims are directly supported by the context.\n\n"
			"Context:\n{context}\n\nAnswer:\n{answer}\n\n"
			"Return ONLY valid JSON with keys: confidence (0-1), unsupported_claims (list of short strings).\n"
			"If all claims are supported, unsupported_claims must be an empty list."
		)

		# HTTP session pooling + retries
		retry_total = int(os.getenv("ACE_HTTP_RETRIES", "2"))
		backoff = float(os.getenv("ACE_HTTP_BACKOFF_S", "0.5"))
		pool_max = int(os.getenv("ACE_HTTP_POOL_MAX", "10"))
		retry = Retry(
			total=retry_total,
			backoff_factor=backoff,
			status_forcelist=(429, 500, 502, 503, 504),
			allowed_methods=("GET", "POST"),
			raise_on_status=False,
		)
		adapter = HTTPAdapter(max_retries=retry, pool_connections=pool_max, pool_maxsize=pool_max)
		self.session = requests.Session()
		self.session.mount("http://", adapter)
		self.session.mount("https://", adapter)

		# Caching
		cache_ttl_s = float(os.getenv("ZYPHERUS_LLM_CACHE_TTL_S", "120"))
		cache_max_items = int(os.getenv("ZYPHERUS_LLM_CACHE_MAX", "256"))
		self.cache = LRUCache(cache_max_items, cache_ttl_s)

		# Circuit breaker + rate limiter
		cb_fail_threshold = int(os.getenv("ACE_CB_FAIL_THRESHOLD", "5"))
		cb_cooldown_s = float(os.getenv("ACE_CB_COOLDOWN_S", "30"))
		self.circuit_breaker = CircuitBreaker(cb_fail_threshold, cb_cooldown_s)

		rate_limit_s = float(os.getenv("ACE_RATE_LIMIT_S", "0"))
		self.rate_limiter = RateLimiter(rate_limit_s)

		self.metrics = LLMetrics()

		logger.info(f"LLMRenderer initialized: model={self.model}, url={self.base_url}")

	def _extract_json_object(self, text: str) -> Optional[dict]:
		"""Extract the first valid JSON object from text."""
		if not text:
			return None
		s = text.strip()

		if s.startswith("{"):
			try:
				obj = json.loads(s)
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

	def _validate_verification_json(self, obj: dict) -> Optional[Dict[str, Any]]:
		if not isinstance(obj, dict):
			return None
		conf = obj.get("confidence")
		unsupported = obj.get("unsupported_claims")
		if not isinstance(conf, (int, float)):
			return None
		if conf < 0.0 or conf > 1.0:
			return None
		if not isinstance(unsupported, list):
			return None
		return {"confidence": float(conf), "unsupported_claims": [str(u) for u in unsupported]}

	def _cache_key(self, model: str, prompt: str, max_tokens: int, stream: bool) -> str:
		h = hashlib.sha256()
		h.update(model.encode("utf-8"))
		h.update(b"|" + str(max_tokens).encode("utf-8"))
		h.update(b"|" + str(int(stream)).encode("utf-8"))
		h.update(b"|" + prompt.encode("utf-8", errors="ignore"))
		return h.hexdigest()

	def _iter_models(self) -> List[str]:
		candidates = [self.active_model] + self.fallback_models
		seen = set()
		ordered = []
		for m in candidates:
			if m and m not in seen:
				ordered.append(m)
				seen.add(m)
		return ordered

	def _post_generate(self, model: str, prompt: str, max_tokens: int, read_timeout: float, stream: bool, temperature: float = 0.7) -> str:
		payload = {
			"model": model,
			"prompt": prompt,
			"stream": bool(stream),
			"num_predict": int(max_tokens),
			"temperature": float(max(0.0, min(2.0, temperature))),
		}
		r = self.session.post(
			f"{self.base_url}/api/generate",
			json=payload,
			timeout=(self.connect_timeout_s, read_timeout),
			stream=stream,
		)
		r.raise_for_status()

		if not stream:
			return (r.json().get("response", "") or "").strip()

		chunks: List[str] = []
		for line in r.iter_lines(decode_unicode=True):
			if not line:
				continue
			try:
				obj = json.loads(line)
			except Exception:
				continue
			if isinstance(obj, dict) and obj.get("response"):
				chunks.append(str(obj["response"]))
			if isinstance(obj, dict) and obj.get("done"):
				break
		return "".join(chunks).strip()

	def generate(
		self,
		prompt: str,
		*,
		max_tokens: Optional[int] = None,
		deadline_s: Optional[float] = None,
		stream: bool = False,
		temperature: float = 0.7,
	) -> str:
		"""Generate text from the configured Ollama model.

		Returns an empty string on failure/timeout.
		Args:
			temperature: 0.1-0.2 for grounded synthesis, 0.7-0.9 for exploration
		"""
		if max_tokens is None:
			max_tokens = self.tokens_default

		if deadline_s is None:
			deadline_s = time.time() + self.total_budget_s
		remaining = max(0.0, deadline_s - time.time())
		if remaining <= 0.0:
			logger.warning("generate() deadline already exceeded")
			return ""

		if self.circuit_breaker.is_open():
			logger.warning("Circuit breaker open; skipping LLM call")
			return ""

		cache_key = self._cache_key(self.active_model, prompt, max_tokens, stream)
		cached = self.cache.get(cache_key)
		if cached is not None:
			self.metrics.cache_hits += 1
			return cached
		self.metrics.cache_misses += 1

		read_timeout = min(self.request_timeout_s, remaining)
		self.rate_limiter.wait()

		for model in self._iter_models():
			start = time.time()
			self.metrics.requests += 1
			try:
				logger.debug(
					f"Generating: {len(prompt)} chars, {max_tokens} tokens, {read_timeout:.1f}s timeout, model={model}, stream={stream}, temp={temperature}"
				)
				response = self._post_generate(model, prompt, max_tokens, read_timeout, stream, temperature)
				latency = time.time() - start
				self.metrics.total_latency_s += latency
				self.metrics.last_latency_s = latency
				self.circuit_breaker.record_success()
				self.active_model = model
				if response:
					self.cache.set(self._cache_key(model, prompt, max_tokens, stream), response)
				logger.debug(f"Generated {len(response)} chars")
				return response
			except requests.exceptions.Timeout:
				self.metrics.timeouts += 1
				self.metrics.last_error = "timeout"
				self.circuit_breaker.record_failure()
				logger.error(f"Ollama timeout after {read_timeout:.1f}s (model={model})")
			except requests.exceptions.ConnectionError:
				self.metrics.connection_errors += 1
				self.metrics.last_error = "connection_error"
				self.circuit_breaker.record_failure()
				logger.error(f"Cannot connect to Ollama at {self.base_url} (model={model})")
			except Exception as e:
				self.metrics.other_errors += 1
				self.metrics.last_error = f"{type(e).__name__}: {e}"
				self.circuit_breaker.record_failure()
				logger.error(f"Ollama error: {type(e).__name__}: {e} (model={model})")

		return ""

	def stream(
		self,
		prompt: str,
		*,
		max_tokens: Optional[int] = None,
		deadline_s: Optional[float] = None,
		temperature: float = 0.7,
	) -> Iterator[str]:
		"""Stream tokens from the configured Ollama model.

		Yields chunks of text as they arrive. Returns silently on failure/timeout.
		"""
		if max_tokens is None:
			max_tokens = self.tokens_default

		if deadline_s is None:
			deadline_s = time.time() + self.total_budget_s
		remaining = max(0.0, deadline_s - time.time())
		if remaining <= 0.0:
			logger.warning("stream() deadline already exceeded")
			return

		if self.circuit_breaker.is_open():
			logger.warning("Circuit breaker open; skipping LLM stream")
			return

		read_timeout = min(self.request_timeout_s, remaining)
		self.rate_limiter.wait()

		for model in self._iter_models():
			start = time.time()
			self.metrics.requests += 1
			try:
				payload = {
					"model": model,
					"prompt": prompt,
					"stream": True,
					"num_predict": int(max_tokens),
					"temperature": float(max(0.0, min(2.0, temperature))),
				}
				r = self.session.post(
					f"{self.base_url}/api/generate",
					json=payload,
					timeout=(self.connect_timeout_s, read_timeout),
					stream=True,
				)
				r.raise_for_status()

				for line in r.iter_lines(decode_unicode=True):
					if not line:
						continue
					try:
						obj = json.loads(line)
					except Exception:
						continue
					if isinstance(obj, dict) and obj.get("response"):
						yield str(obj["response"])
					if isinstance(obj, dict) and obj.get("done"):
						break

				latency = time.time() - start
				self.metrics.total_latency_s += latency
				self.metrics.last_latency_s = latency
				self.circuit_breaker.record_success()
				self.active_model = model
				return
			except requests.exceptions.Timeout:
				self.metrics.timeouts += 1
				self.metrics.last_error = "timeout"
				self.circuit_breaker.record_failure()
				logger.error(f"Ollama timeout after {read_timeout:.1f}s (model={model})")
			except requests.exceptions.ConnectionError:
				self.metrics.connection_errors += 1
				self.metrics.last_error = "connection_error"
				self.circuit_breaker.record_failure()
				logger.error(f"Cannot connect to Ollama at {self.base_url} (model={model})")
			except Exception as e:
				self.metrics.other_errors += 1
				self.metrics.last_error = f"{type(e).__name__}: {e}"
				self.circuit_breaker.record_failure()
				logger.error(f"Ollama error: {type(e).__name__}: {e} (model={model})")

		return

	def answer(
		self,
		question: str,
		context_chunks: List[Any],
		convo_history: str = "",
		reasoning: Any = None,
		style_hint: str = "",
	) -> str:
		"""Generate an answer to a question using context and reasoning."""
		context = "\n\n".join(
			f"[Source: {e.get('source','unknown')}] {e.get('text','')[:self.max_chunk_chars]}"
			for _, e in (context_chunks or [])
		)
		context = context[:self.max_context_chars]

		reasoning_text = ""
		if reasoning:
			if isinstance(reasoning, dict):
				known = "\n".join(f"- {k}" for k in (reasoning.get("known") or []))
				uncertain = "\n".join(f"- {u}" for u in (reasoning.get("uncertain") or []))
				concl = reasoning.get("conclusion", "")
				reasoning_text = (
					f"Reasoning Draft:\nKnown:\n{known}\n\n"
					f"Uncertain:\n{uncertain}\n\n"
					f"Conclusion:\n{concl}\n\n"
				)
			else:
				reasoning_text = f"Reasoning Draft:\n{str(reasoning)}\n\n"

		prompt = self.prompt_answer_instructions
		if style_hint:
			prompt += f"STYLE REQUIREMENT (must follow): {style_hint}\n\n"
		prompt += (
			f"Conversation:\n{convo_history}\n\n"
			f"Context:\n{context}\n\n"
			f"{reasoning_text}"
			f"Question: {question}\n\nAnswer:"
		)

		logger.debug(f"Answering question: {question[:100]}")
		txt = self.generate(prompt, max_tokens=self.tokens_answer)
		if txt:
			logger.info(f"Generated answer: {len(txt)} chars")
			return txt
		logger.warning("LLM unavailable, using fallback")
		return self._fallback_answer(question, context_chunks)

	def verify_answer(self, answer: str, context_chunks: List[Any]) -> Dict[str, Any]:
		"""Verify answer claims against context."""
		context = "\n\n".join(
			f"[Source: {e['source']}] {e['text'][:self.max_chunk_chars]}"
			for _, e in context_chunks
		)
		prompt = self.prompt_verify_template.format(context=context, answer=answer)

		logger.debug(f"Verifying answer: {answer[:100]}")
		try:
			txt = self.generate(prompt, max_tokens=self.tokens_verify)
			if not txt:
				logger.warning("Verification unavailable (LLM offline)")
				return {"confidence": None, "unsupported_claims": [], "status": "unavailable"}

			obj = self._extract_json_object(txt)
			validated = self._validate_verification_json(obj or {})
			if validated:
				logger.debug(f"Verification result: confidence={validated.get('confidence')}")
				return validated

			# Fallback parsing if JSON extraction fails
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

			logger.debug(f"Fallback verification: confidence={conf}")
			return {"confidence": conf or 0.5, "unsupported_claims": unsupported}
		except Exception as e:
			logger.error(f"Verification error: {e}")
			return {"confidence": None, "unsupported_claims": [], "status": "error"}

	def health_check(self) -> Dict[str, Any]:
		"""Check Ollama availability and list models."""
		try:
			r = self.session.get(
				f"{self.base_url}/api/tags",
				timeout=(self.connect_timeout_s, self.request_timeout_s),
			)
			r.raise_for_status()
			data = r.json()
			models = [m.get("name") for m in data.get("models", []) if isinstance(m, dict)]
			return {"ok": True, "models": models}
		except Exception as e:
			return {"ok": False, "error": f"{type(e).__name__}: {e}"}

	def get_metrics(self) -> Dict[str, Any]:
		"""Return telemetry for LLM calls."""
		return {
			"requests": self.metrics.requests,
			"cache_hits": self.metrics.cache_hits,
			"cache_misses": self.metrics.cache_misses,
			"timeouts": self.metrics.timeouts,
			"connection_errors": self.metrics.connection_errors,
			"other_errors": self.metrics.other_errors,
			"total_latency_s": round(self.metrics.total_latency_s, 3),
			"last_latency_s": round(self.metrics.last_latency_s, 3),
			"last_error": self.metrics.last_error,
			"circuit_breaker": self.circuit_breaker.status(),
			"active_model": self.active_model,
		}

	def _fallback_answer(self, question: str, context_chunks: List[Any]) -> str:
		"""Generate a fallback answer when LLM is unavailable."""
		if not context_chunks:
			logger.debug("No context available for fallback")
			return f"(Ollama offline) I don't have context to answer: {question}"

		try:
			first_entry = (
				context_chunks[0][1]
				if isinstance(context_chunks[0], (list, tuple)) and len(context_chunks[0]) >= 2
				else context_chunks[0]
			)
			context_text = (
				(first_entry.get("text", "") or "")[:self.max_fallback_chars]
				if isinstance(first_entry, dict)
				else str(first_entry)[:self.max_fallback_chars]
			)
		except Exception as e:
			logger.warning(f"Error extracting context for fallback: {e}")
			context_text = ""

		q_lower = (question or "").lower()
		if any(word in q_lower for word in ["what", "tell", "explain", "describe"]):
			return (
				f"Based on available context: {context_text}... "
				f"(To get full answers, ensure Ollama is running at {self.base_url})"
			)
		if any(word in q_lower for word in ["how", "can", "do"]):
			return (
				f"I found relevant information: {context_text}... "
				"(Start Ollama for detailed answers)"
			)
		return f"Context found: {context_text}... (Ollama needed for full response)"


__all__ = ["LLMRenderer"]

