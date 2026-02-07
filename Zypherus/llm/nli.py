"""Optional NLI contradiction checker.

Kept defensive and dependency-optional.
"""

from __future__ import annotations

import logging
import time
from typing import Dict


logger = logging.getLogger("ZYPHERUS")


class NLIContradictionChecker:
	"""Optional Natural Language Inference helper.

	Uses a HuggingFace sequence-classification NLI model (MNLI-style) to score
	contradiction/entailment between two short texts.

	Defensive:
	- Lazy-loads model on first use
	- Distinguishes missing-deps vs transient model failures
	- Allows retry on subsequent calls (or explicit reset)
	"""

	def __init__(
		self,
		model_name: str = "facebook/bart-large-mnli",
		contradiction_threshold: float = 0.80,
		max_length: int = 256,
		device: str = "cpu",
		retry_seconds: int = 300,
		bidirectional_margin: float = 0.70,
	):
		self.model_name = model_name
		self.contradiction_threshold = float(contradiction_threshold)
		self.max_length = int(max_length)
		self.device = (device or "cpu").strip().lower()
		self.retry_seconds = int(max(0, retry_seconds))
		self.bidirectional_margin = float(bidirectional_margin)

		self._tokenizer = None
		self._model = None

		self._dependency_missing = False
		self._load_failed = False
		self._last_error = None
		self._next_retry_ts = 0.0

	def reset(self):
		self._dependency_missing = False
		self._load_failed = False
		self._last_error = None
		self._next_retry_ts = 0.0
		self._tokenizer = None
		self._model = None

	def set_device(self, device: str):
		self.device = (device or "cpu").strip().lower()
		try:
			import torch
			if self._model is not None:
				if self.device == "cuda" and torch.cuda.is_available():
					self._model.to("cuda")
				else:
					self._model.to("cpu")
		except Exception:
			pass

	def _ensure_loaded(self) -> bool:
		if self._model is not None and self._tokenizer is not None:
			return True
		if self._dependency_missing:
			return False

		now = time.time()
		if self._load_failed and now < float(self._next_retry_ts or 0.0):
			return False

		try:
			from transformers import AutoModelForSequenceClassification, AutoTokenizer
			import torch
		except Exception:
			self._dependency_missing = True
			self._last_error = "dependency_missing"
			return False

		try:
			self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
			self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
			self._model.eval()
			try:
				if self.device == "cuda" and torch.cuda.is_available():
					self._model.to("cuda")
				else:
					self._model.to("cpu")
			except Exception:
				pass
			self._load_failed = False
			self._last_error = None
			return True
		except Exception:
			logger.exception("Failed to load NLI model; will retry later")
			self._load_failed = True
			self._last_error = "model_failed"
			self._next_retry_ts = time.time() + float(self.retry_seconds)
			self._tokenizer = None
			self._model = None
			return False

	def _label_probs(self, premise: str, hypothesis: str) -> Dict[str, float]:
		if not self._ensure_loaded():
			return {}
		try:
			import torch
		except Exception:
			return {}

		premise = (premise or "").strip()
		hypothesis = (hypothesis or "").strip()
		if not premise or not hypothesis:
			return {}

		try:
			enc = self._tokenizer(
				premise,
				hypothesis,
				return_tensors="pt",
				truncation=True,
				max_length=self.max_length,
			)
			with torch.no_grad():
				out = self._model(**enc)
				logits = out.logits[0]
				probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().tolist()

			id2label = getattr(getattr(self._model, "config", None), "id2label", None) or {}
			label_probs: Dict[str, float] = {}
			for i, p in enumerate(probs):
				raw = str(id2label.get(i, f"LABEL_{i}")).lower()
				if "contrad" in raw:
					label_probs["contradiction"] = float(p)
				elif "entail" in raw:
					label_probs["entailment"] = float(p)
				elif "neutral" in raw:
					label_probs["neutral"] = float(p)
				else:
					label_probs[raw] = float(p)
			return label_probs
		except Exception:
			return {}

	def contradiction_score(self, a: str, b: str) -> float:
		p1 = float(self._label_probs(a, b).get("contradiction", 0.0) or 0.0)
		if p1 >= float(self.contradiction_threshold) * float(self.bidirectional_margin):
			p2 = float(self._label_probs(b, a).get("contradiction", 0.0) or 0.0)
			return float(max(p1, p2))
		return p1

	def is_contradiction(self, a: str, b: str) -> bool:
		return self.contradiction_score(a, b) >= self.contradiction_threshold


__all__ = ["NLIContradictionChecker"]

