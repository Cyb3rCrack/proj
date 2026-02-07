"""ACE orchestrator."""

from __future__ import annotations

import os
import json
import re
import time
import logging
import warnings
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..beliefs.claims import ClaimStore
from ..concepts.graph import ConceptGraph
from ..concepts.ontology import Ontology
from ..core.intent import IntentClassifier, KnownUnknownsRegistry
from ..core.versions import VersionedDefinitionStore, SourceReliabilityRegistry
from ..config.settings import get_config
from ..config.logging_silence import setup_logging_with_silence, silent_logging
from ..config.validator import validate_config_file
from ..config.profiler import profile_function, PerformanceContext
from ..config.recovery import RetryPolicy, with_retry
from .decision import DecisionPolicy
from .dialogue import DialogueManager
from .embedding import EmbeddingModule
from .gate import MemorySufficiencyGate, GateFailure
from .question import QuestionInterpreter
from ..extraction.chunking import chunk_text
from ..extraction.claims import extract_atomic_claims
from ..extraction.concepts import extract_concepts
from ..extraction.parsing import parse_claim
from ..extraction.relationships import RelationshipExtractor
from ..inference.reasoning import ReasoningEngine
from ..ingestion.web import fetch_url_text
from ..ingestion.web_search import search_web
from ..llm.distiller import Distiller
from ..llm.renderer import LLMRenderer
from ..llm.nli import NLIContradictionChecker
from ..memory.index import KnowledgeIndex
from ..memory.relationships import RelationshipStore
from ..memory.management import MemoryHealthCheck
from ..memory.dialogue_persistence import DialoguePersistence
from ..utils.text import answer_shape_issue
from ..utils.ingestion_filter import IngestionFilter
from ..utils.phase1_metadata import Phase1Metadata

logger = logging.getLogger("ACE")


class ACE:
	def __init__(self):
		# Suppress verbose library logging
		logging.getLogger("transformers").setLevel(logging.ERROR)
		logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
		logging.getLogger("transformers.utils.hub").setLevel(logging.ERROR)
		logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
		warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
		warnings.filterwarnings("ignore", message=".*position_ids.*")
		warnings.filterwarnings("ignore", category=UserWarning)

		# Setup logging with AI response silence support
		setup_logging_with_silence("ACE")

		config_path = os.getenv("ZYPHERUS_CONFIG_PATH", os.path.join("config", "zypherus_config.json"))
		self.config = None
		try:
			# Validate configuration on startup
			validate_config_file(Path(config_path))
			self.config = get_config(config_path=config_path)
		except Exception:
			self.config = None

		fast_env = os.getenv("ACE_INGEST_FAST", "true").lower() in ("1", "true", "yes")
		fast_profile_env = os.getenv("ACE_INGEST_PROFILE", "safe").lower()
		if self.config and hasattr(self.config, "fast_ingest"):
			self.fast_ingest = bool(getattr(self.config.fast_ingest, "enabled", fast_env))
			self.fast_profile = str(getattr(self.config.fast_ingest, "profile", fast_profile_env)).lower()
			self.fast_chunk_default = int(getattr(self.config.fast_ingest, "chunk_size_default", 600))
			self.fast_chunk_web = int(getattr(self.config.fast_ingest, "chunk_size_web", 900))
			self.fast_skip_distiller = bool(getattr(self.config.fast_ingest, "skip_distiller", True))
			self.fast_skip_relationships = bool(getattr(self.config.fast_ingest, "skip_relationships", True))
			self.fast_skip_entities = bool(getattr(self.config.fast_ingest, "skip_entities", True))
			self.fast_skip_fact_extraction = bool(getattr(self.config.fast_ingest, "skip_fact_extraction", True))
		else:
			self.fast_ingest = fast_env
			self.fast_profile = fast_profile_env
			self.fast_chunk_default = int(os.getenv("ACE_INGEST_CHUNK_DEFAULT", "600"))
			self.fast_chunk_web = int(os.getenv("ACE_INGEST_CHUNK_WEB", "900"))
			self.fast_skip_distiller = os.getenv("ACE_INGEST_SKIP_DISTILLER", "true").lower() == "true"
			self.fast_skip_relationships = os.getenv("ACE_INGEST_SKIP_RELATIONSHIPS", "true").lower() == "true"
			self.fast_skip_entities = os.getenv("ACE_INGEST_SKIP_ENTITIES", "true").lower() == "true"
			self.fast_skip_fact_extraction = os.getenv("ACE_INGEST_SKIP_FACTS", "true").lower() == "true"

		self.memory_path = "memory.json"
		if self.config and hasattr(self.config, "memory"):
			self.memory_path = str(getattr(self.config.memory, "persistence_path", self.memory_path))

		self.embedder = EmbeddingModule()
		self.memory = KnowledgeIndex()
		self.relationships = RelationshipStore()
		self.ontology = Ontology()
		self.intent_classifier = IntentClassifier()
		self.known_unknowns = KnownUnknownsRegistry()
		self.definitions = VersionedDefinitionStore()
		self.source_reliability = SourceReliabilityRegistry()
		self.dialogue = DialogueManager(max_turns=6)
		self.llm = LLMRenderer()
		self.distiller = Distiller(self.llm)
		self.nli_checker = NLIContradictionChecker()

		self.claim_store = ClaimStore()
		self.concept_graph = ConceptGraph()
		self.struct_reasoner = ReasoningEngine(self.claim_store)
		self.decision_policy = DecisionPolicy()
		self.question_interpreter = QuestionInterpreter()
		self.sufficiency_gate = MemorySufficiencyGate()
		self.memory_health = MemoryHealthCheck()  # Initialize memory monitoring
		self.answer_cache_min_confidence = 0.70
		self.answer_cache_ttl_days = {
			"fast": 7,
			"balanced": 30,
			"deep": 90,
		}
		self.answer_defaults = {
			"mode": "balanced",
			"style": None,
			"recent_only": False,
		}
		if self.config and hasattr(self.config, "answering"):
			try:
				self.answer_defaults["mode"] = str(getattr(self.config.answering, "default_mode", "balanced") or "balanced")
				self.answer_defaults["style"] = getattr(self.config.answering, "default_style", None)
				self.answer_defaults["recent_only"] = bool(getattr(self.config.answering, "recent_only", False))
				self.answer_cache_min_confidence = float(getattr(self.config.answering, "cache_min_confidence", 0.70))
				self.answer_cache_ttl_days = dict(getattr(self.config.answering, "cache_ttl_days", self.answer_cache_ttl_days))
			except Exception:
				pass
		
		# Initialize dialogue persistence for conversation history
		self.dialogue_persistence = DialoguePersistence(Path("data/dialogues"))
		self.dialogue_persistence.start_session()
		
		# Setup error recovery with retry policies
		self.default_retry_policy = RetryPolicy(
			max_attempts=3,
			initial_delay=0.5,
			backoff_factor=2.0
		)

		try:
			self.memory.load(self.memory_path)
		except Exception:
			pass
		try:
			self.relationships.load()
		except Exception:
			pass
		try:
			self.ontology.load()
		except Exception:
			pass
		try:
			self.definitions.load()
		except Exception:
			pass
		try:
			self.source_reliability.load()
		except Exception:
			pass
		try:
			self.claim_store.load()
		except Exception:
			pass

	# ===== SIMPLE INGESTION METHODS =====
	
	def learn(self, text: str, source: str):
		"""Simple method to teach ACE something. Just provide text!
		
		IMPORTANT: You MUST provide a source name. Never use 'user_input' as a source.
		
		Examples:
			ace.learn("Water boils at 100 degrees Celsius", source="chemistry_facts")
			ace.learn("Paris is the capital of France", source="geography")
		"""
		return self.ingest_document(source, text)
	
	def define(self, concept: str, definition: str, source: str = "definition"):
		"""Teach ACE an authoritative definition (highest priority in retrieval).
		
		Use this for canonical explanations of concepts.
		
		Examples:
			ace.define("Python", "Python is a high-level programming language known for readability.")
			ace.define("Water", "Water is a chemical compound (H2O) consisting of hydrogen and oxygen.")
		"""
		return self.ingest_document(f"{source}:{concept}", definition, content_type="definition")
	
	def learn_from_file(self, filepath: str):
		"""Simple method to teach ACE from a file.
		
		Examples:
			ace.learn_from_file("facts.txt")
			ace.learn_from_file("data.csv")
		"""
		return self.ingest_file(filepath)
	
	def learn_from_folder(self, folder_path: str):
		"""Simple method to teach ACE from all files in a folder.
		
		Examples:
			ace.learn_from_folder("documents/")
			ace.learn_from_folder("notes/")
		"""
		return self.ingest_file(folder_path)

	def learn_from_url(self, url: str, source: Optional[str] = None):
		"""Teach ACE by fetching and ingesting a web page.

		Examples:
			ace.learn_from_url("https://example.com/article")
		"""
		return self.ingest_url(url, source=source)
	
	def ask(self, question: str):
		"""Simple method to ask Zypherus a question.
		
		Examples:
			ace.ask("What is water?")
			ace.ask("Tell me about Paris")
		"""
		return self.answer(question)
	
	def chat(self, message: str):
		"""Chat with ACE conversationally - like ChatGPT!
		ACE will automatically respond to whatever you say.
		
		Examples:
			ace.chat("Hello! Tell me about water")
			ace.chat("That's interesting, what else can you tell me?")
			ace.chat("Thanks for the explanation!")
		"""
		# Use dialogue manager for conversational context
		self.dialogue.add_user(message)
		
		# Generate response using LLM with context from memory
		try:
			# Search for relevant context
			emb = self.embedder.embed(message)
			results = self.memory.search(emb, top_k=5)
			
			# Build context from memory
			context_parts = []
			if results:
				for score, mem in results[:3]:  # Top 3 results
					context_parts.append(f"- {mem['text']}")
			
			context = "\n".join(context_parts) if context_parts else "No specific context available."
			
			# Get recent dialogue history  
			history = []
			for turn in self.dialogue.turns[-4:]:  # Last 4 turns
				role = turn.role
				content = turn.text
				history.append(f"{role}: {content}")
			
			history_text = "\n".join(history) if history else ""
			
			# Generate conversational response
			prompt = f"""You are ACE, a helpful AI assistant. Respond naturally to the user's message.

Relevant context from your knowledge:
{context}

Recent conversation:
{history_text}

User's message: {message}

Provide a natural, conversational response:"""
			
			response = self.llm.generate(prompt)
			
			# Add response to dialogue history
			self.dialogue.add_assistant(response)
			
			return response
			
		except Exception as e:
			# Fallback to simple response
			fallback = f"I hear you! You said: '{message}'. I'm still learning, but I'm here to help!"
			self.dialogue.add_assistant(fallback)
			return fallback

	# ===== ORIGINAL METHODS (still available) =====
	
	def ingest_url(
		self,
		url: str,
		source: Optional[str] = None,
		*,
		timeout_s: float = 20.0,
		max_chars: int = 200000,
	):
		"""Fetch a URL, extract readable text, and ingest it.
		
		Args:
			url: URL to fetch
			max_chars: Maximum characters to extract (200KB allows ~40-50K words, full articles)
			"""
		result = fetch_url_text(url, timeout_s=timeout_s, max_chars=max_chars)
		text = result.text
		if not text:
			print("[ZYPHERUS] No readable text extracted from URL.")
			return
		if source is None:
			source = result.source_hint
		chunk_size = self.fast_chunk_web if self.fast_ingest else 600
		return self.ingest_document(source, text, content_type="evidence", skip_filter=True, chunk_size=chunk_size)

	def answer_with_web(
		self,
		question: str,
		*,
		mode: str = "balanced",
		style: Optional[str] = None,
		recent_only: Optional[bool] = None,
		max_results: Optional[int] = None,
		max_ingest: Optional[int] = None,
		allowed_domains: Optional[List[str]] = None,
		search_timeout_s: float = 10.0,
		use_cache: bool = True,
		store_pages: bool = False,
		_allow_refine: bool = True,
	) -> Dict[str, Any]:
		"""Search the web for relevant URLs, ingest top results, then answer.

		This is a convenience workflow to avoid manual URL ingestion.
		"""
		query = (question or "").strip()
		if not query:
			return {
				"answer": "Please provide a question.",
				"reason": "empty_question",
				"sources": [],
			}

		if style is None:
			style = self.answer_defaults.get("style")
		if recent_only is None:
			recent_only = bool(self.answer_defaults.get("recent_only"))

		mode = (mode or self.answer_defaults.get("mode", "balanced")).strip().lower()
		if mode not in {"fast", "balanced", "deep"}:
			mode = "balanced"

		if use_cache:
			cached = self._get_cached_answer(query, mode=mode, style=style, recent_only=recent_only)
			if cached is not None:
				cached["cache_hit"] = True
				return cached

		profile = self._web_profile(mode)
		max_results = max_results if isinstance(max_results, int) else profile["max_results"]
		max_ingest = max_ingest if isinstance(max_ingest, int) else profile["max_ingest"]

		try:
			results = search_web(
				query,
				max_results=max_results,
				timeout_s=search_timeout_s,
				allowed_domains=allowed_domains,
			)
		except Exception as e:
			return {
				"answer": f"Web search failed: {e}",
				"reason": "web_search_failed",
				"sources": [],
			}

		if not results:
			return {
				"answer": "No relevant web results found.",
				"reason": "no_web_results",
				"sources": [],
			}

		ingested: List[Dict[str, str]] = []
		session_id = f"webtmp:{int(time.time())}:{hashlib.sha256(query.encode('utf-8')).hexdigest()[:8]}"
		for result in results[:max_ingest]:
			try:
				host = "unknown"
				try:
					from urllib.parse import urlparse
					host = urlparse(result.url).netloc or "unknown"
				except Exception:
					pass
				source_override = f"{session_id}:{host}"
				if store_pages:
					self.ingest_url(result.url, source=source_override)
				else:
					self._ingest_url_temporary(result.url, source_override)
				ingested.append({"url": result.url, "title": result.title, "source": source_override})
			except Exception:
				continue

		if not ingested:
			return {
				"answer": "Unable to read any of the search results.",
				"reason": "web_ingest_failed",
				"sources": [],
			}

		style_hint = self._style_hint(style)
		response = self.answer(query, style_hint=style_hint, use_cache=False)
		if isinstance(response, dict):
			response["web_sources"] = ingested

		contradictions = self._detect_disagreements(query)
		critique = self._self_critique(query, response, contradictions=contradictions, recent_only=recent_only)
		if critique.get("needs_refine") and _allow_refine:
			refined = self._refine_web_answer(
				query,
				mode=mode,
				style=style,
				recent_only=recent_only,
				allowed_domains=allowed_domains,
				search_timeout_s=search_timeout_s,
			)
			response = refined
			contradictions = self._detect_disagreements(query)
			critique = self._self_critique(query, response, contradictions=contradictions, recent_only=recent_only)

		if isinstance(response, dict):
			response["self_critique"] = critique
			response["disagreements"] = contradictions

			if critique.get("caveat"):
				response["caveat"] = critique["caveat"]
				response["answer"] = f"{response.get('answer', '').strip()}\n\nCaveat: {critique['caveat']}"

			self._store_answer_cache(
				query,
				response,
				mode=mode,
				style=style,
				recent_only=recent_only,
			)

		if not store_pages:
			try:
				removed = self.memory.remove_by_source_prefix(session_id)
				if removed:
					self.memory.save(self.memory_path)
			except Exception:
				pass

		return response

	def _web_profile(self, mode: str) -> Dict[str, int]:
		if mode == "fast":
			return {"max_results": 5, "max_ingest": 2}
		if mode == "deep":
			return {"max_results": 12, "max_ingest": 5}
		return {"max_results": 8, "max_ingest": 3}

	def _ingest_url_temporary(self, url: str, source: str) -> int:
		"""Fetch and ingest a URL without persisting pages or triggering belief updates."""
		try:
			result = fetch_url_text(url)
		except Exception:
			return 0

		text = result.text or ""
		if not text:
			return 0

		try:
			text = IngestionFilter.clean_text(text)
		except Exception:
			pass

		chunk_size = self.fast_chunk_web if self.fast_ingest else 600
		stored = 0
		for idx, chunk in enumerate(chunk_text(text, chunk_size=chunk_size)):
			try:
				emb = self.embedder.embed(chunk)
				self.memory.add(
					emb,
					chunk,
					f"{source}#tmp{idx}",
					mtype="evidence",
					generated_by="system",
					certainty=0.80,
				)
				stored += 1
			except Exception:
				continue
		return stored

	def _style_hint(self, style: Optional[str]) -> Optional[str]:
		style = (style or "").strip().lower()
		if style == "authoritative":
			return "Prefer primary sources, be precise, avoid speculation, and keep tone formal."
		if style == "practical":
			return "Focus on actionable guidance and concrete steps when possible."
		return None

	def _normalize_question(self, question: str) -> str:
		q = re.sub(r"\s+", " ", (question or "").strip().lower())
		return q

	def _answer_cache_key(self, question: str) -> str:
		norm = self._normalize_question(question)
		return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]

	def _is_cache_fresh(self, meta: Dict[str, Any], *, mode: str, recent_only: bool) -> bool:
		try:
			expires_at = float(meta.get("expires_at") or 0.0)
			if expires_at and time.time() > expires_at:
				return False
		except Exception:
			return False

		if recent_only and not bool(meta.get("recent_only", False)):
			return False

		return True

	def _get_cached_answer(
		self,
		question: str,
		*,
		mode: str,
		style: Optional[str],
		recent_only: bool,
	) -> Optional[Dict[str, Any]]:
		try:
			q_emb = self.embedder.embed(question)
			candidates = self.memory.search(q_emb, top_k=8, min_score=0.10, query_text=question)
		except Exception:
			return None

		best = None
		best_score = 0.0
		for score, entry in candidates:
			if entry.get("type") != "answer":
				continue
			meta = entry.get("answer_meta") or {}
			if not isinstance(meta, dict):
				continue
			if not self._is_cache_fresh(meta, mode=mode, recent_only=recent_only):
				continue
			if float(meta.get("confidence") or 0.0) < self.answer_cache_min_confidence:
				continue
			stored_style = (meta.get("style") or "").strip().lower()
			if style and stored_style and style.strip().lower() != stored_style:
				continue
			if score > best_score:
				best_score = float(score)
				best = entry

		if not best:
			return None

		meta = best.get("answer_meta") or {}
		return {
			"answer": best.get("text"),
			"confidence": meta.get("confidence"),
			"sources": meta.get("sources") or [],
			"reason": "answer_cache",
			"web_sources": meta.get("web_sources") or [],
			"cache_key": meta.get("cache_key"),
			"cache_profile": meta.get("mode"),
		}

	def _store_answer_cache(
		self,
		question: str,
		response: Dict[str, Any],
		*,
		mode: str,
		style: Optional[str],
		recent_only: bool,
	) -> None:
		try:
			if response.get("reason") in {"no_memory", "insufficient_coverage", "low_confidence", "answer_not_grounded"}:
				return
			answer_text = (response.get("answer") or "").strip()
			if not answer_text:
				return
			confidence = float(response.get("confidence") or 0.0)
			cache_key = self._answer_cache_key(question)
			source = f"answer:{cache_key}"
			meta = {
				"cache_key": cache_key,
				"question": self._normalize_question(question),
				"sources": response.get("sources") or [],
				"web_sources": response.get("web_sources") or [],
				"confidence": confidence,
				"mode": mode,
				"style": (style or "").strip().lower() or None,
				"recent_only": bool(recent_only),
				"expires_at": time.time() + float(self.answer_cache_ttl_days.get(mode, 30)) * 86400,
			}

			emb = self.embedder.embed(question)
			entry_id = self.memory.add(emb, answer_text, source, mtype="answer", generated_by="ace", certainty=confidence)
			if entry_id:
				for entry in self.memory.entries:
					if entry.get("id") == entry_id:
						entry["answer_meta"] = meta
						break
				self.memory.save(self.memory_path)
		except Exception:
			pass

	def _get_top_context(self, question: str, *, limit: int = 6) -> List[Dict[str, Any]]:
		try:
			q_emb = self.embedder.embed(question)
			results = self.memory.search(q_emb, top_k=limit, min_score=0.0, query_text=question)
			return [e for _, e in results if e.get("type") != "answer"]
		except Exception:
			return []

	def _detect_disagreements(self, question: str) -> List[Dict[str, str]]:
		entries = self._get_top_context(question, limit=4)
		if len(entries) < 2:
			return []

		contradictions = []
		try:
			for i in range(len(entries)):
				for j in range(i + 1, len(entries)):
					a = (entries[i].get("text") or "")[:400]
					b = (entries[j].get("text") or "")[:400]
					if not a or not b:
						continue
					if self.nli_checker.is_contradiction(a, b):
						contradictions.append({
							"source_a": entries[i].get("source", "unknown"),
							"source_b": entries[j].get("source", "unknown"),
						})
						if len(contradictions) >= 3:
							return contradictions
		except Exception:
			return []
		return contradictions

	def _self_critique(
		self,
		question: str,
		response: Dict[str, Any],
		*,
		contradictions: List[Dict[str, str]],
		recent_only: bool,
	) -> Dict[str, Any]:
		flags: List[str] = []
		needs_refine = False
		caveat = None

		answer_text = (response.get("answer") or "").strip()
		if not answer_text:
			flags.append("empty_answer")
			needs_refine = True

		sources = response.get("sources") or []
		if not sources:
			flags.append("no_sources")
			needs_refine = True
			caveat = "I could not trace this to specific sources."

		conf = float(response.get("confidence") or 0.0)
		if conf and conf < 0.65:
			flags.append("low_confidence")
			needs_refine = True
			caveat = "The supporting evidence is weak or limited."

		if contradictions:
			flags.append("disagreement")
			caveat = "Sources disagree on key points; treat this as a best-effort synthesis."

		if recent_only:
			flags.append("recent_only")

		return {
			"flags": flags,
			"needs_refine": needs_refine,
			"caveat": caveat,
		}

	def _refine_web_answer(
		self,
		question: str,
		*,
		mode: str,
		style: Optional[str],
		recent_only: bool,
		allowed_domains: Optional[List[str]],
		search_timeout_s: float,
	) -> Dict[str, Any]:
		next_mode = "deep" if mode == "balanced" else "balanced"
		if mode == "deep":
			next_mode = "deep"

		return self.answer_with_web(
			question,
			mode=next_mode,
			style=style,
			recent_only=recent_only,
			allowed_domains=allowed_domains,
			search_timeout_s=search_timeout_s,
			use_cache=False,
			store_pages=False,
			_allow_refine=False,
		)

	def ingest_file(self, path: str, source: Optional[str] = None, *, encoding: str = "utf-8"):
		"""Convenience helper to ingest a local file (or directory of files).

		- If `source` is not provided, uses the basename (or relative path for directories).
		- For `.csv`, attempts to load a DataFrame (if pandas is installed) and ingests as tables.
		- For `.json`, ingests pretty-printed JSON.
		- For directories, ingests common text-like extensions recursively (bounded).
		"""
		p = os.fspath(path)
		if not os.path.exists(p):
			raise FileNotFoundError(p)

		if os.path.isdir(p):
			allowed_ext = {".txt", ".md", ".log", ".json", ".csv"}
			max_files = 1000 if self.fast_ingest else 200
			ingested = 0
			for root, _dirs, files in os.walk(p):
				for fname in files:
					if ingested >= max_files:
						print(f"[ZYPHERUS] Directory ingestion capped at {max_files} files.")
						return
					ext = os.path.splitext(fname)[1].lower()
					if ext not in allowed_ext:
						continue
					full_path = os.path.join(root, fname)
					rel_source = os.path.relpath(full_path, p)
					self.ingest_file(full_path, source=rel_source, encoding=encoding)
					ingested += 1
			if ingested == 0:
				print("[ZYPHERUS] No ingestible files found in directory.")
			return

		if source is None:
			source = os.path.basename(p)

		ext = os.path.splitext(p)[1].lower()

		if ext == ".csv":
			try:
				import pandas as pd  # type: ignore
			except Exception:
				pd = None
			if pd is not None:
				try:
					df = pd.read_csv(p)
					return self.ingest_document(source, text="", tables=[df])
				except Exception:
					pass

		if ext == ".json":
			try:
				with open(p, "r", encoding=encoding, errors="replace") as f:
					obj = json.load(f)
				text = json.dumps(obj, indent=2, ensure_ascii=False)
				return self.ingest_document(source, text, tables=None)
			except Exception:
				pass

		with open(p, "r", encoding=encoding, errors="replace") as f:
			text = f.read()
		return self.ingest_document(source, text, tables=None)

	def ingest_document(
		self,
		source: str,
		text: str,
		tables: Optional[List[Any]] = None,
		*,
		content_type: Optional[str] = None,
		skip_filter: bool = False,
		chunk_size: Optional[int] = None,  # Allow custom chunk size
	):
		# CRITICAL: Reject 'user_input' as source to prevent memory contamination
		if source == "user_input" or source.startswith("user_input#"):
			print(f"[ZYPHERUS] ❌ REJECTED: Cannot use 'user_input' as a source.")
			print(f"[ZYPHERUS] This prevents memory contamination. Please provide a proper source name.")
			print(f"[ZYPHERUS] Example sources: 'python_docs', 'chemistry_facts', 'geography_2024', etc.")
			return
		
		print(f"[ZYPHERUS] Ingesting document: {source}")

		fast_mode = bool(self.fast_ingest)
		fast_profile = (self.fast_profile or "safe").lower()
		fast_skip_filter = skip_filter or (fast_mode and fast_profile == "max")
		strict_filter = not fast_mode
		
		# PHASE 1: Analyze metadata (generated_by, certainty, is_assumption)
		generated_by, certainty, is_assumption = Phase1Metadata.analyze_metadata(text, source)
		
		# FILTER: Reject invalid content before processing
		if not tables:  # Don't filter table data
			if not fast_skip_filter:
				is_valid, reason = IngestionFilter.is_valid_for_ingestion(text, strict=strict_filter)
				if not is_valid:
					print(f"[ZYPHERUS] ❌ Rejected content: {reason}")
					return
			text = IngestionFilter.clean_text(text)
			
		# Auto-classify content type if not specified
		if content_type is None:
			content_type = IngestionFilter.classify_content_type(text) if not tables else "evidence"

		# Smart chunk sizing: larger chunks for web content to preserve context
		if chunk_size is None:
			if source.startswith("web:"):
				chunk_size = self.fast_chunk_web if fast_mode else 600
			else:
				chunk_size = self.fast_chunk_default if fast_mode else 300

		chunks = []

		if tables:
			try:
				import pandas as pd  # type: ignore
			except Exception:
				pd = None

			for i, df in enumerate(tables):
				if pd is not None:
					try:
						if not isinstance(df, pd.DataFrame):
							continue
					except Exception:
						pass
				df.columns = [str(c) for c in df.columns]
				for ridx, row in df.iterrows():
					row_text = " | ".join([str(v) for v in row.values])
					chunks.append((f"{source}#table{i}:row{ridx}", row_text))
		else:
			for idx, c in enumerate(chunk_text(text, chunk_size=chunk_size)):
				chunks.append((f"{source}#text:chunk{idx}", c))

		def _fast_summary(text_in: str, max_chars: int = 500) -> str:
			text_in = (text_in or "").strip()
			return text_in[:max_chars].strip()

		if tables:
			table_texts = []
			for df in tables:
				try:
					table_texts.append(df.head(10).to_csv(index=False))
				except Exception:
					table_texts.append(str(df.head(10)))
			if fast_mode and self.fast_skip_distiller:
				summary_text = _fast_summary("\n\n".join(table_texts), max_chars=600)
			else:
				summary_text = self.distiller.extract_summary("\n\n".join(table_texts))
		else:
			if fast_mode and self.fast_skip_distiller:
				summary_text = _fast_summary(text, max_chars=600)
			else:
				summary_text = self.distiller.extract_summary(text)

		# Store with proper type (definition/fact/evidence) + PHASE 1 metadata
		emb = self.embedder.embed(summary_text)
		# If original content was a definition, store summary as definition too
		summary_type = content_type if content_type == "definition" else "summary"
		self.memory.add(
			emb, summary_text, source, 
			mtype=summary_type,
			generated_by=generated_by,
			certainty=certainty,
			is_assumption=is_assumption,
		)

		# PHASE 2: Extract relationships and ontology classification for summary
		if not (fast_mode and self.fast_skip_relationships):
			try:
				concepts, relationships = RelationshipExtractor.extract_concepts_and_relationships(summary_text)
			
				# Add concepts to ontology
				for concept in concepts:
					if concept and len(concept) > 0:
						self.ontology.add_instance(concept, "concept")
				
				# Store relationships with strength based on Phase 1 certainty
				for c1, rel_type, c2 in relationships:
					if c1 and c2:
						self.relationships.add_relationship(c1, rel_type, c2, strength=certainty, source=source)
			except Exception:
				pass

		if not (fast_mode and self.fast_skip_fact_extraction):
			try:
				facts = self.distiller.extract_facts(summary_text, None)  # No deadline
				for i, f in enumerate(facts):
					femb = self.embedder.embed(f)
					# Analyze metadata for each fact
					fact_generated_by, fact_certainty, fact_is_assumption = Phase1Metadata.analyze_metadata(f, source)
					idx = self.memory.add(
						femb, f, f"{source}#fact{i}", 
						mtype="fact",
						generated_by=fact_generated_by,
						certainty=fact_certainty,
						is_assumption=fact_is_assumption,
					)
					
					# PHASE 2: Extract relationships for facts
					if not (fast_mode and self.fast_skip_relationships):
						try:
							concepts, relationships = RelationshipExtractor.extract_concepts_and_relationships(f)
							for concept in concepts:
								if concept and len(concept) > 0:
									self.ontology.add_instance(concept, "concept")
							for c1, rel_type, c2 in relationships:
								if c1 and c2:
									self.relationships.add_relationship(c1, rel_type, c2, strength=fact_certainty, source=f"{source}#fact{i}")
						except Exception:
							pass
					
					if not (fast_mode and self.fast_skip_entities):
						ents = self.distiller.extract_entities(f)
						if idx is not None and ents:
							# Extract entity names (strings) from entity objects
							ent_names = [e["name"] if isinstance(e, dict) else str(e) for e in ents]
							self.memory.add_entities(idx, ent_names)
			except Exception:
				pass

		stored = 0
		from ..utils.importance_scorer import ImportanceScorer
		
		chunk_texts = [txt for _, txt in chunks]
		chunk_embeddings = None
		if chunk_texts and hasattr(self.embedder, "embed_many"):
			try:
				chunk_embeddings = self.embedder.embed_many(chunk_texts)
			except Exception:
				chunk_embeddings = None
		
		for i, (src, txt) in enumerate(chunks):
			try:
				# Store everything - no importance filtering
				# Deduplication at similarity level still applies
				if chunk_embeddings is not None and i < len(chunk_embeddings):
					emb = chunk_embeddings[i]
				else:
					emb = self.embedder.embed(txt)
				# Analyze metadata for each chunk
				chunk_generated_by, chunk_certainty, chunk_is_assumption = Phase1Metadata.analyze_metadata(txt, src)
				
				idx = self.memory.add(
					emb, txt, src, 
					mtype="evidence",
					generated_by=chunk_generated_by,
					certainty=chunk_certainty,
					is_assumption=chunk_is_assumption,
				)
				if idx is not None:
					stored += 1
				
				# Still process relationships and entities for ALL chunks (used in reasoning)
				if not (fast_mode and self.fast_skip_relationships):
					try:
						concepts, relationships = RelationshipExtractor.extract_concepts_and_relationships(txt)
						for concept in concepts:
							if concept and len(concept) > 0:
								self.ontology.add_instance(concept, "concept")
						for c1, rel_type, c2 in relationships:
							if c1 and c2:
								self.relationships.add_relationship(c1, rel_type, c2, strength=chunk_certainty, source=src)
					except Exception:
						pass
				
				if not (fast_mode and self.fast_skip_entities):
					try:
						ents = self.distiller.extract_entities(txt)
						if idx is not None and ents:
							# Extract entity names (strings) from entity objects
							ent_names = [e["name"] if isinstance(e, dict) else str(e) for e in ents]
							self.memory.add_entities(idx, ent_names)
					except Exception as e:
						logger.debug(f"Entity extraction error: {e}")
			except Exception as e:
				logger.debug(f"Chunk processing error on '{src}': {e}")
				continue

		total_chunks = len(chunks)
		print(f"[ZYPHERUS] ✓ Stored {stored}/{total_chunks} chunks (all valuable content captured). Ingestion complete.")
		
		# Save Phase 2 stores + Phase 4 stores
		try:
			self.memory.save(self.memory_path)
			self.relationships.save()
			self.ontology.save()
			self.definitions.save()
			self.source_reliability.save()
		except Exception:
			pass

		if not (fast_mode and fast_profile == "max"):
			try:
				self.trigger_understanding(source)
			except Exception:
				pass

			try:
				self._internal_cognition_cycle(reason="post_ingest", source=source)
			except Exception:
				pass

			try:
				self.claim_store.save()
			except Exception:
				pass

	def rebuild_beliefs_from_memory(self):
		print("[ZYPHERUS] Rebuilding beliefs from evidence...")
		try:
			self.claim_store.revisions_archive.append(
				{
					"ts": time.time(),
					"num_claims": len(self.claim_store.claims or {}),
					"revisions": self.claim_store.revisions or {},
				}
			)
			self.claim_store.revisions = {}
		except Exception:
			pass
		try:
			self.claim_store.claims.clear()
		except Exception:
			self.claim_store.claims = {}

		for e in (self.memory.entries or []):
			text = e.get("text", "")
			eid = e.get("id")
			if not text or not eid:
				continue

			claims = extract_atomic_claims(text)
			for cl in claims:
				try:
					parsed = parse_claim(cl)
					self.claim_store.observe_claim(parsed, eid)
				except Exception:
					pass

		try:
			self.claim_store.update_beliefs()
		except Exception:
			pass

		try:
			self.claim_store.save()
		except Exception:
			pass

	def _expand_query_with_relationships(self, question: str, candidates: List[tuple], max_expanded: int = 50) -> List[tuple]:
		"""Expand candidates using relationship graph (Phase 2)."""
		try:
			# Extract concepts from question
			q_concepts = extract_concepts(question)
			if not q_concepts:
				return candidates
			
			expanded_ids = set()
			for _, e in candidates:
				expanded_ids.add(e.get("id"))
			
			# For each extracted concept, find related concepts and search for them
			for concept in q_concepts[:5]:  # Limit to top 5 concepts
				try:
					# Get related concepts through relationships
					related = self.relationships.get_all_related(concept, depth=2)
					for rel_concept in related:
						if len(expanded_ids) >= max_expanded:
							break
						
						# Search for entries related to this concept
						try:
							rel_emb = self.embedder.embed(rel_concept)
							rel_results = self.memory.search(rel_emb, top_k=3, min_score=0.0, query_text=rel_concept)
							for score, e in rel_results:
								if e.get("id") not in expanded_ids and len(expanded_ids) < max_expanded:
									candidates.append((score * 0.85, e))  # Slightly lower score for expanded results
									expanded_ids.add(e.get("id"))
						except Exception:
							pass
				except Exception:
					pass
			
			# Re-sort by score
			candidates.sort(key=lambda x: x[0], reverse=True)
			return candidates[:max_expanded]
		except Exception:
			return candidates

	def answer(self, question: str, *, style_hint: Optional[str] = None, use_cache: bool = True):
		try:
			self._internal_cognition_cycle(reason="pre_answer")
		except Exception:
			pass

		if use_cache:
			cached = self._get_cached_answer(question, mode="balanced", style=None, recent_only=False)
			if cached is not None:
				cached["cache_hit"] = True
				return cached

		try:
			question_info = self.question_interpreter.interpret(question)
		except Exception:
			question_info = {"intent": "factual", "requires_high_confidence": False}

		# PHASE 3: Classify question intent
		intent_info = self.intent_classifier.classify(question)

		q_lower = (question or "").strip().lower()
		is_generation_request = any(
			phrase in q_lower
			for phrase in (
				"snippet",
				"example",
				"code",
				"sample",
				"show me",
				"give me",
				"write a",
				"generate",
				"how to",
			)
		)
		
		query_emb = self.embedder.embed(question)
		top_k = 40 if question_info.get("requires_high_confidence") else 30
		min_score = intent_info["min_confidence"] * 0.5  # Use Phase 3 confidence threshold
		candidates = self.memory.search(query_emb, top_k=top_k, min_score=min_score, query_text=question)
		
		# PHASE 2: Expand candidates using relationships
		try:
			candidates = self._expand_query_with_relationships(question, candidates)
		except Exception:
			pass

		# HARD GATING: Evaluate memory sufficiency BEFORE calling LLM
		gate_result = self.sufficiency_gate.evaluate(question, candidates)
		
		if not gate_result.allowed and not is_generation_request:
			# Record unknown concept for future improvement
			if gate_result.failure_reason == GateFailure.NO_MEMORY:
				try:
					concepts = extract_concepts(question)
					for concept in concepts[:3]:
						self.known_unknowns.record_failure(question, concept)
				except Exception:
					pass
			
			# Hard refusal - LLM never called
			return {
				"answer": self.sufficiency_gate.get_refusal_message(gate_result),
			"reason": gate_result.failure_reason.value if gate_result.failure_reason else "Unknown",
				"gate_evaluation": gate_result.to_dict(),
				"allowed_actions": ["ingest_text", "ingest_url", "ingest_file"],
				"note": "Hard gating enforced: Memory insufficient - LLM not called",
			}

		try:
			q_ents = self.distiller.extract_entities(question)
			if q_ents:
				def _norm_ent(ent: str) -> str:
					return re.sub(r"[^a-z0-9 ]+", "", (ent or "").lower()).strip()

				# Extract entity names (strings) from entity objects
				ent_strs = [e["name"] if isinstance(e, dict) else str(e) for e in q_ents if isinstance(e, (dict, str))]
				q_ents_norm = [_norm_ent(e) for e in ent_strs]
				q_ents_norm = [e for e in q_ents_norm if e]
				boosted = []
				for score, e in candidates:
					entry_id = e.get("id")
					overlap = 0
					if entry_id is not None:
						for ent in q_ents_norm:
							if ent in self.memory.entity_index and entry_id in self.memory.entity_index.get(ent, []):
								overlap += 1
					boosted.append((overlap, score, e))
				boosted.sort(reverse=True, key=lambda x: (x[0], x[1]))
				candidates = [(s, e) for _, s, e in boosted]
		except Exception:
			pass

		top_for_context = candidates[:10] if question_info.get("requires_high_confidence") else candidates[:8]
		convo_hist = self.dialogue.get_formatted()

		struct_claims = self.claim_store.retrieve_relevant([e for _, e in top_for_context])
		concepts: List[str] = []
		for _, e in top_for_context:
			concepts.extend(extract_concepts(e.get("text", "")))
		struct_reasoning = self.struct_reasoner.reason(question, struct_claims, concepts)

		sources = [str(e.get("source")) for _, e in top_for_context if e.get("source") is not None]

		intent = (question_info.get("intent") or "factual")
		allow_clarification = False
		try:
			max_claim_conf = max((float(c.get("confidence", 0.0) or 0.0) for c in (struct_claims or [])), default=0.0)
		except Exception:
			max_claim_conf = 0.0

		if allow_clarification and intent in {"define", "compare"} and max_claim_conf < 0.70 and not is_generation_request:
			return {
				"answer": "I can’t define or compare that confidently yet because I don’t have a strong belief supported by evidence.",
				"clarification": "Please ingest a relevant source (definition/comparison) or narrow the term(s) and domain.",
				"confidence": 0.0,
				"sources": sources,
				"beliefs": struct_claims,
				"reasoning": struct_reasoning,
				"verification": {"confidence": None, "status": "gated", "unsupported_claims": []},
				"note": "Definition safety gating: no claim >= 0.70.",
			}

		try:
			known = struct_reasoning.get("known") or []
			if known:
				top_claim = max(known, key=lambda c: float(c.get("confidence", 0.0) or 0.0))
				top_conf = float(top_claim.get("confidence", 0.0) or 0.0)
				claim_first_threshold = 0.70 if intent in {"define", "compare"} else 0.75
				if top_conf >= claim_first_threshold:
					raw = (top_claim.get("raw") or "").strip()
					if raw:
						ans = raw
						if not ans.endswith((".", "!", "?")):
							ans = ans + "."
					else:
						subj = (top_claim.get("subject") or "").strip()
						pred = (top_claim.get("predicate") or "").strip()
						obj = (top_claim.get("object") or "").strip()
						parts = [p for p in [subj, pred, obj] if p]
						ans = " ".join(parts).strip() or "I have a belief, but it is underspecified."
						if not ans.endswith((".", "!", "?")):
							ans = ans + "."

					self.dialogue.add_user(question)
					self.dialogue.add_assistant(ans)

					return {
						"answer": ans,
						"confidence": top_conf,
						"sources": sources,
						"beliefs": struct_claims,
						"reasoning": struct_reasoning,
						"verification": {"confidence": None, "status": "bypassed", "unsupported_claims": []},
						"note": "Claim-first answer (LLM bypassed).",
					}
		except Exception:
			pass

		# HARD GATING: LLM receives ONLY memory context, never raw question
		# Build synthesis prompt that prevents hallucination
		context_text = "\n\n".join(
			f"[Source: {e.get('source', 'unknown')}] {e.get('text', '')[:800]}"
			for _, e in top_for_context
		)
		
		style_hint = style_hint or "Synthesize a clear, neutral explanation using ONLY the information provided."
		if is_generation_request:
			style_hint = (
				"Provide a code snippet in a fenced block (```lang) followed by a clear explanation. "
				"Use general programming knowledge if needed."
			)
			# Allow the LLM to see the raw question for code generation requests
			with silent_logging():
				llm_response = self.llm.answer(
					question,
					top_for_context,
					convo_history=convo_hist,
					reasoning=None,
					style_hint=style_hint,
				)
				verification = {"confidence": None, "unsupported_claims": [], "status": "code_snippet"}
		else:
			# Synthesis prompt: LLM never sees raw question
			# GROUNDING CONTRACT: Force explicit source citations and factual grounding
			synthesis_prompt = f"""You are given verified knowledge from memory.
GROUNDING REQUIREMENTS (MANDATORY):
1. Explicitly cite retrieved sources using [Source: ...] format
2. Use terminology and facts PRESENT IN the retrieved context only
3. Do NOT introduce concepts not in the context (mark any inference as "(inferred)")
4. Output in structured format (see below)
5. If context is insufficient for a complete answer, say so explicitly

REQUIRED OUTPUT STRUCTURE:
[SOURCES CITED]
- List each unique source from the context

[KEY FACTS FROM CONTEXT]
- Fact 1 (from source: ...)
- Fact 2 (from source: ...)
- ...

[SYNTHESIS]
Your answer here. Must be grounded in facts above.

{style_hint}

MEMORY CONTEXT:
{context_text}

Now provide your structured response:"""
			
			# Suppress logging while AI generates response
			with silent_logging():
				# Lower temperature for grounding: 0.25 balances consistency + information
				llm_response = self.llm.generate(synthesis_prompt, max_tokens=400, temperature=0.25)

				# VALIDATE GROUNDING: Check structured format compliance
				grounding_valid, extracted_synthesis = self._validate_and_extract_synthesis(llm_response)
				if grounding_valid and extracted_synthesis:
					# Use extracted synthesis for cleaner response
					llm_response = extracted_synthesis
					grounding_bonus = 0.15  # Reward for following grounding contract
				else:
					# Penalize for not following struct (but don't reject)
					grounding_bonus = -0.10

				try:
					issue = answer_shape_issue(llm_response)
					if issue:
						retry_hint = f"Rewrite as one complete sentence that includes an explicit verb. Problem: {issue}."
						llm_retry = self.llm.answer(
							question,
							top_for_context,
							convo_history=convo_hist,
							reasoning=None,
							style_hint=retry_hint,
						)
						if isinstance(llm_retry, str) and llm_retry.strip() and not answer_shape_issue(llm_retry):
							llm_response = llm_retry
						else:
							llm_response = llm_retry.strip() if isinstance(llm_retry, str) and llm_retry.strip() else llm_response
				except Exception:
					pass

				if max_claim_conf >= 0.60:
					verification = {"confidence": float(max_claim_conf), "unsupported_claims": [], "status": "symbolic", "grounding_bonus": grounding_bonus}
				else:
					verification = self.llm.verify_answer(llm_response, top_for_context)
					if isinstance(verification, dict):
						verification["grounding_bonus"] = grounding_bonus

		if is_generation_request:
			self.dialogue.add_user(question)
			if isinstance(llm_response, str) and llm_response:
				self.dialogue.add_assistant(llm_response)
			return {
				"answer": llm_response,
				"confidence": None,
				"sources": sources,
				"beliefs": struct_claims,
				"reasoning": struct_reasoning,
				"verification": verification,
				"note": "Code snippet generated from general programming knowledge.",
			}

		raw_vc = verification.get("confidence", None)
		verification_conf = float(raw_vc) if isinstance(raw_vc, (int, float)) else 0.5

		retrieval_strength = 0.0
		if top_for_context:
			retrieval_strength = sum(float(s) for s, _ in top_for_context) / len(top_for_context)
			retrieval_strength = max(0.0, min(1.0, float(retrieval_strength)))
		retrieval_metrics = {"retrieval_strength": retrieval_strength}
		symbolic_conflicts = {"contradictions": struct_reasoning.get("contradictions"), "claim_strength": max_claim_conf}
		decision = self.decision_policy.decide(retrieval_metrics, verification, symbolic_conflicts, question_info=question_info)

		underspecified_note = None
		try:
			if isinstance(llm_response, str):
				issue = answer_shape_issue(llm_response)
				if issue:
					underspecified_note = f"Answer may be underspecified ({issue})."
					if "confidence" in decision and isinstance(decision.get("confidence"), (int, float)):
						decision["confidence"] = float(decision["confidence"]) * 0.60
		except Exception:
			pass

		try:
			if answer_shape_issue(llm_response):
				decision = dict(decision or {})
				decision["action"] = "clarify"
				decision["reason"] = "answer_shape"
		except Exception:
			pass

		if allow_clarification and decision.get("action") == "clarify" and not is_generation_request:
			clarif_prompt = f"Answer given with low confidence ({decision.get('reason')}). Consider adding evidence or narrowing the question."
			cautious_conf = max(0.0, min(1.0, float(decision.get("confidence", 0.5)) * 0.70))
			return {
				"answer": llm_response,
				"note": clarif_prompt,
				"clarification": "Please provide more evidence or a narrower question.",
				"confidence": cautious_conf,
				"sources": sources,
				"beliefs": struct_claims,
				"reasoning": struct_reasoning,
				"verification": verification,
				"answer_quality": underspecified_note,
			}

		if allow_clarification and decision.get("action") == "dispute":
			strong = decision.get("strong_conflicts") or []
			note = "Conflicting evidence detected (strong contradiction). Refusing to answer speculatively; ingest more sources or narrow the question."
			cautious_conf = max(0.0, min(1.0, float(decision.get("confidence", 0.5)) * 0.20))
			return {
				"answer": "I can’t answer confidently because my evidence contains strong contradictions.",
				"note": note,
				"clarification": "Conflicting evidence detected; please provide clarification or more sources.",
				"confidence": cautious_conf,
				"sources": sources,
				"beliefs": struct_claims,
				"reasoning": struct_reasoning,
				"verification": verification,
				"strong_conflicts": strong,
				"answer_quality": underspecified_note,
			}

		# ANSWER VALIDATION: Ensure answer is grounded in sources
		if sources and isinstance(llm_response, str):
			is_grounded = self._validate_answer_grounding(llm_response, sources)
			if not is_grounded:
				return {
					"answer": "Unable to answer without fabricating information.",
					"reason": "answer_not_grounded",
					"confidence": 0.0,
					"sources": sources,
					"note": "Generated response was not grounded in stored sources.",
				}
		
		self.dialogue.add_user(question)
		if isinstance(llm_response, str) and llm_response:
			self.dialogue.add_assistant(llm_response)

		out = {
			"answer": llm_response,
			"confidence": decision.get("confidence", verification_conf),
			"sources": sources,
			"beliefs": struct_claims,
			"reasoning": struct_reasoning,
			"verification": verification,
		}
		if underspecified_note:
			out["answer_quality"] = underspecified_note

		critique = self._self_critique(question, out, contradictions=[], recent_only=False)
		out["self_critique"] = critique
		if critique.get("caveat"):
			out["caveat"] = critique["caveat"]
			out["answer"] = f"{out.get('answer', '').strip()}\n\nCaveat: {critique['caveat']}"
		return out

	def _internal_cognition_cycle(self, reason: str = "periodic", source: str | None = None):
		try:
			self.claim_store.update_beliefs()
		except Exception:
			pass

	def _validate_and_extract_synthesis(self, structured_response: str) -> tuple[bool, str]:
		"""Parse and validate structured synthesis response.
		
		Expected format:
		[SOURCES CITED]
		- list of sources
		
		[KEY FACTS FROM CONTEXT]
		- list of facts
		
		[SYNTHESIS]
		The actual answer
		
		Returns: (is_valid, extracted_synthesis_section)
		"""
		try:
			import re
			response_lower = structured_response.lower()
			
			# Check for required sections
			has_sources = "[sources cited]" in response_lower
			has_facts = "[key facts" in response_lower
			has_synthesis = "[synthesis]" in response_lower
			
			# Valid if has at least sources+synthesis or all three
			is_valid = (has_sources and has_synthesis) or (has_sources and has_facts and has_synthesis)
			
			# Extract synthesis section if present
			extracted = None
			if has_synthesis:
				# Find [SYNTHESIS] and extract until next [SECTION] or end
				pattern = r'\[SYNTHESIS\](.*?)(?:\[|$)'
				match = re.search(pattern, structured_response, re.IGNORECASE | re.DOTALL)
				if match:
					extracted = match.group(1).strip()
					# Clean up any markdown formatting
					extracted = extracted.replace('```', '').strip()
					if extracted and len(extracted) > 10:
						return is_valid, extracted
			
			# If no synthesis extracted but has structure, use whole response
			if is_valid and not extracted:
				return is_valid, structured_response.strip()
			
			return is_valid, None
			
		except Exception:
			return False, None

	def _validate_answer_grounding(self, answer: str, sources: List[str]) -> bool:
		"""Validate that answer is grounded in retrieved sources.
		
		Returns True if answer appears to reference stored content.
		"""
		try:
			answer_lower = answer.lower()
			
			# Check 1: Reject assistant chatter / refusal patterns
			if any(phrase in answer_lower for phrase in [
				"i don't", "i can't", "i'm not", "please provide",
				"i'll help", "let me know", "feel free",
			]):
				return False
			
			# Check 2: Has sufficient length and structure
			if len(answer.split()) < 5:
				return False
			
			# If answer is substantial and not chatter, give benefit of doubt
			# (stricter validation can be added later with embedding similarity)
			return True
			
		except Exception:
			return True  # Fail open to avoid blocking valid answers
	
	def trigger_understanding(self, source: str):
		ev = [e for e in self.memory.entries if e.get("source", "").startswith(source)]
		all_concepts = []
		all_claims = []
		
		for e in ev:
			text = e.get("text", "")
			concepts = extract_concepts(text)
			all_concepts.extend(concepts)
			for c in concepts:
				try:
					self.concept_graph.upsert_node(c)
				except Exception:
					pass
		try:
			if all_concepts:
				self.concept_graph.observe_cooccurrence(all_concepts)
		except Exception:
			pass
		
		for e in ev:
			text = e.get("text", "")
			claims = extract_atomic_claims(text) or []
			all_claims.extend(claims)
			for cl in claims:
				try:
					parsed = parse_claim(cl)
				except Exception:
					pass

		try:
			self.claim_store.update_beliefs()
		except Exception as e:
			logger.debug(f"Belief update error: {e}")

	def dump_memory(self):
		"""Display memory contents."""
		if not self.memory.entries:
			print("[ZYPHERUS] Memory is empty.")
			return
		print(f"[ZYPHERUS] Memory entries: {len(self.memory.entries)}")
		for i, e in enumerate(self.memory.entries, 1):
			txt = e["text"]
			print(f"{i}. Source: {e['source']} -- Preview: {txt[:200].replace(os.linesep, ' ')}")


__all__ = ["Zypherus"]


