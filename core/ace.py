"""ACE orchestrator."""

from __future__ import annotations

import os
import json
import re
import time
from typing import Any, Dict, List, Optional

from ace.beliefs.claims import ClaimStore
from ace.concepts.graph import ConceptGraph
from ace.core.decision import DecisionPolicy
from ace.core.dialogue import DialogueManager
from ace.core.embedding import EmbeddingModule
from ace.core.question import QuestionInterpreter
from ace.extraction.chunking import chunk_text
from ace.extraction.claims import extract_atomic_claims
from ace.extraction.concepts import extract_concepts
from ace.extraction.parsing import parse_claim
from ace.inference.reasoning import ReasoningEngine
from ace.llm.distiller import Distiller
from ace.llm.renderer import LLMRenderer
from ace.memory.index import KnowledgeIndex
from ace.utils.text import answer_shape_issue


class ACE:
	def __init__(self):
		print("\n--- ACE ONLINE ---")

		self.embedder = EmbeddingModule()
		self.memory = KnowledgeIndex()
		self.dialogue = DialogueManager(max_turns=6)
		self.llm = LLMRenderer()
		self.distiller = Distiller(self.llm)

		self.claim_store = ClaimStore()
		self.concept_graph = ConceptGraph()
		self.struct_reasoner = ReasoningEngine(self.claim_store)
		self.decision_policy = DecisionPolicy()
		self.question_interpreter = QuestionInterpreter()

		try:
			self.memory.load()
		except Exception:
			pass
		try:
			self.claim_store.load()
		except Exception:
			pass

	def ingest_url(self, url):
		print("[ACE] ingest_url is deprecated. ACE accepts documents, not URLs.")
		print("Use a Fetcher to obtain text and call ingest_document(source, text, tables).")

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
			  max_files = 200
			  ingested = 0
			  for root, _dirs, files in os.walk(p):
				  for fname in files:
					  if ingested >= max_files:
						  print(f"[ACE] Directory ingestion capped at {max_files} files.")
						  return
					  ext = os.path.splitext(fname)[1].lower()
					  if ext not in allowed_ext:
						  continue
					  full_path = os.path.join(root, fname)
					  rel_source = os.path.relpath(full_path, p)
					  self.ingest_file(full_path, source=rel_source, encoding=encoding)
					  ingested += 1
			  if ingested == 0:
				  print("[ACE] No ingestible files found in directory.")
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

	def ingest_document(self, source: str, text: str, tables: Optional[List[Any]] = None):
		print(f"[ACE] Ingesting document: {source}")

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
			for idx, c in enumerate(chunk_text(text, chunk_size=200)):
				chunks.append((f"{source}#text:chunk{idx}", c))

		if tables:
			table_texts = []
			for df in tables:
				try:
					table_texts.append(df.head(10).to_csv(index=False))
				except Exception:
					table_texts.append(str(df.head(10)))
			summary_text = self.distiller.extract_summary("\n\n".join(table_texts))
		else:
			summary_text = self.distiller.extract_summary(text)

		emb = self.embedder.embed(summary_text)
		self.memory.add(emb, summary_text, source, mtype="summary")

		try:
			facts = self.distiller.extract_facts(summary_text, source)
			for i, f in enumerate(facts):
				femb = self.embedder.embed(f)
				idx = self.memory.add(femb, f, f"{source}#fact{i}", mtype="fact")
				ents = self.distiller.extract_entities(f)
				if idx is not None and ents:
					self.memory.add_entities(idx, ents)
		except Exception:
			pass

		stored = 0
		for src, txt in chunks:
			try:
				emb = self.embedder.embed(txt)
				idx = self.memory.add(emb, txt, src, mtype="evidence")
				try:
					ents = self.distiller.extract_entities(txt)
					if idx is not None and ents:
						self.memory.add_entities(idx, ents)
				except Exception:
					pass
				stored += 1
			except Exception:
				continue

		print(f"[ACE] Stored {stored} evidence chunks from {source}.")
		try:
			self.memory.save()
		except Exception:
			pass

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
		print("[ACE] Rebuilding beliefs from evidence...")
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

	def answer(self, question: str):
		try:
			self._internal_cognition_cycle(reason="pre_answer")
		except Exception:
			pass

		try:
			question_info = self.question_interpreter.interpret(question)
		except Exception:
			question_info = {"intent": "factual", "requires_high_confidence": False}

		query_emb = self.embedder.embed(question)
		top_k = 40 if question_info.get("requires_high_confidence") else 30
		min_score = 0.02 if question_info.get("requires_high_confidence") else 0.01
		candidates = self.memory.search(query_emb, top_k=top_k, min_score=min_score, query_text=question)

		if not candidates:
			return {
				"answer": "I don’t have relevant knowledge yet. Please ingest a document first.",
				"confidence": 0.0,
				"sources": [],
			}

		try:
			q_ents = self.distiller.extract_entities(question)
			if q_ents:
				def _norm_ent(ent: str) -> str:
					return re.sub(r"[^a-z0-9 ]+", "", (ent or "").lower()).strip()

				q_ents_norm = [_norm_ent(e) for e in q_ents]
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

		sources = [e.get("source") for _, e in top_for_context]

		intent = (question_info.get("intent") or "factual")
		try:
			max_claim_conf = max((float(c.get("confidence", 0.0) or 0.0) for c in (struct_claims or [])), default=0.0)
		except Exception:
			max_claim_conf = 0.0

		if intent in {"define", "compare"} and max_claim_conf < 0.70:
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

		llm_response = self.llm.answer(question, top_for_context, convo_history=convo_hist, reasoning=None)

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
			verification = {"confidence": float(max_claim_conf), "unsupported_claims": [], "status": "symbolic"}
		else:
			verification = self.llm.verify_answer(llm_response, top_for_context)

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

		if decision.get("action") == "clarify":
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

		if decision.get("action") == "dispute":
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
		return out

	def _internal_cognition_cycle(self, reason: str = "periodic", source: str | None = None):
		try:
			self.claim_store.update_beliefs()
		except Exception:
			pass

	def trigger_understanding(self, source: str):
		ev = [e for e in self.memory.entries if e.get("source", "").startswith(source)]
		for e in ev:
			text = e.get("text", "")
			concepts = extract_concepts(text)
			for c in concepts:
				try:
					self.concept_graph.upsert_node(c)
				except Exception:
					pass
			try:
				if concepts:
					self.concept_graph.observe_cooccurrence(concepts)
			except Exception:
				pass
			claims = extract_atomic_claims(text)
			for cl in claims:
				try:
					parsed = parse_claim(cl)
					self.claim_store.observe_claim(parsed, e.get("id"))
				except Exception:
					pass

		try:
			self.claim_store.update_beliefs()
		except Exception:
			pass

	def dump_memory(self):
		if not self.memory.entries:
			print("[ACE] Memory is empty.")
			return
		print(f"[ACE] Memory entries: {len(self.memory.entries)}")
		for i, e in enumerate(self.memory.entries, 1):
			txt = e["text"]
			print(f"{i}. Source: {e['source']} -- Preview: {txt[:200].replace(os.linesep, ' ')}")


__all__ = ["ACE"]

